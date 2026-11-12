"""
Step 1 - Build a YOLO dataset from the Airbus Ship Detection CSV.

* RLE masks -> bounding boxes (single class "ship")
* Samples ship images + empty-sea images, REUSING anything already in data/raw_images
* Downloads missing images from Kaggle slowly, with back-off when Kaggle says "429 Too Many Requests"
* Safe to stop (Ctrl+C) and re-run: downloaded images are never fetched twice
"""
import argparse, random, shutil, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

IMG = 768
ROOT = Path(__file__).parent.resolve()


def rle_to_bbox(rle: str):
    """Airbus RLE (column-major, 1-indexed) -> (x1, y1, x2, y2) in pixels."""
    s = np.asarray(rle.split(), dtype=np.int64)
    starts, lengths = s[0::2] - 1, s[1::2]
    mask = np.zeros(IMG * IMG, dtype=np.uint8)
    for st, ln in zip(starts, lengths):
        mask[st: st + ln] = 1
    ys, xs = np.where(mask.reshape(IMG, IMG).T)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def download_one(api, comp, name, out_dir):
    dst = out_dir / name
    if dst.exists():
        return True
    err = None
    for attempt in range(8):
        try:
            api.competition_download_file(comp, f"train_v2/{name}", path=str(out_dir), quiet=True)
        except Exception as e:  # noqa
            err = e
            # 429 = rate limited -> wait longer each time (5, 10, 20, 40, 80, 120 s ...)
            time.sleep(min(120, 5 * 2 ** attempt) if "429" in str(e) else 3)
            continue
        # Kaggle may save under another name/folder or as a zip: find it
        for p in [p for p in out_dir.rglob(f"*{name}*") if p.is_file() and p != dst]:
            if p.suffix == ".zip":
                with zipfile.ZipFile(p) as zf:
                    for m in zf.namelist():
                        if m.endswith(name):
                            dst.write_bytes(zf.read(m))
                p.unlink()
            else:
                shutil.move(str(p), str(dst))
        if dst.exists():
            time.sleep(0.3)           # be gentle with the API
            return True
        err = "downloaded, but file not found"
    print(f"  failed: {name} ({str(err)[:90]})")
    return False


def pick(ids, have, n, rng):
    """Reuse images already on disk first, then top up with a seeded shuffle."""
    ids = list(ids)
    rng.shuffle(ids)
    got = [i for i in ids if i in have][:n]
    rest = [i for i in ids if i not in set(got)]
    return got + rest[: max(0, n - len(got))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "train_ship_segmentations_v2.csv"))
    ap.add_argument("--images_dir", default=None, help="Folder that already contains the Airbus train_v2 jpgs")
    ap.add_argument("--n_ship", type=int, default=3000, help="images that contain ships")
    ap.add_argument("--n_empty", type=int, default=600, help="images without ships (negatives)")
    ap.add_argument("--workers", type=int, default=1, help="keep at 1-2 or Kaggle rate-limits you")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    df = pd.read_csv(a.csv)
    has_ship = df.dropna(subset=["EncodedPixels"])
    ship_ids = has_ship.ImageId.unique().tolist()
    empty_ids = df[df.EncodedPixels.isna()].ImageId.unique().tolist()
    print(f"CSV: {df.ImageId.nunique()} images | {len(ship_ids)} with ships | {len(has_ship)} ship masks")

    raw = ROOT / "data" / "raw_images"
    src = Path(a.images_dir) if a.images_dir else raw
    have = {p.name for p in src.glob("*.jpg")} if src.exists() else set()
    print(f"Already on disk: {len(have)} images")
    ids = pick(ship_ids, have, a.n_ship, rng) + pick(empty_ids, have, a.n_empty, rng)
    rng.shuffle(ids)

    if not a.images_dir:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        raw.mkdir(parents=True, exist_ok=True)
        todo = [i for i in ids if i not in have]
        print(f"Downloading {len(todo)} missing images (Ctrl+C is safe; re-run to resume)...")
        with ThreadPoolExecutor(a.workers) as ex:
            list(tqdm(ex.map(lambda n: download_one(api, "airbus-ship-detection", n, raw), todo), total=len(todo)))

    ids = [i for i in ids if (src / i).exists()]
    print(f"{len(ids)} images available for the dataset")

    boxes, idset = {}, set(ids)
    for img_id, rle in tqdm(zip(has_ship.ImageId, has_ship.EncodedPixels), total=len(has_ship), desc="RLE->boxes"):
        if img_id in idset:
            b = rle_to_bbox(rle)
            if b:
                boxes.setdefault(img_id, []).append(b)

    out = ROOT / "datasets" / "ships"
    if out.exists():
        shutil.rmtree(out)
    n = len(ids)
    split = {"train": ids[: int(.8 * n)], "val": ids[int(.8 * n): int(.9 * n)], "test": ids[int(.9 * n):]}
    for sp, lst in split.items():
        (out / "images" / sp).mkdir(parents=True)
        (out / "labels" / sp).mkdir(parents=True)
        for img_id in lst:
            shutil.copy(src / img_id, out / "images" / sp / img_id)
            lines = [f"0 {(x1+x2)/2/IMG:.6f} {(y1+y2)/2/IMG:.6f} {(x2-x1)/IMG:.6f} {(y2-y1)/IMG:.6f}"
                     for x1, y1, x2, y2 in boxes.get(img_id, [])]
            (out / "labels" / sp / (Path(img_id).stem + ".txt")).write_text("\n".join(lines))
        print(f"{sp}: {len(lst)} images, {sum(len(boxes.get(i, [])) for i in lst)} ships")

    (ROOT / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n")
    print("Done -> data.yaml written. Next: python train.py")


if __name__ == "__main__":
    main()
