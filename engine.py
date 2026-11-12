"""Model side: YOLO loading, multi-scale + tiled inference, zero-shot ship-type classifier."""
import functools
import time
from pathlib import Path

import cv2
import numpy as np

import ship_utils as U

ROOT = Path(__file__).parent.resolve()
CUSTOM = ROOT / "models" / "best.pt"


def custom_available():
    return CUSTOM.exists()


def _device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=None)
def load_yolo(which):
    from ultralytics import YOLO
    return YOLO(str(CUSTOM) if which == "custom" else "yolov8s.pt")


def run_yolo(model, bgr, imgsz, conf, iou, classes=None):
    r = model.predict(bgr, imgsz=imgsz, conf=conf, iou=iou, classes=classes,
                      device=_device(), verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy().astype(float), r.boxes.conf.cpu().numpy().astype(float)


def tile_positions(L, T, stride):
    if L <= T:
        return [0]
    pos = list(range(0, L - T + 1, stride))
    if pos[-1] != L - T:
        pos.append(L - T)
    return pos


def detect_image(img_rgb, which, conf, iou, enhanced):
    """Returns dict(boxes, confs, ms, model_label, passes)."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    model = load_yolo(which)
    classes = [8] if which == "coco" else None
    base = 768 if which == "custom" else 640
    t0 = time.time()
    all_b, all_c, passes = [], [], []

    def add(b, c, dx=0, dy=0):
        if len(b):
            all_b.append(b + np.array([dx, dy, dx, dy], float))
            all_c.append(c)

    add(*run_yolo(model, bgr, base, conf, iou, classes))
    passes.append(f"full image @{base}px")
    if enhanced:
        sizes = [1024] if which == "custom" else [960]
        if max(H, W) > 1400:
            sizes.append(1280)
        for s in sizes:
            add(*run_yolo(model, bgr, s, conf, iou, classes))
            passes.append(f"full image @{s}px")
        if which == "custom" and max(H, W) > 1000:
            T, stride = 768, 576
            for y0 in tile_positions(H, T, stride):
                for x0 in tile_positions(W, T, stride):
                    add(*run_yolo(model, bgr[y0:y0 + T, x0:x0 + T], T, conf, iou, classes), x0, y0)
            passes.append("overlapping 768px tiles")

    if all_b:
        b, c = np.concatenate(all_b), np.concatenate(all_c)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        keep = U.nms(b, c, iou)
        b, c = b[keep], c[keep]
    else:
        b, c = np.zeros((0, 4)), np.zeros(0)
    label = ("Custom YOLOv8 (satellite / aerial)" if which == "custom"
             else "YOLOv8 COCO 'boat' (ground-level photos)")
    return {"boxes": b, "confs": c, "ms": (time.time() - t0) * 1000, "model_label": label, "passes": passes}


# ------------------------------------------------------------ ship-type classifier
@functools.lru_cache(maxsize=1)
def _load_clip():
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(name).to(dev).eval()
        return model, CLIPProcessor.from_pretrained(name), dev
    except Exception as e:  # noqa
        print("CLIP classifier not available:", e)
        return None


def clip_available():
    return _load_clip() is not None


def classify_crops(crops_rgb):
    """Zero-shot ship type probabilities, shape (n, len(SHIP_TYPES)); None if unavailable."""
    c = _load_clip()
    if c is None or not crops_rgb:
        return None
    import torch
    from PIL import Image
    model, proc, dev = c
    imgs = [Image.fromarray(x) for x in crops_rgb if x.size]
    inputs = proc(text=U.TYPE_PROMPTS, images=imgs, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
    return logits.softmax(dim=1).cpu().numpy()
