"""Step 3 - Evaluate on the held-out TEST split and save proof for the demo."""
import json, shutil
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()


def main():
    model = YOLO(str(ROOT / "models" / "best.pt"))
    r = model.val(data=str(ROOT / "data.yaml"), split="test", imgsz=768, plots=True,
                  device=0 if torch.cuda.is_available() else "cpu",
                  project=str(ROOT / "runs"), name="eval", exist_ok=True)
    p, rec = float(r.box.mp), float(r.box.mr)
    metrics = {
        "precision": p, "recall": rec,
        "f1": 2 * p * rec / (p + rec + 1e-9),
        "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
        "test_images": len(list((ROOT / "datasets/ships/images/test").glob("*.jpg"))),
    }
    res = ROOT / "results"; res.mkdir(exist_ok=True)
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ev = ROOT / "runs" / "eval"
    for f in ["BoxPR_curve.png", "BoxF1_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg"]:
        if (ev / f).exists():
            shutil.copy(ev / f, res / f)
    tr = ROOT / "runs" / "ship"
    if (tr / "results.png").exists():
        shutil.copy(tr / "results.png", res / "training_curves.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
