"""Step 2 - Fine-tune YOLOv8 on the ship dataset (single class: ship).
Examples:   python train.py --epochs 50 --batch 8
            python train.py --model yolov8m.pt --epochs 50 --batch 4     (bigger, usually more accurate)
"""
import argparse, shutil
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=768)
    a = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Training on:", "GPU" if device == 0 else "CPU (slow - consider Google Colab GPU)")
    model = YOLO(a.model)
    model.train(
        data=str(ROOT / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=device, project=str(ROOT / "runs"), name="ship", exist_ok=True,
        patience=15, workers=2, cos_lr=True, close_mosaic=10,
        # satellite imagery: ships can point in any direction, scales vary a lot
        flipud=0.5, fliplr=0.5, degrees=45, scale=0.6, mosaic=1.0, mixup=0.1, hsv_v=0.3,
    )
    best = ROOT / "runs" / "ship" / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "best.pt")
    print("Saved models/best.pt  -> next: python evaluate.py")


if __name__ == "__main__":
    main()
