# 🚢 Maritime Ship Detection (YOLOv8 + Streamlit)

Pipeline: Airbus CSV (RLE masks) -> YOLO boxes -> train YOLOv8 -> evaluate -> web app.

## 0. IMPORTANT - Kaggle key
Create a NEW key: kaggle.com -> Settings -> API -> "Create New Token" (the old one was shared in chat, so expire it).
Put the new file at `C:\Users\<YourName>\.kaggle\kaggle.json`.
Also open https://www.kaggle.com/c/airbus-ship-detection/rules and click **"I Understand and Accept"** (otherwise downloads fail with 403).

## 1. Setup in VS Code (Windows)
1. Install Python 3.10 or 3.11 from python.org (tick "Add Python to PATH"). Open this folder in VS Code (File -> Open Folder).
2. Open terminal in VS Code (Ctrl + `) and run:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
(If activate is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.)
NVIDIA GPU users: first install CUDA PyTorch from pytorch.org, then the requirements.

## 2. Build dataset (downloads ~6,000 sampled images)
```
python prepare_data.py
```
Smaller/faster test: `python prepare_data.py --n_ship 800 --n_empty 200`
Already have the Airbus `train_v2` folder? `python prepare_data.py --images_dir D:\path\train_v2`

## 3. Train (use Google Colab GPU if you have no NVIDIA GPU)
```
python train.py --epochs 40
```
CPU training is very slow. On Colab (Runtime -> T4 GPU): upload this folder + kaggle.json, `!pip install -r requirements.txt`,
run steps 2-4 there, then download `models/best.pt` and `results/` back into this folder.

## 4. Evaluate (creates the proof shown in the app)
```
python evaluate.py
```

## 5. Run the app
```
streamlit run app.py
```
Opens at http://localhost:8501 - upload images, see boxes, confidence, a written description, and the "Model performance" tab.

## Notes for your viva
- Class is single "ship". Size classes (cargo/tanker vs fishing) in the app are shape/size heuristics, not a classifier.
- YOLOv8 is anchor-free, so no anchor tuning; elongated ships are handled by direct box regression + rotation/flip augmentation.
- Metrics: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 on a held-out test split. Plain "accuracy" is not meaningful for detection.
- The custom model is trained on SATELLITE images. For normal Google side-view photos the app uses the COCO "boat" class automatically (Auto mode).
