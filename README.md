# 🏗️ YOLOv8 - Door and Window Detection from Floor Plans

This project uses the [YOLOv8](https://github.com/ultralytics/ultralytics) object detection model to identify **doors** and **windows** in architectural floor plan images. It can be useful for automating CAD parsing, blueprint analysis, or BIM data extraction.

---

## 📂 Project Structure
- ├── data/
- │ ├── images/ # Training & test images
- │ └── labels/ # YOLO-format annotations
- ├── best.pt # yolov8 model trained file
- ├── main.py # Python script
- ├── requirements.txt 
- └── README.md # This file

## ✅ Requirements

Install dependencies:

```bash
pip install ultralytics opencv-python matplotlib
```
Make sure you're using Python 3.8+.

## 📝 Dataset Format
Annotation format: YOLO format (1 label per line: class x_center y_center width height)

- Classes:
  - 0: door
  - 1: window
## 🔧 Dataset Config File (yolov8_config.yaml)
- path: ./data
- train: images/train
- val: images/val
- test: images/test

- names:
    - 0: door
    - 1: window
 
 ## 🚀 Training
Train YOLOv8 on your dataset:
```
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # You can choose yolov8n, yolov8s, yolov8m, etc.
model.train(data="yolov8_config.yaml", epochs=100, imgsz=640, batch=16)
```

## 🔍 Inference
```
model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict(source="data/images/test/floor1.png", conf=0.3, save=True) 
```

## 📊 Evaluation
```
model.val()
```

## 🖼️ Output Example
- After prediction, detected doors and windows will be marked with bounding boxes:
  - 🟩 Green box for doors
  - 🟦 Blue box for windows

## ✍️ Notes
- Ensure floor plan images are clean and high-resolution.
- Label data carefully for best results.
- Try different YOLOv8 models (nano, small, medium) depending on speed vs accuracy.

