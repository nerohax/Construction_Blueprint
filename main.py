from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()
model = YOLO("best.pt")  # Make sure this file is in the same directory

@app.get("/")
async def root():
    return {"message": "YOLO API is live!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(image)[0]
    detections = []

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        detections.append({
            "class": results.names[int(cls)],
            "confidence": round(conf, 2),
            "box": [round(x1), round(y1), round(x2), round(y2)]
        })

    return JSONResponse(content={"detections": detections})
