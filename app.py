import gradio as gr
from PIL import Image
import os
import uuid
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # ensure best.pt is in the same directory

# Detection function
def detect_objects(image):
    temp_filename = f"input_{uuid.uuid4().hex[:8]}.jpg"
    image.save(temp_filename)

    # Run detection
    results = model(temp_filename)

    # Save the result image
    result_image_path = f"result_{uuid.uuid4().hex[:8]}.jpg"
    results[0].save(filename=result_image_path)

    # Load and return result image
    result_image = Image.open(result_image_path)

    # Cleanup
    os.remove(temp_filename)
    return result_image

# Gradio Interface
title = "Blueprint Door & Window Detector"
description = "Upload a construction blueprint image to detect doors and windows using a custom YOLOv8 model."

gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title=title,
    description=description,
    allow_flagging="never"
).launch()
