from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

# Load your trained segmentation model
model = YOLO("best.pt")  # or wherever your model is saved

# Path to validation/test image folder
image_folder = "images"  # replace with actual path
output_folder = "predicted_masks"  # to save visual results (optional)
os.makedirs(output_folder, exist_ok=True)

# Valid image formats
valid_extensions = (".jpg", ".jpeg", ".png")
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

# Class name mapping
names = model.names  # Index to class name dictionary

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    results = model(img_path)[0]

    # Load original image
    img = cv2.imread(img_path)

    # Object counters
    class_counts = {"door": 0, "window": 0}

    # Loop through each detection
    for box in results.boxes:
        cls_id = int(box.cls)
        label = names[cls_id]

        if label not in class_counts:
            continue

        class_counts[label] += 1

        # Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw rectangle
        color = (0, 255, 0) if label == "door" else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label (just name and count, no confidence)
        count_label = f"{label} #{class_counts[label]}"
        cv2.putText(img, count_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save output
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, img)

    # Display with matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"{img_name} - Doors: {class_counts['door']} | Windows: {class_counts['window']}")
    plt.show()

    # Print counts
    print(f" {img_name} ➜ Doors: {class_counts['door']}, Windows: {class_counts['window']}")
