from ultralytics import YOLO

# Load a pretrained YOLOv10n model
model = YOLO("yolov10n.pt")

# Perform object detection on an image
results = model("/home/nkumar/Downloads/boxing.mp4")

# Display the results
results[0].show()