from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('./runs/detect/train/weights/best.pt')


model.predict(source='./datasets/images/test', save=True, conf=0.25)