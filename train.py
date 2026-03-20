from ultralytics import YOLO

# load model
model = YOLO("yolov8n.pt")

# train
model.train(
    data="o.v7i.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    device=0  # GPU (RTX 3050)
)

