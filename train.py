from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="final_v5/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    patience=30,

    # augmentations (optimized)
    degrees=10,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.5,
    mixup=0.1,

    cache="disk"
)