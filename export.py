from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/final5/weights/best.pt")

# Export to ONNX
model.export(
    format="onnx",
    imgsz=640,
    opset=13,        # stable ONNX version
    simplify=True,   # optimize graph
    dynamic=False    # fixed input size (faster on CPU)
)

print("✅ Model exported to ONNX successfully")