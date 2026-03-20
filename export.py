from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train5/weights/best.pt")

# Export to ONNX
model.export(
    format="onnx",
    opset=12,        # stable ONNX version
    simplify=True,   # optimize graph
    dynamic=False    # fixed input size (faster on CPU)
)

print("✅ Model exported to ONNX successfully")