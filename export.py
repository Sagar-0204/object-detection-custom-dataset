from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/final5/weights/best.pt")

# Export to ONNX
model.export(
    format="onnx",
    imgsz=640,
    opset=13,        
    simplify=True,   
    dynamic=False 
)

print("Model exported to ONNX")
