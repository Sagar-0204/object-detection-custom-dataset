import cv2
from ultralytics import YOLO
import time

# Load ONNX model
model = YOLO("runs/detect/train5/weights/best.onnx", task="detect")

cv2.setUseOptimized(True)
cv2.setNumThreads(16)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cur_frame = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 640))

    # Inference (CPU)
    results = model(frame, verbose=False, device="cpu")

    # Draw results
    annotated_frame = results[0].plot()

    # FPS calculation
    cur_frame += 1
    elapsed_time = time.time() - start_time
    fps = cur_frame / elapsed_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Display
    cv2.imshow("Real-Time Detection (ONNX)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
