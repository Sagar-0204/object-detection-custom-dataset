# Object Detection - Custom Dataset Training & Deployment

## Overview
This project focuses on creating a **custom object detection system** for identifying objects commonly found in collapsed building scenarios.
The final system performs real-time detection using a CPU-based ONNX model.

## Dataset
A custom dataset was created using publicly available images and manually annotated.

### Dataset Details:
- Total Images: **2170**
- Annotation Tool: **Roboflow**
- Task Type: **Object Detection (Bounding Boxes)**

### Classes:

The dataset includes the following object classes:

- reinforcement bars  
- rubble *(bricks, tiles, stones grouped)*  
- window  
- door  
- cylinder  
- switch  
- cupboard (almirah)  
- fire  
- curtains  
- bed  
- fridge  
- fire extinguisher  
- humans  

### Key Notes:
- **Rubble class** combines bricks, tiles, and stones for better generalization
- Dataset includes both **isolated objects and multi-object scenes**

## Observations

Some misclassifications were observed in real-time (e.g., TV detected as window). This is because of the gap between training data and real-world scenarios.

## Training

The model was trained using **YOLOv8n** on the custom dataset.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640
)

## How to Run

### Step 1: Create Virtual Environment
python3 -m venv cv_env  
source cv_env/bin/activate  

### Step 2: Install Dependencies
pip install -r requirements.txt  

### Step 3: Run
python3 custom_yolo.py  
