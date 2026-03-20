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

Due to size constraints, the dataset is not included in this repository.
Dataset link: [[Google Drive Link]](https://drive.google.com/drive/folders/1VvcouMpLDoPu3CXYi-q-HiKedxlysrTe?usp=drive_link)

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

## Performance Metrics

### Training Performance:
- mAP@50: ~73.3%

### Deployment Performance:
- Inference Speed: ~15 FPS (CPU)
- RAM Usage: ~1 GB

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
```
## How to run

### Step 1: Create Virtual Environment
```python

python3 -m venv cv_env  
source cv_env/bin/activate  
```
### Step 2: Install Dependencies
```python

pip install -r requirements.txt  
```
### Step 3: Run
```python

python3 custom_yolo.py  
```
