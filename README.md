# Shelf Space and Competition Analysis for SIM Cards with AI

Image annotation details with dataset:
```
https://app.roboflow.com/ganeshml/simcard-shelf-space-final/7
```

Code repo:
```
https://github.com/AIML-Model/simcard-shelf-space-cnn.git
```

Team Member:
```
Ganesh Giri
Ashwini Thantri
Aruna H K
Venkatesh Reddy
Kavitha Jindam
```

# Project Descriptiion

## Software Resources:
 - Image Analytics Software:
    - OpenCV: Image processing.
    - YOLO (You Only Look Once): Object detection.
    - Roboflow: Image annotation learning.
      
## Cloud Infrastructure:
 - AWS, Google Cloud, or Microsoft Azure: Scalable storage and compute resources for processing retail images.

## Database Systems:
 - MongoDB: Manage inventory data and shelf positioning.

## Dashboard/Visualization Tools:
 - HTML Application: Insights and analytics visualization.

## Custom Web Application:
 - Built with Flask (backend) and HTML (frontend) for real-time shelf space monitoring.


# Step 1: Data preprocessing:
 - Image filter based on quality of image.
 - Remove duplicate image.

# Step2: Image annotation using roboflow

https://app.roboflow.com/ganeshml/simcard-shelf-space-final/7

<img width="1495" alt="image" src="https://github.com/user-attachments/assets/9304c60d-9cbf-45cf-b5ac-21d0853f1c61">

# Step3: Download the dataset for training.

<img width="1496" alt="image" src="https://github.com/user-attachments/assets/7d776595-916f-41a2-b7d7-23b47b3c19f9">

# Step4: Train the model yousing YOLOv8 premodel.

## Import the lib
```
from ultralytics import YOLO
from IPython.display import display, Image
```
## Download the data set.
```
from roboflow import Roboflow
rf = Roboflow(api_key="xxxxxx")
project = rf.workspace("ganeshml").project("simcard-shelf-space-final")
version = project.version(7)
dataset = version.download("yolov8")
```

## Set the location for dataset and config file.
```
import yaml

with open(f"{dataset.location}/data.yaml", 'r') as f:
    dataset_yaml = yaml.safe_load(f)
dataset_yaml["train"] = "../train/images"
dataset_yaml["val"] = "../valid/images"
dataset_yaml["test"] = "../test/images"
with open(f"{dataset.location}/data.yaml", 'w') as f:
    yaml.dump(dataset_yaml, f)
```

## Train the model with 100 epochs, image size = 800
```
%cd {HOME}

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=100 imgsz=800 plots=True
```


# Step5: Deploy the model Flask API and HTML dashboard.

- Change the mongodb script which data need to store and execute the script to insert the data.
./ mogodb.py

- Execure the application
./ shelf-space-cnn.py
 

<img width="1164" alt="image" src="https://github.com/user-attachments/assets/d3e97f1d-e3e6-410a-9df8-52e4679e3b6e">


<img width="1490" alt="image" src="https://github.com/user-attachments/assets/402814cf-e30f-4c30-a388-de9844211780">


<img width="1496" alt="image" src="https://github.com/user-attachments/assets/2ddf2070-68d3-42e7-a490-656a0dfccedd">


<img width="1477" alt="image" src="https://github.com/user-attachments/assets/84339589-3890-411b-888c-50c54ab8fe9c">

