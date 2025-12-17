# [Prototype] Human Tracking Camera System
---
## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [System Design](#system-design)
6. [Model Architecture](#model-architecture)
7. [Dataset](#dataset)
8. [Results and Evaluation](#results-and-evaluation)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)
---

## Introduction
<p align="justify">
This project presents a <b>human tracking camera system</b> based on human detection and face recognition.  
The system integrates <b>Haar Cascade</b> for face detection, <b>LBPH (Local Binary Pattern Histogram)</b> for face recognition, <b>YOLOv8n</b> for human detection, and <b>DeepSort</b> for ID labeling and tracking.  

The camera is capable of tracking a recognized person in real time using a dual servo motor controlled by a Raspberry Pi.
</p>

## Features
- Real-time face detection and recognition  
- Real-time human detection and tracking  
- Threshold-based confidence score and histogram distance  
- Dual servo motor control for camera tracking  
- Integrated with Raspberry Pi  

## Installation
- Clone this repository
- Download and train the human detection dataset from Roboflow:  [YOLOv8]
  (https://app.roboflow.com/skripsi-cqegk/yolov8-mjqbp/1) 
  Train the model using `train_model_yolo.py`
- Or download the pretrained YOLOv8n model from Hugging Face: [panjigema/YOLOv8n_Human_Class] 
  (https://huggingface.co/panjigema/YOLOv8n_Human_Class)
- Run `kumpul_data_wajah.py` to collect face images
- Train the face recognition model using `training_face_model.py`
- Connect Raspberry Pi to the internet and run `pwm_servo.py`
- Run `face_recognition.py` to start the system

## Usage
- Register face images using the face data collection script
- When a user appears in front of the camera, the system:
  - Detects and recognizes the face using Haar Cascade and LBPH
  - Detects humans using YOLOv8n
  - Assigns tracking IDs using DeepSort
  - Tracks the recognized person using a servo motor

## System Design
The system consists of hardware and software components designed to operate together through Raspberry Pi URL-based communication.  
The tracking mechanism is driven by an <b>MG996 servo motor</b>, controlled by a Raspberry Pi, and integrated with a webcam to capture video input.

- The physical design includes a 3D-printed prototype
- All wiring and layout are integrated into a compact and functional housing

### Prototype Views

- Device Prototype  
<p align="center">
  <img width="480" height="360" src="https://github.com/ice-bear14/Human_tracking/blob/main/assets/imp1.jpg">
</p>

- Electrical Implementation  
<p align="center">
  <img width="480" height="360" src="https://github.com/ice-bear14/Human_tracking/blob/main/assets/imp2.jpg">
</p>

- Wiring Schematic of the Human Tracking Camera  
<p align="center">
  <img width="480" height="360" src="https://github.com/ice-bear14/Human_tracking/blob/main/assets/imp3.png">
</p>

- Exploded View, 3D Assembly, and Technical Drawing  
<p align="center">
  <img width="480" height="360" src="https://github.com/ice-bear14/Human_tracking/blob/main/assets/imp4.png">
  <img width="480" height="360" src="https://github.com/ice-bear14/Human_tracking/blob/main/assets/imp5.png">
  <img width="480" height="360" src="https://github.com/ice-bear14/Human_tracking/blob/main/assets/imp6.png">
</p>

## Model Architecture
- Face Detection: Haar Cascade Classifier  
- Face Recognition: Local Binary Pattern Histogram (LBPH)  
- Human Detection: YOLOv8n  
- Tracking and ID Labeling: DeepSort  

## Dataset
- Human Detection Dataset:  
  https://app.roboflow.com/skripsi-cqegk/yolov8-mjqbp/1
- Face Recognition Dataset:  
  Custom face images collected manually using `kumpul_data_wajah.py`

## Results and Evaluation
- Human Detection Performance:
  - Recall: 85%
  - Precision: 80%
  - Accuracy: 70%
  - F1-Score: 82.5%
- Face Recognition:
  - Best LBPH distance threshold: 50
- Tracking Performance:
  - FPS range during tracking: approximately 10–17 FPS

## Future Work
- Replace CIoU loss in YOLOv8n with <b>WIoU (Wise-IoU)</b> for better bounding box accuracy
- Use <b>SGD optimizer</b> to improve training performance
- Apply <b>CLAHE (Contrast Limited Adaptive Histogram Equalization)</b> to enhance face image quality before training

## Contributing
Contributions are welcome.  
Please open issues or submit pull requests to collaborate or improve the system.

## License
This project is licensed under the MIT License - see the [LICENSE] 
(LICENSE)  
file for details.

## Contact
Author: Panji Gema Romadan  
Email: panjigemaramadan@gmail.com  
GitHub: https://github.com/ice-bear14  
LinkedIn: https://www.linkedin.com/in/panji-gema-romadan

