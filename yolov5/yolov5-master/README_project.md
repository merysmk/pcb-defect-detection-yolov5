🚀 Intelligent PCB Defect Detection System

An AI-based system for automatic and real-time detection of defects in Printed Circuit Boards (PCB) using deep learning and computer vision.

📌 📖 Overview

PCB quality is critical in industrial environments. Manual inspection is slow, error-prone, and inefficient.

This project introduces an intelligent inspection system that:

Detects PCB defects using AI
Performs real-time analysis via camera
Stores results in a database
Visualizes results using Augmented Reality


🧠 ⚙️ Technologies Used
Python
PyTorch
YOLOv5
OpenCV
Unity (AR)
Vuforia (AR tracking)

🎯 🧪 Features
🔍 AI-Based Detection
YOLOv5 model trained on PCB dataset
Detects 6 types of defects:
missing_hole
mouse_bite
open_circuit
short
spur
spurious_copper

🎥 Real-Time Detection
Camera input via OpenCV
Live detection with bounding boxes
FPS monitoring for performance

🗄️ Database Storage
Stores:
detected defects
confidence scores
images / reports

🧩 Augmented Reality (AR)
QR code scanning
Visualization of defects in real environment
Two modes:
Defect Visualization
Report Visualization

🏗️ ⚙️ System Architecture
Capture PCB image
Run YOLO detection
Store results 
Load data in AR app
Display defects via AR interface

📊 📈 Performance
Good precision & recall
Real-time inference capability
Balanced speed/accuracy (YOLOv5s)

⚠️ Limitations
Small dataset → affects rare defects detection
Sensitive to lighting conditions
Possible confusion between similar defects

🚀 Future Improvements
Larger dataset
Better generalization
Deployment on edge devices
Predictive maintenance extension

▶️ How to Run
pip install -r requirements.txt
python CAM.py
📸 Demo

(Add screenshots here 👇 — je vais te dire lesquelles juste après)

👩‍💻 Author
Meryem Semak
ENSAM Casablanca
