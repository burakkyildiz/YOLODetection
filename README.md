# 🎯 Real-Time Webcam Object Detection with YOLO

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)

A lightweight and fast Python application for **real-time object detection** using **Ultralytics YOLO** and **OpenCV**. It captures live video from your webcam, detects visible objects, overlays bounding boxes with labels on screen, and prints detection summaries directly to the terminal console.

---

## ✨ Features

- ⚡ **Real-Time Detection:** Live inference using state-of-the-art YOLO models (YOLO11 / YOLOv8).
- 📊 **Terminal Logging:** Real-time console summaries showing detected objects and counts (e.g., `person x1, laptop x1, cell phone x1`).
- 🎨 **Visual HUD Overlay:** On-screen FPS counter, total detected object count, and annotated bounding boxes.
- ⚙️ **Highly Configurable:** Command-line options to customize models, confidence threshold, camera source, and resolution.
- 🛡️ **Cross-Platform:** Works seamlessly on macOS (Apple Silicon / Intel), Windows, and Linux.

---

## 📋 Requirements

- **Python 3.8+**
- A working webcam (built-in or USB external)

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/YoloDetection.git
   cd YoloDetection
   ```

2. **Create and activate a virtual environment:**
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Quick Start
Run the detector with default settings (uses `yolo11n.pt` and default camera `0`):

```bash
python main.py
```

> 💡 **Note:** On the first run, the YOLO model weights will be downloaded automatically.

---

### 🛠️ CLI Options & Customization

You can customize the detection parameters using command-line arguments:

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--model` | `str` | `yolo11n.pt` | YOLO model name or path to custom `.pt` file (e.g., `yolov8n.pt`, `yolo11s.pt`) |
| `--camera` | `int` | `0` | Camera device index (use `1` or `2` for external webcams) |
| `--conf` | `float` | `0.45` | Minimum confidence threshold for detections (between `0.0` and `1.0`) |
| `--width` | `int` | `1280` | Webcam capture width |
| `--height` | `int` | `720` | Webcam capture height |

#### Examples

- **Use YOLOv8 instead of YOLO11:**
  ```bash
  python main.py --model yolov8n.pt
  ```

- **Set higher confidence threshold to filter weak detections:**
  ```bash
  python main.py --conf 0.65
  ```

- **Use an external webcam (Camera Index 1) with 1080p resolution:**
  ```bash
  python main.py --camera 1 --width 1920 --height 1080
  ```

- **Use a custom-trained model:**
  ```bash
  python main.py --model path/to/best.pt
  ```

---

## ⌨️ Controls

| Key | Action |
| :---: | :--- |
| **`q`** or **`ESC`** | Safely stop detection and close the webcam window |

---

## 📁 Project Structure

```text
YoloDetection/
├── main.py              # Main application script
├── requirements.txt     # Python dependencies (ultralytics, opencv-python)
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
```

---

## 🍎 macOS Camera Permission Note

If you are running this on macOS, ensure that your **Terminal** or **IDE (VS Code, PyCharm, etc.)** has permission to access your camera:
- Navigate to **System Settings > Privacy & Security > Camera** and toggle access **ON**.
