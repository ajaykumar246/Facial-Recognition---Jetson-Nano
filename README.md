
# Jetson Face Recognition (4th_batch)

**Project overview**
This repository contains a face-recognition pipeline prepared for deployment on NVIDIA Jetson devices (project files are inside `4th_batch/`).  
It includes training, testing, conversion to TensorFlow Lite, and a small Flask server to expose predictions.

**What's inside**
```
4th_batch/
  ├─ train_face_recognition.py      # Train a MobileNetV2-based classifier on a directory of face images
  ├─ test_face_recognition.py       # Live webcam inference + visualization (face_check function)
  ├─ tester.py                      # Simple wrapper that calls face_check()
  ├─ convert.py                     # Convert saved Keras model (.h5) to .tflite
  ├─ server.py                      # Minimal Flask server exposing /predict -> face_check()
  ├─ face_recognition_model.h5      # (Provided) trained Keras model file
  └─ __pycache__/                   # compiled python cache files
```

---

## Quick start (assumes a Jetson device or any Linux machine with a webcam)

> Warning: TensorFlow builds and versions on Jetson devices vary by JetPack version. See NVIDIA's official JetPack and TensorFlow installation guides for the Jetson platform. citeturn0search2turn0search1

### 1) Clone / extract
If you uploaded a ZIP, extract it. If cloning from a repo:
```bash
git clone <your-repo-url>
cd <repo-root>/4th_batch
```

### 2) System prerequisites (Jetson-specific notes)
- Flash your Jetson with an appropriate JetPack (Jetson Linux) image using SDK Manager. JetPack provides the GPU drivers, CUDA, cuDNN and TensorRT required for acceleration. citeturn0search2  
- Install a Jetson-compatible TensorFlow package (NVIDIA provides pip wheels and install instructions for supported JetPack releases). Check the official guide for instructions and compatible TF versions for your JetPack. citeturn0search1turn0search12

If you are not on Jetson (desktop Linux), install Python3, pip and create a venv:
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
python3 -m venv venv
source venv/bin/activate
```

### 3) Python dependencies
Install the packages required by the scripts:
```bash
pip install --upgrade pip
pip install opencv-python numpy flask tensorflow
```
**Jetson note:** On Jetson, install the NVIDIA-provided TensorFlow wheel for the correct JetPack / Python combination (see NVIDIA docs). Attempting to `pip install tensorflow` blindly may not provide GPU support on Jetson. citeturn0search1turn0search6

---

## Dataset format (expected by `train_face_recognition.py`)
`train_face_recognition.py` uses `ImageDataGenerator.flow_from_directory()` (Keras) so your dataset should follow the standard directory structure:

```
dataset/
  train/
    person1/
      img001.jpg
      img002.jpg
    person2/
      img001.jpg
  validation/   # optional (the script may use validation_split)
    person1/
    person2/
```

If you only have one `dataset/` folder with subfolders per person, set the `flow_from_directory` parameters (or use `validation_split=0.2`) — the training script already uses Keras generators/augmentation.

---

## How to train
Run the training script (ensure you are in the `4th_batch/` folder):
```bash
python3 train_face_recognition.py
```
Typical behavior:
- The script builds a MobileNetV2-based classifier (pretrained base + global average pooling + final Dense layers).
- Uses `ImageDataGenerator` for augmentation and a validation split.
- Saves the final Keras model as `face_recognition_model.h5`.

Training tips:
- Use small input image size for Jetson (e.g., 160×160 or 224×224) to reduce compute & memory.
- Use GPU-accelerated TF on Jetson for faster training if supported.

---

## Convert to TensorFlow Lite
To deploy a lightweight model on Jetson (or other edge devices), convert `.h5` to `.tflite`:
```bash
python3 convert.py
# Output: face_recognition_model.tflite
```
`convert.py` uses `tf.lite.TFLiteConverter.from_keras_model(...)` with `Optimize.DEFAULT`. If you need TensorRT acceleration on Jetson, consider converting to TensorRT engine using TensorFlow-TensorRT (TF-TRT) or use NVIDIA TensorRT directly — see NVIDIA docs. citeturn0search1turn0search19

---

## Run inference / demo
There are a few entry points:

1. **Local interactive demo (webcam)**
```bash
python3 test_face_recognition.py
# or
python3 tester.py
```
This runs the `face_check()` function — loads the Keras model (`face_recognition_model.h5`), opens a webcam feed, detects faces, predicts labels and shows FPS and bounding boxes.

2. **HTTP server**
```bash
python3 server.py
# server available at http://<jetson-ip>:4123/predict
```
`server.py` exposes a single `/predict` route that calls `face_check()` and returns its result (suitable for simple integration or remote testing).

---

## Files you may want to edit
- `train_face_recognition.py` — tweak epochs, batch size, image size, and augmentation.
- `test_face_recognition.py` — modify camera index, confidence threshold, or the way predictions are returned (e.g., JSON for API use).
- `server.py` — expand the Flask app, add authentication, or return JSON results.

---

## Troubleshooting & Jetson tips
- If TensorFlow doesn't detect the GPU on Jetson, verify your JetPack/CUDA/cudnn/TensorRT versions match the TensorFlow wheel. NVIDIA's JetPack installer includes compatible versions. citeturn0search1turn0search19  
- For inference speed, convert TFLite models to TensorRT engines (TF-TRT) or use TensorRT directly.  
- If `opencv-python` wheel fails on Jetson, install OpenCV from source or use the Jetson-optimized OpenCV packages.

---

## Development notes & license
This project appears to be a student / demo project prepared for a Jetson-based face recognition demo. The included `face_recognition_model.h5` is a trained model (keep it in `4th_batch/` or update paths in the scripts). No explicit license file was found — add `LICENSE` if you plan to open-source.

---

## Where to go next (suggestions)
- Add a `requirements.txt` with exact package versions used for reproducibility.
- Add a `Dockerfile` or `container.yaml` for reproducible deployment (Jetson supports containers).
- Add unit tests for inference functions and a CLI wrapper for batch inference.
- Export a TensorRT engine for faster inference on Jetson.

---

## References
- NVIDIA: JetPack (Jetson Linux) downloads & docs. citeturn0search2  
- NVIDIA: Installing TensorFlow for Jetson Platform (pip wheel & instructions). citeturn0search1  
- NVIDIA Developer Forums — community notes on TF / JetPack compatibility. citeturn0search3turn0search6


