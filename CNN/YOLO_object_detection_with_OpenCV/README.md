<div align="center">

```text
██╗   ██╗██║███████╗██╗ ██████╗ ███╗   ██╗███████╗████████╗██╗   ██╗██████╗ ██╗ ██████╗      █████╗ ██╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║██╔═══██╗    ██╔══██╗██║
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║███████╗   ██║   ██║   ██║██║  ██║██║██║   ██║    ███████║██║
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║╚════██║   ██║   ██║   ██║██║  ██║██║██║   ██║    ██╔══██║██║
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║███████║   ██║   ╚██████╔╝██████╔╝██║╚██████╔╝██╗██║  ██║██║
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝
```

### *Zero-Lag, Production-Grade Edge AI and Live Biometric Telemetry Hub*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ONNX](https://img.shields.io/badge/ONNX-005C99?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai)

---

**VisionStudio AI** is a real-time computer vision platform designed for processing live streams (Webcams, RTSP, and IP camera feeds) with zero latency accumulation. By separating frame ingestion from model inference, it eliminates the buffer-drift common in standard CV implementations, providing smooth, production-grade tracking and vitals estimation.

[🚀 Quick Start](#-quick-start) • [⚡ CLI Desktop Studio](#-cli-desktop-studio) • [🧬 Vitals Science](#-rppg-biometric-science) • [🔬 Diagnostic Verification](#-diagnostic-verification)

</div>

---

## ✨ Features

- **⚡ Thread-Decoupled Frame Grabbing:** An asynchronous background frame grabber (`VideoCaptureThread`) continuously flushes OpenCV's pipeline buffer, guaranteeing zero input latency even under high inference loads.
- **👁️ Dynamic Object Tracking & Counting:** Integrated ByteTrack engine with configurable horizontal/vertical counting gates and motion trail visualizations.
- **❤️ Contact-Free Vitals (rPPG):** Real-time remote photoplethysmography face tracking that extracts heart rate (BPM) and blood volume pulse (BVP) waveforms directly from ambient skin reflections.
- **⚙️ Edge Optimization & Quantization:** Profile inference latency (ms) and throughput (FPS) across multiple YOLO scales directly on local hardware, and export models into quantized half-precision (FP16) ONNX engines.
- **📊 Real-time Hardware Telemetry:** Inline HUD monitoring CPU usage, RAM utilization, CUDA GPU state, and VRAM allocation.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Model Ingestion** | `ultralytics` (YOLOv8 & YOLO-World) | Object tracking & Person detection |
| **Video Pipe** | `OpenCV (cv2)` (Multi-threaded) | Decoupled background frame retrieval |
| **Biometrics** | `NumPy`, `FFT` Spectral Analysis | Green-spectrum mean detrending and frequency analysis |
| **Dashboard** | `Streamlit` (Glassmorphic Theme) | Interactive web interface |
| **Charts** | `Plotly (Express/GraphObjects)` | ECG-like pulse waveforms and benchmarking visualizations |
| **Compilation** | `ONNX Runtime` | Quantized FP16 edge engine exporter |

---

## 📐 Pipeline Architecture

```
                       [ Live Video Stream (USB / RTSP) ]
                                      │
                                      ▼ (Asynchronous Thread)
                       ┌──────────────────────────────┐
                       │   VideoCaptureThread Daemon  │
                       └──────────────┬───────────────┘
                                      │ Overwrites (thread-safe copy)
                                      ▼
                       ┌──────────────────────────────┐
                       │    Shared Frame Memory       │
                       └──────────────┬───────────────┘
                                      │ Reads latest frame (no lag)
                                      ▼
                       ┌──────────────────────────────┐
                       │  Inference & Tracking Engine │
                       │    (NVIDIA RTX 2050 CUDA)    │
                       └──────────────┬───────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│   ByteTrack Tracking  │                           │   rPPG Face ROI Crop  │
│   & Gate Counting     │                           │   & Green Spectral FFT│
└───────────────────────┘                           └───────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation
Install the modernized requirements:
```bash
pip install -r requirements.txt
```
*(Ensure CUDA toolkits are installed if running with GPU hardware acceleration)*

### 2. Launch the Web App
Run the interactive glassmorphic dashboard:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## ⚡ CLI Desktop Studio

For maximum framerate, run the native C++ style OpenCV desktop wrapper to bypass browser rendering overhead.

### 🔹 Tracking & Gate Counting
Run multi-object tracking and count targets crossing a horizontal counting line at 50% screen height:
```bash
python live_studio.py --mode tracking --source 0 --line --line-pos 50
```

### 🔹 rPPG Face Vitals Monitor
Run contact-free biometric pulse tracking:
```bash
python live_studio.py --mode rppg --source 0
```

### 🎛️ CLI Parameters Reference
- `--mode`: `tracking` or `rppg` (default: `tracking`)
- `--source`: Webcam index (e.g. `0`) or RTSP URL (`rtsp://...`)
- `--model-mode`: `coco` or `world` (custom vocabularies)
- `--model-scale`: `n` (Nano), `s` (Small), `m` (Medium), `l` (Large)
- `--conf`: Confidence threshold (default: `0.25`)
- `--iou`: NMS IoU threshold (default: `0.45`)
- `--line`: Include counting line gate
- `--line-orient`: `horizontal` or `vertical` (default: `horizontal`)

---

## 🧬 rPPG Biometric Science

Hemoglobin in blood absorbs green light spectrums (approx. 500-600 nm) significantly more than surrounding tissue. When the heart beats, blood volume increases, changing the amount of green light reflected off the face.

1. **Face Isolation:** YOLO detects and crops the face region (top 20% of the person's bounding box).
2. **Mean Estimation:** Computes the spatial average of the green channel pixels over a rolling window.
3. **Signal Detrending:** Subtracts the mean (detrending low-frequency movements) and normalizes by the standard deviation.
4. **Spectral Decomposition:** Computes a Fast Fourier Transform (FFT) on the signal.
5. **Frequency Isolation:** Limits the search window to `[0.75 Hz, 3.0 Hz]` (equivalent to `[45 BPM, 180 BPM]`). The highest amplitude frequency peak is resolved as the final heart rate (BPM).

---

## 🔬 Diagnostic Verification

Run the test suite to verify CUDA acceleration, geometric crossing intersections, and FFT algorithm math validity:
```bash
python verify_advanced.py
```

### Expected Output
```text
[INFO] Verifying advanced VisionStudio capabilities...
[SUCCESS] Hardware diagnostics fetched:
 - CUDA: True (Device: NVIDIA GeForce RTX 2050)
[SUCCESS] Crossing math verified (Intersection cross1=True, cross2=False, direction=1)
[SUCCESS] YOLO imported successfully
 - Synthetic heart rate signal frequency: 90.0 BPM
 - Estimated rPPG frequency: 90.00 BPM
[SUCCESS] rPPG FFT analysis math verified successfully!
[INFO] All advanced backend features verified successfully!
```
