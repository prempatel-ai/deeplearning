import streamlit as st
import cv2
import numpy as np
import time
import os
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from typing import Optional, Union, List

# Import custom helpers
from utils import get_system_info, do_intersect, get_crossing_direction, estimate_bpm
from video_stream import VideoCaptureThread

# Try importing YOLO-World, fallback to standard YOLO
try:
    from ultralytics import YOLOWorld
    YOLO_WORLD_AVAILABLE = True
except ImportError:
    YOLO_WORLD_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="VisionStudio AI - Production Live Hub",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium glassmorphic theme
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1e38 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers & Title */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    
    /* Gradient Title Text */
    .gradient-text {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 900;
        margin-bottom: 5px;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.98);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphic Metrics Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 15px;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.6);
        background: rgba(99, 102, 241, 0.04);
        transform: translateY(-3px);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.15);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #818cf8;
        letter-spacing: -0.03em;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #94a3b8;
        font-weight: 600;
    }

    /* Diagnostics Card inside Sidebar */
    .diag-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 15px;
    }
    .diag-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .diag-label {
        color: #94a3b8;
    }
    .diag-value {
        color: #e2e8f0;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        padding: 10px;
        background: rgba(11, 15, 25, 0.85);
        backdrop-filter: blur(8px);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<p class="gradient-text">VisionStudio AI</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8; font-size:1.15rem; margin-top:-15px; margin-bottom: 25px;">Zero-Lag Production-Grade Live Stream Computer Vision Studio</p>', unsafe_allow_html=True)
st.write("---")

# Session state initialization
if "stream" not in st.session_state:
    st.session_state.stream = None
if "current_src" not in st.session_state:
    st.session_state.current_src = None
if "count_in" not in st.session_state:
    st.session_state.count_in = 0
if "count_out" not in st.session_state:
    st.session_state.count_out = 0
if "counted_ids" not in st.session_state:
    st.session_state.counted_ids = set()
if "track_history" not in st.session_state:
    st.session_state.track_history = {}
if "rppg_buffer" not in st.session_state:
    st.session_state.rppg_buffer = []

# Sidebar Navigation (Studio Suites)
st.sidebar.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/logo-yolo-only.png", width=70)
st.sidebar.markdown("### Studio Control Center")
active_suite = st.sidebar.selectbox(
    "Select Active Suite",
    ["👁️ Live Tracking & Counting", "❤️ rPPG Vitals Monitor", "⚡ Inference Optimizer & Benchmark"]
)

# Fetch diagnostics
sys_info = get_system_info()
device_status = "💚 GPU (CUDA) Active" if sys_info["cuda_available"] else "💙 Running on CPU"

st.sidebar.markdown(f"""
<div class="diag-container">
    <div style="font-weight:bold; color:#f8fafc; font-size:0.9rem; margin-bottom:10px;">{device_status}</div>
    <div class="diag-row">
        <span class="diag-label">GPU:</span>
        <span class="diag-value" style="font-size:0.75rem;">{sys_info['cuda_device_name']}</span>
    </div>
    <div class="diag-row">
        <span class="diag-label">CPU Usage:</span>
        <span class="diag-value">{sys_info['cpu_percent']}%</span>
    </div>
    <div class="diag-row">
        <span class="diag-label">RAM Usage:</span>
        <span class="diag-value">{sys_info['ram_percent']}% ({sys_info['ram_used_gb']}/{sys_info['ram_total_gb']} GB)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Cache model weights to avoid heavy file reload cycles
@st.cache_resource
def load_vision_model(mode: str, scale: str, custom_cls_str: Optional[str] = None) -> Union[YOLO, "YOLOWorld"]:
    """Loads YOLO or YOLO-World checkpoints and offloads them to CUDA/CPU device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "COCO (80 standard classes)":
        model_name = f"yolov8{scale}.pt"
        model = YOLO(model_name)
    else:  # Custom Vocabulary YOLO-World
        model_name = f"yolov8{scale}-worldv2.pt"
        if YOLO_WORLD_AVAILABLE:
            model = YOLOWorld(model_name)
        else:
            model = YOLO(model_name)
    model.to(device)
    return model


if active_suite == "👁️ Live Tracking & Counting":
    st.subheader("👁️ Live Object Tracking & Line Boundary Counting")
    st.markdown("This suite enables zero-lag multi-object tracking and counts objects crossing virtual boundaries using your local GPU.")
    
    st.sidebar.markdown("### Model & Class Settings")
    model_mode = st.sidebar.selectbox(
        "Select Model Mode",
        ["COCO (80 standard classes)", "Custom Vocabulary (YOLO-World)"],
        key="tracking_model_mode"
    )

    custom_classes: List[str] = []
    if model_mode == "Custom Vocabulary (YOLO-World)":
        classes_input = st.sidebar.text_input(
            "Enter Custom Classes (comma separated)",
            "person, laptop, cup, chair, phone",
            key="tracking_custom_classes"
        )
        custom_classes = [c.strip() for c in classes_input.split(",") if c.strip()]

    model_scale = st.sidebar.selectbox(
        "Select Model Scale",
        ["Nano (Fastest)", "Small (Balanced)", "Medium (Accurate)", "Large (Heavy)"],
        key="tracking_model_scale"
    )

    scale_map = {"Nano (Fastest)": "n", "Small (Balanced)": "s", "Medium (Accurate)": "m", "Large (Heavy)": "l"}
    suffix = scale_map[model_scale]

    model = load_vision_model(model_mode, suffix, str(custom_classes) if model_mode == "Custom Vocabulary (YOLO-World)" else None)
    if model_mode == "Custom Vocabulary (YOLO-World)" and custom_classes:
        model.set_classes(custom_classes)

    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="tracking_conf")
    iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05, key="tracking_iou")

    st.sidebar.markdown("### Line Crossing Config")
    enable_line_crossing = st.sidebar.checkbox("Enable Counting Line", False, key="tracking_enable_line")
    line_orientation = st.sidebar.selectbox("Line Orientation", ["Horizontal", "Vertical"], disabled=not enable_line_crossing, key="tracking_line_orient")
    line_position = st.sidebar.slider("Line Position %", 10, 90, 50, disabled=not enable_line_crossing, key="tracking_line_pos")

    # Ingestion configuration
    st.markdown("#### Stream Source Configuration")
    stream_type = st.radio("Select Ingestion Input", ["Local USB Webcam", "IP Camera / RTSP Stream"], horizontal=True, key="tracking_stream_type")

    if stream_type == "Local USB Webcam":
        webcam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=10, value=0, step=1)
        src = webcam_idx
    else:
        rtsp_url = st.text_input("RTSP Stream URL", "rtsp://admin:password@192.168.1.100:554/stream1")
        src = rtsp_url

    col_control1, col_control2 = st.columns([1, 4])
    with col_control1:
        run_stream = st.toggle("🔴 ACTIVATE LIVE STREAM", False)
    with col_control2:
        if st.button("Reset Counter Metrics"):
            st.session_state.count_in = 0
            st.session_state.count_out = 0
            st.session_state.counted_ids = set()
            st.session_state.track_history = {}
            st.success("Counters reset!")

    # Start or restart camera thread safely without visual stutter
    if run_stream:
        if st.session_state.stream is None or st.session_state.current_src != src:
            if st.session_state.stream is not None:
                st.session_state.stream.stop()
            st.session_state.stream = VideoCaptureThread(src)
            st.session_state.stream.start()
            st.session_state.current_src = src
            st.session_state.count_in = 0
            st.session_state.count_out = 0
            st.session_state.counted_ids = set()
            st.session_state.track_history = {}
            time.sleep(0.5)
    else:
        if st.session_state.stream is not None:
            st.session_state.stream.stop()
            st.session_state.stream = None
            st.session_state.current_src = None

    # Render dynamic metric containers
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        fps_metric = st.empty()
    with m2:
        detect_metric = st.empty()
    with m3:
        in_metric = st.empty()
    with m4:
        out_metric = st.empty()

    frame_view = st.empty()

    if run_stream and st.session_state.stream is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prev_time = time.time()
        frame_count = 0
        fps_val = 0.0

        while run_stream:
            grabbed, frame = st.session_state.stream.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            t0 = time.time()
            results = model.track(frame, conf=conf_thresh, iou=iou_thresh, persist=True, device=device, verbose=False)
            t_inf = (time.time() - t0) * 1000

            res = results[0]
            plotted = frame.copy()
            names = res.names
            H, W = frame.shape[:2]

            if enable_line_crossing:
                if line_orientation == "Horizontal":
                    ly = int(H * (line_position / 100.0))
                    line_p1 = (0, ly)
                    line_q1 = (W, ly)
                else:
                    lx = int(W * (line_position / 100.0))
                    line_p1 = (lx, 0)
                    line_q1 = (lx, H)

            active_count = 0
            if res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else None
                active_count = len(boxes)

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cls_idx = int(clss[idx])
                    cls_name = names[cls_idx]
                    conf = confs[idx]

                    cv2.rectangle(plotted, (x1, y1), (x2, y2), (240, 32, 160), 2)

                    tid = int(ids[idx]) if ids is not None else None
                    if tid is not None:
                        lbl = f"ID:{tid} {cls_name} {conf:.2f}"
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        curr_pt = (cx, cy)

                        if tid not in st.session_state.track_history:
                            st.session_state.track_history[tid] = []
                        st.session_state.track_history[tid].append(curr_pt)
                        if len(st.session_state.track_history[tid]) > 15:
                            st.session_state.track_history[tid].pop(0)

                        for p_idx, pt in enumerate(st.session_state.track_history[tid]):
                            alpha = (p_idx + 1) / len(st.session_state.track_history[tid])
                            color = (int(99 * alpha), int(102 * alpha), int(241 * alpha))
                            cv2.circle(plotted, pt, 3, color, -1)

                        if enable_line_crossing and len(st.session_state.track_history[tid]) > 1:
                            prev_pt = st.session_state.track_history[tid][-2]
                            if do_intersect(line_p1, line_q1, prev_pt, curr_pt):
                                if tid not in st.session_state.counted_ids:
                                    st.session_state.counted_ids.add(tid)
                                    direction = get_crossing_direction(line_p1, line_q1, prev_pt, curr_pt)
                                    if direction > 0:
                                        st.session_state.count_in += 1
                                    else:
                                        st.session_state.count_out += 1
                    else:
                        lbl = f"{cls_name} {conf:.2f}"

                    cv2.putText(plotted, lbl, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if enable_line_crossing:
                cv2.line(plotted, line_p1, line_q1, (255, 255, 0), 2, cv2.LINE_AA)

            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed >= 0.5:
                fps_val = frame_count / elapsed
                frame_count = 0
                prev_time = curr_time

            fps_metric.markdown(f'<div class="metric-card"><div class="metric-value">{fps_val:.1f}</div><div class="metric-label">Stream FPS</div></div>', unsafe_allow_html=True)
            detect_metric.markdown(f'<div class="metric-card"><div class="metric-value">{active_count}</div><div class="metric-label">Active Objects</div></div>', unsafe_allow_html=True)
            in_metric.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.count_in}</div><div class="metric-label">IN Counter</div></div>', unsafe_allow_html=True)
            out_metric.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.count_out}</div><div class="metric-label">OUT Counter</div></div>', unsafe_allow_html=True)

            cv2.putText(plotted, f"INFERENCE: {t_inf:.1f}ms", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            frame_view.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            time.sleep(0.01)
    else:
        fps_metric.markdown('<div class="metric-card"><div class="metric-value">0.0</div><div class="metric-label">Stream FPS</div></div>', unsafe_allow_html=True)
        detect_metric.markdown('<div class="metric-card"><div class="metric-value">0</div><div class="metric-label">Active Objects</div></div>', unsafe_allow_html=True)
        in_metric.markdown('<div class="metric-card"><div class="metric-value">0</div><div class="metric-label">IN Counter</div></div>', unsafe_allow_html=True)
        out_metric.markdown('<div class="metric-card"><div class="metric-value">0</div><div class="metric-label">OUT Counter</div></div>', unsafe_allow_html=True)
        frame_view.info("Stream offline. Click Toggle to run.")


elif active_suite == "❤️ rPPG Vitals Monitor":
    st.subheader("❤️ Remote Photoplethysmography (rPPG) Vitals Monitor")
    st.markdown("""
    This biometric module isolates facial regions dynamically, extracts the **Green spectrum blood volume pulse (BVP) signal**, 
    and applies a real-time spectral Fast Fourier Transform (FFT) to measure heart rates without physical contact.
    """)

    st.markdown("#### Stream Source Configuration")
    stream_type = st.radio("Select Ingestion Input", ["Local USB Webcam", "IP Camera / RTSP Stream"], horizontal=True, key="rppg_stream_type")

    if stream_type == "Local USB Webcam":
        webcam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=10, value=0, step=1, key="rppg_webcam_idx")
        src = webcam_idx
    else:
        rtsp_url = st.text_input("RTSP Stream URL", "rtsp://admin:password@192.168.1.100:554/stream1", key="rppg_rtsp_url")
        src = rtsp_url

    col_control1, col_control2 = st.columns([1, 4])
    with col_control1:
        run_rppg = st.toggle("🔴 ACTIVATE VITALS MONITOR", False, key="run_rppg_toggle")
    with col_control2:
        if st.button("Reset Pulse Buffer"):
            st.session_state.rppg_buffer = []
            st.success("Signal buffer cleared!")

    # Start camera thread
    if run_rppg:
        if st.session_state.stream is None or st.session_state.current_src != src:
            if st.session_state.stream is not None:
                st.session_state.stream.stop()
            st.session_state.stream = VideoCaptureThread(src)
            st.session_state.stream.start()
            st.session_state.current_src = src
            st.session_state.rppg_buffer = []
            time.sleep(0.5)
    else:
        if st.session_state.stream is not None:
            st.session_state.stream.stop()
            st.session_state.stream = None
            st.session_state.current_src = None

    c_stream, c_graph = st.columns([1, 1])
    with c_stream:
        frame_view = st.empty()
    with c_graph:
        bpm_card = st.empty()
        pulse_chart = st.empty()

    if run_rppg and st.session_state.stream is not None:
        face_model = load_vision_model("COCO (80 standard classes)", "n")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        prev_time = time.time()
        frame_count = 0
        fps_val = 30.0  # Default fallback
        
        while run_rppg:
            grabbed, frame = st.session_state.stream.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            H, W = frame.shape[:2]
            plotted = frame.copy()

            results = face_model.predict(frame, conf=0.4, classes=[0], device=device, verbose=False)
            res = results[0]
            
            target_detected = False
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
                max_idx = np.argmax(areas)
                box = boxes[max_idx]
                target_detected = True
                
                bx1, by1, bx2, by2 = map(int, box)
                bw = bx2 - bx1
                bh = by2 - by1
                
                # Estimate Face Region of Interest (ROI) inside person bounding box (top 20%)
                fx1 = int(bx1 + 0.3 * bw)
                fx2 = int(bx1 + 0.7 * bw)
                fy1 = int(by1 + 0.05 * bh)
                fy2 = int(by1 + 0.22 * bh)
                
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(W, fx2), min(H, fy2)
                
                if fx2 > fx1 and fy2 > fy1:
                    cv2.rectangle(plotted, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
                    cv2.rectangle(plotted, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                    cv2.putText(plotted, "rPPG FACE ROI", (fx1, fy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    roi = frame[fy1:fy2, fx1:fx2]
                    avg_g = float(np.mean(roi[:, :, 1]))  # Channel 1 is green
                    
                    st.session_state.rppg_buffer.append(avg_g)
                    if len(st.session_state.rppg_buffer) > 150:
                        st.session_state.rppg_buffer.pop(0)

            # Estimate FPS
            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed >= 0.5:
                fps_val = frame_count / elapsed
                frame_count = 0
                prev_time = curr_time

            buf_len = len(st.session_state.rppg_buffer)
            if target_detected and buf_len >= 60:
                bpm, filtered_signal = estimate_bpm(st.session_state.rppg_buffer, fps=fps_val)
                
                bpm_card.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #ef4444; font-size: 3.5rem; text-shadow: 0 0 10px rgba(239, 68, 68, 0.3);">❤️ {bpm:.1f} BPM</div>
                    <div class="metric-label">Estimated Pulse Frequency</div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_pulse = go.Figure()
                fig_pulse.add_trace(go.Scatter(
                    y=filtered_signal,
                    mode='lines',
                    line=dict(color='#ef4444', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.05)'
                ))
                fig_pulse.update_layout(
                    title="Real-time Blood Volume Pulse (BVP) Waveform",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Intensity Variance (Normalized)")
                )
                pulse_chart.plotly_chart(fig_pulse, use_container_width=True)
            elif target_detected:
                calib_percent = int((buf_len / 60.0) * 100)
                bpm_card.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #e2e8f0; font-size: 2.2rem;">CALIBRATING...</div>
                    <div class="metric-label">Signal Buffer Loading ({buf_len}/60 frames)</div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_loading = go.Figure()
                fig_loading.add_trace(go.Bar(x=[calib_percent], y=['Buffer'], orientation='h', marker_color='#a855f7'))
                fig_loading.update_layout(
                    xaxis=dict(range=[0, 100], showgrid=False),
                    yaxis=dict(showgrid=False),
                    height=180,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0"
                )
                pulse_chart.plotly_chart(fig_loading, use_container_width=True)
            else:
                cv2.putText(plotted, "NO SUBJECT DETECTED", (30, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                bpm_card.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #ef4444; font-size: 2.0rem;">🔴 NO TARGET</div>
                    <div class="metric-label">Stand inside webcam window</div>
                </div>
                """, unsafe_allow_html=True)

            frame_view.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            time.sleep(0.01)
    else:
        frame_view.info("Monitor offline. Click Toggle to run biometric detection.")


elif active_suite == "⚡ Inference Optimizer & Benchmark":
    st.subheader("⚡ Model Deployment, Quantization & Performance Benchmark Suite")
    st.markdown("""
    Optimize your deep learning workflows. Benchmark standard YOLO models directly on your hardware and export them into 
    lightweight, quantized inference engines like **ONNX Runtime** for production rollouts.
    """)

    col_bench, col_export = st.columns(2)
    
    with col_bench:
        st.markdown("### 📊 GPU Performance Benchmarking")
        st.markdown("Profile deep learning scale tradeoffs directly on your hardware. We will measure exact inference latency (ms) on your CUDA GPU.")
        
        bench_models = st.multiselect(
            "Select Scales to Benchmark",
            ["YOLOv8 Nano", "YOLOv8 Small", "YOLOv8 Medium", "YOLOv8 Large"],
            default=["YOLOv8 Nano", "YOLOv8 Small"]
        )
        
        run_bench = st.button("Start Live GPU Benchmark")
        
        if run_bench:
            if not bench_models:
                st.warning("Please select at least one scale to benchmark.")
            else:
                progress_bar = st.progress(0)
                status_txt = st.empty()
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                latencies = []
                names_mapped = []
                
                scale_codes = {
                    "YOLOv8 Nano": "n",
                    "YOLOv8 Small": "s",
                    "YOLOv8 Medium": "m",
                    "YOLOv8 Large": "l"
                }
                
                dummy_input = torch.randn(1, 3, 640, 640).to(device)
                
                for idx, m_label in enumerate(bench_models):
                    code = scale_codes[m_label]
                    status_txt.markdown(f"Loading and warming up **{m_label}** model...")
                    
                    m_instance = load_vision_model("COCO (80 standard classes)", code)
                    
                    # Warm up cycles
                    for _ in range(10):
                        _ = m_instance(dummy_input, verbose=False)
                        
                    status_txt.markdown(f"Profiling inference latency for **{m_label}**...")
                    t_runs = []
                    for _ in range(30):
                        t_start = time.time()
                        _ = m_instance(dummy_input, verbose=False)
                        t_runs.append((time.time() - t_start) * 1000)
                        
                    avg_latency = float(np.mean(t_runs))
                    latencies.append(avg_latency)
                    names_mapped.append(m_label)
                    
                    progress_bar.progress(int(((idx + 1) / len(bench_models)) * 100))
                
                status_txt.success("GPU Benchmarking complete!")
                
                df_bench = pd.DataFrame({
                    "Model Scale": names_mapped,
                    "Inference Latency (ms)": latencies,
                    "Throughput (FPS)": [1000.0 / lat for lat in latencies]
                })
                
                fig_bench = px.bar(df_bench, x="Model Scale", y="Inference Latency (ms)", text="Inference Latency (ms)", color="Inference Latency (ms)", color_continuous_scale="Purples")
                fig_bench.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0"
                )
                st.plotly_chart(fig_bench, use_container_width=True)
                
                st.markdown("#### Performance Summary Table")
                st.dataframe(df_bench, use_container_width=True)
                
    with col_export:
        st.markdown("### 🚀 Production Model Quantization & Exporter")
        st.markdown("Compile PyTorch neural network checkpoints into optimized standalone formats.")
        
        export_scale = st.selectbox(
            "Select Scale to Export",
            ["Nano (6MB)", "Small (22MB)", "Medium (52MB)", "Large (87MB)"]
        )
        
        export_format = st.selectbox(
            "Select Deployment Target",
            ["ONNX (FP16 Quantized/GPU Accelerated)", "ONNX (Standard FP32)"]
        )
        
        run_export = st.button("🚀 Run Model Optimization")
        
        if run_export:
            scale_map_exp = {"Nano (6MB)": "n", "Small (22MB)": "s", "Medium (52MB)": "m", "Large (87MB)": "l"}
            code = scale_map_exp[export_scale]
            
            with st.spinner("Optimizing model graphs & exporting to ONNX format... (this can take up to a minute)"):
                try:
                    model_to_opt = YOLO(f"yolov8{code}.pt")
                    half_val = (export_format == "ONNX (FP16 Quantized/GPU Accelerated)")
                    onnx_path = model_to_opt.export(format="onnx", half=half_val, imgsz=640)
                    
                    if os.path.exists(onnx_path):
                        st.success(f"Model successfully optimized and exported to: `{onnx_path}`")
                        
                        with open(onnx_path, "rb") as f:
                            st.download_button(
                                label="📥 Download ONNX Model Engine",
                                data=f,
                                file_name=os.path.basename(onnx_path),
                                mime="application/octet-stream"
                            )
                    else:
                        st.error("Export process completed but output model was not found.")
                except Exception as e:
                    st.error(f"Compilation pipeline failed: {e}")

# Footer
st.markdown("""
<div class="footer">
    <p>VisionStudio AI | Production-Grade Zero-Lag Streaming | 2026</p>
</div>
""", unsafe_allow_html=True)
