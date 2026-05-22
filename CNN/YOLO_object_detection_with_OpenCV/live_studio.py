import argparse
import cv2
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Union

from video_stream import VideoCaptureThread
from utils import get_system_info, do_intersect, get_crossing_direction, estimate_bpm

try:
    from ultralytics import YOLOWorld
    from ultralytics import YOLO
    YOLO_WORLD_AVAILABLE = True
except ImportError:
    from ultralytics import YOLO
    YOLO_WORLD_AVAILABLE = False

def run_live_studio(args: argparse.Namespace) -> None:
    """Runs high-performance native window computer vision suite (Tracking or rPPG)."""
    # 1. Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using Device: {device.upper()}")
    if device == "cuda":
        print(f" - GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Model Loading
    model_scale = args.model_scale
    if args.model_mode == "world":
        model_name = f"yolov8{model_scale}-worldv2.pt"
        print(f"[INFO] Loading YOLO-World model: {model_name}...")
        if YOLO_WORLD_AVAILABLE:
            model = YOLOWorld(model_name)
        else:
            print("[WARNING] YOLOWorld import failed, falling back to standard YOLO class.")
            model = YOLO(model_name)
        
        classes = [c.strip() for c in args.custom_classes.split(",") if c.strip()]
        print(f"[INFO] Setting YOLO-World custom classes: {classes}")
        model.set_classes(classes)
    else:
        model_name = f"yolov8{model_scale}.pt"
        print(f"[INFO] Loading YOLO COCO model: {model_name}...")
        model = YOLO(model_name)

    model.to(device)

    # 3. Source Selection & Thread Initialization
    try:
        source: Union[int, str] = int(args.source)
        print(f"[INFO] Initializing webcam stream (Index: {source})...")
    except ValueError:
        source = args.source
        print(f"[INFO] Initializing RTSP stream: {source}...")

    stream = VideoCaptureThread(source)
    stream.start()

    # Wait for the first frame to load
    print("[INFO] Waiting for video stream to initialize...")
    retries = 10
    frame = None
    while retries > 0:
        grabbed, frame = stream.read()
        if grabbed and frame is not None:
            break
        time.sleep(0.5)
        retries -= 1

    if frame is None:
        print("[ERROR] Failed to retrieve frame from camera stream. Exiting.")
        stream.stop()
        sys.exit(1)

    print("[SUCCESS] Stream connected successfully!")
    H, W = frame.shape[:2]
    print(f" - Resolution: {W}x{H}")

    # 4. Counting Line Setup
    enable_line = args.line
    line_orient = args.line_orient
    line_pos = args.line_pos / 100.0

    if enable_line and args.mode == "tracking":
        if line_orient == "horizontal":
            ly = int(H * line_pos)
            line_p1 = (0, ly)
            line_q1 = (W, ly)
        else:
            lx = int(W * line_pos)
            line_p1 = (lx, 0)
            line_q1 = (lx, H)
        print(f"[INFO] Counting Line set from {line_p1} to {line_q1}")

    # Tracking states
    counted_ids: Set[int] = set()
    count_in = 0
    count_out = 0
    track_history: Dict[int, List[Tuple[int, int]]] = {}
    
    # rPPG states
    rppg_buffer: List[float] = []

    # Performance telemetry
    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    inference_ms = 0.0

    print("\n" + "="*50)
    print(f" VISIONSTUDIO LIVE STUDIO RUNNING [MODE: {args.mode.upper()}]")
    print(" - Press 'q' or 'ESC' in the video window to exit.")
    print("="*50 + "\n")

    # Native Window setup
    window_name = f"VisionStudio Live Feed ({args.mode.capitalize()} Mode)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    try:
        while True:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            t_inference_start = time.time()

            if args.mode == "rppg":
                # ------------------- rPPG MODE -------------------
                results = model.predict(
                    frame,
                    conf=0.4,
                    classes=[0],
                    device=device,
                    verbose=False
                )
                t_inference_end = time.time()
                inference_ms = (t_inference_end - t_inference_start) * 1000

                res = results[0]
                plotted = frame.copy()

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
                    
                    # Estimate Face ROI (top 20% of person box)
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
                        avg_g = float(np.mean(roi[:, :, 1]))
                        rppg_buffer.append(avg_g)
                        if len(rppg_buffer) > 150:
                            rppg_buffer.pop(0)

                # Estimate FPS
                curr_time = time.time()
                elapsed = curr_time - prev_time
                if elapsed >= 0.5:
                    fps = frame_count / elapsed
                    frame_count = 0
                    prev_time = curr_time

                buf_len = len(rppg_buffer)
                if target_detected and buf_len >= 60:
                    bpm, filtered_sig = estimate_bpm(rppg_buffer, fps=fps if fps > 0 else 30.0)
                    cv2.putText(plotted, f"HEART RATE: {bpm:.1f} BPM", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Waveform drawing overlay
                    graph_h = 100
                    cv2.rectangle(plotted, (0, H - graph_h), (W, H), (15, 15, 15), -1)
                    
                    if len(filtered_sig) >= 2:
                        step = max(1, W // len(filtered_sig))
                        for i in range(1, len(filtered_sig)):
                            x1 = (i - 1) * step
                            y1 = int(H - (graph_h / 2) - filtered_sig[i - 1] * (graph_h / 4))
                            x2 = i * step
                            y2 = int(H - (graph_h / 2) - filtered_sig[i] * (graph_h / 4))
                            cv2.line(plotted, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
                            
                elif target_detected:
                    cv2.putText(plotted, f"CALIBRATING BIOMETRIC BUFFER: {buf_len}/60", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(plotted, "NO SUBJECT DETECTED", (30, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            else:
                # ------------------- TRACKING MODE -------------------
                results = model.track(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    persist=True,
                    device=device,
                    verbose=False
                )
                t_inference_end = time.time()
                inference_ms = (t_inference_end - t_inference_start) * 1000

                res = results[0]
                plotted = frame.copy()
                names = res.names

                if res.boxes is not None and res.boxes.id is not None:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    ids = res.boxes.id.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()

                    for box, tid, cls_idx, conf in zip(boxes, ids, classes, confs):
                        tid = int(tid)
                        cls_name = names[int(cls_idx)]
                        
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        curr_pt = (cx, cy)

                        cv2.rectangle(plotted, (x1, y1), (x2, y2), (240, 32, 160), 2)
                        lbl = f"ID:{tid} {cls_name} {conf:.2f}"
                        cv2.putText(plotted, lbl, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        if tid not in track_history:
                            track_history[tid] = []
                        track_history[tid].append(curr_pt)
                        if len(track_history[tid]) > 20:
                            track_history[tid].pop(0)

                        for idx, pt in enumerate(track_history[tid]):
                            alpha = (idx + 1) / len(track_history[tid])
                            color = (int(99 * alpha), int(102 * alpha), int(241 * alpha))
                            cv2.circle(plotted, pt, 3, color, -1)

                        if enable_line and len(track_history[tid]) > 1:
                            prev_pt = track_history[tid][-2]
                            if do_intersect(line_p1, line_q1, prev_pt, curr_pt):
                                if tid not in counted_ids:
                                    counted_ids.add(tid)
                                    direction = get_crossing_direction(line_p1, line_q1, prev_pt, curr_pt)
                                    if direction > 0:
                                        count_in += 1
                                    else:
                                        count_out += 1

                if enable_line:
                    cv2.line(plotted, line_p1, line_q1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(plotted, f"IN: {count_in}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(plotted, f"OUT: {count_out}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                curr_time = time.time()
                elapsed = curr_time - prev_time
                if elapsed >= 0.5:
                    fps = frame_count / elapsed
                    frame_count = 0
                    prev_time = curr_time

            # Draw HUD Telemetry Diagnostics overlay (Top-right corner)
            hud_y = 35
            sys_stats = get_system_info()
            hud_bg = np.zeros_like(plotted[:130, W-300:W])
            cv2.addWeighted(plotted[:130, W-300:W], 0.4, hud_bg, 0.6, 0, plotted[:130, W-300:W])
            
            cv2.putText(plotted, f"STREAM FPS: {fps:.1f}", (W - 280, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(plotted, f"INFERENCE: {inference_ms:.1f}ms", (W - 280, hud_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(plotted, f"CPU USAGE: {sys_stats['cpu_percent']}%", (W - 280, hud_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(plotted, f"RAM USAGE: {sys_stats['ram_percent']}%", (W - 280, hud_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, plotted)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("[INFO] Terminating studio loop...")
    finally:
        print("[INFO] Shutting down streams and window services...")
        stream.stop()
        cv2.destroyAllWindows()
        print("[SUCCESS] Studio cleanup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionStudio - High-Performance Live Video Studio")
    parser.add_argument("--mode", type=str, choices=["tracking", "rppg"], default="tracking", help="Studio run mode: tracking or rppg")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or RTSP URL")
    parser.add_argument("--model-mode", type=str, choices=["coco", "world"], default="coco", help="Model mode: coco or world")
    parser.add_argument("--custom-classes", type=str, default="person, laptop, cup, chair, phone", help="YOLO-World custom vocabulary")
    parser.add_argument("--model-scale", type=str, choices=["n", "s", "m", "l"], default="n", help="Model scale: n, s, m, l")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--line", action="store_true", help="Enable counting line")
    parser.add_argument("--line-orient", type=str, choices=["horizontal", "vertical"], default="horizontal", help="Counting line orientation")
    parser.add_argument("--line-pos", type=int, default=50, help="Counting line position percentage (10 to 90)")

    args = parser.parse_args()
    run_live_studio(args)
