import sys
import torch
import numpy as np
from utils import get_system_info, do_intersect, get_crossing_direction, estimate_bpm

def main() -> None:
    print("[INFO] Verifying advanced VisionStudio capabilities...")

    # 1. Test Telemetry
    try:
        info = get_system_info()
        print("[SUCCESS] Hardware diagnostics fetched:")
        print(f" - CPU: {info['cpu_percent']}%")
        print(f" - RAM: {info['ram_percent']}% ({info['ram_used_gb']}/{info['ram_total_gb']} GB)")
        print(f" - CUDA: {info['cuda_available']} (Device: {info['cuda_device_name']})")
    except Exception as e:
        print(f"[ERROR] Telemetry fetching failed: {e}")
        sys.exit(1)

    # 2. Test Line-Crossing Intersection Math
    try:
        # Segment 1: counting line from (0, 100) to (200, 100)
        line_p1 = (0, 100)
        line_q1 = (200, 100)
        
        # Path 1: crosses line downwards from (50, 50) to (50, 150) -> should intersect
        path_p1 = (50, 50)
        path_q1 = (50, 150)
        
        # Path 2: does not cross line (moves parallel) from (50, 50) to (150, 50) -> should not intersect
        path_p2 = (50, 50)
        path_q2 = (150, 50)
        
        cross1 = do_intersect(line_p1, line_q1, path_p1, path_q1)
        cross2 = do_intersect(line_p1, line_q1, path_p2, path_q2)
        
        assert cross1 is True, "Path 1 should cross the line"
        assert cross2 is False, "Path 2 should not cross the line"
        
        dir1 = get_crossing_direction(line_p1, line_q1, path_p1, path_q1)
        print(f"[SUCCESS] Crossing math verified (Intersection cross1={cross1}, cross2={cross2}, direction={dir1})")
    except Exception as e:
        print(f"[ERROR] Crossing math failed: {e}")
        sys.exit(1)

    # 3. Check YOLO
    try:
        from ultralytics import YOLO
        print("[SUCCESS] YOLO imported successfully")
    except Exception as e:
        print(f"[ERROR] YOLO import failed: {e}")
        sys.exit(1)

    # 4. Test rPPG Signal detrending & FFT Heart Rate Estimation
    try:
        # Generate a pure 1.5 Hz sine wave (90 BPM) sampled at 30 Hz for 4 seconds (120 samples)
        fps = 30.0
        t = np.arange(120) / fps
        pure_signal = list(np.sin(2 * np.pi * 1.5 * t))
        
        bpm, filtered = estimate_bpm(pure_signal, fps=fps)
        print(f" - Synthetic heart rate signal frequency: 90.0 BPM")
        print(f" - Estimated rPPG frequency: {bpm:.2f} BPM")
        assert abs(bpm - 90.0) < 5.0, f"BPM estimation error too large: {bpm}"
        print(f"[SUCCESS] rPPG FFT analysis math verified successfully!")
    except Exception as e:
        print(f"[ERROR] rPPG verification failed: {e}")
        sys.exit(1)

    print("[INFO] All advanced backend features verified successfully!")

if __name__ == "__main__":
    main()
