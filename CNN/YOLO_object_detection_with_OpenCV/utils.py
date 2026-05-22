import psutil
import torch
import numpy as np
from typing import Dict, Tuple, List, Union

# --- LINE-CROSSING MATH HELPERS ---

def on_segment(p: Tuple[int, int], q: Tuple[int, int], r: Tuple[int, int]) -> bool:
    """Checks if point q lies on line segment 'pr'."""
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p: Tuple[int, int], q: Tuple[int, int], r: Tuple[int, int]) -> int:
    """
    Finds orientation of ordered triplet (p, q, r).
    0 -> p, q and r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def do_intersect(p1: Tuple[int, int], q1: Tuple[int, int], p2: Tuple[int, int], q2: Tuple[int, int]) -> bool:
    """Returns True if line segment 'p1q1' and 'p2q2' intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

def get_crossing_direction(p1: Tuple[int, int], q1: Tuple[int, int], p2: Tuple[int, int], q2: Tuple[int, int]) -> int:
    """
    Determines crossing direction of path p2->q2 across line p1->q1.
    p1, q1 define the counting line.
    p2, q2 define the object's movement path.
    Returns +1 for one direction, -1 for the other.
    """
    # Vector of the line: dx, dy
    line_dx = q1[0] - p1[0]
    line_dy = q1[1] - p1[1]
    
    # Normal vector to the line pointing to one side (90 deg CCW)
    normal_x = -line_dy
    normal_y = line_dx
    
    # Movement vector
    move_dx = q2[0] - p2[0]
    move_dy = q2[1] - p2[1]
    
    # Dot product of movement and normal vector
    dot_product = move_dx * normal_x + move_dy * normal_y
    
    return 1 if dot_product >= 0 else -1


# --- SYSTEM TELEMETRY HELPERS ---

def get_system_info() -> Dict[str, Union[float, bool, str]]:
    """Retrieves current CPU, RAM, and CUDA GPU stats."""
    info = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": "None",
        "cuda_vram_allocated_gb": 0.0,
        "cuda_vram_reserved_gb": 0.0
    }
    
    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        info["cuda_vram_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        
    return info


# --- rPPG SIGNAL PROCESSING HELPERS ---

def estimate_bpm(raw_signal: List[float], fps: float = 30.0) -> Tuple[float, np.ndarray]:
    """
    Estimates heart rate (BPM) from a raw PPG green channel signal.
    Applies detrending (mean subtraction) and Fast Fourier Transform (FFT).
    Filters frequencies to human heart range: [0.75 Hz, 3.0 Hz] -> [45 BPM, 180 BPM].
    """
    if len(raw_signal) < 30:
        # Not enough samples yet to run FFT reliably, return baseline
        return 72.0, np.zeros(len(raw_signal), dtype=np.float32)

    signal = np.array(raw_signal, dtype=np.float32)
    
    # 1. Detrend signal (remove slow drift by subtracting mean)
    signal = signal - np.mean(signal)
    
    # Normalize signal amplitude for clean visualization
    std_val = np.std(signal)
    if std_val > 0:
        signal = signal / std_val
        
    n = len(signal)
    # Compute FFT
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(n, d=1.0/fps)
    
    # Filter for positive frequencies in human heart rate band: [0.75 Hz, 3.0 Hz]
    valid_idx = np.where((freqs >= 0.75) & (freqs <= 3.0))[0]
    
    if len(valid_idx) == 0:
        return 72.0, signal
        
    valid_freqs = freqs[valid_idx]
    valid_fft = fft_vals[valid_idx]
    
    # Locate peak frequency
    peak_idx = np.argmax(valid_fft)
    bpm = valid_freqs[peak_idx] * 60.0
    
    return float(bpm), signal
