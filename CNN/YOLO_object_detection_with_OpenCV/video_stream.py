import cv2
import threading
import time
import numpy as np
from typing import Union, Tuple, Optional

class VideoCaptureThread:
    """
    A production-grade, thread-safe asynchronous camera frame grabber.
    Decouples camera frame ingestion from model inference to ensure zero latency
    accumulation in real-time video processing pipelines.
    """
    def __init__(self, src: Union[int, str] = 0) -> None:
        self.src: Union[int, str] = src
        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.src)
        self.grabbed: bool = False
        self.frame: Optional[np.ndarray] = None
        self.started: bool = False
        self.read_lock: threading.Lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "VideoCaptureThread":
        """Starts the background frame grabber thread."""
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self) -> None:
        """Continuously reads frames from cv2.VideoCapture in the background."""
        while self.started:
            if not self.cap.isOpened():
                time.sleep(2.0)
                self.cap.open(self.src)
                continue
                
            grabbed, frame = self.cap.read()
            if not grabbed or frame is None:
                # Attempt reconnection in case of connection drop
                time.sleep(0.5)
                self.cap.open(self.src)
                continue
                
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame.copy()
            
            # Brief sleep to prevent CPU starvation
            time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the latest frame and a status boolean indicating if retrieval succeeded."""
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self) -> None:
        """Stops the grabber thread and releases the cv2.VideoCapture resource."""
        self.started = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
