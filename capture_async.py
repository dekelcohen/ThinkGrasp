import os
import time
import threading
from queue import Queue, Full, Empty

import cv2
import numpy as np


class AsyncImageWriter:
    """Encode and write images off the sim thread.

    Push RGB frames to a bounded queue via offer(); a background thread
    encodes (PNG/JPEG) and writes to disk. On queue full, drops frames
    without blocking the simulator.
    """

    def __init__(
        self,
        out_dir: str,
        max_queue: int = 128,
        codec: str = "png",
        quality: int = 90,
        drop_newest: bool = True,
    ) -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.q: Queue = Queue(maxsize=max_queue)
        self.codec = codec.lower()
        self.quality = int(quality)
        self.drop_newest = drop_newest
        self._stop = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def offer(self, base_name: str, rgb: np.ndarray, ts: float | None = None, idx: int | None = None) -> bool:
        """Offer a frame to be written asynchronously.

        Args:
            base_name: filename prefix (e.g., 'observer_upper').
            rgb: HxWx3 uint8 RGB image.
            ts: optional timestamp; used if idx not provided.
            idx: optional integer index for filename sequencing.

        Returns:
            True if the frame was accepted into the queue; False if dropped.
        """
        if ts is None:
            ts = time.time()
        if idx is None:
            idx = int(ts * 1000)
        # Ensure contiguous shape and type
        if not isinstance(rgb, np.ndarray):
            return False
        item = (base_name, rgb, idx)
        try:
            self.q.put(item, block=False)
            return True
        except Full:
            if self.drop_newest:
                return False
            # Drop an oldest frame to make space
            try:
                _ = self.q.get_nowait()
                self.q.put(item, block=False)
                return True
            except Empty:
                return False

    def _run(self) -> None:
        while not self._stop or not self.q.empty():
            try:
                base_name, rgb, idx = self.q.get(timeout=0.2)
            except Empty:
                continue
            try:
                # Convert RGB -> BGR for OpenCV encoders
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if self.codec == "jpg" or self.codec == "jpeg":
                    ok, buf = cv2.imencode(
                        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                    )
                    if ok:
                        path = os.path.join(self.out_dir, f"{base_name}_{idx}.jpg")
                        with open(path, "wb") as f:
                            f.write(buf.tobytes())
                else:
                    # Default to PNG for compatibility with existing pipelines
                    ok, buf = cv2.imencode(".png", bgr)
                    if ok:
                        path = os.path.join(self.out_dir, f"{base_name}_{idx}.png")
                        with open(path, "wb") as f:
                            f.write(buf.tobytes())
            finally:
                self.q.task_done()

    def close(self, wait: bool = True) -> None:
        """Signal the worker to stop and optionally wait for drain."""
        self._stop = True
        if wait:
            self._worker.join()

