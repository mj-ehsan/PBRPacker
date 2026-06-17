import os
import time
import traceback
import numpy as np
import imagecodecs
from PyQt5.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor, as_completed

class PackWorker(QThread):
    """
    Saves the pre-composed BaseAOTransparency and NMS textures to disk.
    The heavy lifting (AO, normal generation, packing) is done on the GPU
    by pbr_renderer.py; this worker only handles file I/O.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, object, object)
    log = pyqtSignal(str)

    def __init__(self, jobs, out_dir, max_workers=None):
        super().__init__()
        self.jobs = jobs
        self.out_dir = out_dir
        self.max_workers = max_workers or 2

    def run(self):
        try:
            self.progress.emit(5, "Saving textures…")
            os.makedirs(self.out_dir, exist_ok=True)
            total_files = max(len(self.jobs) * 2, 1)
            completed_files = 0

            def save_image(array, filepath):
                start_time = time.perf_counter()
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                imagecodecs.imwrite(filepath, array, level=3)
                elapsed_time = time.perf_counter() - start_time
                filename = os.path.basename(filepath)
                self.log.emit(f"[Save] {filename} in {elapsed_time:.3f}s")
                return filepath

            futures = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for job in self.jobs:
                    base_path = os.path.join(self.out_dir, job["base_path"])
                    nms_path = os.path.join(self.out_dir, job["nms_path"])
                    futures.append(executor.submit(save_image, job["base_alpha"], base_path))
                    futures.append(executor.submit(save_image, job["nms"], nms_path))
                for future in as_completed(futures):
                    future.result()
                    completed_files += 1
                    progress = 5 + int((completed_files / total_files) * 95)
                    self.progress.emit(progress, f"Saved {completed_files}/{total_files} files…")
            self.progress.emit(100, "Done!")
            preview_base = self.jobs[-1]["base_alpha"] if self.jobs else None
            preview_nms = self.jobs[-1]["nms"] if self.jobs else None
            self.finished.emit(True, "Textures saved successfully.", preview_base, preview_nms)

        except Exception as error:
            self.log.emit(traceback.format_exc().strip())
            self.finished.emit(False, str(error), None, None)
