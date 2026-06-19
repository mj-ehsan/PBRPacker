import os
import time
import numpy as np
import imagecodecs
from PyQt5.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- pack_worker.py (new class) ---

class BatchPackWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, items):
        """
        items: list of dicts with keys:
            'base_name', 'base_alpha' (array), 'nms' (array), 'output_dir' (str)
        """
        super().__init__()
        self.items = items

    def run(self):
        try:
            total = len(self.items)
            # We'll save two files per item, so total tasks = total * 2
            tasks = []
            for idx, item in enumerate(self.items):
                out_dir = item['output_dir']
                os.makedirs(out_dir, exist_ok=True)
                base_path = os.path.join(out_dir, "BaseAOTransparency.png")
                nms_path = os.path.join(out_dir, "NMS.png")
                tasks.append((item['base_alpha'], base_path, idx, "BaseAO"))
                tasks.append((item['nms'], nms_path, idx, "NMS"))

            # Use ThreadPoolExecutor to save in parallel
            total_tasks = len(tasks)
            completed = 0
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for arr, path, idx, name in tasks:
                    futures.append(executor.submit(self._save_image, arr, path, idx, name))
                for future in as_completed(futures):
                    future.result()  # raise any exception
                    completed += 1
                    percent = int((completed / total_tasks) * 100)
                    self.progress.emit(percent, f"Saving {completed}/{total_tasks}")
            self.finished.emit(True, "Batch saved successfully.")
        except Exception as e:
            self.finished.emit(False, str(e))

    def _save_image(self, array, filepath, idx, name):
        # imagecodecs write
        imagecodecs.imwrite(filepath, array, level=3)
        # Optionally log time
        # print(f"Saved {os.path.basename(filepath)}")

class PackWorker(QThread):
    """
    Saves the pre-composed BaseAOTransparency and NMS textures to disk.
    The heavy lifting (AO, normal generation, packing) is done on the GPU
    by pbr_renderer.py; this worker only handles file I/O.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, object, object)

    def __init__(self, base_alpha_array, nms_array, out_dir):
        super().__init__()
        self.base_alpha_array = base_alpha_array
        self.nms_array = nms_array
        self.out_dir = out_dir

    def run(self):
        try:
            self.progress.emit(10, "Saving textures…")
            os.makedirs(self.out_dir, exist_ok=True)

            base_path = os.path.join(self.out_dir, "BaseAOTransparency.png")
            nms_path  = os.path.join(self.out_dir, "NMS.png")
            
            def save_image(array, filepath):
                # Track the start time on this specific thread
                start_time = time.perf_counter()
                
                # imagecodecs writes directly from the numpy buffer.
                imagecodecs.imwrite(filepath, array, level=3)
                
                # Calculate elapsed time
                elapsed_time = time.perf_counter() - start_time
                filename = os.path.basename(filepath)
                print(f"[Thread Log] Compression/Write for {filename} took {elapsed_time:.4f} seconds.")

            # Write both files in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(save_image, self.base_alpha_array, base_path),
                    executor.submit(save_image, self.nms_array, nms_path),
                ]
                for future in as_completed(futures):
                    future.result()  # raise any exception that occurred

            self.progress.emit(100, "Done!")
            self.finished.emit(True, "Textures saved successfully.",
                               self.base_alpha_array, self.nms_array)

        except Exception as error:
            self.finished.emit(False, str(error), None, None)