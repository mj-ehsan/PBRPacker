import os
import numpy as np
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor, as_completed


class PackWorker(QThread):
    """
    Saves the pre-composed BaseAOTransparency and NMS textures to disk.
    The heavy lifting (AO, normal generation, packing) is done on the GPU
    by pbr_renderer.py; this worker only handles file I/O.
    """
    progress = pyqtSignal(int, str)
    # finished: success (bool), message (str),
    #           base_alpha_array (np.uint8), nms_array (np.uint8)
    finished = pyqtSignal(bool, str, object, object)

    def __init__(self, base_alpha_array, nms_array, out_dir):
        """
        :param base_alpha_array: numpy uint8 RGBA array – BaseAOTransparency
        :param nms_array:        numpy uint8 RGBA array – NMS (normal, metallic, smoothness)
        :param out_dir:          directory where BaseAOTransparency.png and NMS.png are saved
        """
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
                Image.fromarray(array, "RGBA").save(
                    filepath, optimize=True, compress_level=9
                )

            # Write both files in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(save_image, self.base_alpha_array, base_path),
                    executor.submit(save_image, self.nms_array, nms_path),
                ]
                for future in as_completed(futures):
                    future.result()   # raise any exception that occurred

            self.progress.emit(100, "Done!")
            self.finished.emit(True, "Textures saved successfully.",
                               self.base_alpha_array, self.nms_array)

        except Exception as error:
            self.finished.emit(False, str(error), None, None)