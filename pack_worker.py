import os

import numpy as np
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal


def build_packed_arrays(paths, ao_intensity=1.0, invert_normal_y=False):
    images = {}
    max_w, max_h = 1, 1
    for key, path in paths.items():
        if path and os.path.exists(path):
            img = Image.open(path).convert("RGBA")
            images[key] = img
            max_w = max(max_w, img.width)
            max_h = max(max_h, img.height)

    if not images:
        raise ValueError("No valid images provided.")

    def get_array(key, default_color):
        if key in images:
            img = images[key]
            if img.width != max_w or img.height != max_h:
                img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
            return np.array(img, dtype=np.float32)
        array = np.empty((max_h, max_w, 4), dtype=np.float32)
        array[:] = default_color
        return array

    base_color = get_array("BaseColor", (255, 255, 255, 255))
    ao = get_array("AO", (255, 255, 255, 255))[..., 0]
    ao_channel = np.clip((ao / 255.0) ** max(ao_intensity, 0.00001), 0.0, 1.0)
    base_ao_rgb = base_color[..., :3] * ao_channel[..., np.newaxis]

    alpha_arr = get_array("Alpha", (255, 255, 255, 255))
    transparency = alpha_arr[..., 0] if "Alpha" in images else base_color[..., 3]

    out_base_alpha = np.empty((max_h, max_w, 4), dtype=np.uint8)
    out_base_alpha[..., :3] = np.clip(base_ao_rgb, 0, 255).astype(np.uint8)
    out_base_alpha[..., 3] = np.clip(transparency, 0, 255).astype(np.uint8)

    normal = get_array("Normal", (127, 127, 255, 255))
    if invert_normal_y:
        normal = normal.copy()
        normal[..., 1] = 255 - normal[..., 1]
    metallic = get_array("Metallic", (0, 0, 0, 255))[..., 0]
    smoothness = get_array("Smoothness", (127, 127, 127, 255))[..., 0]

    out_nms = np.empty((max_h, max_w, 4), dtype=np.uint8)
    out_nms[..., 0] = normal[..., 0]
    out_nms[..., 1] = normal[..., 1]
    out_nms[..., 2] = metallic
    out_nms[..., 3] = smoothness
    return out_base_alpha, out_nms


class PackWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, object, object)

    def __init__(self, paths, out_dir, ao_intensity=1.0, invert_normal_y=False):
        super().__init__()
        self.paths = paths
        self.out_dir = out_dir
        self.ao_intensity = ao_intensity
        self.invert_normal_y = invert_normal_y

    def run(self):
        try:
            self.progress.emit(10, "Loading images...")
            self.progress.emit(50, "Processing BaseAOTransparency...")
            self.progress.emit(70, "Processing NMS...")
            out_base_alpha, out_nms = build_packed_arrays(self.paths, self.ao_intensity, self.invert_normal_y)
            self.progress.emit(85, "Saving textures...")
            os.makedirs(self.out_dir, exist_ok=True)
            Image.fromarray(out_base_alpha, "RGBA").save(
                os.path.join(self.out_dir, "BaseAOTransparency.png"),
                optimize=True,
                compress_level=9,
            )
            Image.fromarray(out_nms, "RGBA").save(
                os.path.join(self.out_dir, "NMS.png"),
                optimize=True,
                compress_level=9,
            )
            self.progress.emit(100, "Done!")
            self.finished.emit(True, "Textures packed successfully.", out_base_alpha, out_nms)
        except Exception as error:
            self.finished.emit(False, str(error), None, None)
