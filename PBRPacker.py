import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QProgressBar, 
                             QCheckBox, QMessageBox, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap, QDesktopServices

# --- STYLESHEET ---
dark_theme_qss = """
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 14px;
}
QLabel[previewZone="true"] {
    background-color: #252526;
    border: 2px dashed #3e3e42;
    border-radius: 8px;
    padding: 10px;
    qproperty-alignment: AlignCenter;
}
QLabel[previewZone="true"]:hover {
    background-color: #2d2d30;
    border: 2px dashed #007acc;
}
QPushButton {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
}
QPushButton:hover { background-color: #0098ff; }
QPushButton:pressed { background-color: #005a9e; }
QPushButton:disabled { background-color: #333333; color: #777777; }
QProgressBar {
    background-color: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    text-align: center;
    color: white;
}
QProgressBar::chunk { background-color: #007acc; border-radius: 5px; }
QCheckBox::indicator {
    width: 18px; height: 18px;
    border-radius: 4px;
    border: 2px solid #3e3e42;
    background-color: #252526;
}
QCheckBox::indicator:checked { background-color: #007acc; border: 2px solid #007acc; }
"""

# --- DRAG AND DROP LABEL ---
class ImagePreviewLabel(QLabel):
    file_loaded = pyqtSignal(str)

    def __init__(self, text):
        super().__init__(text)
        self.setProperty("previewZone", "true")
        self.setAcceptDrops(True)
        self.filepath = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            self.load_image(filepath)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            filepath, _ = QFileDialog.getOpenFileName(self, "Select Texture", "", "Images (*.png *.jpg *.jpeg *.tga *.tif)")
            if filepath:
                self.load_image(filepath)

    def load_image(self, filepath):
        self.filepath = filepath
        pixmap = QPixmap(filepath).scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        self.file_loaded.emit(filepath)


# --- PROCESSING THREAD ---
class ProcessThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, paths, output_dir):
        super().__init__()
        self.paths = paths
        self.output_dir = output_dir

    def run(self):
        try:
            self.progress.emit(10)
            
            # Determine target size based on first loaded texture
            target_size = None
            for p in self.paths.values():
                if p:
                    with Image.open(p) as img:
                        target_size = img.size
                        break
            
            if not target_size:
                raise ValueError("No textures loaded!")

            self.progress.emit(20)

            # Helper to load, resize, and convert to numpy array (float 0-1)
            def load_channel(path, default_val, channels=1):
                if path and os.path.exists(path):
                    img = Image.open(path).convert('RGBA').resize(target_size, Image.Resampling.LANCZOS)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    if channels == 1:
                        return arr[:, :, 0] # Return Red channel for grayscale
                    return arr[:, :, :channels]
                else:
                    shape = (target_size[1], target_size[0]) if channels == 1 else (target_size[1], target_size[0], channels)
                    return np.full(shape, default_val, dtype=np.float32)

            self.progress.emit(30)
            
            # Load arrays
            base_color = load_channel(self.paths['basecolor'], 1.0, channels=3)
            ao = load_channel(self.paths['ao'], 1.0, channels=1)
            transparency = load_channel(self.paths['transparency'], 1.0, channels=1)
            
            self.progress.emit(50)
            
            normal = load_channel(self.paths['normal'], [0.5, 1.0, 0.5], channels=3)
            metallic = load_channel(self.paths['metallic'], 0.0, channels=1)
            smoothness = load_channel(self.paths['smoothness'], 0.5, channels=1)

            self.progress.emit(70)

            # 1. BaseAOTransparency (RGB = BaseColor * AO, A = Transparency)
            out_color = np.zeros((target_size[1], target_size[0], 4), dtype=np.float32)
            out_color[:, :, :3] = base_color * ao[..., np.newaxis]
            out_color[:, :, 3] = transparency
            
            img_color = Image.fromarray((np.clip(out_color, 0, 1) * 255).astype(np.uint8), 'RGBA')
            color_path = os.path.join(self.output_dir, "BaseAOTransparency.png")
            img_color.save(color_path, optimize=True, compress_level=9)

            self.progress.emit(85)

            # 2. NMS (R=Normal.x, G=Normal.y, B=Metallic, A=Smoothness)
            out_nms = np.zeros((target_size[1], target_size[0], 4), dtype=np.float32)
            out_nms[:, :, 0] = normal[:, :, 0] # Normal X
            out_nms[:, :, 1] = normal[:, :, 1] # Normal Y
            out_nms[:, :, 2] = metallic        # Metallic
            out_nms[:, :, 3] = smoothness      # Smoothness

            img_nms = Image.fromarray((np.clip(out_nms, 0, 1) * 255).astype(np.uint8), 'RGBA')
            nms_path = os.path.join(self.output_dir, "NMS.png")
            img_nms.save(nms_path, optimize=True, compress_level=9)

            self.progress.emit(100)
            self.finished.emit(self.output_dir)

        except Exception as e:
            self.error.emit(str(e))


# --- MAIN APP WINDOW ---
class TexturePackerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PBR Texture Packer Pro")
        self.resize(600, 500)
        self.setStyleSheet(dark_theme_qss)
        self.labels = {}
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        grid = QGridLayout()
        slots = ['BaseColor', 'AO', 'Transparency', 'Normal', 'Metallic', 'Smoothness']
        
        for i, name in enumerate(slots):
            lbl = ImagePreviewLabel(f"Click or Drop\n{name}")
            self.labels[name.lower()] = lbl
            grid.addWidget(lbl, i // 3, i % 3)
            
        layout.addLayout(grid)

        # Options
        self.auto_open_cb = QCheckBox("Open output folder when done")
        self.auto_open_cb.setChecked(True)
        layout.addWidget(self.auto_open_cb)

        # Progress and Button
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.btn_pack = QPushButton("Process and Pack Textures")
        self.btn_pack.clicked.connect(self.process_textures)
        layout.addWidget(self.btn_pack)

        self.setLayout(layout)

    def process_textures(self):
        paths = {name: lbl.filepath for name, lbl in self.labels.items()}
        
        if not any(paths.values()):
            QMessageBox.warning(self, "Error", "Please load at least one texture.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir:
            return

        self.btn_pack.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.thread = ProcessThread(paths, out_dir)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_finished(self, output_dir):
        self.btn_pack.setEnabled(True)
        self.progress_bar.hide()
        QMessageBox.information(self, "Success", "Textures packed successfully!")
        
        if self.auto_open_cb.isChecked():
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def on_error(self, err_msg):
        self.btn_pack.setEnabled(True)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err_msg}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TexturePackerApp()
    window.show()
    sys.exit(app.exec_())
