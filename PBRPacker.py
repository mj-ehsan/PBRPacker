import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QProgressBar, QCheckBox, QGridLayout)
from PyQt5.QtGui import (QPixmap, QPainter, QColor, QRadialGradient, QBrush, 
                         QImage, QPainterPath, QPen, QFont)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QRect, QRectF
from PyQt5.QtGui import QDesktopServices

# ---------------------------------------------------------
# Worker Thread for Texture Packing Logic
# ---------------------------------------------------------
class PackWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, paths, out_dir):
        super().__init__()
        self.paths = paths
        self.out_dir = out_dir

    def run(self):
        try:
            self.progress.emit(10, "Loading images...")
            images = {}
            max_w, max_h = 1, 1

            # Load images and find max dimensions
            for key, path in self.paths.items():
                if path and os.path.exists(path):
                    img = Image.open(path).convert("RGBA")
                    images[key] = img
                    max_w = max(max_w, img.width)
                    max_h = max(max_h, img.height)

            if not images:
                self.finished.emit(False, "No valid images provided.")
                return

            self.progress.emit(30, f"Resizing to {max_w}x{max_h}...")
            
            # Helper to get numpy array of an image or a default solid color
            def get_array(key, default_color):
                if key in images:
                    img = images[key]
                    if img.width != max_w or img.height != max_h:
                        img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
                    return np.array(img, dtype=np.float32)
                else:
                    arr = np.empty((max_h, max_w, 4), dtype=np.float32)
                    arr[:] = default_color
                    return arr

            # Defaults
            # BaseColor: (255, 255, 255, 255)
            # AO: 255
            # Metallic: 0
            # Smoothness: 127
            # Normal: (127, 255, 127, 255) -> vec3(0.5, 1.0, 0.5)
            # Alpha: uses BaseColor.a if not provided
            
            self.progress.emit(50, "Processing BaseAOTransparency...")
            base_color = get_array("BaseColor", (255, 255, 255, 255))
            ao = get_array("AO", (255, 255, 255, 255))[..., 0] # Use R channel
            
            # BaseAOTransparency = vec4(BaseColor.rgb * AO, Transparency)
            # $$ BaseColor.rgb \times AO $$
            base_ao_rgb = base_color[..., :3] * (ao[..., np.newaxis] / 255.0)
            
            alpha_arr = get_array("Alpha", (255, 255, 255, 255))
            if "Alpha" in images:
                transparency = alpha_arr[..., 0] # Use R channel if explicit map provided
            else:
                transparency = base_color[..., 3] # Otherwise use BaseColor alpha

            out_base_ao = np.empty((max_h, max_w, 4), dtype=np.uint8)
            out_base_ao[..., :3] = np.clip(base_ao_rgb, 0, 255).astype(np.uint8)
            out_base_ao[..., 3] = np.clip(transparency, 0, 255).astype(np.uint8)

            self.progress.emit(70, "Processing NMS...")
            normal = get_array("Normal", (127, 255, 127, 255))
            metallic = get_array("Metallic", (0, 0, 0, 255))[..., 0]
            smoothness = get_array("Smoothness", (127, 127, 127, 255))[..., 0]

            # NMS = vec4(Normal.x, Normal.y, Metallic, Smoothness)
            out_nms = np.empty((max_h, max_w, 4), dtype=np.uint8)
            out_nms[..., 0] = normal[..., 0]   # Normal X
            out_nms[..., 1] = normal[..., 1]   # Normal Y
            out_nms[..., 2] = metallic         # Metallic
            out_nms[..., 3] = smoothness       # Smoothness

            self.progress.emit(85, "Saving textures...")
            
            os.makedirs(self.out_dir, exist_ok=True)
            
            img_base_ao = Image.fromarray(out_base_ao, "RGBA")
            img_base_ao.save(os.path.join(self.out_dir, "BaseAOTransparency.png"), 
                             optimize=True, compress_level=9)

            img_nms = Image.fromarray(out_nms, "RGBA")
            img_nms.save(os.path.join(self.out_dir, "NMS.png"), 
                         optimize=True, compress_level=9)

            self.progress.emit(100, "Done!")
            self.finished.emit(True, "Textures packed successfully.")

        except Exception as e:
            self.finished.emit(False, str(e))


# ---------------------------------------------------------
# UI: Custom Image Preview Widget
# ---------------------------------------------------------
class ImagePreviewWidget(QWidget):
    fileDropped = pyqtSignal(str, str) # map_type, file_path

    def __init__(self, map_type):
        super().__init__()
        self.map_type = map_type
        self.file_path = None
        self.pixmap = None
        self.setAcceptDrops(True)
        self.setMinimumSize(140, 140)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)  # Enable mouse tracking

    def set_image(self, path):
        self.file_path = path
        if path and os.path.exists(path):
            self.pixmap = QPixmap(path)
        else:
            self.pixmap = None
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.set_image(path)
            self.fileDropped.emit(self.map_type, path)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            path, _ = QFileDialog.getOpenFileName(self, f"Select {self.map_type} Map", "", "Images (*.png *.jpg *.jpeg *.tga *.tif)")
            if path:
                self.set_image(path)
                self.fileDropped.emit(self.map_type, path)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = self.rect()
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 10, 10)

        # Draw Background
        painter.fillPath(path, QColor("#2a2a30"))

        # Draw Image if exists (Full Frame, KeepAspectRatioByExpanding)
        if self.pixmap and not self.pixmap.isNull():
            painter.setClipPath(path)
            scaled_pix = self.pixmap.scaled(rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            
            # Center the image
            x = (rect.width() - scaled_pix.width()) // 2
            y = (rect.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)
            
            # Draw gradient at the bottom for text readability
            grad = QRadialGradient(rect.width()/2, rect.height(), rect.width())
            grad.setColorAt(0, QColor(0, 0, 0, 180))
            grad.setColorAt(1, QColor(0, 0, 0, 0))
            painter.fillRect(rect.x(), rect.height() - 40, rect.width(), 40, QBrush(grad))
            painter.setClipping(False)
        else:
            # Dashed border for empty state
            pen = QPen(QColor("#4a4a55"))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPath(path)

        # Draw Map Type Label inside at the bottom
        painter.setPen(QColor("#ffffff"))
        font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(font)
        text_rect = QRect(rect.x(), rect.height() - 30, rect.width(), 30)
        painter.drawText(text_rect, Qt.AlignCenter, self.map_type)

        painter.end()


# ---------------------------------------------------------
# UI: Main Window
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PBR Texture Packer")
        self.resize(700, 500)
        
        # Mouse tracking for background glow
        self.mouse_pos = None
        self.setMouseTracking(True)
        
        # Generate static noise texture for sparkles
        noise_size = 256
        noise_arr = np.random.randint(0, 255, (noise_size, noise_size), dtype=np.uint8)
        rgba_noise = np.zeros((noise_size, noise_size, 4), dtype=np.uint8)
        rgba_noise[..., 0:3] = 255 # White sparkles
        rgba_noise[..., 3] = noise_arr // 3 # Subtle opacity
        self.noise_image = QImage(rgba_noise.data, noise_size, noise_size, QImage.Format_ARGB32)
        self.noise_brush = QBrush(self.noise_image)

        self.paths = {
            "BaseColor": None, "AO": None, "Metallic": None, 
            "Smoothness": None, "Normal": None, "Alpha": None
        }

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_widget.setMouseTracking(True)
        
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Previews Grid
        grid = QGridLayout()
        grid.setSpacing(15)
        
        self.previews = {}
        maps = ["BaseColor", "AO", "Metallic", "Smoothness", "Normal", "Alpha"]
        
        for i, m in enumerate(maps):
            pw = ImagePreviewWidget(m)
            pw.fileDropped.connect(self.update_path)
            self.previews[m] = pw
            grid.addWidget(pw, i // 3, i % 3)
            
        layout.addLayout(grid)

        # Output Directory
        out_layout = QHBoxLayout()
        self.btn_out = QPushButton("Select Output Directory")
        self.btn_out.clicked.connect(self.select_output)
        self.lbl_out = QLabel("No directory selected")
        out_layout.addWidget(self.btn_out)
        out_layout.addWidget(self.lbl_out, 1)
        layout.addLayout(out_layout)

        # Options & Progress
        bot_layout = QHBoxLayout()
        self.chk_open = QCheckBox("Open Output Folder when done")
        self.chk_open.setChecked(True)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - Waiting")
        self.progress_bar.hide()
        
        self.btn_pack = QPushButton("Pack Textures")
        self.btn_pack.setMinimumHeight(40)
        self.btn_pack.clicked.connect(self.start_packing)

        bot_layout.addWidget(self.chk_open)
        bot_layout.addWidget(self.progress_bar, 1)
        bot_layout.addWidget(self.btn_pack)
        layout.addLayout(bot_layout)

        # Install event filters on all child widgets after they're created
        self.installEventFilters(self)

    def installEventFilters(self, widget):
        """Recursively install event filters on all child widgets."""
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)
            child.setMouseTracking(True)

    def eventFilter(self, obj, event):
        if event.type() == event.MouseMove:
            # Convert global position to main window coordinates
            self.mouse_pos = self.mapFromGlobal(event.globalPos())
            self.update()
        return super().eventFilter(obj, event)

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Base dark background
        painter.fillRect(self.rect(), QColor("#1e1e22"))
        
        if self.mouse_pos:
            x, y = self.mouse_pos.x(), self.mouse_pos.y()
            radius = 450.0 # Wider glow
            
            gradient = QRadialGradient(x, y, radius)
            base_color = QColor(100, 150, 255) # Soft bluish glow
            
            # Inverse Square Approximation via color stops
            c0 = QColor(base_color); c0.setAlpha(45)
            c1 = QColor(base_color); c1.setAlpha(20)
            c2 = QColor(base_color); c2.setAlpha(8)
            c3 = QColor(base_color); c3.setAlpha(2)
            c4 = QColor(base_color); c4.setAlpha(0)
            
            gradient.setColorAt(0.0, c0)
            gradient.setColorAt(0.15, c1)
            gradient.setColorAt(0.35, c2)
            gradient.setColorAt(0.6, c3)
            gradient.setColorAt(1.0, c4)

            # Draw smooth glow
            painter.fillRect(self.rect(), QBrush(gradient))

            # Masked Shiny Grain Setup
            # Create a temporary pixmap to mask the grain perfectly with the glow
            glow_pix = QPixmap(self.size())
            glow_pix.fill(Qt.transparent)
            glow_painter = QPainter(glow_pix)
            
            # Draw grain tile
            glow_painter.fillRect(glow_pix.rect(), self.noise_brush)
            
            # Mask it out using the gradient
            glow_painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            glow_painter.fillRect(glow_pix.rect(), QBrush(gradient))
            glow_painter.end()
            
            # Draw the masked sparkles over the background
            painter.drawPixmap(0, 0, glow_pix)
            
        painter.end()

    def apply_theme(self):
        # We handle the main window background in paintEvent. 
        # Here we style buttons, labels, and progress bar.
        self.setStyleSheet("""
            QWidget {
                color: #e0e0e0;
                font-family: "Segoe UI", sans-serif;
            }
            QPushButton {
                background-color: #3a3a45;
                border: 1px solid #5a5a65;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a55;
                border: 1px solid #7a7a85;
            }
            QPushButton:pressed {
                background-color: #2a2a35;
            }
            QProgressBar {
                background-color: #2a2a30;
                border: 1px solid #4a4a55;
                border-radius: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background-color: #2a2a30;
                border: 1px solid #4a4a55;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 1px solid #4CAF50;
            }
        """)

    def update_path(self, map_type, path):
        self.paths[map_type] = path

    def select_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.out_dir = dir_path
            self.lbl_out.setText(dir_path)

    def start_packing(self):
        if not hasattr(self, 'out_dir') or not self.out_dir:
            self.select_output()
            if not hasattr(self, 'out_dir') or not self.out_dir:
                return

        self.btn_pack.setEnabled(False)
        self.progress_bar.show()
        
        self.worker = PackWorker(self.paths, self.out_dir)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.packing_finished)
        self.worker.start()

    def update_progress(self, val, text):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(f"%p% - {text}")

    def packing_finished(self, success, msg):
        self.btn_pack.setEnabled(True)
        if success:
            self.progress_bar.setFormat("100% - Finished!")
            if self.chk_open.isChecked():
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))
        else:
            self.progress_bar.setFormat(f"Error: {msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())