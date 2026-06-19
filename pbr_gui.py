import os
import time   # <-- added for log timestamps
import numpy as np
from PyQt5.QtCore import pyqtSignal, QRect, QRectF, Qt, QUrl, QTimer  # QTimer added
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
    QLinearGradient,          # <-- added for gradient fade
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pbr_renderer import PBRRendererWidget
from pack_worker import BatchPackWorker

# -------------------------------------------------------------------
# Helper: common suffixes per map type (unchanged)
# -------------------------------------------------------------------
MAP_SUFFIXES = {
    "BaseColor": [
        "_BaseColor", "_Albedo", "_Diffuse", "_Color", "_D", "_col",
        "_basecolor", "_albedo", "_diffuse", "_color", "_diff",
    ],
    "AO": [
        "_AO", "_AmbientOcclusion", "_Occlusion",
        "_ao", "_ambientocclusion", "_occlusion",
    ],
    "Metallic": [
        "_Metallic", "_Metalness", "_Metal",
        "_metallic", "_metalness", "_metal",
    ],
    "Smoothness": [
        "_Smoothness", "_Roughness", "_Smooth", "_Rough", "_rgh",
        "_smoothness", "_roughness", "_smooth", "_rough",
    ],
    "Normal": [
        "_Normal", "_NRM", "_N",
        "_normal", "_nrm", "_n",
    ],
    "Alpha": [
        "_Alpha", "_Opacity", "_Mask",
        "_alpha", "_opacity", "_mask",
    ],
}

# -------------------------------------------------------------------
# Auto‑assignment dialog (unchanged)
# -------------------------------------------------------------------
class AutoAssignDialog(QDialog):
    def __init__(self, base_name, candidates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto‑assign Texture Maps")
        self.setMinimumWidth(400)
        self.candidates = candidates

        layout = QVBoxLayout(self)

        info = QLabel(
            f"<b>Base name:</b> {base_name}<br>"
            f"Found {len(candidates)} additional map(s) in the same folder.<br>"
            "Choose the ones you want to assign:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.checkboxes = {}
        for map_type, file_path in candidates.items():
            cb = QCheckBox(f"{map_type}: {os.path.basename(file_path)}")
            cb.setChecked(True)
            self.checkboxes[map_type] = cb
            layout.addWidget(cb)

        toggle_layout = QHBoxLayout()
        self.select_all_cb = QCheckBox("Select all")
        self.select_all_cb.setChecked(True)
        self.select_all_cb.toggled.connect(self._toggle_all)
        toggle_layout.addWidget(self.select_all_cb)
        toggle_layout.addStretch()
        layout.addLayout(toggle_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if parent:
            self.setStyleSheet(parent.styleSheet())

    def _toggle_all(self, checked):
        for cb in self.checkboxes.values():
            cb.setChecked(checked)

    def selected_maps(self):
        return {
            mt: path for mt, path in self.candidates.items()
            if self.checkboxes[mt].isChecked()
        }


# -------------------------------------------------------------------
# ImagePreviewWidget (unchanged)
# -------------------------------------------------------------------
class ImagePreviewWidget(QWidget):
    fileDropped = pyqtSignal(str, str)
    cleared = pyqtSignal(str)

    def __init__(self, map_type):
        super().__init__()
        self.map_type = map_type
        self.file_path = None
        self.pixmap = None
        self.setAcceptDrops(True)
        self.setMinimumSize(140, 140)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.clear_button_rect = QRect(8, 8, 22, 22)

    def set_image(self, path):
        self.file_path = path
        self.pixmap = QPixmap(path) if path and os.path.exists(path) else None
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
            if self.pixmap and self.clear_button_rect.contains(event.pos()):
                self.set_image(None)
                self.cleared.emit(self.map_type)
                return
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {self.map_type} Map", "",
                "Images (*.png *.jpg *.jpeg *.tga *.tif)"
            )
            if path:
                self.set_image(path)
                self.fileDropped.emit(self.map_type, path)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = self.rect()
        rounded_path = QPainterPath()
        rounded_path.addRoundedRect(QRectF(rect), 10, 10)
        painter.fillPath(rounded_path, QColor("#2a2a30"))
        if self.pixmap and not self.pixmap.isNull():
            painter.setClipPath(rounded_path)
            scaled_pix = self.pixmap.scaled(
                rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            x = (rect.width() - scaled_pix.width()) // 2
            y = (rect.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)
            gradient = QRadialGradient(rect.width() / 2, rect.height(), rect.width())
            gradient.setColorAt(0, QColor(0, 0, 0, 180))
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            painter.fillRect(
                rect.x(), rect.height() - 40, rect.width(), 40, QBrush(gradient)
            )
            painter.setClipping(False)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(24, 24, 28, 215))
            painter.drawEllipse(self.clear_button_rect)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            inset = 7
            painter.drawLine(
                self.clear_button_rect.left() + inset,
                self.clear_button_rect.top() + inset,
                self.clear_button_rect.right() - inset,
                self.clear_button_rect.bottom() - inset,
            )
            painter.drawLine(
                self.clear_button_rect.right() - inset,
                self.clear_button_rect.top() + inset,
                self.clear_button_rect.left() + inset,
                self.clear_button_rect.bottom() - inset,
            )
        else:
            pen = QPen(QColor("#4a4a55"))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPath(rounded_path)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        text_rect = QRect(rect.x(), rect.height() - 30, rect.width(), 30)
        painter.drawText(text_rect, Qt.AlignCenter, self.map_type)
        painter.end()


# -------------------------------------------------------------------
# LogOverlay : transparent terminal‑log overlay (NEW)
# -------------------------------------------------------------------
class LogOverlay(QWidget):
    """
    A completely transparent widget that draws up to 10 recent log lines.
    Each line fades out after 15 seconds.
    After the 6th line a linear gradient fades the remaining text to transparent.
    No frame, no background – just shadowed white text.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)   # clicks pass through
        self.entries = []          # (message, perf_counter timestamp)
        self.font = QFont("Consolas", 10)
        self.line_height = QFontMetrics(self.font).height() + 2
        self.max_visible_lines = 10

        # Timer to repaint for smooth opacity animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)       # ~20 fps

    def add_log(self, message):
        self.entries.append((message, time.perf_counter()))
        # Keep the list from growing indefinitely
        if len(self.entries) > 200:
            self.entries = self.entries[-100:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setFont(self.font)

        now = time.perf_counter()
        # Remove entries older than 15 seconds (fully faded)
        self.entries = [(msg, t) for msg, t in self.entries if now - t <= 15.0]
        if not self.entries:
            return

        # Show newest lines at the top
        recent = list(reversed(self.entries))[:self.max_visible_lines]

        x = 10
        y = 5 + self.fontMetrics().ascent()          # first line baseline

        for i, (msg, t) in enumerate(recent):
            age = now - t
            alpha = max(0.0, 1.0 - age / 15.0)
            if alpha < 0.01:
                continue

            # Shadow (dark, semi‑transparent)
            shadow_color = QColor(0, 0, 0, int(alpha * 120))
            painter.setPen(shadow_color)
            painter.drawText(x + 1, y + 1, msg)

            # Main text (white)
            text_color = QColor(255, 255, 255, int(alpha * 255))
            painter.setPen(text_color)
            painter.drawText(x, y, msg)

            y += self.line_height

        # Gradient fade‑out after the 6th line
        if len(recent) > 6:
            start_y = 5 + 6 * self.line_height   # top of 7th line area
            gradient = QLinearGradient(0, start_y, 0, self.height())
            gradient.setColorAt(0.0, QColor(0, 0, 0, 255))   # opaque mask
            gradient.setColorAt(1.0, QColor(0, 0, 0, 0))     # transparent mask

            painter.save()
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.fillRect(0, start_y, self.width(), self.height() - start_y, gradient)
            painter.restore()


# -------------------------------------------------------------------
# MainWindow (updated with log overlay and log() helper)
# -------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, worker_class):
        super().__init__()
        self.batch_dir = None
        self.worker_class = worker_class
        self.setWindowTitle("PBR Texture Packer for Unity")
        self.resize(1400, 800)
        self.mouse_pos = None
        self.ao_intensity = 1.0
        self.normal_gen_sigma = 1.0
        self.normal_gen_height = 1.0
        self.invert_normal_y = False
        self.setMouseTracking(True)
        noise_size = 256
        noise_arr = np.random.randint(0, 255, (noise_size, noise_size), dtype=np.uint8)
        rgba_noise = np.zeros((noise_size, noise_size, 4), dtype=np.uint8)
        rgba_noise[..., 0:3] = 255
        rgba_noise[..., 3] = noise_arr // 3
        self.noise_image = QImage(rgba_noise.data, noise_size, noise_size, QImage.Format_ARGB32)
        self.noise_brush = QBrush(self.noise_image)
        self.paths = {
            "BaseColor": None,
            "AO": None,
            "Metallic": None,
            "Smoothness": None,
            "Normal": None,
            "Alpha": None,
        }
        self.out_dir = None
        self._suppress_auto_assign = False
        self.init_ui()
        self.apply_theme()

    # ---------------------------------------------------------------
    # UI setup (added log overlay after the renderer)
    # ---------------------------------------------------------------
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_widget.setMouseTracking(True)
        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_panel.setMouseTracking(True)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        title_label = QLabel("Texture Maps")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.previews = {}
        for index, map_name in enumerate(
            ["BaseColor", "AO", "Metallic", "Smoothness", "Normal", "Alpha"]
        ):
            preview = ImagePreviewWidget(map_name)
            preview.fileDropped.connect(self.update_texture)
            preview.cleared.connect(self.clear_texture)
            self.previews[map_name] = preview
            grid.addWidget(preview, index // 3, index % 3)
        left_layout.addLayout(grid)

        normal_toggle_layout = QHBoxLayout()
        normal_toggle_layout.addStretch()
        self.chk_invert_normal_y = QCheckBox("Invert Y")
        self.chk_invert_normal_y.toggled.connect(self.update_normal_invert)
        normal_toggle_layout.addWidget(self.chk_invert_normal_y)
        normal_toggle_layout.addStretch()
        left_layout.addLayout(normal_toggle_layout)

        material_group = QGroupBox("Material Properties")
        material_layout = QVBoxLayout()
        ao_label_layout = QHBoxLayout()
        ao_label_layout.addWidget(QLabel("AO Intensity"))
        self.ao_value_label = QLabel("1.00x")
        self.ao_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ao_label_layout.addWidget(self.ao_value_label)
        material_layout.addLayout(ao_label_layout)
        self.ao_slider = QSlider(Qt.Horizontal)
        self.ao_slider.setRange(0, 200)
        self.ao_slider.setValue(100)
        self.ao_slider.valueChanged.connect(self.update_ao_intensity)
        material_layout.addWidget(self.ao_slider)
        sigma_label_layout = QHBoxLayout()
        sigma_label_layout.addWidget(QLabel("Normal Sigma"))
        self.normal_sigma_value_label = QLabel("1.00")
        self.normal_sigma_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sigma_label_layout.addWidget(self.normal_sigma_value_label)
        material_layout.addLayout(sigma_label_layout)
        self.normal_sigma_slider = QSlider(Qt.Horizontal)
        self.normal_sigma_slider.setRange(1, 500)
        self.normal_sigma_slider.setValue(100)
        self.normal_sigma_slider.valueChanged.connect(self.update_normal_generation)
        material_layout.addWidget(self.normal_sigma_slider)
        height_label_layout = QHBoxLayout()
        height_label_layout.addWidget(QLabel("Normal Height"))
        self.normal_height_value_label = QLabel("1.00")
        self.normal_height_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        height_label_layout.addWidget(self.normal_height_value_label)
        material_layout.addLayout(height_label_layout)
        self.normal_height_slider = QSlider(Qt.Horizontal)
        self.normal_height_slider.setRange(0, 200)
        self.normal_height_slider.setValue(100)
        self.normal_height_slider.valueChanged.connect(self.update_normal_generation)
        material_layout.addWidget(self.normal_height_slider)
        material_group.setLayout(material_layout)
        left_layout.addWidget(material_group)

        out_group = QGroupBox("Directories")
        out_layout = QVBoxLayout()
        out_dir_layout = QHBoxLayout()
        self.btn_out = QPushButton("Select Output Directory")
        self.btn_out.clicked.connect(self.select_output)
        self.lbl_out = QLabel("No directory selected")
        self.lbl_out.setWordWrap(True)
        out_dir_layout.addWidget(self.btn_out)
        out_dir_layout.addWidget(self.lbl_out, 1)

        batch_dir_layout = QHBoxLayout()
        self.btn_batch = QPushButton("Select Batch Directory")
        self.btn_batch.clicked.connect(self.select_batch_directory)
        self.lbl_batch = QLabel("No directory selected")
        self.lbl_batch.setWordWrap(True)
        batch_dir_layout.addWidget(self.btn_batch)
        batch_dir_layout.addWidget(self.lbl_batch, 1)
        out_layout.addLayout(batch_dir_layout)

        out_layout.addLayout(out_dir_layout)
        options_layout = QHBoxLayout()
        self.chk_open = QCheckBox("Open folder when done")
        self.chk_open.setChecked(True)
        options_layout.addWidget(self.chk_open)
        out_layout.addLayout(options_layout)
        out_group.setLayout(out_layout)
        left_layout.addWidget(out_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - Waiting")
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        self.btn_pack = QPushButton("Pack Textures")
        self.btn_pack.setMinimumHeight(40)
        self.btn_pack.clicked.connect(self.start_packing)
        left_layout.addWidget(self.btn_pack)
        left_layout.addStretch()

        self.btn_batch_process = QPushButton("Batch Process")
        self.btn_batch_process.setMinimumHeight(40)
        self.btn_batch_process.clicked.connect(self.process_batch)
        left_layout.addWidget(self.btn_batch_process)

        right_panel = QWidget()
        right_panel.setMouseTracking(True)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        preview_title = QLabel("3D PBR Preview")
        preview_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        preview_title.setAlignment(Qt.AlignCenter)
        preview_title.setFixedHeight(20)
        right_layout.addWidget(preview_title)
        self.preview_mode_label = QLabel("Previewing live input textures")
        self.preview_mode_label.setAlignment(Qt.AlignCenter)
        self.preview_mode_label.setFixedHeight(20)
        right_layout.addWidget(self.preview_mode_label)

        # Renderer widget
        self.pbr_renderer = PBRRendererWidget()
        right_layout.addWidget(self.pbr_renderer)

        # --- Log overlay placed on top of the renderer (top‑left) ---
        self.log_overlay = LogOverlay(self.pbr_renderer)
        self.log_overlay.setFixedSize(420, 200)
        self.log_overlay.move(10, 10)
        self.log_overlay.show()
        self.log("Ready.")    # initial message

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        right_layout.addWidget(reset_btn)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 900])
        layout = QVBoxLayout(main_widget)
        layout.addWidget(splitter)
        self.installEventFilters(self)

    # ---------------------------------------------------------------
    # Log helper (new)
    # ---------------------------------------------------------------
    def log(self, message):
        """Add a line to the transparent terminal overlay."""
        self.log_overlay.add_log(message)

    # ---------------------------------------------------------------
    # Batch Processing (now logs its progress)
    # ---------------------------------------------------------------
    def select_batch_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Batch Directory (containing texture sets)")
        if dir_path:
            self.batch_dir = dir_path
            self.lbl_batch.setText(dir_path)

    def _find_texture_set_in_folder(self, folder_path, base_name):
        result = {}
        for map_type, suffixes in MAP_SUFFIXES.items():
            for suffix in suffixes:
                for ext in ['.png', '.jpg', '.jpeg', '.tga', '.tif']:
                    candidate = os.path.join(folder_path, base_name + suffix + ext)
                    if os.path.isfile(candidate):
                        result[map_type] = candidate
                        break
                if map_type in result:
                    break
        return result

    def process_batch(self):
        if not self.batch_dir:
            QMessageBox.warning(self, "No Batch Directory", "Please select a batch directory first.")
            return
        if not self.out_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory first.")
            return

        self.btn_batch_process.setEnabled(False)
        self.btn_pack.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Scanning batch folder...")
        self.log("Batch processing started.")

        subdirs = [d for d in os.listdir(self.batch_dir)
                   if os.path.isdir(os.path.join(self.batch_dir, d))]
        if not subdirs:
            QMessageBox.information(self, "No Sets", "No subdirectories found in batch folder.")
            self.btn_batch_process.setEnabled(True)
            self.btn_pack.setEnabled(True)
            self.progress_bar.hide()
            self.log("No sets found.")
            return

        sets = []
        total = len(subdirs)
        for i, sub in enumerate(subdirs):
            self.progress_bar.setValue(int((i / total) * 50))
            self.progress_bar.setFormat(f"Scanning {sub}...")
            self.log(f"Scanning {sub}...")
            QApplication.processEvents()

            folder_path = os.path.join(self.batch_dir, sub)
            base_name = sub
            maps = self._find_texture_set_in_folder(folder_path, base_name)
            if not maps:
                all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                found_base = None
                for f in all_files:
                    name, ext = os.path.splitext(f)
                    for suffixes in MAP_SUFFIXES.values():
                        for suf in suffixes:
                            if name.lower().endswith(suf.lower()):
                                found_base = name[:-len(suf)]
                                break
                        if found_base:
                            break
                    if found_base:
                        break
                if found_base:
                    maps = self._find_texture_set_in_folder(folder_path, found_base)
                else:
                    continue

            sets.append({
                'base_name': sub,
                'maps': maps,
                'output_subdir': sub
            })

        if not sets:
            QMessageBox.information(self, "No Texture Sets", "No valid texture sets found.")
            self.btn_batch_process.setEnabled(True)
            self.btn_pack.setEnabled(True)
            self.progress_bar.hide()
            self.log("No valid sets found.")
            return

        results = []
        for idx, set_info in enumerate(sets):
            self.progress_bar.setValue(50 + int((idx / len(sets)) * 40))
            self.progress_bar.setFormat(f"Composing {set_info['base_name']}...")
            self.log(f"Composing {set_info['base_name']}...")
            QApplication.processEvents()

            # Clear previous textures to avoid cross‑contamination
            for map_name in self.paths:
                self.paths[map_name] = None
                self.pbr_renderer.load_input_texture(map_name, None)

            for map_type, path in set_info['maps'].items():
                self.paths[map_type] = path
                self.pbr_renderer.load_input_texture(map_type, path)

            self.pbr_renderer.set_ao_intensity(self.ao_intensity)
            self.pbr_renderer.set_normal_generation(self.normal_gen_sigma, self.normal_gen_height)
            self.pbr_renderer.set_normal_y_inverted(self.invert_normal_y)

            base_alpha, nms = self.pbr_renderer.get_composed_data()
            if base_alpha is None or nms is None:
                continue

            output_dir = os.path.join(self.out_dir, set_info['output_subdir'])
            results.append({
                'base_name': set_info['base_name'],
                'base_alpha': base_alpha,
                'nms': nms,
                'output_dir': output_dir
            })

        self.progress_bar.setValue(90)
        self.progress_bar.setFormat("Saving textures...")
        self.log("Saving packed textures...")
        QApplication.processEvents()

        self.batch_worker = BatchPackWorker(results)
        self.batch_worker.progress.connect(self.update_progress)
        self.batch_worker.finished.connect(self.batch_finished)
        self.batch_worker.start()

    def batch_finished(self, success, msg):
        self.btn_batch_process.setEnabled(True)
        self.btn_pack.setEnabled(True)

        # --- Clear previous textures to avoid cross‑contamination after batch ---
        for map_name in self.paths:
            self.paths[map_name] = None
            self.pbr_renderer.load_input_texture(map_name, None)

        # Also clear the preview widgets so they reflect the empty state
        for preview in self.previews.values():
            preview.set_image(None)
        # ---------------------------------------------------------------------

        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Batch complete!")
            self.log("Batch complete.")
            if self.chk_open.isChecked():
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))
        else:
            self.progress_bar.setFormat(f"Batch error: {msg}")
            self.log(f"Batch error: {msg}")
            QMessageBox.critical(self, "Batch Error", msg)
    
    # ---------------------------------------------------------------
    # Texture management (unchanged except for a log on auto‑assign)
    # ---------------------------------------------------------------
    def update_texture(self, map_type, path):
        self.paths[map_type] = path
        self.pbr_renderer.load_input_texture(map_type, path)
        self.preview_mode_label.setText("Previewing live input textures")
        if not self._suppress_auto_assign:
            self.attempt_auto_assign(map_type, path)

    def clear_texture(self, map_type):
        self.paths[map_type] = None
        self.pbr_renderer.load_input_texture(map_type, None)
        self.preview_mode_label.setText("Previewing live input textures")

    def attempt_auto_assign(self, dropped_map_type, dropped_path):
        missing = [t for t in self.paths if t != dropped_map_type and self.paths[t] is None]
        if not missing:
            return

        dir_path, filename = os.path.split(dropped_path)
        base, ext = os.path.splitext(filename)

        stripped_base = base
        suffixes_for_type = sorted(
            MAP_SUFFIXES.get(dropped_map_type, []), key=len, reverse=True
        )
        for suffix in suffixes_for_type:
            if base.lower().endswith(suffix.lower()):
                stripped_base = base[: -len(suffix)]
                break

        candidates = {}
        for mtype in missing:
            for suffix in MAP_SUFFIXES.get(mtype, []):
                candidate_name = stripped_base + suffix + ext
                candidate_path = os.path.join(dir_path, candidate_name)
                if os.path.isfile(candidate_path):
                    candidates[mtype] = candidate_path
                    break

        if not candidates:
            return

        dlg = AutoAssignDialog(stripped_base, candidates, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            selected = dlg.selected_maps()
            if selected:
                self._suppress_auto_assign = True
                for mtype, fpath in selected.items():
                    self.previews[mtype].set_image(fpath)
                    self.paths[mtype] = fpath
                    self.pbr_renderer.load_input_texture(mtype, fpath)
                self._suppress_auto_assign = False
                self.preview_mode_label.setText("Previewing live input textures")
                self.log(f"Auto‑assigned {len(selected)} map(s) for base '{stripped_base}'")

    # ---------------------------------------------------------------
    # Other UI handlers (unchanged)
    # ---------------------------------------------------------------
    def update_normal_invert(self, checked):
        self.invert_normal_y = checked
        self.pbr_renderer.set_normal_y_inverted(checked)
        self.preview_mode_label.setText("Previewing live input textures")

    def update_ao_intensity(self):
        self.ao_intensity = self.ao_slider.value() / 100.0
        self.ao_value_label.setText(f"{self.ao_intensity:.2f}x")
        self.pbr_renderer.use_input_preview()
        self.pbr_renderer.set_ao_intensity(self.ao_intensity)
        self.preview_mode_label.setText("Previewing live input textures")

    def update_normal_generation(self):
        self.normal_gen_sigma = self.normal_sigma_slider.value() / 100.0
        self.normal_gen_height = self.normal_height_slider.value() / 100.0
        self.normal_sigma_value_label.setText(f"{self.normal_gen_sigma:.2f}")
        self.normal_height_value_label.setText(f"{self.normal_gen_height:.2f}")
        self.pbr_renderer.use_input_preview()
        self.pbr_renderer.set_normal_generation(self.normal_gen_sigma, self.normal_gen_height)
        self.preview_mode_label.setText("Previewing live input textures")

    def reset_view(self):
        self.pbr_renderer.rotation_x = -30.0
        self.pbr_renderer.rotation_y = -45.0
        self.pbr_renderer.zoom = -5.0
        self.pbr_renderer.update()

    def installEventFilters(self, widget):
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)
            child.setMouseTracking(True)

    def eventFilter(self, obj, event):
        if event.type() == event.MouseMove:
            self.mouse_pos = self.mapFromGlobal(event.globalPos())
            self.update()
        return super().eventFilter(obj, event)

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1e1e22"))
        if self.mouse_pos:
            x, y = self.mouse_pos.x(), self.mouse_pos.y()
            radius = 450.0
            gradient = QRadialGradient(x, y, radius)
            base_color = QColor(100, 150, 255)
            for stop, alpha in zip((0.0, 0.15, 0.35, 0.6, 1.0), (45, 20, 8, 2, 0)):
                color = QColor(base_color)
                color.setAlpha(alpha)
                gradient.setColorAt(stop, color)
            painter.fillRect(self.rect(), QBrush(gradient))
            glow_pix = QPixmap(self.size())
            glow_pix.fill(Qt.transparent)
            glow_painter = QPainter(glow_pix)
            glow_painter.fillRect(glow_pix.rect(), self.noise_brush)
            glow_painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            glow_painter.fillRect(glow_pix.rect(), QBrush(gradient))
            glow_painter.end()
            painter.drawPixmap(0, 0, glow_pix)
        painter.end()

    def apply_theme(self):
        self.setStyleSheet("""
            QWidget {
                color: #e0e0e0;
                font-family: "Segoe UI", sans-serif;
            }
            QDialog {
                background-color: #1e1e22;
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
            QSlider::groove:horizontal {
                height: 6px;
                background: #2a2a30;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -5px 0;
                background: #4CAF50;
                border-radius: 8px;
            }
            QGroupBox {
                border: 1px solid #4a4a55;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

    def select_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.out_dir = dir_path
            self.lbl_out.setText(dir_path)

    def start_packing(self):
        if not self.out_dir:
            self.select_output()
            if not self.out_dir:
                return

        base_alpha, nms = self.pbr_renderer.get_composed_data()
        self.btn_pack.setEnabled(False)
        self.progress_bar.show()
        self.log("Packing started...")

        self.worker = self.worker_class(base_alpha, nms, self.out_dir)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.packing_finished)
        self.worker.start()

    def update_progress(self, val, text):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(f"%p% - {text}")

    def packing_finished(self, success, msg, base_ao_data, nms_data):
        self.btn_pack.setEnabled(True)
        if success:
            self.progress_bar.setFormat("100% - Finished!")
            self.log("Packing finished successfully.")
            self.pbr_renderer.set_packed_textures(base_ao_data, nms_data)
            self.preview_mode_label.setText("Previewing packed output textures")
            if self.chk_open.isChecked():
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))
        else:
            self.progress_bar.setFormat(f"Error: {msg}")
            self.log(f"Packing error: {msg}")