import os
from datetime import datetime

import numpy as np
from PyQt5.QtCore import pyqtSignal, QRect, QRectF, QSettings, Qt, QUrl
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
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
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pbr_renderer import PBRRendererWidget

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
            path, _ = QFileDialog.getOpenFileName(self, f"Select {self.map_type} Map", "", "Images (*.png *.jpg *.jpeg *.tga *.tif)")
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
            scaled_pix = self.pixmap.scaled(rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (rect.width() - scaled_pix.width()) // 2
            y = (rect.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)
            gradient = QRadialGradient(rect.width() / 2, rect.height(), rect.width())
            gradient.setColorAt(0, QColor(0, 0, 0, 180))
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            painter.fillRect(rect.x(), rect.height() - 40, rect.width(), 40, QBrush(gradient))
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


class MainWindow(QMainWindow):
    MAP_TYPES = ["BaseColor", "AO", "Metallic", "Smoothness", "Normal", "Alpha"]
    MAP_ALIASES = {
        "BaseColor": (
            ("basecolor",),
            ("base", "color"),
            ("albedo",),
            ("diffuse",),
            ("basecol",),
        ),
        "AO": (
            ("ambientocclusion",),
            ("ambient", "occlusion"),
            ("occlusion",),
            ("ao",),
        ),
        "Metallic": (
            ("metallic",),
            ("metalness",),
            ("metal",),
            ("mtl",),
        ),
        "Smoothness": (
            ("smoothness",),
            ("smooth",),
            ("roughness",),
            ("rough",),
            ("glossiness",),
            ("gloss",),
            ("rgh",),
        ),
        "Normal": (
            ("normalgl",),
            ("normaldx",),
            ("normal",),
            ("nrm",),
            ("nor",),
        ),
        "Alpha": (
            ("alpha",),
            ("opacity",),
            ("transparency",),
            ("trans",),
        ),
    }

    def __init__(self, worker_class):
        super().__init__()
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
        self.settings = QSettings()
        self.auto_assign_prompt_enabled = self.settings.value("auto_assign_prompt_enabled", True, type=bool)
        self.batch_input_dir = None
        self.batch_options = {
            "ignore_missing_basecolor": False,
            "ignore_ao": False,
        }
        self.out_dir = None
        self.max_compression_workers = max((os.cpu_count() or 2) - 1, 1)
        self.init_ui()
        self.apply_theme()

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
        for index, map_name in enumerate(self.MAP_TYPES):
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

        out_group = QGroupBox("Directory Settings")
        out_layout = QVBoxLayout()
        batch_dir_layout = QHBoxLayout()
        self.btn_batch_in = QPushButton("Batch Processing Folder Input")
        self.btn_batch_in.clicked.connect(self.select_batch_input)
        self.txt_batch_in = QLineEdit()
        self.txt_batch_in.setReadOnly(True)
        self.txt_batch_in.setPlaceholderText("No directory selected")
        batch_dir_layout.addWidget(self.btn_batch_in)
        batch_dir_layout.addWidget(self.txt_batch_in, 1)
        out_layout.addLayout(batch_dir_layout)
        out_dir_layout = QHBoxLayout()
        self.btn_out = QPushButton("Select Output Directory")
        self.btn_out.clicked.connect(self.select_output)
        self.lbl_out = QLabel("No directory selected")
        self.lbl_out.setWordWrap(True)
        out_dir_layout.addWidget(self.btn_out)
        out_dir_layout.addWidget(self.lbl_out, 1)
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
        self.pbr_renderer = PBRRendererWidget()
        right_layout.addWidget(self.pbr_renderer)
        self.log_overlay = QTextEdit(self.pbr_renderer)
        self.log_overlay.setReadOnly(True)
        self.log_overlay.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log_overlay.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log_overlay.setFixedSize(360, 140)
        self.log_overlay.move(12, 12)
        self.log_overlay.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 0, 0, 0);
                color: rgba(224, 224, 224, 220);
                border: 1px solid rgba(90, 90, 101, 120);
                border-radius: 6px;
                padding: 6px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 11px;
            }
        """)
        self.log_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.log_overlay.raise_()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        right_layout.addWidget(reset_btn)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 900])
        layout = QVBoxLayout(main_widget)
        layout.addWidget(splitter)
        self.installEventFilters(self)

    def update_texture(self, map_type, path):
        was_empty_before = not any(existing_path for existing_path in self.paths.values())
        self.apply_texture(map_type, path)
        if was_empty_before and path:
            self.maybe_auto_assign_related_textures(map_type, path)

    def clear_texture(self, map_type):
        self.paths[map_type] = None
        self.pbr_renderer.load_input_texture(map_type, None)
        self.preview_mode_label.setText("Previewing live input textures")

    def apply_texture(self, map_type, path):
        self.paths[map_type] = path
        preview = self.previews.get(map_type)
        if preview and preview.file_path != path:
            preview.set_image(path)
        self.pbr_renderer.load_input_texture(map_type, path)
        self.preview_mode_label.setText("Previewing live input textures")

    def maybe_auto_assign_related_textures(self, selected_map_type, selected_path):
        matches = self.find_related_textures(selected_map_type, selected_path)
        if not matches:
            return
        if self.auto_assign_prompt_enabled:
            allow, dont_ask_again = self.ask_auto_assign_permission(matches)
            if dont_ask_again:
                self.auto_assign_prompt_enabled = False
                self.settings.setValue("auto_assign_prompt_enabled", False)
            if not allow:
                return
        for map_type, path in matches.items():
            self.apply_texture(map_type, path)

    def find_related_textures(self, selected_map_type, selected_path, ignore_gui_state=False):
        directory = os.path.dirname(selected_path)
        filename = os.path.basename(selected_path)
        stem, _ = os.path.splitext(filename)
        parsed_selected = self.parse_texture_name(stem)
        if not parsed_selected or parsed_selected["map_type"] != selected_map_type:
            return {}
        selected_base = parsed_selected["base_name"]

        matches = {}
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            candidate_stem, candidate_ext = os.path.splitext(entry.name)
            if candidate_ext.lower() not in {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp"}:
                continue
            if entry.path == selected_path:
                continue
            parsed_candidate = self.parse_texture_name(candidate_stem)
            if not parsed_candidate:
                continue
            candidate_map = parsed_candidate["map_type"]
            if candidate_map == selected_map_type:
                continue
            if parsed_candidate["base_name"] != selected_base:
                continue
            if not ignore_gui_state and self.paths[candidate_map]:
                continue
            matches[candidate_map] = entry.path
        return matches

    def build_texture_set_from_seed(self, map_type, path):
        texture_set = {name: None for name in self.MAP_TYPES}
        if not path:
            return texture_set
        texture_set[map_type] = path
        texture_set.update(self.find_related_textures(map_type, path, ignore_gui_state=True))
        return texture_set

    def normalize_texture_name(self, stem):
        chars = []
        for char in stem.lower():
            chars.append(char if char.isalnum() else " ")
        return " ".join("".join(chars).split())

    def tokenize_texture_name(self, stem):
        normalized = self.normalize_texture_name(stem)
        return normalized.split() if normalized else []

    def parse_texture_name(self, stem):
        tokens = self.tokenize_texture_name(stem)
        if not tokens:
            return None

        best_match = None
        for map_type, aliases in self.MAP_ALIASES.items():
            for alias_tokens in aliases:
                alias_len = len(alias_tokens)
                if len(tokens) >= alias_len and tuple(tokens[-alias_len:]) == alias_tokens:
                    base_tokens = tokens[:-alias_len]
                    score = (3, alias_len)
                elif len(tokens) >= alias_len and tuple(tokens[:alias_len]) == alias_tokens:
                    base_tokens = tokens[alias_len:]
                    score = (2, alias_len)
                elif alias_len == 1 and alias_tokens[0] in tokens:
                    alias_index = tokens.index(alias_tokens[0])
                    base_tokens = tokens[:alias_index] + tokens[alias_index + 1:]
                    score = (1, alias_len)
                else:
                    continue

                if not base_tokens:
                    continue
                candidate = {
                    "map_type": map_type,
                    "base_name": " ".join(base_tokens),
                    "score": score,
                }
                if best_match is None or candidate["score"] > best_match["score"]:
                    best_match = candidate
        return best_match

    def ask_auto_assign_permission(self, matches):
        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-assign matching textures?")
        dialog.setModal(True)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e22;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QCheckBox {
                color: #e0e0e0;
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
            QPushButton {
                background-color: #3a3a45;
                border: 1px solid #5a5a65;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4a4a55;
                border: 1px solid #7a7a85;
            }
            QPushButton:pressed {
                background-color: #2a2a35;
            }
        """)
        layout = QVBoxLayout(dialog)
        names = ", ".join(sorted(matches.keys()))
        message = QLabel(f"Found matching textures for {names} in the same folder. Auto-assign them?")
        message.setWordWrap(True)
        layout.addWidget(message)
        dont_ask_again = QCheckBox("Don't ask again")
        layout.addWidget(dont_ask_again)
        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        allow = dialog.exec_() == QDialog.Accepted
        return allow, dont_ask_again.isChecked()

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

    def select_batch_input(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Batch Input Directory")
        if dir_path:
            self.batch_input_dir = dir_path
            self.txt_batch_in.setText(dir_path)

    def start_packing(self):
        if not self.out_dir:
            self.select_output()
            if not self.out_dir:
                return
        if self.batch_input_dir:
            self.start_batch_packing()
            return

        self.append_log("Baking current material…")
        base_alpha, nms = self.pbr_renderer.get_composed_data()

        self.btn_pack.setEnabled(False)
        self.progress_bar.show()

        jobs = [{
            "base_alpha": base_alpha,
            "nms": nms,
            "base_path": "BaseAOTransparency.png",
            "nms_path": "NMS.png",
        }]
        self.worker = self.worker_class(jobs, self.out_dir, max_workers=min(self.max_compression_workers, 2))
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.packing_finished)
        self.worker.start()

    def start_batch_packing(self):
        if not self.batch_input_dir:
            self.append_log("Batch input folder not set.")
            return
        options = self.ask_batch_processing_options()
        if options is None:
            return
        self.batch_options = options
        groups = self.collect_batch_groups(self.batch_input_dir)
        if not groups:
            self.append_log("No texture groups found for batch processing.")
            return

        self.btn_pack.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - Baking batch…")
        jobs = []
        total_groups = len(groups)
        baked_count = 0
        for index, (group_name, texture_paths) in enumerate(groups.items(), start=1):
            expanded_paths = self.expand_batch_texture_set(texture_paths)
            if self.batch_options["ignore_missing_basecolor"] and not expanded_paths.get("BaseColor"):
                self.append_log(f"[Batch] Skipping {group_name} (no BaseColor)")
                continue
            baked_count += 1
            self.append_log(f"[Batch] Baking {group_name} ({index}/{total_groups})")
            self.load_texture_set(expanded_paths)
            base_alpha, nms = self.pbr_renderer.get_composed_data()
            safe_name = self.make_safe_output_name(group_name)
            jobs.append({
                "base_alpha": base_alpha,
                "nms": nms,
                "base_path": os.path.join(safe_name, "BaseAOTransparency.png"),
                "nms_path": os.path.join(safe_name, "NMS.png"),
            })
            progress = int((index / total_groups) * 50)
            self.update_progress(progress, f"Baked {baked_count}/{total_groups} sets…")
            QApplication.processEvents()

        if not jobs:
            self.btn_pack.setEnabled(True)
            self.progress_bar.setFormat("0% - No valid batch sets found")
            self.append_log("No valid batch sets remained after filtering.")
            return

        self.worker = self.worker_class(jobs, self.out_dir, max_workers=self.max_compression_workers)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.packing_finished)
        self.worker.start()

    def collect_batch_groups(self, root_dir):
        groups = {}
        supported_exts = {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp"}
        for current_root, _, files in os.walk(root_dir):
            for filename in files:
                stem, ext = os.path.splitext(filename)
                if ext.lower() not in supported_exts:
                    continue
                parsed_texture = self.parse_texture_name(stem)
                if not parsed_texture:
                    continue
                map_type = parsed_texture["map_type"]
                base_name = parsed_texture["base_name"]
                rel_root = os.path.relpath(current_root, root_dir)
                rel_root = "" if rel_root == "." else rel_root
                group_key = os.path.join(rel_root, base_name).replace("\\", "/")
                groups.setdefault(group_key, {})
                groups[group_key].setdefault(map_type, os.path.join(current_root, filename))
        return dict(sorted(groups.items()))

    def load_texture_set(self, texture_paths):
        for map_type in self.MAP_TYPES:
            if map_type == "AO" and self.batch_options["ignore_ao"]:
                self.apply_texture(map_type, None)
                continue
            path = texture_paths.get(map_type)
            self.apply_texture(map_type, path)

    def expand_batch_texture_set(self, texture_paths):
        if texture_paths.get("BaseColor"):
            expanded = self.build_texture_set_from_seed("BaseColor", texture_paths["BaseColor"])
        else:
            expanded = {name: None for name in self.MAP_TYPES}
            for map_type in self.MAP_TYPES:
                seed_path = texture_paths.get(map_type)
                if seed_path:
                    expanded = self.build_texture_set_from_seed(map_type, seed_path)
                    break
        for map_type, path in texture_paths.items():
            if path and not expanded.get(map_type):
                expanded[map_type] = path
        return expanded

    def make_safe_output_name(self, name):
        cleaned = []
        for char in name:
            cleaned.append(char if char.isalnum() or char in ("-", "_", "/", os.sep) else "_")
        return "".join(cleaned).strip("/\\") or "TextureSet"

    def append_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_overlay.append(f"[{timestamp}] {message}")
        scrollbar = self.log_overlay.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def ask_batch_processing_options(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Processing Options")
        dialog.setModal(True)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e22;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QCheckBox {
                color: #e0e0e0;
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
            QPushButton {
                background-color: #3a3a45;
                border: 1px solid #5a5a65;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4a4a55;
                border: 1px solid #7a7a85;
            }
            QPushButton:pressed {
                background-color: #2a2a35;
            }
        """)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Choose how batch processing should handle incomplete texture sets."))

        ignore_missing_basecolor = QCheckBox("Ignore textures with no Base Color")
        ignore_missing_basecolor.setChecked(self.batch_options["ignore_missing_basecolor"])
        layout.addWidget(ignore_missing_basecolor)

        ignore_ao = QCheckBox("Ignore AO")
        ignore_ao.setChecked(self.batch_options["ignore_ao"])
        layout.addWidget(ignore_ao)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return None
        return {
            "ignore_missing_basecolor": ignore_missing_basecolor.isChecked(),
            "ignore_ao": ignore_ao.isChecked(),
        }

    def update_progress(self, val, text):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(f"%p% - {text}")

    def packing_finished(self, success, msg, base_ao_data, nms_data):
        self.btn_pack.setEnabled(True)
        if success:
            self.progress_bar.setFormat("100% - Finished!")
            if base_ao_data is not None and nms_data is not None:
                self.pbr_renderer.set_packed_textures(base_ao_data, nms_data)
                self.preview_mode_label.setText("Previewing packed output textures")
            self.append_log(msg)
            if self.chk_open.isChecked():
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))
        else:
            self.progress_bar.setFormat(f"Error: {msg}")
            self.append_log(f"[Error] {msg}")
