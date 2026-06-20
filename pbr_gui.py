import json
import os
import time
import numpy as np
from PyQt5.QtCore import pyqtSignal, QRect, QRectF, Qt, QUrl, QTimer
from PyQt5.QtGui import (QBrush, QColor, QDesktopServices, QFont, QFontMetrics,
                          QImage, QPainter, QPainterPath, QPen, QPixmap,
                          QRadialGradient, QLinearGradient)
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
                             QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QListWidget, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QSlider, QSplitter,
                             QTabWidget, QVBoxLayout, QWidget)
from pbr_renderer import PBRRendererWidget
from pack_worker import BatchPackWorker

# -------------------------------------------------------------------
# Load configuration files
# -------------------------------------------------------------------
def _load_json_config(filename):
    """Load a JSON config file. Looks next to the script, then in CWD."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(script_dir, filename), os.path.join(os.getcwd(), filename)]
    for p in paths:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f"Configuration file '{filename}' not found in {paths}")

# Load texture aliases
_aliases = _load_json_config("texture_aliases.json")
MAP_TYPES = _aliases["map_types"]
MAP_SUFFIXES = _aliases["suffixes"]
VALID_EXTENSIONS = _aliases["extensions"]

# Load theme
_theme = _load_json_config("pbr_theme.json")

def build_stylesheet():
    """Convert theme JSON dict into a Qt stylesheet string."""
    parts = []
    for selector, props in _theme.items():
        prop_str = "; ".join(f"{k}: {v}" for k, v in props.items())
        parts.append(f"{selector} {{ {prop_str}; }}")
    return "\n".join(parts)

def find_texture_set_in_folder(folder, base):
    """Given a folder and base name, return a dict of map_type -> filepath"""
    result = {}
    for mt, suffs in MAP_SUFFIXES.items():
        for sf in suffs:
            for ext in VALID_EXTENSIONS:
                cand = os.path.join(folder, base + sf + ext)
                if os.path.isfile(cand):
                    result[mt] = cand
                    break
            if mt in result:
                break
    return result

# -------------------------------------------------------------------
# Auto‑assignment dialog (unchanged)
# -------------------------------------------------------------------
class AutoAssignDialog(QDialog):
    def __init__(self, base_name, candidates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto‑assign Texture Maps")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"<b>Base name:</b> {base_name}<br>"
            f"Found {len(candidates)} additional map(s) in the same folder.<br>"
            "Choose the ones you want to assign:", wordWrap=True))
        self.checkboxes = {}
        for map_type, file_path in candidates.items():
            cb = QCheckBox(f"{map_type}: {os.path.basename(file_path)}")
            cb.setChecked(True)
            self.checkboxes[map_type] = cb
            layout.addWidget(cb)
        toggle_layout = QHBoxLayout()
        self.select_all_cb = QCheckBox("Select all")
        self.select_all_cb.setChecked(True)
        self.select_all_cb.toggled.connect(lambda ch: [cb.setChecked(ch) for cb in self.checkboxes.values()])
        toggle_layout.addWidget(self.select_all_cb)
        toggle_layout.addStretch()
        layout.addLayout(toggle_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if parent: self.setStyleSheet(parent.styleSheet())
    def selected_maps(self):
        return {mt: p for mt, p in self.candidates.items() if self.checkboxes[mt].isChecked()}

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
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.set_image(path)
            self.fileDropped.emit(self.map_type, path)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.pixmap and self.clear_button_rect.contains(event.pos()):
                self.set_image(None); self.cleared.emit(self.map_type); return
            path, _ = QFileDialog.getOpenFileName(self, f"Select {self.map_type} Map", "", "Images (*.png *.jpg *.jpeg *.tga *.tif)")
            if path: self.set_image(path); self.fileDropped.emit(self.map_type, path)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        rect = self.rect()
        rounded_path = QPainterPath(); rounded_path.addRoundedRect(QRectF(rect), 10, 10)
        painter.fillPath(rounded_path, QColor("#2a2a30"))
        if self.pixmap and not self.pixmap.isNull():
            painter.setClipPath(rounded_path)
            scaled = self.pixmap.scaled(rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            painter.drawPixmap((rect.width()-scaled.width())//2, (rect.height()-scaled.height())//2, scaled)
            gradient = QRadialGradient(rect.width()/2, rect.height(), rect.width())
            gradient.setColorAt(0, QColor(0,0,0,180)); gradient.setColorAt(1, QColor(0,0,0,0))
            painter.fillRect(rect.x(), rect.height()-40, rect.width(), 40, QBrush(gradient))
            painter.setClipping(False); painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(24,24,28,215)); painter.drawEllipse(self.clear_button_rect)
            painter.setPen(QPen(QColor("#ffffff"),2))
            inset=7; painter.drawLine(self.clear_button_rect.left()+inset, self.clear_button_rect.top()+inset, self.clear_button_rect.right()-inset, self.clear_button_rect.bottom()-inset)
            painter.drawLine(self.clear_button_rect.right()-inset, self.clear_button_rect.top()+inset, self.clear_button_rect.left()+inset, self.clear_button_rect.bottom()-inset)
        else:
            pen = QPen(QColor("#4a4a55"), 2, Qt.DashLine); painter.setPen(pen); painter.drawPath(rounded_path)
        painter.setPen(QColor("#ffffff")); painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(QRect(rect.x(), rect.height()-30, rect.width(), 30), Qt.AlignCenter, self.map_type)
        painter.end()

# -------------------------------------------------------------------
# LogOverlay (unchanged)
# -------------------------------------------------------------------
class LogOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.entries = []   # (msg, perf_counter)
        self.font = QFont("Consolas", 10)
        self.line_height = QFontMetrics(self.font).height()+2
        self.max_visible_lines = 10
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
    def add_log(self, message):
        self.entries.append((message, time.perf_counter()))
        if len(self.entries)>200: self.entries = self.entries[-100:]
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.setFont(self.font)
        now = time.perf_counter()
        self.entries = [(m,t) for m,t in self.entries if now - t <= 15.0]
        if not self.entries: return
        recent = list(reversed(self.entries))[:self.max_visible_lines]
        x, y = 10, 5 + self.fontMetrics().ascent()
        for i, (msg, t) in enumerate(recent):
            alpha = max(0.0, 1.0 - (now - t)/15.0)
            if alpha < 0.01: continue
            painter.setPen(QColor(0,0,0,int(alpha*120))); painter.drawText(x+1,y+1,msg)
            painter.setPen(QColor(255,255,255,int(alpha*255))); painter.drawText(x,y,msg)
            y += self.line_height
        if len(recent)>6:
            start_y = 5 + 6*self.line_height
            gradient = QLinearGradient(0, start_y, 0, self.height())
            gradient.setColorAt(0.0, QColor(0,0,0,255)); gradient.setColorAt(1.0, QColor(0,0,0,0))
            painter.save(); painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.fillRect(0, start_y, self.width(), self.height()-start_y, gradient)
            painter.restore()

# -------------------------------------------------------------------
# MainWindow (with preview on the right, tabs on the left, batch preview)
# -------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, worker_class):
        super().__init__()
        self.worker_class = worker_class
        self.setWindowTitle("PBR Texture Packer for Unity"); self.resize(1400,800)
        self.batch_dir = None; self.mouse_pos = None
        self.ao_intensity=1.0; self.normal_gen_sigma=1.0; self.normal_gen_height=1.0
        self.invert_normal_y=False
        noise = np.random.randint(0,255,(256,256),dtype=np.uint8)
        rgba = np.zeros((256,256,4),dtype=np.uint8); rgba[...,:3]=255; rgba[...,3]=noise//3
        self.noise_image = QImage(rgba.data,256,256,QImage.Format_ARGB32)
        self.noise_brush = QBrush(self.noise_image)
        self.paths = {t:None for t in MAP_TYPES}
        self.out_dir = None; self._suppress_auto_assign = False
        self.batch_sets = []   # holds scanned sets
        self.setMouseTracking(True); self.init_ui(); self.apply_theme()

    # ---------- helpers ----------
    def _make_slider(self, text, range_min, range_max, init, value_label, slot, layout):
        row = QHBoxLayout(); row.addWidget(QLabel(text))
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(value_label); layout.addLayout(row)
        slider = QSlider(Qt.Horizontal); slider.setRange(range_min, range_max)
        slider.setValue(init); slider.valueChanged.connect(slot)
        layout.addWidget(slider)
        return slider

    def _make_output_group(self):
        grp = QGroupBox("Output")
        layout = QVBoxLayout()
        od = QHBoxLayout()
        btn_out = QPushButton("Select Output Directory")
        btn_out.clicked.connect(self.select_output)
        self.lbl_out = QLabel("No directory selected")
        self.lbl_out.setWordWrap(True)
        od.addWidget(btn_out); od.addWidget(self.lbl_out,1)
        layout.addLayout(od)
        self.chk_open = QCheckBox("Open folder when done")
        self.chk_open.setChecked(True)
        layout.addWidget(self.chk_open)
        grp.setLayout(layout)
        return grp

    def _make_material_group(self):
        grp = QGroupBox("Material Properties")
        layout = QVBoxLayout()
        self.ao_value_label = QLabel("1.00x")
        self.ao_slider = self._make_slider("AO Intensity",0,200,100,self.ao_value_label,self.update_ao_intensity,layout)
        self.normal_sigma_value_label = QLabel("1.00")
        self.normal_sigma_slider = self._make_slider("Normal Sigma",1,500,100,self.normal_sigma_value_label,self.update_normal_generation,layout)
        self.normal_height_value_label = QLabel("1.00")
        self.normal_height_slider = self._make_slider("Normal Height",0,200,100,self.normal_height_value_label,self.update_normal_generation,layout)
        nt_layout = QHBoxLayout(); nt_layout.addStretch()
        self.chk_invert_normal_y = QCheckBox("Invert Y")
        self.chk_invert_normal_y.toggled.connect(self.update_normal_invert)
        nt_layout.addWidget(self.chk_invert_normal_y); nt_layout.addStretch()
        layout.addLayout(nt_layout)
        grp.setLayout(layout)
        return grp

    # ---------- UI creation ----------
    def init_ui(self):
        main = QWidget(self); self.setCentralWidget(main)
        main_layout = QHBoxLayout(main)
        splitter = QSplitter(Qt.Horizontal)

        # --- LEFT SIDE: tabbed controls ---
        self.tab_widget = QTabWidget()
        # Tab 1: Single Process
        single_tab = QWidget(); single_layout = QVBoxLayout(single_tab)
        single_layout.setSpacing(10); single_layout.setContentsMargins(10,10,10,10)
        # texture previews
        grid = QGridLayout(); grid.setSpacing(10)
        self.previews = {}
        for i, map_name in enumerate(MAP_TYPES):
            pv = ImagePreviewWidget(map_name)
            pv.fileDropped.connect(self.update_texture); pv.cleared.connect(self.clear_texture)
            self.previews[map_name] = pv
            grid.addWidget(pv, i//3, i%3)
        single_layout.addLayout(grid)
        single_layout.addWidget(self._make_material_group())
        single_layout.addWidget(self._make_output_group())
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        single_layout.addWidget(self.progress_bar)
        self.btn_pack = QPushButton("Pack Textures"); self.btn_pack.setMinimumHeight(40)
        self.btn_pack.clicked.connect(self.start_packing)
        single_layout.addWidget(self.btn_pack)
        single_layout.addStretch()
        self.tab_widget.addTab(single_tab, "Single Process")

        # Tab 2: Batch Process
        batch_tab = QWidget(); batch_layout = QVBoxLayout(batch_tab)
        batch_layout.setSpacing(10); batch_layout.setContentsMargins(10,10,10,10)
        # batch directory
        bgrp = QGroupBox("Batch Directory"); bdl = QVBoxLayout()
        bd = QHBoxLayout()
        self.btn_batch = QPushButton("Select Batch Directory")
        self.btn_batch.clicked.connect(self.select_batch_directory)
        self.lbl_batch = QLabel("No directory selected"); self.lbl_batch.setWordWrap(True)
        bd.addWidget(self.btn_batch); bd.addWidget(self.lbl_batch,1)
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.clicked.connect(self._scan_batch_folder)
        bd.addWidget(self.btn_scan)
        bdl.addLayout(bd); bgrp.setLayout(bdl)
        batch_layout.addWidget(bgrp)

        # selected item preview
        self.batch_preview_group = QGroupBox("Selected Item Preview")
        preview_grid = QGridLayout(); preview_grid.setSpacing(10)
        self.batch_previews = {}
        for i, map_name in enumerate(MAP_TYPES):
            pv = ImagePreviewWidget(map_name)
            pv.setAcceptDrops(False); pv.setCursor(Qt.ArrowCursor)
            self.batch_previews[map_name] = pv
            preview_grid.addWidget(pv, i//3, i%3)
        self.batch_preview_group.setLayout(preview_grid)
        batch_layout.addWidget(self.batch_preview_group)

        # batch items list
        self.batch_items_group = QGroupBox("Batch Items")
        items_layout = QVBoxLayout()
        self.batch_list_widget = QListWidget()
        self.batch_list_widget.currentRowChanged.connect(self._on_batch_item_selected)
        items_layout.addWidget(self.batch_list_widget)
        self.batch_items_group.setLayout(items_layout)
        batch_layout.addWidget(self.batch_items_group)

        batch_layout.addWidget(self._make_material_group())   # same material sliders (shared state)
        batch_layout.addWidget(self._make_output_group())     # same output selection
        self.batch_progress_bar = QProgressBar(); self.batch_progress_bar.setVisible(False)
        batch_layout.addWidget(self.batch_progress_bar)
        self.btn_batch_process = QPushButton("Batch Process"); self.btn_batch_process.setMinimumHeight(40)
        self.btn_batch_process.clicked.connect(self.process_batch)
        batch_layout.addWidget(self.btn_batch_process)
        batch_layout.addStretch()
        self.tab_widget.addTab(batch_tab, "Batch Process")

        # --- RIGHT SIDE: 3D preview ---
        right = QWidget(); right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10,10,10,10)
        title = QLabel("3D PBR Preview"); title.setFont(QFont("Segoe UI",14,QFont.Bold)); title.setAlignment(Qt.AlignCenter)
        title.setFixedHeight(20)
        right_layout.addWidget(title)
        self.preview_mode_label = QLabel("Previewing live input textures")
        self.preview_mode_label.setAlignment(Qt.AlignCenter); self.preview_mode_label.setFixedHeight(20)
        right_layout.addWidget(self.preview_mode_label)
        self.pbr_renderer = PBRRendererWidget(); right_layout.addWidget(self.pbr_renderer)
        self.log_overlay = LogOverlay(self.pbr_renderer)
        self.log_overlay.setFixedSize(420,200); self.log_overlay.move(10,10); self.log_overlay.show()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        right_layout.addWidget(reset_btn)

        splitter.addWidget(self.tab_widget)      # left side: controls
        splitter.addWidget(right)               # right side: preview
        splitter.setSizes([500, 900])           # give more space to preview
        main_layout.addWidget(splitter)

        # tab change to restore single preview when switching to Single Process
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        for child in self.findChildren(QWidget):
            child.installEventFilter(self); child.setMouseTracking(True)

    # ---------- tab change handling ----------
    def _on_tab_changed(self, index):
        if index == 0:  # Single Process tab
            self._restore_single_preview()

    def _restore_single_preview(self):
        """Reload single-process paths into renderer and preview widgets."""
        for map_type, path in self.paths.items():
            self.pbr_renderer.load_input_texture(map_type, path)
        self.preview_mode_label.setText("Previewing live input textures")

    # ---------- logging ----------
    def log(self, msg):
        self.log_overlay.add_log(msg)

    # ---------- batch scanning & preview ----------
    def select_batch_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Batch Directory")
        if dir_path: self.batch_dir = dir_path; self.lbl_batch.setText(dir_path)

    def _scan_batch_folder(self):
        if not self.batch_dir:
            QMessageBox.warning(self, "No Batch Directory", "Please select a batch directory first.")
            return
        self.log("Scanning batch folder...")
        self.batch_list_widget.clear()
        self.batch_sets = []
        subdirs = [d for d in os.listdir(self.batch_dir) if os.path.isdir(os.path.join(self.batch_dir,d))]
        if not subdirs:
            self.log("No subdirectories found.")
            QMessageBox.information(self,"No Sets","No subdirectories found in batch folder.")
            return
        for sub in subdirs:
            folder = os.path.join(self.batch_dir, sub)
            maps = find_texture_set_in_folder(folder, sub)
            if not maps:
                # try to guess base name
                all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
                base = None
                for f in all_files:
                    name, ext = os.path.splitext(f)
                    for suffs in MAP_SUFFIXES.values():
                        for sf in suffs:
                            if name.lower().endswith(sf.lower()):
                                base = name[:-len(sf)]; break
                        if base: break
                    if base: break
                if base: maps = find_texture_set_in_folder(folder, base)
            if maps:
                self.batch_sets.append({'base_name': sub, 'maps': maps, 'output_subdir': sub})
                self.batch_list_widget.addItem(sub)
        if not self.batch_sets:
            self.log("No valid texture sets found.")
            QMessageBox.information(self,"No Texture Sets","No valid texture sets found.")
        else:
            self.log(f"Found {len(self.batch_sets)} set(s).")
            # Select first item automatically to show preview
            self.batch_list_widget.setCurrentRow(0)

    def _on_batch_item_selected(self, row):
        if row < 0 or row >= len(self.batch_sets):
            # Clear preview
            for pv in self.batch_previews.values():
                pv.set_image(None)
            return
        set_info = self.batch_sets[row]
        maps = set_info['maps']
        for map_type in MAP_TYPES:
            path = maps.get(map_type, None)
            self.batch_previews[map_type].set_image(path)
            self.pbr_renderer.load_input_texture(map_type, path)
        self.preview_mode_label.setText(f"Previewing batch item: {set_info['base_name']}")

    # ---------- batch processing ----------
    def process_batch(self):
        # If not scanned yet, scan automatically
        if not self.batch_sets:
            self._scan_batch_folder()
        if not self.batch_sets:
            QMessageBox.warning(self, "No Sets", "No valid texture sets found.")
            return
        if not self.out_dir:
            QMessageBox.warning(self, "Missing", "Select an output directory first.")
            return

        self.btn_batch_process.setEnabled(False); self.btn_pack.setEnabled(False)
        self.batch_progress_bar.setVisible(True); self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat("Composing textures...")
        self.log("Batch processing started.")

        results = []
        total = len(self.batch_sets)
        for idx, s in enumerate(self.batch_sets):
            self.batch_progress_bar.setValue(int((idx+1)/total*80))
            self.batch_progress_bar.setFormat(f"Composing {s['base_name']}...")
            self.log(f"Composing {s['base_name']}..."); QApplication.processEvents()

            # Load this set's textures into the renderer (temporary)
            for mn in self.paths: self.paths[mn]=None; self.pbr_renderer.load_input_texture(mn,None)
            for mt, p in s['maps'].items(): self.paths[mt]=p; self.pbr_renderer.load_input_texture(mt,p)
            self.pbr_renderer.set_ao_intensity(self.ao_intensity)
            self.pbr_renderer.set_normal_generation(self.normal_gen_sigma, self.normal_gen_height)
            self.pbr_renderer.set_normal_y_inverted(self.invert_normal_y)
            ba, nms = self.pbr_renderer.get_composed_data()
            if ba is None or nms is None: continue
            results.append({'base_name':s['base_name'], 'base_alpha':ba, 'nms':nms,
                            'output_dir':os.path.join(self.out_dir, s['output_subdir'])})

        self.batch_progress_bar.setValue(90); self.batch_progress_bar.setFormat("Saving textures...")
        self.log("Saving packed textures..."); QApplication.processEvents()
        self.batch_worker = BatchPackWorker(results)
        self.batch_worker.progress.connect(self.update_batch_progress)
        self.batch_worker.finished.connect(self.batch_finished)
        self.batch_worker.start()

    def update_batch_progress(self, val, text):
        self.batch_progress_bar.setValue(val); self.batch_progress_bar.setFormat(f"%p% - {text}")

    def batch_finished(self, success, msg):
        self.btn_batch_process.setEnabled(True); self.btn_pack.setEnabled(True)
        for mn in self.paths: self.paths[mn]=None; self.pbr_renderer.load_input_texture(mn,None)
        for pv in self.previews.values(): pv.set_image(None)
        if success:
            self.batch_progress_bar.setValue(100); self.batch_progress_bar.setFormat("Batch complete!")
            self.log("Batch complete.")
            if self.chk_open.isChecked(): QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))
        else:
            self.batch_progress_bar.setFormat(f"Batch error: {msg}"); self.log(f"Batch error: {msg}")
            QMessageBox.critical(self,"Batch Error",msg)

    # ---------- texture management ----------
    def update_texture(self, map_type, path):
        self.paths[map_type]=path; self.pbr_renderer.load_input_texture(map_type,path)
        self.preview_mode_label.setText("Previewing live input textures")
        if not self._suppress_auto_assign: self.attempt_auto_assign(map_type, path)

    def clear_texture(self, map_type):
        self.paths[map_type]=None; self.pbr_renderer.load_input_texture(map_type,None)
        self.preview_mode_label.setText("Previewing live input textures")

    def attempt_auto_assign(self, dropped_type, dropped_path):
        missing = [t for t,p in self.paths.items() if t!=dropped_type and p is None]
        if not missing: return
        folder, fname = os.path.split(dropped_path)
        base, ext = os.path.splitext(fname)
        stripped = base
        for sf in sorted(MAP_SUFFIXES.get(dropped_type,[]), key=len, reverse=True):
            if base.lower().endswith(sf.lower()): stripped = base[:-len(sf)]; break
        candidates = {}
        for mt in missing:
            for sf in MAP_SUFFIXES.get(mt,[]):
                cand = os.path.join(folder, stripped+sf+ext)
                if os.path.isfile(cand): candidates[mt]=cand; break
        if not candidates: return
        dlg = AutoAssignDialog(stripped, candidates, self)
        if dlg.exec_()==QDialog.Accepted:
            self._suppress_auto_assign = True
            for mt, fp in dlg.selected_maps().items():
                self.previews[mt].set_image(fp); self.paths[mt]=fp; self.pbr_renderer.load_input_texture(mt,fp)
            self._suppress_auto_assign = False
            self.preview_mode_label.setText("Previewing live input textures")
            self.log(f"Auto‑assigned {len(candidates)} map(s) for base '{stripped}'")

    # ---------- material sliders ----------
    def update_normal_invert(self, ch): self.invert_normal_y=ch; self.pbr_renderer.set_normal_y_inverted(ch); self.preview_mode_label.setText("Previewing live input textures")
    def update_ao_intensity(self):
        self.ao_intensity=self.ao_slider.value()/100.0; self.ao_value_label.setText(f"{self.ao_intensity:.2f}x")
        self.pbr_renderer.use_input_preview(); self.pbr_renderer.set_ao_intensity(self.ao_intensity)
        self.preview_mode_label.setText("Previewing live input textures")
    def update_normal_generation(self):
        self.normal_gen_sigma=self.normal_sigma_slider.value()/100.0; self.normal_gen_height=self.normal_height_slider.value()/100.0
        self.normal_sigma_value_label.setText(f"{self.normal_gen_sigma:.2f}"); self.normal_height_value_label.setText(f"{self.normal_gen_height:.2f}")
        self.pbr_renderer.use_input_preview(); self.pbr_renderer.set_normal_generation(self.normal_gen_sigma, self.normal_gen_height)
        self.preview_mode_label.setText("Previewing live input textures")

    def reset_view(self):
        self.pbr_renderer.rotation_x=-30.0; self.pbr_renderer.rotation_y=-45.0; self.pbr_renderer.zoom=-5.0; self.pbr_renderer.update()

    # ---------- output & packing ----------
    def select_output(self):
        d = QFileDialog.getExistingDirectory(self,"Select Output Directory")
        if d: self.out_dir=d; self.lbl_out.setText(d)

    def start_packing(self):
        if not self.out_dir:
            QMessageBox.warning(self,"Missing","Please select an output directory first."); return
        ba, nms = self.pbr_renderer.get_composed_data()
        self.btn_pack.setEnabled(False); self.progress_bar.setVisible(True); self.log("Packing started...")
        self.worker = self.worker_class(ba, nms, self.out_dir)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.packing_finished)
        self.worker.start()

    def update_progress(self, val, text): self.progress_bar.setValue(val); self.progress_bar.setFormat(f"%p% - {text}")

    def packing_finished(self, success, msg, bao, nms):
        self.btn_pack.setEnabled(True)
        if success:
            self.progress_bar.setFormat("100% - Finished!"); self.log("Packing finished successfully.")
            self.pbr_renderer.set_packed_textures(bao, nms); self.preview_mode_label.setText("Previewing packed output textures")
            if self.chk_open.isChecked(): QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))
        else:
            self.progress_bar.setFormat(f"Error: {msg}"); self.log(f"Packing error: {msg}")

    # ---------- background effect ----------
    def eventFilter(self, obj, event):
        if event.type() == event.MouseMove: self.mouse_pos = self.mapFromGlobal(event.globalPos()); self.update()
        return super().eventFilter(obj, event)
    def mouseMoveEvent(self, event): self.mouse_pos = event.pos(); self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing); painter.fillRect(self.rect(), QColor("#1e1e22"))
        if self.mouse_pos:
            x,y = self.mouse_pos.x(), self.mouse_pos.y()
            radius=450.0; grad = QRadialGradient(x,y,radius)
            base_color=QColor(100,150,255)
            for stop,alpha in zip((0.0,0.15,0.35,0.6,1.0),(45,20,8,2,0)):
                c=QColor(base_color); c.setAlpha(alpha); grad.setColorAt(stop,c)
            painter.fillRect(self.rect(), QBrush(grad))
            glow = QPixmap(self.size()); glow.fill(Qt.transparent)
            gp = QPainter(glow); gp.fillRect(glow.rect(), self.noise_brush)
            gp.setCompositionMode(QPainter.CompositionMode_DestinationIn); gp.fillRect(glow.rect(), QBrush(grad))
            gp.end(); painter.drawPixmap(0,0,glow)
        painter.end()

    def apply_theme(self):
        self.setStyleSheet(build_stylesheet())