import os, time
import numpy as np
from PyQt5.QtCore import pyqtSignal, QRect, QRectF, Qt, QUrl, QTimer
from PyQt5.QtGui import (QBrush, QColor, QDesktopServices, QFont, QFontMetrics,
                          QImage, QPainter, QPainterPath, QPen, QPixmap,
                          QRadialGradient, QLinearGradient)
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
                             QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QMainWindow, QMessageBox, QProgressBar,
                             QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget)
from pbr_renderer import PBRRendererWidget
from pack_worker import BatchPackWorker

MAP_SUFFIXES = {
    "BaseColor": ["_BaseColor","_Albedo","_Diffuse","_Color","_D","_col",
                  "_basecolor","_albedo","_diffuse","_color","_diff"],
    "AO": ["_AO","_AmbientOcclusion","_Occlusion",
           "_ao","_ambientocclusion","_occlusion"],
    "Metallic": ["_Metallic","_Metalness","_Metal",
                 "_metallic","_metalness","_metal"],
    "Smoothness": ["_Smoothness","_Roughness","_Smooth","_Rough","_rgh",
                   "_smoothness","_roughness","_smooth","_rough"],
    "Normal": ["_Normal","_NRM","_N","_normal","_nrm","_n"],
    "Alpha": ["_Alpha","_Opacity","_Mask","_alpha","_opacity","_mask"],
}

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
        self.paths = {t:None for t in MAP_SUFFIXES}
        self.out_dir = None; self._suppress_auto_assign = False
        self.setMouseTracking(True); self.init_ui(); self.apply_theme()

    # ---------- UI helpers ----------
    def _make_slider(self, text, range_min, range_max, init, value_label, slot, layout):
        row = QHBoxLayout(); row.addWidget(QLabel(text))
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(value_label); layout.addLayout(row)
        slider = QSlider(Qt.Horizontal); slider.setRange(range_min, range_max)
        slider.setValue(init); slider.valueChanged.connect(slot)
        layout.addWidget(slider)
        return slider

    # ---------- UI creation ----------
    def init_ui(self):
        main = QWidget(self); self.setCentralWidget(main)
        splitter = QSplitter(Qt.Horizontal)

        # left panel
        left = QWidget(); left.setMouseTracking(True)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10,10,10,10); left_layout.setSpacing(10)
        title = QLabel("Texture Maps"); title.setFont(QFont("Segoe UI",14,QFont.Bold)); title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        grid = QGridLayout(); grid.setSpacing(10)
        self.previews = {}
        for i, map_name in enumerate(MAP_SUFFIXES):
            pv = ImagePreviewWidget(map_name)
            pv.fileDropped.connect(self.update_texture); pv.cleared.connect(self.clear_texture)
            self.previews[map_name] = pv
            grid.addWidget(pv, i//3, i%3)
        left_layout.addLayout(grid)

        nt_layout = QHBoxLayout(); nt_layout.addStretch()
        self.chk_invert_normal_y = QCheckBox("Invert Y")
        self.chk_invert_normal_y.toggled.connect(self.update_normal_invert)
        nt_layout.addWidget(self.chk_invert_normal_y); nt_layout.addStretch()
        left_layout.addLayout(nt_layout)

        # material group
        mat_group = QGroupBox("Material Properties"); mat_layout = QVBoxLayout()
        self.ao_value_label = QLabel("1.00x")
        self.ao_slider = self._make_slider("AO Intensity",0,200,100,self.ao_value_label,self.update_ao_intensity,mat_layout)
        self.normal_sigma_value_label = QLabel("1.00")
        self.normal_sigma_slider = self._make_slider("Normal Sigma",1,500,100,self.normal_sigma_value_label,self.update_normal_generation,mat_layout)
        self.normal_height_value_label = QLabel("1.00")
        self.normal_height_slider = self._make_slider("Normal Height",0,200,100,self.normal_height_value_label,self.update_normal_generation,mat_layout)
        mat_group.setLayout(mat_layout)
        left_layout.addWidget(mat_group)

        # directories group
        out_group = QGroupBox("Directories"); out_layout = QVBoxLayout()
        bd = QHBoxLayout()
        self.btn_batch = QPushButton("Select Batch Directory")
        self.btn_batch.clicked.connect(self.select_batch_directory)
        self.lbl_batch = QLabel("No directory selected")
        self.lbl_batch.setWordWrap(True)
        bd.addWidget(self.btn_batch); bd.addWidget(self.lbl_batch,1); out_layout.addLayout(bd)
        od = QHBoxLayout()
        self.btn_out = QPushButton("Select Output Directory")
        self.btn_out.clicked.connect(self.select_output)
        self.lbl_out = QLabel("No directory selected")
        self.lbl_out.setWordWrap(True)
        od.addWidget(self.btn_out); od.addWidget(self.lbl_out,1); out_layout.addLayout(od)
        opts = QHBoxLayout(); self.chk_open=QCheckBox("Open folder when done"); self.chk_open.setChecked(True)
        opts.addWidget(self.chk_open); out_layout.addLayout(opts)
        out_group.setLayout(out_layout)
        left_layout.addWidget(out_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
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

        # right panel
        right = QWidget(); right.setMouseTracking(True); right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10,10,10,10)
        pt = QLabel("3D PBR Preview"); pt.setFont(QFont("Segoe UI",14,QFont.Bold)); pt.setAlignment(Qt.AlignCenter); pt.setFixedHeight(20)
        right_layout.addWidget(pt)
        self.preview_mode_label = QLabel("Previewing live input textures")
        self.preview_mode_label.setAlignment(Qt.AlignCenter)
        self.preview_mode_label.setFixedHeight(20)
        right_layout.addWidget(self.preview_mode_label)
        self.pbr_renderer = PBRRendererWidget(); right_layout.addWidget(self.pbr_renderer)

        self.log_overlay = LogOverlay(self.pbr_renderer)
        self.log_overlay.setFixedSize(420,200); self.log_overlay.move(10,10); self.log_overlay.show()
        self.log("Ready.")

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        right_layout.addWidget(reset_btn)

        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([500,900])
        layout = QVBoxLayout(main); layout.addWidget(splitter)

        for child in self.findChildren(QWidget):
            child.installEventFilter(self); child.setMouseTracking(True)
    # ---------- logging ----------
    def log(self, msg):
        self.log_overlay.add_log(msg)

    # ---------- batch processing ----------
    def select_batch_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Batch Directory")
        if dir_path: self.batch_dir = dir_path; self.lbl_batch.setText(dir_path)

    def _find_texture_set_in_folder(self, folder, base):
        result = {}
        for mt, suffs in MAP_SUFFIXES.items():
            for sf in suffs:
                for ext in ['.png','.jpg','.jpeg','.tga','.tif']:
                    cand = os.path.join(folder, base+sf+ext)
                    if os.path.isfile(cand): result[mt]=cand; break
                if mt in result: break
        return result

    def process_batch(self):
        if not self.batch_dir or not self.out_dir:
            QMessageBox.warning(self, "Missing", "Select batch and output directories first."); return
        self.btn_batch_process.setEnabled(False); self.btn_pack.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0); self.progress_bar.setFormat("Scanning batch folder...")
        self.log("Batch processing started.")

        subdirs = [d for d in os.listdir(self.batch_dir) if os.path.isdir(os.path.join(self.batch_dir,d))]
        if not subdirs:
            QMessageBox.information(self,"No Sets","No subdirectories found in batch folder.")
            self.btn_batch_process.setEnabled(True); self.btn_pack.setEnabled(True); self.progress_bar.setVisible(False)
            self.log("No sets found."); return

        sets = []; total = len(subdirs)
        for i, sub in enumerate(subdirs):
            self.progress_bar.setValue(int(i/total*50)); self.progress_bar.setFormat(f"Scanning {sub}...")
            self.log(f"Scanning {sub}..."); QApplication.processEvents()
            folder = os.path.join(self.batch_dir, sub)
            maps = self._find_texture_set_in_folder(folder, sub)
            if not maps:
                all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
                base = None
                for f in all_files:
                    name, ext = os.path.splitext(f)
                    for suffs in MAP_SUFFIXES.values():
                        for sf in suffs:
                            if name.lower().endswith(sf.lower()): base = name[:-len(sf)]; break
                        if base: break
                    if base: break
                if base: maps = self._find_texture_set_in_folder(folder, base)
                else: continue
            if maps: sets.append({'base_name':sub, 'maps':maps, 'output_subdir':sub})
        if not sets:
            QMessageBox.information(self,"No Texture Sets","No valid texture sets found.")
            self.btn_batch_process.setEnabled(True); self.btn_pack.setEnabled(True); self.progress_bar.setVisible(False)
            self.log("No valid sets found."); return

        results = []
        for idx, s in enumerate(sets):
            self.progress_bar.setValue(50+int(idx/len(sets)*40)); self.progress_bar.setFormat(f"Composing {s['base_name']}...")
            self.log(f"Composing {s['base_name']}..."); QApplication.processEvents()
            for mn in self.paths: self.paths[mn]=None; self.pbr_renderer.load_input_texture(mn,None)
            for mt, p in s['maps'].items(): self.paths[mt]=p; self.pbr_renderer.load_input_texture(mt,p)
            self.pbr_renderer.set_ao_intensity(self.ao_intensity)
            self.pbr_renderer.set_normal_generation(self.normal_gen_sigma, self.normal_gen_height)
            self.pbr_renderer.set_normal_y_inverted(self.invert_normal_y)
            ba, nms = self.pbr_renderer.get_composed_data()
            if ba is None or nms is None: continue
            results.append({'base_name':s['base_name'], 'base_alpha':ba, 'nms':nms, 'output_dir':os.path.join(self.out_dir, s['output_subdir'])})
        self.progress_bar.setValue(90); self.progress_bar.setFormat("Saving textures...")
        self.log("Saving packed textures..."); QApplication.processEvents()
        self.batch_worker = BatchPackWorker(results)
        self.batch_worker.progress.connect(self.update_progress)
        self.batch_worker.finished.connect(self.batch_finished)
        self.batch_worker.start()

    def batch_finished(self, success, msg):
        self.btn_batch_process.setEnabled(True); self.btn_pack.setEnabled(True)
        for mn in self.paths: self.paths[mn]=None; self.pbr_renderer.load_input_texture(mn,None)
        for pv in self.previews.values(): pv.set_image(None)
        if success:
            self.progress_bar.setValue(100); self.progress_bar.setFormat("Batch complete!")
            self.log("Batch complete.")
            if self.chk_open.isChecked(): QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))
        else:
            self.progress_bar.setFormat(f"Batch error: {msg}"); self.log(f"Batch error: {msg}")
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
        if not self.out_dir: self.select_output(); return
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
        self.setStyleSheet("""
            QWidget { color: #e0e0e0; font-family: "Segoe UI", sans-serif; }
            QDialog { background-color: #1e1e22; }
            QPushButton { background-color: #3a3a45; border: 1px solid #5a5a65; border-radius: 6px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #4a4a55; border: 1px solid #7a7a85; }
            QPushButton:pressed { background-color: #2a2a35; }
            QProgressBar { background-color: #2a2a30; border: 1px solid #4a4a55; border-radius: 6px; text-align: center; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 5px; }
            QCheckBox { spacing: 8px; } QCheckBox::indicator { width: 18px; height: 18px; background-color: #2a2a30; border: 1px solid #4a4a55; border-radius: 4px; }
            QCheckBox::indicator:checked { background-color: #4CAF50; border: 1px solid #4CAF50; }
            QSlider::groove:horizontal { height: 6px; background: #2a2a30; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; height: 16px; margin: -5px 0; background: #4CAF50; border-radius: 8px; }
            QGroupBox { border: 1px solid #4a4a55; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)