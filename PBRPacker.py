import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QProgressBar, QCheckBox, QGridLayout, QSplitter,
                             QSlider, QGroupBox)
from PyQt5.QtGui import (QPixmap, QPainter, QColor, QRadialGradient, QBrush, 
                         QImage, QPainterPath, QPen, QFont, QSurfaceFormat)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QRect, QRectF, QTimer, QPoint
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from io import BytesIO

# ---------------------------------------------------------
# Worker Thread for Texture Packing Logic
# ---------------------------------------------------------
class PackWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, object, object)  # Added texture data output

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
                self.finished.emit(False, "No valid images provided.", None, None)
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

            self.progress.emit(50, "Processing BaseAOTransparency...")
            base_color = get_array("BaseColor", (255, 255, 255, 255))
            ao = get_array("AO", (255, 255, 255, 255))[..., 0]
            
            base_ao_rgb = base_color[..., :3] * (ao[..., np.newaxis] / 255.0)
            
            alpha_arr = get_array("Alpha", (255, 255, 255, 255))
            if "Alpha" in images:
                transparency = alpha_arr[..., 0]
            else:
                transparency = base_color[..., 3]

            out_base_ao = np.empty((max_h, max_w, 4), dtype=np.uint8)
            out_base_ao[..., :3] = np.clip(base_ao_rgb, 0, 255).astype(np.uint8)
            out_base_ao[..., 3] = np.clip(transparency, 0, 255).astype(np.uint8)

            self.progress.emit(70, "Processing NMS...")
            normal = get_array("Normal", (127, 255, 127, 255))
            metallic = get_array("Metallic", (0, 0, 0, 255))[..., 0]
            smoothness = get_array("Smoothness", (127, 127, 127, 255))[..., 0]

            out_nms = np.empty((max_h, max_w, 4), dtype=np.uint8)
            out_nms[..., 0] = normal[..., 0]
            out_nms[..., 1] = normal[..., 1]
            out_nms[..., 2] = metallic
            out_nms[..., 3] = smoothness

            self.progress.emit(85, "Saving textures...")
            
            os.makedirs(self.out_dir, exist_ok=True)
            
            img_base_ao = Image.fromarray(out_base_ao, "RGBA")
            img_base_ao.save(os.path.join(self.out_dir, "BaseAOTransparency.png"), 
                             optimize=True, compress_level=9)

            img_nms = Image.fromarray(out_nms, "RGBA")
            img_nms.save(os.path.join(self.out_dir, "NMS.png"), 
                         optimize=True, compress_level=9)

            self.progress.emit(100, "Done!")
            self.finished.emit(True, "Textures packed successfully.", out_base_ao, out_nms)

        except Exception as e:
            self.finished.emit(False, str(e), None, None)


# ---------------------------------------------------------
# PBR Renderer Widget
# ---------------------------------------------------------
class PBRRendererWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        
        # Rotation variables
        self.rotation_x = -30.0
        self.rotation_y = -45.0
        self.last_pos = QPoint()
        
        # Zoom
        self.zoom = -5.0
        
        # Light position
        self.light_pos = [5.0, 5.0, 5.0]
        self.light_color = [1.0, 1.0, 1.0]
        self.ambient_light = [0.2, 0.2, 0.2]
        
        # Textures
        self.base_ao_texture = None
        self.nms_texture = None
        self.textures_loaded = False
        
        # Sphere display list
        self.sphere_list = None
        self.wireframe_list = None
        
        # Timer for auto-rotation
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_rotate)
        self.auto_rotate_enabled = True
        
    def initializeGL(self):
        # Set up OpenGL
        glClearColor(0.12, 0.12, 0.13, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_NORMALIZE)
        
        # Set up light
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # Create sphere display lists
        self.create_sphere()
        self.create_wireframe_sphere()
        
        # Start auto-rotation timer
        self.timer.start(16)  # ~60 FPS
        
    def create_sphere(self):
        """Create a solid sphere with normals and texture coordinates"""
        self.sphere_list = glGenLists(1)
        glNewList(self.sphere_list, GL_COMPILE)
        
        radius = 1.5
        slices = 64
        stacks = 64
        
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                
                # Normal
                glNormal3f(x * zr0, y * zr0, z0)
                glTexCoord2f(float(j) / slices, float(i) / stacks)
                glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
                
                glNormal3f(x * zr1, y * zr1, z1)
                glTexCoord2f(float(j) / slices, float(i + 1) / stacks)
                glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
            glEnd()
            
        glEndList()
        
    def create_wireframe_sphere(self):
        """Create a wireframe sphere for when no textures are loaded"""
        self.wireframe_list = glGenLists(1)
        glNewList(self.wireframe_list, GL_COMPILE)
        
        radius = 1.5
        slices = 32
        stacks = 32
        
        # Draw latitude lines
        for i in range(stacks):
            lat = math.pi * (-0.5 + float(i) / stacks)
            z = radius * math.sin(lat)
            r = radius * math.cos(lat)
            
            glBegin(GL_LINE_LOOP)
            for j in range(slices):
                lng = 2 * math.pi * float(j) / slices
                x = r * math.cos(lng)
                y = r * math.sin(lng)
                glVertex3f(x, y, z)
            glEnd()
            
        # Draw longitude lines
        for j in range(slices):
            lng = 2 * math.pi * float(j) / slices
            
            glBegin(GL_LINE_STRIP)
            for i in range(stacks + 1):
                lat = math.pi * (-0.5 + float(i) / stacks)
                x = radius * math.cos(lat) * math.cos(lng)
                y = radius * math.cos(lat) * math.sin(lng)
                z = radius * math.sin(lat)
                glVertex3f(x, y, z)
            glEnd()
            
        glEndList()
        
    def setup_textures(self, base_ao_data, nms_data):
        # Convert numpy arrays to OpenGL textures
        if base_ao_data is not None:
            # Delete old texture if exists
            if self.base_ao_texture is not None:
                glDeleteTextures([self.base_ao_texture])
                
            # BaseAO texture
            self.base_ao_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.base_ao_texture)
            
            # Flip image for OpenGL
            img_data = np.flipud(base_ao_data)
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, img_data.shape[1], img_data.shape[0],
                             GL_RGBA, GL_UNSIGNED_BYTE, img_data.tobytes())
            
        if nms_data is not None:
            # Delete old texture if exists
            if self.nms_texture is not None:
                glDeleteTextures([self.nms_texture])
                
            # NMS texture
            self.nms_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.nms_texture)
            
            img_data = np.flipud(nms_data)
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, img_data.shape[1], img_data.shape[0],
                             GL_RGBA, GL_UNSIGNED_BYTE, img_data.tobytes())
            
        self.textures_loaded = True
        self.update()
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply zoom
        glTranslatef(0.0, 0.0, self.zoom)
        
        # Apply rotation
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        
        # Update light position
        glLightfv(GL_LIGHT0, GL_POSITION, [self.light_pos[0], self.light_pos[1], self.light_pos[2], 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [self.light_color[0], self.light_color[1], self.light_color[2], 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [self.ambient_light[0], self.ambient_light[1], self.ambient_light[2], 1.0])
        
        # Draw grid
        self.draw_grid()
        
        # Draw sphere with textures
        if self.textures_loaded and self.sphere_list is not None:
            if self.base_ao_texture is not None:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self.base_ao_texture)
                
                # Set material properties for PBR-like rendering
                glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
                glColor4f(1.0, 1.0, 1.0, 1.0)
                
                # Use specular from NMS texture if available
                if self.nms_texture is not None:
                    # We'll use metallic/smoothness to modify specular
                    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
                    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
                else:
                    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
                    glMaterialf(GL_FRONT, GL_SHININESS, 25.0)
                
                glCallList(self.sphere_list)
                
                # Unbind texture
                glBindTexture(GL_TEXTURE_2D, 0)
                glDisable(GL_TEXTURE_2D)
        else:
            # Draw wireframe sphere if no textures
            glColor3f(0.5, 0.5, 0.5)
            glDisable(GL_LIGHTING)
            glCallList(self.wireframe_list)
            glEnable(GL_LIGHTING)
            
        # Draw light indicator
        self.draw_light_indicator()
        
    def draw_light_indicator(self):
        """Draw a small sphere at the light position"""
        glPushMatrix()
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        
        # Draw light position indicator
        glTranslatef(self.light_pos[0], self.light_pos[1], self.light_pos[2])
        glColor3f(1.0, 1.0, 0.0)  # Yellow indicator
        
        # Draw a small sphere using GLU quadric
        quad = gluNewQuadric()
        gluSphere(quad, 0.1, 8, 8)
        gluDeleteQuadric(quad)
        
        glEnable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glPopMatrix()
        
    def draw_grid(self):
        glPushMatrix()
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        
        # Draw ground grid with fading
        glBegin(GL_LINES)
        for i in range(-10, 11):
            # Calculate alpha based on distance from center
            alpha = max(0.1, 0.3 - abs(i) * 0.02)
            glColor4f(0.2, 0.2, 0.22, alpha)
            
            # X-axis lines
            glVertex3f(i, -2.0, -10)
            glVertex3f(i, -2.0, 10)
            
            # Z-axis lines
            glVertex3f(-10, -2.0, i)
            glVertex3f(10, -2.0, i)
            
        # Draw center lines with brighter color
        glColor3f(0.3, 0.3, 0.33)
        glVertex3f(-10, -2.0, 0)
        glVertex3f(10, -2.0, 0)
        glVertex3f(0, -2.0, -10)
        glVertex3f(0, -2.0, 10)
        
        glEnd()
        
        glEnable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glPopMatrix()
        
    def resizeGL(self, w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        self.auto_rotate_enabled = False
        
    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            self.update()
            
        self.last_pos = event.pos()
        
    def mouseReleaseEvent(self, event):
        # Resume auto-rotation after a short delay
        QTimer.singleShot(2000, lambda: setattr(self, 'auto_rotate_enabled', True))
        
    def wheelEvent(self, event):
        self.zoom += event.angleDelta().y() / 120.0
        self.zoom = max(-15.0, min(-2.0, self.zoom))
        self.update()
        
    def auto_rotate(self):
        if self.auto_rotate_enabled:
            self.rotation_y += 0.5
            if self.rotation_y >= 360:
                self.rotation_y -= 360
            self.update()
            
    def update_textures_from_files(self, out_dir):
        """Try to load textures from saved files"""
        try:
            base_path = os.path.join(out_dir, "BaseAOTransparency.png")
            nms_path = os.path.join(out_dir, "NMS.png")
            
            if os.path.exists(base_path) and os.path.exists(nms_path):
                base_img = Image.open(base_path).convert("RGBA")
                nms_img = Image.open(nms_path).convert("RGBA")
                
                base_data = np.array(base_img)
                nms_data = np.array(nms_img)
                
                self.setup_textures(base_data, nms_data)
                return True
        except Exception as e:
            print(f"Error loading textures: {e}")
        return False


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
        self.setMouseTracking(True)

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
        self.setWindowTitle("PBR Texture Packer with 3D Preview")
        self.resize(1200, 700)
        
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
        
        # Create splitter for resizable layout
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Texture inputs
        left_panel = QWidget()
        left_panel.setMouseTracking(True)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Title
        title_label = QLabel("Texture Maps")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)

        # Previews Grid
        grid = QGridLayout()
        grid.setSpacing(10)
        
        self.previews = {}
        maps = ["BaseColor", "AO", "Metallic", "Smoothness", "Normal", "Alpha"]
        
        for i, m in enumerate(maps):
            pw = ImagePreviewWidget(m)
            pw.fileDropped.connect(self.update_path)
            self.previews[m] = pw
            grid.addWidget(pw, i // 3, i % 3)
            
        left_layout.addLayout(grid)

        # Output Directory
        out_group = QGroupBox("Output Settings")
        out_layout = QVBoxLayout()
        
        out_dir_layout = QHBoxLayout()
        self.btn_out = QPushButton("Select Output Directory")
        self.btn_out.clicked.connect(self.select_output)
        self.lbl_out = QLabel("No directory selected")
        self.lbl_out.setWordWrap(True)
        out_dir_layout.addWidget(self.btn_out)
        out_dir_layout.addWidget(self.lbl_out, 1)
        out_layout.addLayout(out_dir_layout)
        
        # Options
        options_layout = QHBoxLayout()
        self.chk_open = QCheckBox("Open folder when done")
        self.chk_open.setChecked(True)
        options_layout.addWidget(self.chk_open)
        out_layout.addLayout(options_layout)
        
        out_group.setLayout(out_layout)
        left_layout.addWidget(out_group)

        # Progress and Pack button
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
        
        # Add stretch
        left_layout.addStretch()
        
        # Right panel - 3D Preview
        right_panel = QWidget()
        right_panel.setMouseTracking(True)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Preview title
        preview_title = QLabel("3D PBR Preview")
        preview_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        preview_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(preview_title)
        
        # PBR Renderer
        self.pbr_renderer = PBRRendererWidget()
        right_layout.addWidget(self.pbr_renderer)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Light controls
        light_group = QGroupBox("Light Position")
        light_layout = QVBoxLayout()
        
        self.light_x_slider = QSlider(Qt.Horizontal)
        self.light_x_slider.setRange(-10, 10)
        self.light_x_slider.setValue(5)
        self.light_x_slider.valueChanged.connect(self.update_light)
        light_layout.addWidget(QLabel("X"))
        light_layout.addWidget(self.light_x_slider)
        
        self.light_y_slider = QSlider(Qt.Horizontal)
        self.light_y_slider.setRange(-10, 10)
        self.light_y_slider.setValue(5)
        self.light_y_slider.valueChanged.connect(self.update_light)
        light_layout.addWidget(QLabel("Y"))
        light_layout.addWidget(self.light_y_slider)
        
        self.light_z_slider = QSlider(Qt.Horizontal)
        self.light_z_slider.setRange(-10, 10)
        self.light_z_slider.setValue(5)
        self.light_z_slider.valueChanged.connect(self.update_light)
        light_layout.addWidget(QLabel("Z"))
        light_layout.addWidget(self.light_z_slider)
        
        light_group.setLayout(light_layout)
        controls_layout.addWidget(light_group)
        
        # Reset view button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        controls_layout.addWidget(reset_btn)
        
        right_layout.addLayout(controls_layout)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])
        
        # Main layout
        layout = QVBoxLayout(main_widget)
        layout.addWidget(splitter)

        # Install event filters on all child widgets after they're created
        self.installEventFilters(self)

    def update_light(self):
        if hasattr(self, 'pbr_renderer'):
            self.pbr_renderer.light_pos = [
                self.light_x_slider.value(),
                self.light_y_slider.value(),
                self.light_z_slider.value()
            ]
            self.pbr_renderer.update()
            
    def reset_view(self):
        if hasattr(self, 'pbr_renderer'):
            self.pbr_renderer.rotation_x = -30.0
            self.pbr_renderer.rotation_y = -45.0
            self.pbr_renderer.zoom = -5.0
            self.pbr_renderer.update()
            
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

    def packing_finished(self, success, msg, base_ao_data, nms_data):
        self.btn_pack.setEnabled(True)
        if success:
            self.progress_bar.setFormat("100% - Finished!")
            
            # Update PBR renderer with the new textures
            if base_ao_data is not None and nms_data is not None:
                # Make OpenGL context current
                self.pbr_renderer.makeCurrent()
                self.pbr_renderer.setup_textures(base_ao_data, nms_data)
                self.pbr_renderer.doneCurrent()
            
            if self.chk_open.isChecked():
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))
        else:
            self.progress_bar.setFormat(f"Error: {msg}")


if __name__ == "__main__":
    # Set OpenGL format before creating application
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())