import math
import os

import numpy as np
from PIL import Image
from PyQt5.QtCore import QPoint, QTimer, Qt
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QSurfaceFormat
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.EXT.texture_filter_anisotropic import *


class PBRRendererWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        fmt = QSurfaceFormat()
        fmt.setSamples(8)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)

        self.setMinimumSize(400, 400)
        self.rotation_x = -30.0
        self.rotation_y = -45.0
        self.last_pos = QPoint()
        self.zoom = -5.0
        self.key_light = {'pos': [6.0, 7.0, 8.0], 'color': [1.0, 0.97, 0.92], 'intensity': 18.0}
        self.fill_light = {'pos': [-5.0, 1.5, 2.5], 'color': [0.65, 0.75, 1.0], 'intensity': 4.5}
        self.rim_light = {'pos': [-2.0, 4.0, -7.0], 'color': [0.95, 0.98, 1.0], 'intensity': 9.0}
        self.ao_intensity = 1.0
        self.invert_normal_y = False
        self.base_ao_tex = None
        self.nms_tex = None
        self.textures_loaded = False
        self.preview_mode = "input"
        self.input_textures = {
            "BaseColor": None,
            "AO": None,
            "Metallic": None,
            "Smoothness": None,
            "Normal": None,
            "Alpha": None,
        }
        self.packed_base_ao_data = None
        self.packed_nms_data = None
        self.sphere_list = None
        self.wireframe_list = None
        self.has_anisotropy = False
        self.max_anisotropy = 1.0
        self.shader_program = None
        self.shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_rotate)
        self.auto_rotate_enabled = True

    def initializeGL(self):
        glEnable(GL_MULTISAMPLE)
        glClearColor(0.12, 0.12, 0.13, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        self.check_anisotropy_support()
        self.create_sphere()
        self.create_wireframe_sphere()
        self.create_shader_program()
        self.refresh_preview_textures()
        self.timer.start(16)

    def create_shader_program(self):
        vertex_shader = self.compile_shader(self.load_shader_source("pbr_preview.vert"), GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(self.load_shader_source("pbr_preview.frag"), GL_FRAGMENT_SHADER)
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        if glGetProgramiv(self.shader_program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.shader_program).decode())
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def load_shader_source(self, filename):
        shader_path = os.path.join(self.shader_dir, filename)
        with open(shader_path, "r", encoding="utf-8") as shader_file:
            return shader_file.read()

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(shader).decode())
        return shader

    def check_anisotropy_support(self):
        try:
            extensions = glGetString(GL_EXTENSIONS)
            if extensions:
                ext_string = extensions.decode() if isinstance(extensions, bytes) else extensions
                if 'GL_EXT_texture_filter_anisotropic' in ext_string:
                    self.has_anisotropy = True
                    self.max_anisotropy = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                else:
                    self.has_anisotropy = False
            else:
                self.has_anisotropy = False
        except Exception:
            self.has_anisotropy = False
            self.max_anisotropy = 1.0

    def create_sphere(self):
        self.sphere_list = glGenLists(1)
        glNewList(self.sphere_list, GL_COMPILE)
        radius = 1.5
        slices = 128
        stacks = 128
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
                glNormal3f(x * zr0, y * zr0, z0)
                glTexCoord2f(float(j) / slices, float(i) / stacks)
                glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
                glNormal3f(x * zr1, y * zr1, z1)
                glTexCoord2f(float(j) / slices, float(i + 1) / stacks)
                glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
            glEnd()
        glEndList()

    def create_wireframe_sphere(self):
        self.wireframe_list = glGenLists(1)
        glNewList(self.wireframe_list, GL_COMPILE)
        radius = 1.5
        slices = 32
        stacks = 32
        for i in range(stacks):
            lat = math.pi * (-0.5 + float(i) / stacks)
            z = radius * math.sin(lat)
            r = radius * math.cos(lat)
            glBegin(GL_LINE_LOOP)
            for j in range(slices):
                lng = 2 * math.pi * float(j) / slices
                glVertex3f(r * math.cos(lng), r * math.sin(lng), z)
            glEnd()
        for j in range(slices):
            lng = 2 * math.pi * float(j) / slices
            glBegin(GL_LINE_STRIP)
            for i in range(stacks + 1):
                lat = math.pi * (-0.5 + float(i) / stacks)
                glVertex3f(
                    radius * math.cos(lat) * math.cos(lng),
                    radius * math.cos(lat) * math.sin(lng),
                    radius * math.sin(lat),
                )
            glEnd()
        glEndList()

    def load_input_texture(self, name, path):
        if path and os.path.exists(path):
            try:
                image = Image.open(path).convert("RGBA")
                self.input_textures[name] = np.array(image, dtype=np.uint8)
            except Exception:
                self.input_textures[name] = None
        else:
            self.input_textures[name] = None
        self.preview_mode = "input"
        self.refresh_preview_textures()

    def create_default_texture(self, color):
        return np.array([[color]], dtype=np.uint8)

    def apply_normal_y_inversion(self, normal):
        if not self.invert_normal_y:
            return normal
        inverted = normal.copy()
        inverted[..., 1] = 255 - inverted[..., 1]
        return inverted

    def refresh_preview_textures(self):
        if self.preview_mode == "packed" and self.packed_base_ao_data is not None and self.packed_nms_data is not None:
            self.upload_texture_set(self.packed_base_ao_data, self.packed_nms_data)
            return
        self.upload_texture_set(*self.compose_live_texture_set())

    def compose_live_texture_set(self):
        base_color = self.input_textures["BaseColor"]
        if base_color is None:
            base_color = self.create_default_texture([255, 255, 255, 255])
        ao = self.input_textures["AO"]
        if ao is None:
            ao = self.create_default_texture([255, 255, 255, 255])
        metallic = self.input_textures["Metallic"]
        if metallic is None:
            metallic = self.create_default_texture([0, 0, 0, 255])
        smoothness = self.input_textures["Smoothness"]
        if smoothness is None:
            smoothness = self.create_default_texture([127, 127, 127, 255])
        normal = self.input_textures["Normal"]
        if normal is None:
            normal = self.create_default_texture([127, 127, 255, 255])
        alpha = self.input_textures["Alpha"]
        if alpha is None:
            alpha = self.create_default_texture([255, 255, 255, 255])

        max_h = max(tex.shape[0] for tex in [base_color, ao, metallic, smoothness, normal, alpha])
        max_w = max(tex.shape[1] for tex in [base_color, ao, metallic, smoothness, normal, alpha])

        def resize_texture(texture, width, height):
            if texture.shape[0] != height or texture.shape[1] != width:
                image = Image.fromarray(texture)
                image = image.resize((width, height), Image.Resampling.LANCZOS)
                return np.array(image, dtype=np.float32)
            return texture.astype(np.float32)

        base_color = resize_texture(base_color, max_w, max_h)
        ao = resize_texture(ao, max_w, max_h)
        metallic = resize_texture(metallic, max_w, max_h)
        smoothness = resize_texture(smoothness, max_w, max_h)
        normal = resize_texture(normal, max_w, max_h)
        alpha = resize_texture(alpha, max_w, max_h)
        normal = self.apply_normal_y_inversion(normal)

        ao_channel = np.clip((ao[..., 0] / 255.0) * self.ao_intensity, 0.0, 1.0)
        base_ao_rgb = base_color[..., :3] * ao_channel[..., np.newaxis]
        transparency = alpha[..., 0] if self.input_textures["Alpha"] is not None else base_color[..., 3]

        base_ao_data = np.zeros((max_h, max_w, 4), dtype=np.uint8)
        base_ao_data[..., :3] = np.clip(base_ao_rgb, 0, 255).astype(np.uint8)
        base_ao_data[..., 3] = np.clip(transparency, 0, 255).astype(np.uint8)

        nms_data = np.zeros((max_h, max_w, 4), dtype=np.uint8)
        nms_data[..., 0] = normal[..., 0]
        nms_data[..., 1] = normal[..., 1]
        nms_data[..., 2] = metallic[..., 0]
        nms_data[..., 3] = smoothness[..., 0]
        return base_ao_data, nms_data

    def upload_texture_set(self, base_ao_data, nms_data):
        self.makeCurrent()
        if self.base_ao_tex is not None:
            glDeleteTextures([self.base_ao_tex])
        if self.nms_tex is not None:
            glDeleteTextures([self.nms_tex])
        self.base_ao_tex = self.create_gl_texture(base_ao_data)
        self.nms_tex = self.create_gl_texture(nms_data)
        self.textures_loaded = True
        self.doneCurrent()
        self.update()

    def create_gl_texture(self, data):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        image_data = np.flipud(data)
        if self.has_anisotropy:
            try:
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, self.max_anisotropy)
            except Exception:
                pass
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            image_data.shape[1],
            image_data.shape[0],
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            image_data,
        )
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture_id

    def set_ao_intensity(self, intensity):
        self.ao_intensity = intensity
        if self.preview_mode == "input":
            self.refresh_preview_textures()

    def set_normal_y_inverted(self, inverted):
        self.invert_normal_y = inverted
        if self.preview_mode == "input":
            self.refresh_preview_textures()

    def set_packed_textures(self, base_ao_data, nms_data):
        self.packed_base_ao_data = base_ao_data
        self.packed_nms_data = nms_data
        self.preview_mode = "packed"
        self.refresh_preview_textures()

    def use_input_preview(self):
        self.preview_mode = "input"
        self.refresh_preview_textures()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        #self.draw_grid()
        if self.textures_loaded and self.sphere_list is not None and self.base_ao_tex is not None and self.shader_program is not None:
            glUseProgram(self.shader_program)
            self.set_shader_uniforms()
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.base_ao_tex)
            glUniform1i(glGetUniformLocation(self.shader_program, "base_ao_tex"), 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.nms_tex)
            glUniform1i(glGetUniformLocation(self.shader_program, "nms_tex"), 1)
            glCallList(self.sphere_list)
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0)
            glUseProgram(0)
        else:
            glColor3f(0.5, 0.5, 0.5)
            glCallList(self.wireframe_list)
        #self.draw_light_indicators()

    def get_camera_world_pos(self):
        # Read the current modelview matrix (column‑major)
        mv_mat = glGetFloatv(GL_MODELVIEW_MATRIX)
        # Reshape to 4x4 and transpose to row‑major for numpy
        M = np.array(mv_mat).reshape(4, 4).T
        # Invert the view matrix to get the camera’s transformation
        invM = np.linalg.inv(M)
        # The translation part of the inverse is the camera world position
        #return invM[:3, 3].tolist()
        return [0.0, 0.0, 0.0]

    def set_shader_uniforms(self):
        glUniform3f(glGetUniformLocation(self.shader_program, "key_light_pos"), *self.key_light['pos'])
        glUniform3f(glGetUniformLocation(self.shader_program, "key_light_color"), *self.key_light['color'])
        glUniform1f(glGetUniformLocation(self.shader_program, "key_light_intensity"), self.key_light['intensity'])
        glUniform3f(glGetUniformLocation(self.shader_program, "fill_light_pos"), *self.fill_light['pos'])
        glUniform3f(glGetUniformLocation(self.shader_program, "fill_light_color"), *self.fill_light['color'])
        glUniform1f(glGetUniformLocation(self.shader_program, "fill_light_intensity"), self.fill_light['intensity'])
        glUniform3f(glGetUniformLocation(self.shader_program, "rim_light_pos"), *self.rim_light['pos'])
        glUniform3f(glGetUniformLocation(self.shader_program, "rim_light_color"), *self.rim_light['color'])
        glUniform1f(glGetUniformLocation(self.shader_program, "rim_light_intensity"), self.rim_light['intensity'])
        cam_pos = self.get_camera_world_pos()
        glUniform3f(glGetUniformLocation(self.shader_program, "camera_pos"), *cam_pos)

    def draw_light_indicators(self):
        glUseProgram(0)
        glDisable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 0.0)
        self.draw_light_sphere(self.key_light['pos'])
        glColor3f(0.5, 0.5, 1.0)
        self.draw_light_sphere(self.fill_light['pos'])
        glColor3f(1.0, 1.0, 1.0)
        self.draw_light_sphere(self.rim_light['pos'])
        glEnable(GL_TEXTURE_2D)

    def draw_light_sphere(self, pos):
        glPushMatrix()
        glTranslatef(*pos)
        quadric = gluNewQuadric()
        gluSphere(quadric, 0.15, 16, 16)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def draw_grid(self):
        glUseProgram(0)
        glPushMatrix()
        glDisable(GL_TEXTURE_2D)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            alpha = max(0.05, 0.2 - abs(i) * 0.015)
            glColor4f(0.2, 0.2, 0.22, alpha)
            glVertex3f(i, -2.0, -10)
            glVertex3f(i, -2.0, 10)
            glVertex3f(-10, -2.0, i)
            glVertex3f(10, -2.0, i)
        glColor3f(0.3, 0.3, 0.33)
        glVertex3f(-10, -2.0, 0)
        glVertex3f(10, -2.0, 0)
        glVertex3f(0, -2.0, -10)
        glVertex3f(0, -2.0, 10)
        glEnd()
        glEnable(GL_TEXTURE_2D)
        glPopMatrix()

    def resizeGL(self, w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 0.1, 100.0)
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
