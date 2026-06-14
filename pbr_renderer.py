import math
import os
import ctypes
import numpy as np
import glm
from PIL import Image
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QSurfaceFormat
from OpenGL.GL import *
from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT, GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT

# ----------------------------------------------------------------------
# helper to convert numpy array to ctypes for VBO upload
def np_to_gl_array(arr, dtype=np.float32):
    return arr.astype(dtype).flatten().tobytes()
# ----------------------------------------------------------------------

class CompositionWorker(QThread):
    resultReady = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, renderer):
        super().__init__()
        # copy all needed data because we'll run in a different thread
        self.input_textures = {k: v.copy() if v is not None else None
                               for k, v in renderer.input_textures.items()}
        self.ao_intensity = renderer.ao_intensity
        self.invert_normal_y = renderer.invert_normal_y
        self.default_colors = {
            "BaseColor": [255, 255, 255, 255],
            "AO": [255, 255, 255, 255],
            "Metallic": [0, 0, 0, 255],
            "Smoothness": [127, 127, 127, 255],
            "Normal": [127, 127, 255, 255],
            "Alpha": [255, 255, 255, 255],
        }

    def run(self):
        # identical logic to the old compose_live_texture_set,
        # but using self.input_textures etc.
        # (Move the entire compose_live_texture_set here, returning base_ao_data, nms_data)
        base_ao_data, nms_data = self.compose()
        self.resultReady.emit(base_ao_data, nms_data)

    def create_default_texture(self, color):
        return np.array([[color]], dtype=np.uint8)
    
    def apply_normal_y_inversion(self, normal):
        if not self.invert_normal_y:
            return normal
        inverted = normal.copy()
        inverted[..., 1] = 255 - inverted[..., 1]
        return inverted
    
    def compose(self):
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

        ao_channel = np.clip((ao[..., 0] / 255.0) ** max(self.ao_intensity, 0.00001), 0.0, 1.0)
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

    # Connect the worker inside __init__
    def connect_composition_signal(self):
        # This method must be called once after the widget is created
        pass   # We'll wire it in initializeGL or after

    # In start_composition_thread:
    def start_composition_thread(self):
        if not self.pending_composition:
            return
        self.pending_composition = False
        self.worker = CompositionWorker(self)
        self.worker.resultReady.connect(self.on_composition_done)
        self.worker.start()

    # Slot to upload the textures in the GUI thread
    def on_composition_done(self, base_ao_data, nms_data):
        self.upload_texture_set(base_ao_data, nms_data)

class PBRRendererWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.pending_composition = False
        self.composition_timer = QTimer()
        self.composition_timer.setSingleShot(True)
        self.composition_timer.timeout.connect(self.start_composition_thread)

        fmt = QSurfaceFormat()
        fmt.setSamples(8)
        fmt.setDepthBufferSize(24)
        fmt.setVersion(3, 3)          # request OpenGL 3.3 core profile
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(fmt)

        self.setMinimumSize(400, 400)
        self.rotation_x = -30.0
        self.rotation_y = -45.0
        self.last_pos = None
        self.zoom = -5.0
        self.key_light = {'pos': [6.0, 7.0, 8.0], 'color': [1.0, 0.97, 0.92], 'intensity': 18.0}
        self.fill_light = {'pos': [-5.0, 1.5, 2.5], 'color': [0.65, 0.75, 1.0], 'intensity': 4.5}
        self.rim_light = {'pos': [-2.0, 4.0, -7.0], 'color': [0.95, 0.98, 1.0], 'intensity': 9.0}
        self.ao_intensity = 1.0
        self.invert_normal_y = False
        self.base_alpha_tex = None
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
        self.packed_base_alpha_data = None
        self.packed_nms_data = None

        # --- VAO / VBO / EBO handles ---
        self.sphere_vao = None
        self.sphere_vbo_vertices = None
        self.sphere_vbo_normals = None
        self.sphere_vbo_texcoords = None
        self.sphere_ebo = None
        self.sphere_index_count = 0

        self.wireframe_vao = None
        self.wireframe_vbo = None
        self.wireframe_vertex_count = 0

        self.has_anisotropy = False
        self.max_anisotropy = 1.0

        self.shader_program = None
        self.shader_dir = os.path.join(os.path.dirname(__file__), "shaders")

        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_rotate)
        self.auto_rotate_enabled = True

    # In start_composition_thread:
    def start_composition_thread(self):
        if not self.pending_composition:
            return
        self.pending_composition = False
        self.worker = CompositionWorker(self)
        self.worker.resultReady.connect(self.on_composition_done)
        self.worker.start()

    # Slot to upload the textures in the GUI thread
    def on_composition_done(self, base_ao_data, nms_data):
        self.upload_texture_set(base_ao_data, nms_data)

    def request_refresh(self):
        """Debounce texture changes and start a worker thread."""
        self.pending_composition = True
        self.composition_timer.start(50)   # 50 ms delay, adjust as needed

    def initializeGL(self):
        glEnable(GL_MULTISAMPLE)
        glClearColor(0.12, 0.12, 0.13, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE)

        #self.CompositionWorker.connect_composition_signal()

        self.check_anisotropy_support()
        self.create_sphere_geometry()
        self.create_wireframe_sphere_geometry()
        self.create_shader_program()
        self.refresh_preview_textures()
        self.timer.start(16)

    # ---- shader loading (unchanged) -------------------------------------------------
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

    # ---- anisotropy support ----------------------------------------------------------
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

    # =========================================================================
    #  NEW: VAO‑based sphere creation
    # =========================================================================
    def create_sphere_geometry(self):
        radius = 1.5
        slices = 128
        stacks = 128

        vertices = []
        normals = []
        texcoords = []
        indices = []

        # generate vertices, normals, texcoords
        for i in range(stacks + 1):
            lat = math.pi * (-0.5 + float(i) / stacks)
            z = math.sin(lat)
            zr = math.cos(lat)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                vertices.extend([x * zr * radius, y * zr * radius, z * radius])
                normals.extend([x * zr, y * zr, z])        # unit length
                texcoords.extend([float(j) / slices, float(i) / stacks])

        # generate indices for triangle strips
        for i in range(stacks):
            for j in range(slices):
                first = i * (slices + 1) + j
                second = first + slices + 1
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        self.sphere_index_count = len(indices)

        # --- VAO ---
        self.sphere_vao = glGenVertexArrays(1)
        glBindVertexArray(self.sphere_vao)

        # vertex buffer
        self.sphere_vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, np_to_gl_array(np.array(vertices)), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # normal buffer
        self.sphere_vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, np_to_gl_array(np.array(normals)), GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # texcoord buffer
        self.sphere_vbo_texcoords = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo_texcoords)
        glBufferData(GL_ARRAY_BUFFER, np_to_gl_array(np.array(texcoords)), GL_STATIC_DRAW)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        # index buffer
        self.sphere_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.sphere_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(indices, dtype=np.uint32).tobytes(), GL_STATIC_DRAW)

        glBindVertexArray(0)

    # =========================================================================
    #  NEW: VAO‑based wireframe sphere (just lines)
    # =========================================================================
    def create_wireframe_sphere_geometry(self):
        radius = 1.5
        slices = 32
        stacks = 32
        vertices = []

        # latitude circles
        for i in range(stacks):
            lat = math.pi * (-0.5 + float(i) / stacks)
            z = radius * math.sin(lat)
            r = radius * math.cos(lat)
            for j in range(slices):
                lng = 2 * math.pi * float(j) / slices
                vertices.extend([r * math.cos(lng), r * math.sin(lng), z])
                # next point
                lng2 = 2 * math.pi * float(j + 1) / slices
                vertices.extend([r * math.cos(lng2), r * math.sin(lng2), z])

        # longitude lines
        for j in range(slices):
            lng = 2 * math.pi * float(j) / slices
            for i in range(stacks):
                lat1 = math.pi * (-0.5 + float(i) / stacks)
                lat2 = math.pi * (-0.5 + float(i + 1) / stacks)
                vertices.extend([
                    radius * math.cos(lat1) * math.cos(lng),
                    radius * math.cos(lat1) * math.sin(lng),
                    radius * math.sin(lat1),
                    radius * math.cos(lat2) * math.cos(lng),
                    radius * math.cos(lat2) * math.sin(lng),
                    radius * math.sin(lat2)
                ])

        self.wireframe_vertex_count = len(vertices) // 3

        self.wireframe_vao = glGenVertexArrays(1)
        glBindVertexArray(self.wireframe_vao)
        self.wireframe_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.wireframe_vbo)
        glBufferData(GL_ARRAY_BUFFER, np_to_gl_array(np.array(vertices)), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

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
        self.request_refresh()

    def create_default_texture(self, color):
        return np.array([[color]], dtype=np.uint8)

    def apply_normal_y_inversion(self, normal):
        if not self.invert_normal_y:
            return normal
        inverted = normal.copy()
        inverted[..., 1] = 255 - inverted[..., 1]
        return inverted

    def refresh_preview_textures(self):
        if self.preview_mode == "packed" and self.packed_base_alpha_data is not None and self.packed_nms_data is not None:
            self.upload_texture_set(self.packed_base_alpha_data, self.packed_nms_data)
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
                image = image.resize((width, height), Image.Resampling.BILINEAR)
                return np.array(image, dtype=np.float32)
            return texture.astype(np.float32)

        base_color = resize_texture(base_color, max_w, max_h)
        ao = resize_texture(ao, max_w, max_h)
        metallic = resize_texture(metallic, max_w, max_h)
        smoothness = resize_texture(smoothness, max_w, max_h)
        normal = resize_texture(normal, max_w, max_h)
        alpha = resize_texture(alpha, max_w, max_h)
        normal = self.apply_normal_y_inversion(normal)

        #ao_channel = np.clip((ao[..., 0] / 255.0) ** max(self.ao_intensity, 0.00001), 0.0, 1.0)
        
        ao_lut = (np.arange(256, dtype=np.float32) / 255.0) ** max(self.ao_intensity, 0.00001)
        ao_channel = ao_lut[ao[..., 0].astype(np.uint8)]
        #ao_channel = ao_lut[ao[..., 0]]   # fast integer indexing

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
        # No makeCurrent() here – called from paintGL context
        if self.base_alpha_tex is not None:
            glDeleteTextures([self.base_alpha_tex])
        if self.nms_tex is not None:
            glDeleteTextures([self.nms_tex])
        self.base_alpha_tex = self.create_gl_texture(base_ao_data)
        self.nms_tex = self.create_gl_texture(nms_data)
        self.textures_loaded = True
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
            self.request_refresh()

    def set_normal_y_inverted(self, inverted):
        self.invert_normal_y = inverted
        if self.preview_mode == "input":
            self.request_refresh()

    def set_packed_textures(self, base_ao_data, nms_data):
        self.packed_base_alpha_data = base_ao_data
        self.packed_nms_data = nms_data
        self.preview_mode = "packed"
        self.refresh_preview_textures()

    def use_input_preview(self):
        self.preview_mode = "input"
        self.request_refresh()

    # =========================================================================
    #  NEW: uniform setup + drawing
    # =========================================================================
    def set_shader_uniforms(self):
        # ---------- matrices ------------------------------------------------
        model = (
            glm.rotate(glm.mat4(1.0), glm.radians(self.rotation_x), glm.vec3(1, 0, 0)) *
            glm.rotate(glm.mat4(1.0), glm.radians(self.rotation_y), glm.vec3(0, 1, 0))
        )
        view = glm.lookAt(
            glm.vec3(0.0, 0.0, self.zoom),
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, 1.0, 0.0)
        )
        projection = glm.perspective(
            glm.radians(45.0),
            self.width() / max(self.height(), 1.0),
            0.1,
            100.0
        )

        mvp = projection * view * model
        normal_matrix = glm.transpose(glm.inverse(glm.mat3(model)))

        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "u_mvp"), 1, GL_FALSE, glm.value_ptr(mvp))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "u_model"), 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "u_view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix3fv(glGetUniformLocation(self.shader_program, "u_normal_matrix"), 1, GL_FALSE, glm.value_ptr(normal_matrix))

        # ---------- lights as struct array -----------------------------------
        lights = [
            {
                'pos': self.key_light['pos'],
                'color': self.key_light['color'],
                'intensity': self.key_light['intensity']
            },
            {
                'pos': self.fill_light['pos'],
                'color': self.fill_light['color'],
                'intensity': self.fill_light['intensity']
            },
            {
                'pos': self.rim_light['pos'],
                'color': self.rim_light['color'],
                'intensity': self.rim_light['intensity']
            }
        ]

        glUniform1i(glGetUniformLocation(self.shader_program, "num_lights"), len(lights))

        for i, light in enumerate(lights):
            glUniform3f(
                glGetUniformLocation(self.shader_program, f"lights[{i}].pos"),
                *light['pos']
            )
            glUniform3f(
                glGetUniformLocation(self.shader_program, f"lights[{i}].color"),
                *light['color']
            )
            glUniform1f(
                glGetUniformLocation(self.shader_program, f"lights[{i}].intensity"),
                light['intensity']
            )

        # camera position in world space
        glUniform3f(glGetUniformLocation(self.shader_program, "camera_pos"), 0.0, 0.0, self.zoom)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # upload matrices & lighting uniforms
        self.set_shader_uniforms()

        # ------ draw solid sphere if textures loaded ------
        if self.textures_loaded and self.base_alpha_tex is not None and self.sphere_vao:
            # bind textures
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.base_alpha_tex)
            glUniform1i(glGetUniformLocation(self.shader_program, "base_alpha_tex"), 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.nms_tex)
            glUniform1i(glGetUniformLocation(self.shader_program, "nms_tex"), 1)

            glBindVertexArray(self.sphere_vao)
            glDrawElements(GL_TRIANGLES, self.sphere_index_count, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

            # unbind textures
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 0)
        else:
            # ------ fallback wireframe sphere ------
            if self.wireframe_vao:
                # set a simple untextured uniform (the shader can use a fallback)
                glBindVertexArray(self.wireframe_vao)
                glDrawArrays(GL_LINES, 0, self.wireframe_vertex_count)
                glBindVertexArray(0)

        glUseProgram(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, max(h, 1))

    # =========================================================================
    #  mouse / auto‑rotate (unchanged)
    # =========================================================================
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        self.auto_rotate_enabled = False

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            self.last_pos = event.pos()
            return
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        if event.buttons() & Qt.LeftButton:
            self.rotation_y += dx * 0.5
            self.rotation_x -= dy * 0.5
            self.update()
        self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        QTimer.singleShot(2000, lambda: setattr(self, 'auto_rotate_enabled', True))

    def wheelEvent(self, event):
        self.zoom += event.angleDelta().y() / 120.0
        self.zoom = max(-15.0, min(-2.0, self.zoom))
        self.update()

    def auto_rotate(self):
        self.auto_rotate_enabled = False
        if self.auto_rotate_enabled:
            self.rotation_y += 0.5
            if self.rotation_y >= 360:
                self.rotation_y -= 360
            self.update()