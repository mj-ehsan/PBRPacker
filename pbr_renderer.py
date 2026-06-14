import math
import os
import numpy as np
import glm
from PIL import Image
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QSurfaceFormat
from OpenGL.GL import *
from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT, GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT

# ----------------------------------------------------------------------
# helper to convert numpy array to ctypes for VBO upload
def np_to_gl_array(arr, dtype=np.float32):
    return arr.astype(dtype).flatten().tobytes()
# ----------------------------------------------------------------------

class PBRRendererWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.pending_composition = False
        self.compose_requested = False
        self.compose_vao = None
        self.compose_fbo = None
        self.compose_size = (0, 0)
        self.compose_source_textures = {}
        self.compose_source_size = (1, 1)
        self.default_texture_cache = {}
        self.external_packed_mode = False
        self.composition_timer = QTimer()
        self.composition_timer.setSingleShot(True)
        self.composition_timer.timeout.connect(self.schedule_compose_pass)

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
        self.normal_gen_sigma = 1.0
        self.normal_gen_height = 1.0
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
        self.compose_shader_program = None
        self.shader_dir = os.path.join(os.path.dirname(__file__), "shaders")

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick_preview)
        self.auto_rotate_enabled = True

    def schedule_compose_pass(self):
        if not self.pending_composition:
            return
        self.pending_composition = False
        self.compose_requested = True
        self.update()

    def request_refresh(self):
        """Debounce texture changes and schedule a single compose pass."""
        self.external_packed_mode = False
        self.pending_composition = True
        self.composition_timer.start(50)

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
        self.create_shader_programs()
        self.compose_requested = True
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

    def create_shader_program(self, vertex_name, fragment_name):
        vertex_shader = self.compile_shader(self.load_shader_source(vertex_name), GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(self.load_shader_source(fragment_name), GL_FRAGMENT_SHADER)
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)
        if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(shader_program).decode())
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        return shader_program

    def create_shader_programs(self):
        self.shader_program = self.create_shader_program("pbr_preview.vert", "pbr_preview.frag")
        self.compose_shader_program = self.create_shader_program("compose.vert", "compose.frag")
        self.compose_vao = glGenVertexArrays(1)

    # ---- anisotropy support ----------------------------------------------------------
    def check_anisotropy_support(self):
        try:
            num_extensions = glGetIntegerv(GL_NUM_EXTENSIONS)
            extensions = {
                glGetStringi(GL_EXTENSIONS, index).decode("utf-8")
                for index in range(num_extensions)
            }
            self.has_anisotropy = 'GL_EXT_texture_filter_anisotropic' in extensions
            if self.has_anisotropy:
                self.max_anisotropy = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
            else:
                self.max_anisotropy = 1.0
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

    def create_gl_texture(self, data, generate_mipmaps=True):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        image_data = np.flipud(data)
        if self.has_anisotropy:
            try:
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, self.max_anisotropy)
            except Exception:
                pass
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        min_filter = GL_LINEAR_MIPMAP_LINEAR if generate_mipmaps else GL_LINEAR
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)
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
        if generate_mipmaps:
            glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture_id

    def create_empty_gl_texture(self, width, height):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture_id

    def get_or_create_default_gl_texture(self, name, color):
        texture_id = self.default_texture_cache.get(name)
        if texture_id is None:
            texture_id = self.create_gl_texture(self.create_default_texture(color), generate_mipmaps=True)
            self.default_texture_cache[name] = texture_id
        return texture_id

    def delete_texture(self, texture_id):
        if texture_id is not None:
            glDeleteTextures([texture_id])

    def sync_source_textures(self):
        defaults = {
            "BaseColor": [255, 255, 255, 255],
            "AO": [255, 255, 255, 255],
            "Metallic": [0, 0, 0, 255],
            "Smoothness": [127, 127, 127, 255],
            "Normal": [127, 127, 255, 255],
            "Alpha": [255, 255, 255, 255],
        }

        max_h = 1
        max_w = 1
        for texture_data in self.input_textures.values():
            if texture_data is not None:
                max_h = max(max_h, texture_data.shape[0])
                max_w = max(max_w, texture_data.shape[1])

        default_textures = set(self.default_texture_cache.values())
        for name, color in defaults.items():
            texture_data = self.input_textures[name]
            old_texture = self.compose_source_textures.get(name)
            if old_texture is not None and old_texture not in default_textures:
                self.delete_texture(old_texture)
            if texture_data is None:
                self.compose_source_textures[name] = self.get_or_create_default_gl_texture(name, color)
            else:
                self.compose_source_textures[name] = self.create_gl_texture(texture_data, generate_mipmaps=False)

        self.compose_source_size = (max_w, max_h)

    def ensure_compose_targets(self, width, height):
        if self.compose_size == (width, height) and self.compose_fbo is not None and self.base_alpha_tex is not None and self.nms_tex is not None:
            return

        if self.compose_fbo is not None:
            glDeleteFramebuffers(1, [self.compose_fbo])
            self.compose_fbo = None

        self.delete_texture(self.base_alpha_tex)
        self.delete_texture(self.nms_tex)
        self.base_alpha_tex = self.create_empty_gl_texture(width, height)
        self.nms_tex = self.create_empty_gl_texture(width, height)

        self.compose_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.compose_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.base_alpha_tex, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self.nms_tex, 0)
        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Compose framebuffer incomplete: 0x{status:04X}")

        self.compose_size = (width, height)

    def run_compose_pass(self):
        if self.compose_shader_program is None:
            return

        self.sync_source_textures()
        width, height = self.compose_source_size
        self.ensure_compose_targets(width, height)

        previous_viewport = glGetIntegerv(GL_VIEWPORT)
        previous_framebuffer = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        glBindFramebuffer(GL_FRAMEBUFFER, self.compose_fbo)
        glViewport(0, 0, width, height)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.compose_shader_program)
        glBindVertexArray(self.compose_vao)

        for unit, name in enumerate(["BaseColor", "AO", "Metallic", "Smoothness", "Normal", "Alpha"]):
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, self.compose_source_textures[name])
            glUniform1i(glGetUniformLocation(self.compose_shader_program, name), unit)

        glUniform1f(glGetUniformLocation(self.compose_shader_program, "u_ao_intensity"), max(self.ao_intensity, 0.00001))
        glUniform1f(glGetUniformLocation(self.compose_shader_program, "u_normal_gen_sigma"), max(self.normal_gen_sigma, 0.0001))
        glUniform1f(glGetUniformLocation(self.compose_shader_program, "u_normal_gen_height"), self.normal_gen_height)
        glUniform1i(glGetUniformLocation(self.compose_shader_program, "u_invert_normal_y"), int(self.invert_normal_y))
        glUniform1i(glGetUniformLocation(self.compose_shader_program, "u_use_alpha"), int(self.input_textures["Alpha"] is not None))
        glUniform1i(glGetUniformLocation(self.compose_shader_program, "u_generate_normal_from_luma"), int(self.input_textures["Normal"] is None))

        glDrawArrays(GL_TRIANGLES, 0, 3)

        for unit in range(6):
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, 0)

        glBindVertexArray(0)
        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, self.base_alpha_tex)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.nms_tex)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, previous_framebuffer)
        glViewport(*previous_viewport)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)

        self.textures_loaded = True
        self.compose_requested = False

    def set_ao_intensity(self, intensity):
        self.ao_intensity = intensity
        self.request_refresh()

    def set_normal_generation(self, sigma, height):
        self.normal_gen_sigma = sigma
        self.normal_gen_height = height
        self.request_refresh()

    def set_normal_y_inverted(self, inverted):
        self.invert_normal_y = inverted
        self.request_refresh()

    def set_packed_textures(self, base_alpha_data, nms_data):
        self.packed_base_alpha_data = base_alpha_data
        self.packed_nms_data = nms_data
        self.preview_mode = "packed"
        self.external_packed_mode = True
        if self.compose_fbo is not None:
            glDeleteFramebuffers(1, [self.compose_fbo])
            self.compose_fbo = None
        self.delete_texture(self.base_alpha_tex)
        self.delete_texture(self.nms_tex)
        self.base_alpha_tex = self.create_gl_texture(base_alpha_data)
        self.nms_tex = self.create_gl_texture(nms_data)
        self.compose_size = (base_alpha_data.shape[1], base_alpha_data.shape[0])
        self.textures_loaded = True
        self.update()

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

        if self.preview_mode == "input" and not self.external_packed_mode and self.compose_requested:
            self.run_compose_pass()

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

    def tick_preview(self):
        if self.auto_rotate_enabled:
            self.rotation_y += 0.5
            if self.rotation_y >= 360:
                self.rotation_y -= 360
        self.update()
