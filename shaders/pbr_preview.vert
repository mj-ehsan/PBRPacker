#version 330

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat3 u_normal_matrix;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec3 a_tangent; 

out vec3 v_normal;
out vec3 v_world_pos;
out vec2 v_uv;
out vec3 v_tangent;
out vec3 v_bitangent;

void main() {
    // Calculate world position (model matrix only, not view)
    vec4 world_pos = u_model * vec4(a_position, 1.0);
    v_world_pos = world_pos.xyz;
    
    // Transform normal and tangent to world space (rotation only)
    v_normal = normalize(u_normal_matrix * a_normal);
    v_tangent = normalize(u_normal_matrix * a_tangent);
    v_bitangent = normalize(cross(v_tangent, v_normal));
    
    // Pass UV coordinates
    v_uv = a_uv;
    
    // Calculate final clip position
    gl_Position = u_mvp * vec4(a_position, 1.0);
}