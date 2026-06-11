#version 130
varying vec3 v_normal;
varying vec3 v_world_pos;
varying vec2 v_uv;

void main() {
    vec4 world_pos = gl_ModelViewMatrix * gl_Vertex;   // model matrix * vertex
    v_world_pos = world_pos.xyz;

    // Transform normal to world space (rotation only)
    v_normal = normalize(mat3(gl_ModelViewMatrix) * gl_Normal);
    // Alternatively: v_normal = normalize(gl_NormalMatrix * gl_Normal);
    // Both are correct because view matrix is identity.

    v_uv = gl_MultiTexCoord0.xy;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}