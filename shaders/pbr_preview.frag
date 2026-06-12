#version 130
#extension GL_ARB_derivative_control : require

#define saturate(a) (clamp(a, 0.0, 1.0))
float max3component (vec3 v) { return max(max(v.x,v.y),v.z); }

uniform sampler2D base_ao_tex;
uniform sampler2D nms_tex;
uniform vec3 key_light_pos;
uniform vec3 key_light_color;
uniform float key_light_intensity;
uniform vec3 fill_light_pos;
uniform vec3 fill_light_color;
uniform float fill_light_intensity;
uniform vec3 rim_light_pos;
uniform vec3 rim_light_color;
uniform float rim_light_intensity;
uniform vec3 camera_pos;

varying vec3 v_normal;
varying vec3 v_world_pos;
varying vec2 v_uv;

struct Material {
    vec3 normal;
    vec3 base;
    float alpha;
    float metallic;
    float roughness; 
};

vec3 Fresnel_Schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float Fd_Burley(float NoV, float NoL, float VoH, float roughness)
{
    float fd90 = 0.5 + 2.0 * VoH * VoH * roughness;
    float lightScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NoL, 5.0);
    float viewScatter  = 1.0 + (fd90 - 1.0) * pow(1.0 - NoV, 5.0);
    return lightScatter * viewScatter;
}

float D_GGX(float NoH, float alpha)
{
    float a2 = alpha * alpha;
    float d = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}

float G_SmithGGX_Correlated(float NoV, float NoL, float alpha)
{
    float a2 = alpha * alpha;

    float denomV = NoL * sqrt(a2 + (1.0 - a2) * NoV * NoV);
    float denomL = NoV * sqrt(a2 + (1.0 - a2) * NoL * NoL);

    return (2.0 * NoL * NoV) / max(denomV + denomL, 1e-5);
}

vec3 GGX_MultiScatterEnergy(vec3 F0, float roughness)
{
    float energyBias   = mix(0.0, 0.5, roughness);
    float energyFactor = mix(1.0, 0.8, roughness);
    return F0 * energyFactor + energyBias;
}

vec3 apply_lightPBR(vec3 lp, vec3 lc, float li, vec3 V, Material M) {
    vec3 Lv = lp - v_world_pos;
    float Ll = length(Lv);
    float L_atten = 1.0 / max(Ll * Ll, 0.01); //considering a light radius of 0.1 => 0.1^2 = 0.01
    vec3 radiance = L_atten * lc * li;

    vec3 L = Lv / Ll;
    vec3 H = normalize(V + L);
    
    float NoL = saturate(dot(M.normal, L));
    float NoV = saturate(dot(M.normal, V));
    float NoH = saturate(dot(M.normal, H));
	float VoH = saturate(dot(V, H));

    vec3 dF0 = vec3(0.04,0.04,0.04);
    vec3 F0 = mix(dF0, M.base, vec3(M.metallic));
    
    float alpha = M.roughness * M.roughness;
    vec3 F = Fresnel_Schlick(VoH, F0);

    // ---- Diffuse (Burley, energy aware) ----
    float Fd = Fd_Burley(NoV, NoL, VoH, M.roughness);
	vec3 diffuse = (M.base / 3.14159265) * Fd * NoL * (1.0 - M.metallic);

    // ---- GGX Specular (height-correlated) ----
	float D = D_GGX(NoH, alpha);
	float G = G_SmithGGX_Correlated(NoV, NoL, alpha);
	vec3 specSingle = (D * G * F) / max(4.0 * NoV, 1e-4);

    // ---- Multiscatter compensation ----
	vec3 Fms = GGX_MultiScatterEnergy(F0, M.roughness);
	vec3 specMulti = Fms * mix(NoL * vec3(Fd,Fd,Fd), specSingle, M.metallic);

    // ---- Energy balancing ----
	vec3 specular  = mix(specSingle, specMulti, M.metallic);
	vec3 diffuseBalanced = diffuse * (1.0 - max3component(specular));

    return (diffuseBalanced + specular) * radiance;
}

vec3 ApplyTangentNormal(vec3 O_Normal, vec3 T_Normal, vec2 uv, vec3 WorldPos)
{
    // Get derivatives
    vec3 dp1 = dFdxFine(WorldPos);
    vec3 dp2 = dFdyFine(WorldPos);
    vec2 duv1 = dFdxFine(uv);
    vec2 duv2 = dFdyFine(uv);
    
    // Solve for tangent and bitangent
    vec3 N = normalize(O_Normal);
    vec3 T = normalize(dp1 * duv2.y - dp2 * duv1.y);
    vec3 B = normalize(dp2 * duv1.x - dp1 * duv2.x);
    
    // Orthonormalize
    T = normalize(T - N * dot(N, T));
    B = cross(N, T);
    
    // Apply tangent space normal
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * T_Normal);
}

vec3 ACESFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Apply normal map variance to roughness
// nrm: current filtered normal (normalized, from mip N)
// nrm0: base normal from mip 0 (normalized, highest detail)
// rgh: base roughness [0,1]
// returns: modified roughness with variance applied
float ApplyNrmVarToRgh(vec3 nrm, vec3 nrm0, float rgh)
{
    float d = dot(nrm, nrm0);
    
    d = clamp(d, 0.0, 1.0);
    
    float nrmVar = 1.0 - d;
    
    float alpha = rgh * rgh;
    float fs = 1.0 / (1.0 + alpha * alpha * nrmVar);
    
    float rghEff = rgh * (1.0 + nrmVar);
    
    return clamp(rghEff, 0.0, 1.0);
}

void getMaterial (out Material m)
{
    vec4 base_alpha = texture2D(base_ao_tex, v_uv);
    vec4 nms = texture2D(nms_tex, v_uv);
    

    vec3 n = vec3(nms.r * 2.0 - 1.0, nms.g * 2.0 - 1.0, 0.0);
    n.z = sqrt(1.0 - nms.r*nms.r - nms.g*nms.g);

    vec4 nms0 = textureLod(nms_tex, v_uv, 0);
    vec3 n0 = vec3(nms0.r * 2.0 - 1.0, nms0.g * 2.0 - 1.0, 0.0);
    n0.z = sqrt(1.0 - nms0.r*nms0.r - nms0.g*nms0.g);
    nms.w = ApplyNrmVarToRgh(n, n0, nms.w);

    m.base = base_alpha.rgb;
    m.alpha = base_alpha.a;
    m.normal = n.xyz;
    m.metallic = nms.z;
    m.roughness = nms.w;

    m.normal = ApplyTangentNormal(v_normal, m.normal, v_uv, v_world_pos);
}

void main() {
    Material material;
    getMaterial(material);

    vec3 view_dir = normalize(camera_pos - v_world_pos);

    vec3 color = vec3(0.0, 0.0, 0.0);
    color += apply_lightPBR( key_light_pos,  key_light_color,  key_light_intensity, view_dir, material);
    color += apply_lightPBR(fill_light_pos, fill_light_color, fill_light_intensity, view_dir, material);
    color += apply_lightPBR( rim_light_pos,  rim_light_color,  rim_light_intensity, view_dir, material);

    float exposure = 3.0;
    color *= pow(2.0, exposure);
    color = ACESFilm(color);
    
    gl_FragColor = vec4(saturate(color), 1.0);
}
