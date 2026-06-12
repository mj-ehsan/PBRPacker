#version 330
#extension GL_ARB_derivative_control : require

#define MAX_LIGHTS 16

#define saturate(a) (clamp(a, 0.0, 1.0))
float max3component (vec3 v) { return max(max(v.x,v.y),v.z); }

uniform sampler2D base_alpha_tex;
uniform sampler2D nms_tex;

uniform vec3 camera_pos;

in vec3 v_normal;
in vec3 v_world_pos;
in vec2 v_uv;

out vec4 FragColor;

// Lights as struct array
struct Light {
    vec3 pos;
    vec3 color;
    float intensity;
};

uniform Light lights[MAX_LIGHTS];
uniform int num_lights;

struct Material {
    vec3 normal;
    vec3 base;
    float alpha;
    float metallic;
    vec3 roughness; //with pow2 and pow4
};

vec3 Fresnel_Schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Add to your shader:
float IOR_from_F0(float F0) {
    float sqrtF0 = sqrt(max(F0, 0.0));
    return (1.0 + sqrtF0) / max(1.0 - sqrtF0, 0.001);
}

vec3 Fresnel_Physical(float cosTheta, vec3 F0, float metallic) {
    cosTheta = abs(cosTheta);  // Fix back-facing issue
    
    if (metallic < 0.001) {
        // Pure dielectric: use exact formula
        vec3 F = vec3(0.0);
        for (int i = 0; i < 3; i++) {
            float ior = IOR_from_F0(F0[i]);
            float c = cosTheta;
            float g = sqrt(ior * ior + c * c - 1.0);
            float gc = g - c;
            float gp = g + c;
            F[i] = 0.5 * gc * gc / (gp * gp) * 
                   (1.0 + ((c * gp - 1.0) * (c * gp - 1.0)) / 
                          ((c * gc + 1.0) * (c * gc + 1.0)));
        }
        return F;
    } else {
        // Metallic: use enhanced Schlick with grazing term
        vec3 F90 = mix(vec3(1.0), F0, 0.5);  // Metals are still reflective at grazing
        float c = 1.0 - cosTheta;
        float c5 = c * c * c * c * c;
        return F0 + (F90 - F0) * c5 * (1.0 + (sqrt(max(F0, 0.0)) - 0.5) * c * (1.0 - c));
    }
}

float Fd_Burley(float NoV, float NoL, float VoH, float roughness)
{
    float fd90 = 0.5 + 2.0 * VoH * VoH * roughness;
    float lightScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NoL, 5.0);
    float viewScatter  = 1.0 + (fd90 - 1.0) * pow(1.0 - NoV, 5.0);
    return lightScatter * viewScatter;
}

float D_GGX(float NoH, float alpha2)
{
    float d = (NoH * NoH) * (alpha2 - 1.0) + 1.0;
    return alpha2 / (3.14159265 * d * d);
}

float G_SmithGGX_Correlated(float NoV, float NoL, float alpha2)
{
    float denomV = NoL * sqrt(alpha2 + (1.0 - alpha2) * NoV * NoV);
    float denomL = NoV * sqrt(alpha2 + (1.0 - alpha2) * NoL * NoL);

    return (2.0 * NoL * NoV) / max(denomV + denomL, 1e-5);
}

vec3 GGX_MultiScatterEnergy(vec3 F0, float roughness)
{
    float energyBias   = mix(0.0, 0.5, roughness);
    float energyFactor = mix(1.0, 0.8, roughness);
    return F0 * energyFactor + energyBias;
}

vec3 apply_lightPBR(Light light, vec3 V, Material M) {
    vec3 Lv = light.pos - v_world_pos;
    float Ll = length(Lv);
    float L_atten = 1.0 / max(Ll * Ll, 0.01); //considering a light radius of 0.1 => 0.1^2 = 0.01
    vec3 radiance = L_atten * light.color * light.intensity;

    vec3 L = Lv / Ll;
    vec3 H = normalize(V + L);
    
    float NoL = saturate(dot(M.normal, L));
    float NoV = saturate(dot(M.normal, V));
    float NoH = saturate(dot(M.normal, H));
	float VoH = saturate(dot(V, H));

    vec3 dF0 = vec3(0.04,0.04,0.04);
    vec3 F0 = mix(dF0, M.base, vec3(M.metallic));
    
    vec3 F = Fresnel_Physical(VoH, F0, M.metallic);

    // ---- Diffuse (Burley, energy aware) ----
    float Fd = Fd_Burley(NoV, NoL, VoH, M.roughness.x);
	vec3 diffuse = (M.base / 3.14159265) * Fd * NoL * (1.0 - M.metallic);

    // ---- GGX Specular (height-correlated) ----
	float D = D_GGX(NoH, M.roughness.z);
	float G = G_SmithGGX_Correlated(NoV, NoL, M.roughness.z);
	vec3 specSingle = (D * G * F) / max(4.0 * NoV, 1e-4);

    // ---- Multiscatter compensation ----
	vec3 Fms = GGX_MultiScatterEnergy(F0, M.roughness.x);
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

vec3 unpackNormal(in vec2 normal) {
    vec3 o;
    o.xy = normal * 2.0 - 1.0;
    o.z = sqrt(1.0 - normal.x*normal.x - normal.y*normal.y);
    return o;
}

void getMaterial (out Material m)
{
    vec4 base_alpha = texture(base_alpha_tex, v_uv);

    vec4 nms = texture(nms_tex, v_uv);
    
    vec3 n = unpackNormal(nms.xy);
    n = ApplyTangentNormal(v_normal, n, v_uv, v_world_pos);

    //after derivative calculation so it doesn't mess with the tangent space basis.
    //transparenct is mainly handled by AtoC, so this one's just a perf saver.
    if(base_alpha.a < 1.0 / 255.0) discard;

    vec4 nms0 = textureLod(nms_tex, v_uv, 0);
    vec3 n0 = unpackNormal(nms0.xy);

    nms.w = ApplyNrmVarToRgh(n, n0, nms.w);

    m.base = base_alpha.rgb;
    m.alpha = base_alpha.a;
    m.normal = n.xyz;
    m.metallic = nms.z;
    m.roughness.x = nms.w;
    m.roughness.y = nms.w * nms.w;
    m.roughness.z = m.roughness.y * m.roughness.y;
}

void main() {
    Material material;
    getMaterial(material);
    

    vec3 view_dir = normalize(camera_pos - v_world_pos);

    vec3 color = vec3(0.0, 0.0, 0.0);
    for(int i = 0; i < num_lights; i++) {
        color += apply_lightPBR(lights[i], view_dir, material);
    }

    float exposure = 3.0;
    color *= pow(2.0, exposure);
    color = ACESFilm(color);
    
    FragColor = vec4(saturate(color), material.alpha);
}
