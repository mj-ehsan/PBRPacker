#version 450
#extension GL_ARB_derivative_control : require

#define MAX_LIGHTS 16

#define PI 3.1415926535
#define saturate(a) (clamp(a, 0.0, 1.0))
#define rsqrt(a) inversesqrt(a)
float max3component (vec3 v) { return max(max(v.x,v.y),v.z); }

uniform sampler2D base_alpha_tex;
uniform sampler2D nms_tex;
uniform sampler2D u_environment_map;
uniform bool u_use_environment_map;

const float u_env_mip_count = 0.0; 

uniform vec3 camera_pos;

in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
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

struct LightIndependentLightingData {
    vec3  viewDir;
    float noV;
    float quarterOverNoV;
    float viewAngleFactor;
    float schlickWeight;
    vec3  f0;
    float oneMinusMetallic;
    vec3 ggxMultiScatterEnergy;
    float burleyViewFactor;
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

float Fd_Burley(float vf, float NoL, float VoH, float roughness)
{
    float fd90 = 2.0 * VoH * VoH * roughness - 0.5;
    float lightScatter = 1.0 + fd90 * pow(1.0 - NoL, 5.0);
    float viewScatter  = 1.0 + fd90 * vf;
    return lightScatter * viewScatter;
}

float D_GGX(float NoH, float alpha2)
{
    float d = (NoH * NoH) * (alpha2 - 1.0) + 1.0;
    return alpha2 / (3.14159265 * d * d);
}

float G_SmithGGX_Correlated(LightIndependentLightingData data, float NoL, float alpha2)
{
    float denomV = NoL * data.viewAngleFactor;
    float denomL = data.noV * sqrt(alpha2 + (1.0 - alpha2) * NoL * NoL);

    return (2.0 * NoL * data.noV) / max(denomV + denomL, 1e-5);
}

vec3 GGX_MultiScatterEnergy(vec3 F0, float roughness)
{
    float energyBias   = mix(0.0, 0.5, roughness);
    float energyFactor = mix(1.0, 0.8, roughness);
    return F0 * energyFactor + energyBias;
}

void computeLightIndependentLightingData(in Material M, in vec3 V, out LightIndependentLightingData data) {
    data.viewDir = V;
    data.noV = saturate(dot(M.normal, data.viewDir));
    data.quarterOverNoV = 0.25 / max(data.noV, 1e-5);
    data.viewAngleFactor = sqrt(M.roughness.z + (1.0 - M.roughness.z) * data.noV * data.noV);

    float OneMinusNoV = 1.0 - data.noV;
    float n2 = OneMinusNoV * OneMinusNoV;
    data.schlickWeight = n2 * n2 * OneMinusNoV;

    data.f0 = mix(vec3(0.04), M.base, vec3(M.metallic));
    data.oneMinusMetallic = 1.0 - M.metallic;
    data.ggxMultiScatterEnergy = GGX_MultiScatterEnergy(data.f0, M.roughness.x);
}

vec3 apply_lightPBR(Light light, Material M, LightIndependentLightingData data) {

    vec3 Lv = light.pos - v_world_pos;

    float Ll2 = dot(Lv,Lv);
    float InvL = rsqrt(Ll2);
    vec3 L = Lv * InvL;

    float NoL = dot(M.normal, L);
    if(NoL < 0.0f) return vec3(0.0);

    float L_atten = 1.0 / max(Ll2, 0.01); //considering a light radius of 0.1 => 0.1^2 = 0.01
    vec3 radiance = L_atten * light.color * light.intensity;

    vec3 H = normalize(data.viewDir + L);
    
    float NoH = saturate(dot(M.normal, H));
	float VoH = saturate(dot(data.viewDir, H));
    
    vec3 F = Fresnel_Physical(VoH, data.f0, M.metallic);

    // ---- Diffuse (Burley, energy aware) ----
    float Fd = Fd_Burley(data.schlickWeight, NoL, VoH, M.roughness.x);
	vec3 diffuse = (M.base / 3.14159265) * Fd * NoL * data.oneMinusMetallic;

    // ---- GGX Specular (height-correlated) ----
	float D = D_GGX(NoH, M.roughness.z);
	float G = G_SmithGGX_Correlated(data, NoL, M.roughness.z);
	vec3 specSingle = D * G * F * data.quarterOverNoV;

    // ---- Multiscatter compensation ----
	vec3 Fms = data.ggxMultiScatterEnergy;
	vec3 specMulti = Fms * mix(NoL * vec3(Fd), specSingle, M.metallic);

    // ---- Energy balancing ----
	vec3 specular  = mix(specSingle, specMulti, M.metallic);
	vec3 diffuseBalanced = diffuse * (1.0 - max3component(specular));

    return (diffuseBalanced + specular) * radiance;
    //return specular * radiance;
}

vec3 ApplyTangentNormal(vec3 O_Normal, vec3 T_Normal)
{
    vec3 T = normalize(v_tangent);
    vec3 B = normalize(v_bitangent);
    vec3 N = normalize(O_Normal);

    T = normalize(T - N * dot(N, T));
    B = normalize(B - N * dot(N, B) - T * dot(T, B));

    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * T_Normal);
}

vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 ACESFitted(vec3 color)
{
    const mat3 ACESInputMat = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );

    const mat3 ACESOutputMat = mat3(
        1.60475, -0.10208, -0.00327,
       -0.53108,  1.10813, -0.07276,
       -0.07367, -0.00605,  1.07602
    );

    color = ACESInputMat * color;
    color = RRTAndODTFit(color);
    color = ACESOutputMat * color;

    return clamp(color, 0.0, 1.0);
}


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
    n = ApplyTangentNormal(v_normal, n);

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
    m.roughness.y = m.roughness.x * m.roughness.x;
    m.roughness.z = m.roughness.y * m.roughness.y;
    
}

float bayerDither(ivec2 pos) {
    int x = pos.x % 4;
    int y = pos.y % 4;
    int index = x + y * 4;
    int bayer[16] = int[16](
         0,  8,  2, 10,
        12,  4, 14,  6,
            3, 11,  1,  9,
            15, 7, 13, 5
    );
    return float(bayer[index]) / 16.0;
}

void applyDither(inout vec3 color, ivec2 pixelPos) {
    float ditherValue = bayerDither(pixelPos);
    color += ditherValue / 255.0; // Scale to [0,1] range
}

const bool debug = false;

void main() {
    vec3 view_dir = normalize(camera_pos - v_world_pos);

    Material material;
    LightIndependentLightingData lightingData;

    getMaterial(material);
    computeLightIndependentLightingData(material, view_dir, lightingData);

    vec3 color = vec3(0.0, 0.0, 0.0);
    for(int i = (debug ? 2 : 0); i < (debug ? 3 : num_lights); i++) {
        color += apply_lightPBR(lights[i], material, lightingData);
    }
    //apply ambient light
    color += material.base * 0.02;

    if(!debug) {
        float exposure = 4.0;
        color *= pow(2.0, exposure);
        color = ACESFitted(color);
        applyDither(color, ivec2(gl_FragCoord.xy));
    }

    FragColor = vec4(saturate(color), material.alpha);
}
