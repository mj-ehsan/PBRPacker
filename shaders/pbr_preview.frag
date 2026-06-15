#version 330
#extension GL_ARB_derivative_control : require

#define MAX_LIGHTS 16

#define PI 3.1415926535
#define saturate(a) (clamp(a, 0.0, 1.0))
float max3component (vec3 v) { return max(max(v.x,v.y),v.z); }

uniform sampler2D base_alpha_tex;
uniform sampler2D nms_tex;
uniform sampler2D u_environment_map;
uniform bool u_use_environment_map;

const float u_env_mip_count = 0.0; 

uniform vec3 camera_pos;

in vec3 v_normal;
in vec3 v_tangent;
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

vec3 ApplyTangentNormal(vec3 O_Normal, vec3 T_Normal)
{
    vec3 T = normalize(v_tangent);
    vec3 N = normalize(O_Normal);
    T = normalize(T - N * dot(N, T));
    vec3 B = cross(N, T);

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
    m.roughness.y = nms.w * nms.w;
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

vec2 dirToEquirectUV(vec3 dir) {
    dir = normalize(dir);
    float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    return vec2(u, v);
}

vec3 sampleEnv(vec3 dir) {
    return texture(u_environment_map, dirToEquirectUV(dir)).rgb;
}

vec3 sampleEnvLOD(vec3 dir, float lod) {
    return textureLod(u_environment_map, dirToEquirectUV(dir), lod).rgb;
}

////////////////////////////////////////////////////////////////////////////////
// Low discrepancy sampling for diffuse integral
////////////////////////////////////////////////////////////////////////////////

float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 Hammersley(uint i, uint N) {
    return vec2(float(i) / float(N), RadicalInverse_VdC(i));
}

vec3 tangentToWorld(vec3 v, vec3 N) {
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);
    return T * v.x + B * v.y + N * v.z;
}

////////////////////////////////////////////////////////////////////////////////
// Diffuse irradiance integral from original env
//
// We integrate:
// Lo_diff = integral[ Li * (base/pi) * Fd_Burley * (1-F) * NoL dω ]
//
// This is more view-dependent than classic Lambert irradiance, which is desirable
// for a high quality preview.
////////////////////////////////////////////////////////////////////////////////

vec3 integrateDiffuseIBL(vec3 N, vec3 V, vec3 baseColor, vec3 F0, float metallic, float roughness) {
    const uint SAMPLE_COUNT = 512u;

    float NoV = saturate(dot(N, V));
    vec3 sum = vec3(0.0);

    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);

        // cosine-weighted hemisphere sample
        float phi = 2.0 * PI * Xi.x;
        float cosTheta = sqrt(1.0 - Xi.y);
        float sinTheta = sqrt(Xi.y);

        vec3 Llocal = vec3(cos(phi) * sinTheta,
                           sin(phi) * sinTheta,
                           cosTheta);

        vec3 L = normalize(tangentToWorld(Llocal, N));
        vec3 H = normalize(V + L);

        float NoL = saturate(dot(N, L));
        float VoH = saturate(dot(V, H));

        if (NoL > 0.0) {
            vec3 Li = sampleEnv(L);

            vec3 F = Fresnel_Physical(VoH, F0, metallic);

            // diffuse energy conservation:
            // diffuse should diminish as specular Fresnel rises
            vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);

            float fd = Fd_Burley(NoV, NoL, VoH, roughness);

            // Because cosine-weighted sampling has pdf = NoL / PI,
            // estimator simplifies to Li * kd * baseColor * fd
            sum += Li * kd * baseColor * fd;
        }
    }

    return sum / float(SAMPLE_COUNT);
}

////////////////////////////////////////////////////////////////////////////////
// Specular IBL using prefiltered GGX mip chain
//
// Since the env is already prefiltered with GGX, we sample reflection direction
// with roughness-derived LOD and modulate with a BRDF response term.
//
// For best physical plausibility without a BRDF LUT, we evaluate a local BRDF
// response using NoV and exact/physical Fresnel, then apply multi-scatter
// compensation.
////////////////////////////////////////////////////////////////////////////////

vec3 integrateSpecularIBL(vec3 N, vec3 V, vec3 F0, float metallic, float roughness, float alpha2) {
    float NoV = saturate(dot(N, V));
    if (NoV <= 0.0) return vec3(0.0);

    vec3 R = reflect(-V, N);

    // For GGX-prefiltered env maps, roughness -> mip level
    float lod = roughness * max(u_env_mip_count - 1.0, 0.0);
    vec3 prefiltered = sampleEnvLOD(R, lod);

    // Fresnel at view angle
    vec3 F = Fresnel_Physical(NoV, F0, metallic);

    // Approximate integrated visibility for IBL.
    // This is the weak point without a split-sum BRDF LUT.
    // We use a high-quality approximation based on correlated Smith response.
    //
    // Sample a representative light direction around reflection to estimate G.
    vec3 L = R;
    vec3 H = normalize(V + L);

    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));

    float D = D_GGX(NoH, alpha2);
    float G = G_SmithGGX_Correlated(NoV, NoL, alpha2);

    // Local BRDF normalization heuristic for prefiltered env usage.
    vec3 singleScatter = F * G;

    // Multi-scatter compensation for rough surfaces
    vec3 Fms = GGX_MultiScatterEnergy(F0, roughness);
    vec3 multiScatter = (Fms - F) * 0.04 * (1.0 + roughness + D * alpha2);

    return prefiltered * (singleScatter + multiScatter);
}

////////////////////////////////////////////////////////////////////////////////
// Final environment evaluation
////////////////////////////////////////////////////////////////////////////////

vec3 sampleEnvironmentMap(vec3 view_dir, Material M) {
    if (!u_use_environment_map) return vec3(0.0);

    vec3 N = normalize(M.normal);
    vec3 V = normalize(view_dir); // surface -> camera

    float metallic  = clamp(M.metallic, 0.0, 1.0);
    float roughness = clamp(M.roughness.x, 0.001, 1.0);

    // Prefer provided alpha if it is your actual microfacet alpha,
    // otherwise derive from roughness.
    float alpha = M.alpha > 0.0 ? M.alpha : roughness * roughness;
    float alpha2 = alpha * alpha;

    vec3 baseColor = clamp(M.base, 0.0, 1.0);

    // Standard metallic workflow:
    // - dielectrics have low F0
    // - metals tint F0 by baseColor
    vec3 dielectricF0 = vec3(0.04);
    vec3 F0 = mix(dielectricF0, baseColor, metallic);

    vec3 diffuse  = integrateDiffuseIBL(N, V, baseColor, F0, metallic, roughness);
    vec3 specular = integrateSpecularIBL(N, V, F0, metallic, roughness, alpha2);

    return diffuse + specular;
}

void main() {
    Material material;
    getMaterial(material);
    

    vec3 view_dir = normalize(camera_pos - v_world_pos);

    vec3 color = vec3(0.0, 0.0, 0.0);
    for(int i = 0; i < num_lights; i++) {
        color += apply_lightPBR(lights[i], view_dir, material);
    }
    //color += sampleEnvironmentMap(view_dir, material) * .001;

    float exposure = 3.0;
    color *= pow(2.0, exposure);
    color = ACESFilm(color);
    applyDither(color, ivec2(gl_FragCoord.xy));
    FragColor = vec4(saturate(color), material.alpha);
}
