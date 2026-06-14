#version 330

#define saturate(a) (clamp(a, 0.0, 1.0))
// Input textures (from texture units 0–5)
uniform sampler2D BaseColor;
uniform sampler2D AO;
uniform sampler2D Metallic;
uniform sampler2D Smoothness;
uniform sampler2D Normal;
uniform sampler2D Alpha;

// Composition parameters
uniform float u_ao_intensity;
uniform float u_normal_gen_sigma;
uniform float u_normal_gen_height;
uniform bool  u_invert_normal_y;
uniform bool  u_use_alpha;        // true if Alpha texture is present
uniform bool  u_generate_normal_from_luma; // true if Normal texture is not present and should be generated from BaseColor luma

// MRT outputs sampled later as `base_alpha_tex` and `nms_tex`
layout(location = 0) out vec4 base_alpha_tex;
layout(location = 1) out vec4 nms_tex;

in vec2 texCoord;

float getLuma(vec3 color) {
    //rec.709 luma
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 NormalFromLuma(vec2 uv) {
    vec2 pix = 1.0 / textureSize(BaseColor, 0);
    float sigma = max(u_normal_gen_sigma, 0.0001);
    float sigma_invSqr2 = 0.5 / (sigma * sigma);
    vec2 grad = vec2(0.0);

    for(int xx = -3; xx < 3; xx++) {
        for(int yy = -3; yy < 3; yy++) {

            vec2 offs = vec2(xx, yy);
            float sLenSqr = dot(offs, offs);
            if(sLenSqr < 0.0001 || sLenSqr > 9.0) continue;

            float weight = exp(-sLenSqr * sigma_invSqr2);
            float luma = getLuma(texture(BaseColor, offs * pix + uv).rgb);
            grad += offs * inversesqrt(sLenSqr) * luma * weight;
        }
    }

    vec3 n;
    n.xy = clamp(grad * u_normal_gen_height, -1.0, 1.0);
    n.z = sqrt(max(1.0 - dot(n.xy, n.xy), 0.0));

    return normalize(n) * 0.5 + 0.5;
}

void main() {
    vec4 baseColor = texture(BaseColor, texCoord);
    float ao = texture(AO, texCoord).r;
    float aoFactor = pow(max(ao, 0.0), max(u_ao_intensity, 0.00001));
    baseColor.rgb *= aoFactor;

    float metallic = texture(Metallic, texCoord).r;
    float smoothness = texture(Smoothness, texCoord).r;

    float alpha = baseColor.a;
    if(u_use_alpha) {
        alpha = texture(Alpha, texCoord).r;
    }

    vec3 normal;
    if(u_generate_normal_from_luma) {
        normal = NormalFromLuma(texCoord);
    } else {
        normal = texture(Normal, texCoord).rgb;
        normal.y = u_invert_normal_y ? 1.0 - normal.y : normal.y;
    }

    base_alpha_tex = vec4(baseColor.rgb, alpha);
    nms_tex = vec4(normal.xy, metallic, smoothness);
}
