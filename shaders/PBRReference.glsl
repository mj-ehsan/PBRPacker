//#undef GL_FRAGMENT_PRECISION_HIGH
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

uniform vec2 resolution;
uniform float time;
uniform vec3 pointers[10];

const float ui_scale = 10.;
const float ui_roughness = 0.5;
const float ui_F0 = 0.04;
const float ui_metallic = 0.;
const float ui_touchPressure = 1.0;
const float ui_speed = 1.0;

#define float4 vec4
#define float3 vec3
#define float2 vec2
#define frac fract
#define lerp mix
#define saturate(a) clamp(a,0.,1.)

# define nep 2.81828

float hash3(float3 p)
{
    p = frac(p * float3(0.1031, 0.11369, 0.13787));
    p += dot(p, p.yzx + 33.33);
    return frac(p.x * p.y * p.z);
}

vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

float noise(in float3 pos)
{
	float result = 0.0;
    float3 p0 = floor(pos);
    float3 f  = pos - p0;

    float3 f2 = f * f;
    float3 f3 = f2 * f;

    float3 w[4];
    w[0] = -0.5*f3 + f2 - 0.5*f;
    w[1] =  1.5*f3 - 2.5*f2 + 1.0;
    w[2] = -1.5*f3 + 2.0*f2 + 0.5*f;
    w[3] =  0.5*f3 - 0.5*f2;

    result = 0.0;

    for(int x = -1; x <= 2; x++)
    for(int y = -1; y <= 2; y++)
    for(int z = -1; z <= 2; z++)
    {
        float W =
            w[x+1].x *
            w[y+1].y *
            w[z+1].z;

        float h = hash3(p0 + float3(x,y,z));
        result += h * W;
    }
    return result;
}

float fractalNoise(vec3 pos, int octaves)
{
	float lacunarity = 2.0;
	float invLacunarity = 1.0 / lacunarity;

	float result = 0.0;
	float wSum = 0.0;

	vec3 octPos = pos;
	float octW = 1.0;

	for(int oct = 0; oct < octaves; oct++)
	{
		result += noise(octPos) * octW;
		wSum += octW;

		octPos *= lacunarity;
		octW *= invLacunarity;
	}

	result /= wSum;
	return result;
}

float voronoi_soft(vec3 pos, float k, out vec3 cellId, out vec3 g, out vec2 uv)
{
	vec3 f = floor(pos);
	vec3 minv = vec3(0.0,0.0,0.0);
	float minll = 0.0;
	cellId = g = vec3(0.,0.,0.);
	float wsum = 0.0;
	int r = 3;

	for(int x = -r; x < r; x++){
	for(int y = -r; y < r; y++){
	for(int z = -r; z < r; z++)
	{
		vec3 offs = vec3(x,y,z);
		if(dot(offs,offs) > 3.0)continue;
		vec3 h = hash33(f + offs);
		vec3 point = f + offs + h;
		vec3 v = pos - point;
		float l = sqrt(dot(v,v));
		float w = saturate(exp(-l * k));

		wsum += w;
		g = v * w + g;
		cellId = h * w + cellId;
		minv = v * w + minv;
		minll = l * w + minll;
	}}}
	g /= wsum;
	cellId /= wsum;
	minv /= wsum;
	minll /= wsum;

	g.z = sqrt(1. - minll);
	g = normalize(g);
	uv = minv.xy;

	return sqrt(minll);
}

float D_GGX(float NoH, float alpha)
{
    float a2 = alpha * alpha;
    float d = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}

float G_SchlickGGX(float NoV, float k)
{
    return NoV / (NoV * (1.0 - k) + k);
}

float G_Smith(float NoV, float NoL, float roughness)
{
    float k = (roughness + 1.0);
    k = (k * k) / 8.0;
    return G_SchlickGGX(NoV, k) * G_SchlickGGX(NoL, k);
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

vec3 ACESFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main(void) {
	vec2 uv = gl_FragCoord.xy / resolution.xx;
	vec3 viewDir = vec3(0.,0.,1.);
	float n = fractalNoise(float3(uv.xy*10.0, time), 2);
	vec3 point = pointers[0] / resolution.xxx;
	float2 compression = point.xy - uv.xy;
	float LC = length(compression);
	float CW = abs(-abs(LC*20.) + 2.0);
	CW = exp(-CW * CW * ui_touchPressure);
	compression = compression * CW / LC;

	vec3 col = vec3(0.,0.,0.);
	vec3 normals = vec3(0.,0.,0.);
	vec2 vuv;
	float vor = voronoi_soft(float3(uv.xy * ui_scale - compression, time * ui_speed), 10., col, normals, vuv);

	float roughness = ui_roughness;
	vec3 F0 = lerp(vec3(ui_F0), col, ui_metallic);

	vec3 dirL_dir = normalize(vec3(0.3,0.5,0.4));
	vec3 dirL_col = vec3(1.0, 0.918, 0.651) * 2.0;
	vec3 amb = vec3(0.,0.,0.);

	vec3 N = normalize(normals);
	vec3 V = normalize(viewDir);

	// Force correct hemisphere
	vec3 L = normalize(dirL_dir);
	float NoL = dot(N, L);

	vec3 H = normalize(V + L);

	float NoV = saturate(dot(N, V));
	NoL = saturate(NoL);
	float NoH = saturate(dot(N, H));
	float VoH = saturate(dot(V, H));

	// artistic control
	float alpha = roughness * roughness;

	// Fresnel
	vec3 F = Fresnel_Schlick(VoH, F0);

	// ---- Diffuse (Burley, energy aware) ----
	float Fd = Fd_Burley(NoV, NoL, VoH, roughness);
	vec3 diffuse = (col / 3.14159265) * Fd * NoL * (1.-ui_metallic);

	// ---- GGX Specular (height-correlated) ----
	float D = D_GGX(NoH, alpha);
	float G = G_SmithGGX_Correlated(NoV, NoL, alpha);
	vec3 specSingle = (D * G * F) / max(4.0 * NoV * NoL, 1e-4);

	// ---- Multiscatter compensation ----
	vec3 Fms = GGX_MultiScatterEnergy(F0, roughness);
	vec3 specMulti = Fms * lerp(NoL * vec3(Fd,Fd,Fd), specSingle, ui_metallic);

	// ---- Energy balancing ----
	vec3 specular = specSingle + specMulti;
	vec3 diffuseBalanced = diffuse * (1.0 - max(max(specular.r, specular.g), specular.b));

	// ---- Final lighting ----
	vec3 lit = (diffuseBalanced + specular) * dirL_col + amb * col;

	// Tonemapping
	lit *= 0.5;
	lit = ACESFilm(lit);

	gl_FragColor = float4(lit,0.);
}
