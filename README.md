# PBRPacker
PBRPacker is a Python mini-app that packs 6 PBR Textures into 2 for Unity. 
This can help reduce draw calls, RAM/VRAM usage, and texture setup time.

You need a specific shadergraph setup for unpacking the packed shaders in Unity that you can find here.

====
Core logic:

It takes 6 textures:
Base Color
Transparency
Roughness
Metallic
Normals
Ambient Occlusion

And outputs 2 textures:
1- unorm8_rgba(BaseColor.rgb * AmbeintOcclusion, Transparency)
2- unorm8_rgba(Normal.xy, Metallic, Smoothness)

Smoothness = 1 - Roughness. URP lit shader expects smoothness, so converting roughness before runtime is a more efficient way. It saves a single ALU instruction.
Normal maps can be simplified to just .xy, then we can reconstruct .z using sqrt(1-dot(n.xy,n.xy)) in the shader. This saves 1 byte per pixel while remaining efficient (5 ALU instructions) during unpacking.

====
Build guide:
1- Clone this repository
2- Open CMD at the local directory of the repo.
3- Run these commands one by one:
  `pip install -r requirements.txt`
  `pyinstaller --onefile --windowed --icon=app_icon.ico PBRPacker.py`
4- PBRPacker.exe will appear in the dist folder in the root path.

====
Credits:
ChatGPT 5.4 was the coder.
Nano Banana Pro was the icon designer.
DeepSeek wrote the readme (except credits).
And I was the prompter.
