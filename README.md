PBR Packer

A GPU-accelerated texture packing tool for modern PBR workflows.

PBR Packer combines multiple PBR texture maps into optimized texture sets while providing a real-time physically based preview of the final material. The entire processing pipeline runs on the GPU, allowing fast iteration and responsive editing even with high-resolution textures.

---

Features

Texture Packing

Pack up to six source textures into two optimized output textures:

Source Map| Destination
Base Color| Albedo Texture
Ambient Occlusion| Albedo Alpha
Roughness| Packed Texture
Metallic| Packed Texture
Height| Packed Texture
Normal Map| Packed Texture

---

GPU Accelerated Processing

All texture processing is performed on the GPU:

- Channel packing
- AO application
- Roughness conversion
- Normal map reconstruction
- Texture compositing
- Preview rendering

This allows near-instant updates while adjusting settings.

---

Real-Time PBR Preview

Preview your material before exporting with:

- Physically Based Rendering (PBR)
- HDR environment lighting
- Multiple dynamic lights
- Metallic/Roughness workflow
- Real-time material updates
- Adjustable environment rotation

No need to switch back and forth between your engine and texture editor.

---

Built-In Utilities

Ambient Occlusion Integration

Optionally multiply AO into Base Color before export:

BaseColor.rgb *= AO

Useful for workflows that prefer baked ambient occlusion.

Roughness / Smoothness Conversion

Automatically convert between:

- Roughness workflows
- Smoothness workflows

Ideal for Unity and custom engine pipelines.

Normal Map Generation

Generate normal maps from height maps directly inside the application.

Channel Packing

Create optimized packed textures for reduced memory usage and fewer texture fetches.

---

Why Use PBR Packer?

Traditional workflow:

1. Open several textures in Photoshop/GIMP
2. Pack channels manually
3. Export textures
4. Import into engine
5. Discover issues
6. Repeat

With PBR Packer:

1. Drag textures into the application
2. Adjust settings
3. Preview material instantly
4. Export

Done.

---

Performance

The application is designed around GPU processing rather than CPU image manipulation.

Benefits include:

- Fast preview updates
- Responsive UI
- Efficient handling of large textures
- Background exporting
- Minimal workflow interruptions

---

Supported Maps

Inputs

- Base Color
- Ambient Occlusion
- Roughness
- Metallic
- Height
- Normal

Outputs

- Albedo Texture
- Packed Material Texture

---

Target Use Cases

- Unity projects (rn only a specific VR workflow
TODO:
- Unreal Engine projects
- Custom engines
- Real-time rendering pipelines
- Game asset optimization
- Material authoring workflows

---

Screenshots

TODO: Add screenshots here:

- Main UI
- Material Preview
- Packed Texture Output
- Export Results

---

Installation

Download the latest release and run:

PBRPacker.exe

No additional setup required.
Or build the latest push in dev branch using by cloning and running build.bat

---

License

See LICENSE for details.

---

Acknowledgements

Built for artists, technical artists, and rendering programmers who are tired of manually packing textures.