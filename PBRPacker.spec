# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['PBRPacker.py'],
    pathex=[],
    binaries=[],
    datas=[('shaders', 'shaders'), ('hdri', 'hdri')],
    hiddenimports=['pack_worker', 'pbr_gui', 'pbr_renderer', 'imagecodecs', 'imageio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PBRPacker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.ico'],
)
