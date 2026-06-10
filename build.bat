@echo off
echo Checking and installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Building PBR Texture Packer...
pyinstaller --noconfirm --onefile --windowed --icon=app_icon.ico PBRPacker.py

echo.
echo Build complete! Your executable is located in the "dist" folder.
pause
