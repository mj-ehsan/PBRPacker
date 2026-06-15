@echo off
setlocal enabledelayedexpansion

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed.
    goto :activate_venv
)

:: Python not found - ask before downloading
echo.
echo Python is not installed or not in PATH.
echo This script can download and install Python 3.11.9 automatically.
echo The installer includes pip.
echo.
set /p INSTALL_PYTHON="Download and install Python 3.11.9? [y/N]: "

if /i not "!INSTALL_PYTHON!"=="y" (
    echo.
    echo Please install Python 3.8 or later manually from:
    echo https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Download Python installer
echo.
echo Downloading Python 3.11.9...
set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=%TEMP%\python-3.11.9-amd64.exe

curl -L -o "%PYTHON_INSTALLER%" "%PYTHON_URL%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download Python installer.
    echo Please install manually from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Install Python silently (adds to PATH, installs pip)
echo Installing Python 3.11.9...
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0
if %errorlevel% neq 0 (
    echo ERROR: Python installation failed.
    pause
    exit /b 1
)

:: Clean up installer
del "%PYTHON_INSTALLER%"

:: Refresh environment variables
echo Refreshing environment...
call :refresh_env

:: Verify installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python was installed but not detected in PATH.
    echo Please restart your command prompt or manually add Python to PATH.
    pause
    exit /b 1
)

echo Python installed successfully.

:activate_venv
:: Create venv if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b %errorlevel%
    )
)

:: Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements.
    call venv\Scripts\deactivate.bat
    pause
    exit /b %errorlevel%
)

:: Build
echo.
echo Building with PyInstaller...
pyinstaller ^
  --noconfirm ^
  --clean ^
  PBRPacker.spec

:: Deactivate venv
call venv\Scripts\deactivate.bat

if %errorlevel% neq 0 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b %errorlevel%
)

echo.
echo Build completed successfully.
pause
endlocal
exit /b 0

:: Function to refresh environment variables
:refresh_env
    :: Refresh PATH and other env vars from registry
    for /f "skip=2 tokens=1,2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do (
        if "%%a"=="PATH" set "PATH=%%c;%PATH%"
    )
    for /f "skip=2 tokens=1,2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do (
        if "%%a"=="PATH" set "PATH=%%c;%PATH%"
    )
goto :eof
