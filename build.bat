@echo off
setlocal

pyinstaller ^
  --noconfirm ^
  --clean ^
  PBRPacker.spec

endlocal
