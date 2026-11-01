@echo off
setlocal
rem Explicit first-install entry point: install both services, then open ComfyUI.
if defined VISION_WORKBENCH_PYTHON (
    "%VISION_WORKBENCH_PYTHON%" "%~dp0setup.py" --install
    goto :installed
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0setup.py" --install
    goto :installed
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0setup.py" --install
    goto :installed
)
echo Install Python 3.11 and Node.js LTS, then run INSTALL_AND_START.cmd again.
goto :failed
:installed
if errorlevel 1 goto :failed
call "%~dp0START.cmd"
exit /b %errorlevel%
:failed
echo Installation stopped. Read the error above; existing files are preserved.
pause
exit /b 1
