@echo off
setlocal
if defined VISION_WORKBENCH_PYTHON (
    "%VISION_WORKBENCH_PYTHON%" "%~dp0launch.py" %*
    goto :done
)
if defined VISION_WORKBENCH_VENV (
    "%VISION_WORKBENCH_VENV%\Scripts\python.exe" "%~dp0launch.py" %*
    goto :done
)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0launch.py" %*
    goto :done
)
rem Compatibility with the original local installation; not required by a clone.
if exist "%~dp0..\..\work\vision_workbench\.venv\Scripts\python.exe" (
    "%~dp0..\..\work\vision_workbench\.venv\Scripts\python.exe" "%~dp0launch.py" %*
    goto :done
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0launch.py" %*
    goto :done
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0launch.py" %*
    goto :done
)
echo Install Python 3.11, then run: python setup.py --install
exit /b 1
:done
if errorlevel 1 pause
