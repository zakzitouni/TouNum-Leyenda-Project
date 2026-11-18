@echo off
rem Starts the Leyenda Project page on http://localhost:8000. The first run installs the Python packages (a few minutes).
rem Close this window to stop the page.
cd /d "%~dp0"
set VENV=%LOCALAPPDATA%\leyenda-venv
if not exist "%VENV%\Scripts\python.exe" (
  echo Creating the Python environment in %VENV% ...
  py -3 -m venv "%VENV%" || python -m venv "%VENV%" || (echo Python 3.10 to 3.13 is required: https://www.python.org/downloads/ & pause & exit /b 1)
  "%VENV%\Scripts\python" -m pip install --disable-pip-version-check -r ..\requirements.txt || (pause & exit /b 1)
)
set TF_CPP_MIN_LOG_LEVEL=3
"%VENV%\Scripts\python" server.py %1
pause
