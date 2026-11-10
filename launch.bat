@echo off
rem SimplAr launcher for Windows: double-click to start.
rem The first start installs the Python packages listed in requirements.txt.
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul && set "PY=py -3"

%PY% -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
if errorlevel 1 goto nopython

%PY% -c "import numpy, scipy, pandas, openpyxl, matplotlib, PyQt5" >nul 2>nul
if not errorlevel 1 goto run
echo Installing the packages SimplAr needs. This happens once and takes a few minutes.
echo.
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto failed

:run
%PY% launcher.py
if errorlevel 1 pause
exit /b

:nopython
echo SimplAr needs Python 3.10 or later. Install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" on the first page of the installer.
echo Then double-click launch.bat again.
pause
exit /b 1

:failed
echo.
echo The installation failed; the messages above say why.
pause
exit /b 1
