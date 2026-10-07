@echo off
setlocal
rem Starts the guided app (python -m app.server): its own window with pywebview installed, else the browser.
rem On a new computer it first gets what is missing, asking before each download: Python itself
rem (through winget, Windows' own installer), then the one package the app needs.
cd /d "%~dp0"

call :find_python
if not defined PY (
  echo.
  echo  Python was not found on this computer. The app needs it to run.
  where winget >nul 2>nul
  if errorlevel 1 goto no_winget
  echo  Windows can install Python 3.12 for you now ^(from python.org, about 27 MB, for this user only^).
  choice /c YN /m "  Install Python"
  if errorlevel 2 goto by_hand
  call winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
  call :find_python
)
if not defined PY goto by_hand

"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo.
  echo  This computer's Python is older than 3.10, which the app needs:
  "%PY%" --version
  echo  Install Python 3.12 or newer from https://www.python.org/downloads/ and start OC2VRM.bat again.
  echo.
  pause
  exit /b 1
)
"%PY%" -c "import PIL" >nul 2>nul
if errorlevel 1 (
  echo.
  echo  First start: the app needs Python's picture library, Pillow ^(about 3 MB, from pypi.org^).
  choice /c YN /m "  Install it now"
  if not errorlevel 2 "%PY%" -m pip install -r requirements.txt
  echo.
)

rem The app asks to be started again (exit code 75) after it has updated itself. A new launcher is
rem waiting as OC2VRM.bat.new then: a script can't be written over while it runs, so it is taken
rem here, in one line, and started in place of this one.
set OC2VRM_LAUNCHER=1
:run
"%PY%" -m app.server %*
if errorlevel 76 goto ended
if not errorlevel 75 goto ended
set OC2VRM_RESTARTED=1
if exist "%~f0.new" (move /y "%~f0.new" "%~f0" >nul & "%~f0" %* & exit /b)
goto run
:ended
if errorlevel 1 pause
exit /b

:no_winget
echo  This Windows has no winget to install it with.
:by_hand
echo.
echo  To install Python yourself:
echo  1. Get Python 3.12 or newer from https://www.python.org/downloads/
echo  2. In its installer, tick "Add python.exe to PATH".
echo  3. Start OC2VRM.bat again.
echo.
pause
exit /b 1

rem Sets PY to a Python that runs: the one on PATH, else the one winget or python.org's installer
rem put in this user's folder (a window open since before the install doesn't have it on PATH yet).
rem The Store's stand-in python.exe, which only opens the Store, is not one: it is asked to run.
:find_python
set "PY="
for %%P in (python.exe) do if not "%%~$PATH:P"=="" (
  "%%~$PATH:P" -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PY=%%~$PATH:P"
)
if defined PY exit /b
for %%V in (313 312 311 310) do if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
exit /b
