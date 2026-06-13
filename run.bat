@echo off
setlocal enabledelayedexpansion

set "PY314=%USERPROFILE%\AppData\Local\Programs\Python\Python314\python.exe"

if exist "%PY314%" (
    "%PY314%" src\prompt_runner.py %*
    goto :done
)

where python 2>/dev/null
if %ERRORLEVEL% equ 0 (
    python src\prompt_runner.py %*
    goto :done
)

echo Python not found. Please install Python 3.12+.
echo Or edit run.bat and set PY314 to your python.exe path.
:done
endlocal
