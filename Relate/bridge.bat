@echo off
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"

:: Try to find a Python with nibabel
set "PYTHON="

:: 1. Check if 'python' on PATH has nibabel
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -c "import nibabel" >nul 2>&1
    if !ERRORLEVEL! equ 0 set "PYTHON=python"
)

:: 2. Try 'python3'
if not defined PYTHON (
    where python3 >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        python3 -c "import nibabel" >nul 2>&1
        if !ERRORLEVEL! equ 0 set "PYTHON=python3"
    )
)

:: 3. Try common Python3xx install locations
if not defined PYTHON (
    for %%v in (314 313 312 311 310) do (
        for %%d in (
            "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe"
            "C:\Python%%v\python.exe"
            "%ProgramFiles%\Python%%v\python.exe"
        ) do (
            if exist %%d (
                %%d -c "import nibabel" >nul 2>&1
                if !ERRORLEVEL! equ 0 (
                    set "PYTHON=%%d"
                    goto :found_python
                )
            )
        )
    )
)
:found_python

if defined PYTHON (
    "%PYTHON%" "%SCRIPT_DIR%bridge_maisi_mask.py" %*
    goto :done
)

echo Python with nibabel not found.
echo Please install: pip install nibabel numpy scipy
echo.
echo Or set MASK_BRIDGE_PYTHON to your python.exe path.
if defined MASK_BRIDGE_PYTHON (
    "%MASK_BRIDGE_PYTHON%" "%SCRIPT_DIR%bridge_maisi_mask.py" %*
    goto :done
)

:done
endlocal
