@echo off
rem ============================================================
rem gsvtts.dll native Windows build (mingw-w64 gcc + Ninja)
rem Prereqs:
rem   1. mingw-w64 gcc (-posix) -- w64devkit works
rem   2. Ninja
rem   3. CMake >= 3.16
rem   4. onnxruntime*.dll under third_party\onnxruntime\lib\
rem      (download per README "Build" section)
rem If gcc/ninja are not on PATH, uncomment and edit the next line:
rem set "PATH=<w64devkit>\bin;<ninja>\bin;%PATH%"
rem ============================================================

where gcc >nul 2>nul
if errorlevel 1 (
    echo [ERROR] gcc not found on PATH. Install mingw-w64 ^(e.g. w64devkit^)
    echo         and add its bin dir to PATH, or edit the PATH line in this script.
    exit /b 1
)
where ninja >nul 2>nul
if errorlevel 1 (
    echo [ERROR] ninja not found on PATH. Install Ninja and add it to PATH,
    echo         or edit the PATH line in this script.
    exit /b 1
)
if not exist "%~dp0third_party\onnxruntime\lib\onnxruntime.dll" (
    echo [ERROR] third_party\onnxruntime\lib\onnxruntime.dll not found.
    echo         Download ONNX Runtime per the README "Build" section and place
    echo         the DLLs there. CPU-only builds can drop -DGSV_USE_CUDA=ON.
    exit /b 1
)

cmake -S "%~dp0." -B "%~dp0build-win" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGSV_USE_CUDA=ON || exit /b 1
cmake --build "%~dp0build-win" || exit /b 1

echo.
echo Done: %~dp0build-win\gsvtts.dll
