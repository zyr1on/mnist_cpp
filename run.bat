@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo    MNIST C++ Engine - Build and Run
echo ======================================================

:: 1. Check and create build directory
if not exist "build" (
    echo [+] Creating 'build' directory...
    mkdir build
)

:: 2. CMake configuration (Release mode)
echo [+] Configuring CMake (Release)...
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo [!] CMake configuration failed!
    pause
    exit /b %ERRORLEVEL%
)

:: 3. Build step
echo [+] Building project...
cmake --build build --config Release --parallel
if %ERRORLEVEL% NEQ 0 (
    echo [!] Build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo ======================================================
echo [+] Build successful! Launching program...
echo ======================================================
echo.

:: 4. Run executable (keep root as working directory so data/ is found)
if exist "build\Release\mnist_engine.exe" (
    build\Release\mnist_engine.exe
) else if exist "build\mnist_engine.exe" (
    build\mnist_engine.exe
) else (
    echo [!] 'mnist_engine.exe' not found!
    pause
    exit /b 1
)

echo.
echo ======================================================
echo [+] Program execution finished successfully.
echo ======================================================
pause
