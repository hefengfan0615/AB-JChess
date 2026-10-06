@echo off
set "V11_REPO=%~dp0"
if "%V11_BUILD%"=="" set "V11_BUILD=%V11_REPO%build"
if "%JQV4_COMMON_ROOT%"=="" set "JQV4_COMMON_ROOT=%V11_REPO%..\..\jieqi_v4_common"
cmake --fresh -S "%V11_REPO%." -B "%V11_BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DJQV4_COMMON_ROOT="%JQV4_COMMON_ROOT%" -DBUILD_TESTING=ON
if errorlevel 1 exit /b 1
cmake --build "%V11_BUILD%" -j4
if errorlevel 1 exit /b 1
ctest --test-dir "%V11_BUILD%" --output-on-failure
