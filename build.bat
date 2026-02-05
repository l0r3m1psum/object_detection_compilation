@echo off

setlocal

if not exist build mkdir build

cl /LD /nologo /std:c++17 /ZI ^
    /I %installdir%\Programs\TVM\include ^
    /I %installdir%\Programs\dlpack\include ^
    /I %installdir%\Programs\dmlc\include ^
    src\my_device_api.cc ^
    tvm.lib ^
    /MDd ^
    /wd4005 ^
    /EHsc ^
    /Fo".\build\my_device_api.obj" ^
    /Fe".\build\my_device_api.dll" ^
    /Fd".\build\my_device_api.pdb" ^
    /link ^
    "/LIBPATH:%installdir%\Programs\TVM\lib" ^
    || goto :exit

rem     /fsanitize=address /MTd ^
REM https://learn.microsoft.com/en-us/cpp/build/reference/output-file-f-options
cl /nologo /std:c++17 /ZI ^
    /I %installdir%\Programs\TVM\include ^
    /I %installdir%\Programs\dlpack\include ^
    /I %installdir%\Programs\dmlc\include ^
    /I %installdir%\Programs\safetensors\include ^
    src/launcher.cpp ^
    tvm.lib ^
    safetensors_cpp.lib ^
    /MDd ^
    /wd4005 ^
    /EHsc ^
    /Fo".\build\launcher.obj" ^
    /Fe".\build\launcher.exe" ^
    /Fd".\build\launcher.pdb" ^
    /link ^
    "/LIBPATH:%installdir%\Programs\TVM\lib" ^
    "/LIBPATH:%installdir%\Programs\safetensors\lib" ^
    || goto :exit

set "PATH=%installdir%\Programs\TVM\lib;%PATH%"
REM build\launcher || goto :exit

endlocal

:exit
exit /b %ERRORLEVEL%