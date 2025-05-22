@echo off

setlocal

call config.bat

if not exist build mkdir build

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
    /EHsc ^
    /Fo".\build\launcher.obj" ^
    /Fe".\build\launcher.exe" ^
    /Fd".\build\launcher.pdb" ^
    /link ^
    "/LIBPATH:%installdir%\Programs\TVM\lib" ^
    "/LIBPATH:%installdir%\Programs\safetensors\lib" ^
    || goto :exit

rem python python\export_resnet_model_and_weights.py || goto :exit

set TVM_LOG_DEBUG=1
set "PATH=%installdir%\Programs\TVM\lib;%PATH%"
build\launcher || goto :exit

endlocal

:exit
exit /b %ERRORLEVEL%