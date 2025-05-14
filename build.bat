@echo off

setlocal

call config.bat

if not exist build mkdir build

REM https://learn.microsoft.com/en-us/cpp/build/reference/output-file-f-options
cl /std:c++17 ^
    /I %installdir%\Programs\TVM\include ^
    /I %installdir%\Programs\dlpack\include ^
    /I %installdir%\Programs\dmlc\include ^
    src/launcher.cpp ^
    /Fo".\build\launcher.obj" ^
    /Fe".\build\launcher.exe" ^
    /Fd".\build\launcher.pdb" ^
    /link ^
    /LIBPATH:%installdir%\Programs\TVM\lib

endlocal