@echo off
cd llvm-project\llvm
rmdir /S /Q build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=%LOCALAPPDATA%\Programs\LLVM -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --target install --parallel 24
REM needed to build TVM
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\LLVM\bin"
cd ..\..\..
