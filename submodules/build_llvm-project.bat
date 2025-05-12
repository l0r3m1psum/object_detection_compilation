@echo off

REM TVM depends on TVM to build
cd llvm-project\llvm
if exist build (rmdir /S /Q build)
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=%LOCALAPPDATA%\Programs\LLVM -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --target install --parallel 24
REM needed to build TVM
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\LLVM\bin"

REM TVM on Windows requires clang as a C compiler
cd ..\clang
if exist build (rmdir /S /Q build)
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=%LOCALAPPDATA%\Programs\LLVM -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_INCLUDE_TESTS=OFF ..
cmake --build . --target install --parallel 24

cd ..\..\..
