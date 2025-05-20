@echo off

setlocal

call config.bat

rmdir /s /q submodules\cpython\PCbuild\amd64
rmdir /s /q submodules\cpython\PCbuild\obj
rmdir /s /q submodules\cpython\PCbuild\win32

rmdir /s /q submodules\llvm-project\llvm\build
rmdir /s /q submodules\llvm-project\clang\build

rmdir /s /q submodules\tvm\build
rmdir /s /q submodules\tvm\3rdparty\dlpack\build
rmdir /s /q submodules\tvm\3rdparty\dmlc-core\build

rmdir /s /q submodules\safetensors\build

rmdir /s /q "%installdir%\Programs\LLVM"
rmdir /s /q "%installdir%\Programs\Python"
rmdir /s /q "%installdir%\Programs\TVM"
rmdir /s /q "%installdir%\Programs\dlpack"
rmdir /s /q "%installdir%\Programs\dmlc"
rmdir /s /q "%installdir%\Programs\safetensors"

endlocal