@echo off

REM TODO: determine the cmake build type dynamically
REM TODO: rename this to clang_vta_fsim_wrapper
REM This is supposet to be the TVM_WIN_CC.
clang -v -Wl,-verbose ^
-l "%cd%\submodules\tvm\build\Debug\vta_fsim" ^
-l "%cd%\submodules\tvm\build\Debug\tvm_runtime" ^
%*