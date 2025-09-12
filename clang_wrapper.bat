@echo off

REM TODO: rename this to clang_vta_fsim_wrapper
REM This is supposet to be the TVM_WIN_CC.
clang -v -Wl,-verbose ^
-l "%cd%\submodules\tvm\build\RelWithDebInfo\vta_fsim" ^
-l "%cd%\submodules\tvm\build\RelWithDebInfo\tvm_runtime" ^
%*