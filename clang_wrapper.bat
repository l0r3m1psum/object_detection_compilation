@echo off

REM This is supposet to be the TVM_WIN_CC.
clang -l "%cd%\submodules\tvm\build\RelWithDebInfo\vta_fsim" ^
-l "%cd%\submodules\tvm\build\RelWithDebInfo\tvm_runtime" ^
%*