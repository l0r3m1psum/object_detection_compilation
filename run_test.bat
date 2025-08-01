@echo off

setlocal

set TVM_WIN_CC=clang_wrapper.bat
set PYTHONPATH=%CD%\submodules\tvm\vta\python
python submodules\tvm\vta\tests\python\integration\test_benchmark_topi_conv2d.py

endlocal