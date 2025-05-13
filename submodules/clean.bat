@echo off

rmdir /s /q cpython/PCbuild/amd64
rmdir /s /q cpython/PCbuild/obj
rmdir /s /q cpython/PCbuild/win32

rmdir /s /q llvm-project/llvm/build
rmdir /s /q llvm-project/clang/build

rmdir /s /q tvm/build