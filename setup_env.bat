@echo off

set installdir=D:
set venvdir=nnc

set "PATH=%installdir%\Programs\LLVM\bin;%PATH%"
set "PATH=%installdir%\Programs\Python;%PATH%"
if not exist %venvdir% (python -m venv %venvdir%)
%venvdir%\Scripts\Activate

set installdir=
set venvdir=