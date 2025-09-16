@echo off

REM TODO: rendere questa chiamata idempotente come fa CPython
call vcvarsall x64

call config.bat

set "PATH=%installdir%\Programs\LLVM\bin;%PATH%"
set "PATH=%installdir%\Programs\Python;%PATH%"

if not exist %venvdir% (python -m venv %venvdir%)
call %venvdir%\Scripts\Activate
python -m pip install --no-index --find-links "%installdir%\Programs\wheelhouse" -r projreq.txt
python -m pip install --no-index --find-links "%installdir%\Programs\wheelhouse" torch torchvision