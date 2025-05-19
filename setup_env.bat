@echo off

call config.bat

set "PATH=%installdir%\Programs\LLVM\bin;%PATH%"
set "PATH=%installdir%\Programs\Python;%PATH%"
if not exist %venvdir% (python -m venv %venvdir%)
%venvdir%\Scripts\Activate
pip install --no-index --find-links "%installdir%\Programs\wheelhouse" -r projreq.txt
pip install --no-index --find-links "%installdir%\Programs\wheelhouse" torch torchvision