@echo off

cd pytorch
pip install -r requirements.txt
git submodule update --init --recursive
if exist build (rmdir /S /Q build)
mkdir build
cd build
REM FIXME: disable many dependencies since torch_cpu.lib is more than 4GiB
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --parallel 24
cd ..\..
