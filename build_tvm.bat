@echo off
REM https://tvm.apache.org/docs/install/from_source.html

REM We expect to be in a Python's venv virutal environment and that
REM vcvarsall.bat has been invoked to correctly setup the build environment.

setlocal

set installdir=D:
set ncores=24

REM pip install -r requirements_tvm.txt || goto :exit
pushd submodules\tvm || goto :exit
    git submodule update --init --recursive || goto :exit
    if not exist build (mkdir build || goto :exit)
    pushd build || goto :exit
        copy /y ..\cmake\config.cmake . || goto :exit
        echo set(CMAKE_BUILD_TYPE RelWithDebInfo) >> config.cmake || goto :exit
        echo set(USE_LLVM "llvm-config --ignore-libllvm --link-static") >> config.cmake || goto :exit
        echo set(HIDE_PRIVATE_SYMBOLS ON) >> config.cmake || goto :exit
        echo set(USE_CUDA ON) >> config.cmake || goto :exit
        echo set(USE_CUBLAS ON) >> config.cmake || goto :exit
        cmake -DCMAKE_INSTALL_PREFIX=%installdir%\Programs\TVM .. || goto :exit
        REM Needed because otherwise LINK.EXE cannot find tvm.lib
        set "LINK=/LIBPATH:%CD%\Debug" || goto :exit
        cmake --build . --target install --parallel %ncores% || goto :exit
        set "TVM_LIBRARY_PATH=%CD%\Debug" || goto :exit
    popd || goto :exit
    pip install .\python || goto :exit
    REM FIXME: it does not print!
    python -c "import tvm; " ^
        "print(tvm.__file__); " ^
        "print(tvm._ffi.base._LIB); " ^
        "print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))" || goto :exit
popd || goto :exit

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%
