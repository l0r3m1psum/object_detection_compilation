@echo off
REM https://tvm.apache.org/docs/install/from_source.html

REM We expect to be in a Python's venv virutal environment and that
REM vcvarsall.bat has been invoked to correctly setup the build environment.

setlocal

set "VTA_HW_PATH=%CD:\=/%/submodules/tvm-vta"

REM TODO: pip install stuff like in setup env.
pushd submodules\tvm || goto :exit
    REM git submodule update --init --recursive || goto :exit
    if not exist build (mkdir build || goto :exit)
    pushd build || goto :exit
        copy /y ..\cmake\config.cmake . || goto :exit
        copy /y ..\..\..\vtar\VTAR.cmake
        REM Ignored by msbuild
        REM echo set(CMAKE_BUILD_TYPE RelWithDebInfo) >> config.cmake || goto :exit
        echo set(USE_LLVM "llvm-config --ignore-libllvm --link-static") >> config.cmake || goto :exit
        echo set(HIDE_PRIVATE_SYMBOLS ON) >> config.cmake || goto :exit
        REM echo set(USE_CUDA ON) >> config.cmake || goto :exit
        REM echo set(USE_CUBLAS ON) >> config.cmake || goto :exit
        echo set(USE_VTA_FSIM ON) >> config.cmake || goto :exit
        echo set(Python_ROOT_DIR "%installdir:\=\\%") >> config.cmake || goto :exit
        echo set(Python_FIND_REGISTRY NEVER) >> config.cmake || goto :exit
        echo set(Python_FIND_STRATEGY LOCATION) >> config.cmake || goto :exit

        REM echo set(TVM_LOG_DEBUG ON) >> config.cmake || goto :exit
        REM echo add_compile_definitions(DMLC_LOG_BEFORE_THROW) >> config.cmake || goto :exit

        REM When LLVM is compiled in debug mode this is needed when compiling TVM in Release or RelWithDebInfo mode
        REM echo set(USE_MSVC_MT ON) >> config.cmake
        REM echo add_compile_options("/MT") >> config.cmake || goto :exit
        echo add_compile_options("/MDd")  >> config.cmake || goto :exit

        REM echo set(SUMMARIZE ON) >> config.cmake || goto :exit
        REM echo set(CMAKE_C_COMPILER "%installdir:\=\\%\\Programs\\LLVM\\bin\\clang-cl.exe") >> config.cmake || goto :exit
        REM echo set(CMAKE_CXX_COMPILER "%installdir:\=\\%\\Programs\\LLVM\\bin\\clang-cl.exe") >> config.cmake || goto :exit

        cmake "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\TVM" .. || goto :exit
        REM Needed because otherwise LINK.EXE cannot find tvm.lib
        set "LINK=/LIBPATH:%CD%\RelWithDebInfo /LIBPATH:%installdir%\Programs\Python\libs" || goto :exit
        set "INCLUDE=%cd%\..\ffi\include"
        set UseEnv=true
        cmake --build . --config RelWithDebInfo --verbose --target install --parallel %NUMBER_OF_PROCESSORS% || goto :exit
        set "TVM_LIBRARY_PATH=%CD%\RelWithDebInfo" || goto :exit
    popd || goto :exit
    set "LINK=/LIBPATH:%installdir%\Programs\Python\libs /LIBPATH:%installdir%\Programs\TVM\lib" || goto :exit
    REM --no-build-isolation makes it faster
    python -m pip install --no-build-isolation --no-index --find-links "%installdir%\Programs\wheelhouse" .\python || goto :exit
    python -c "import tvm; print(tvm.__file__); print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))" || goto :exit
popd || goto :exit

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%
