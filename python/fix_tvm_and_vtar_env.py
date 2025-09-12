import os, sys
# sys.path.append(os.path.join(os.getcwd(), "submodules/tvm/vta/python"))
# Leave this above any tvm import!
os.environ["TVM_WIN_CC"] = "clang_wrapper.bat"
# os.environ["CXX"] = "clang"
os.environ["CXX"] = "clang_wrapper.bat"
os.environ["TVM_HOME"] = os.path.join(os.getcwd(), "submodules/tvm")
os.environ["VTA_HW_PATH"] = os.path.join(os.getcwd(), "submodules/tvm/3rdparty/vta-hw")
os.environ["TVM_LIBRARY_PATH"] = os.path.join(os.getcwd(), "submodules/tvm/build")
