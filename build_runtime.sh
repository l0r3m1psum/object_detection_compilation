#!/bin/sh

set -e
export VTA_HW_PATH="`pwd`/submodules/tvm-vta"

(
	cd submodules/tvm
	if [ ! -e build ]
	then
		mkdir build
	fi
	# TODO: add zcu104 branch to VTAR.cmake
	cp -f ../tvm-vta/config/pynq_sample.json ../tvm-vta/config/vta_sample.json
	(
		cd build
		cp -f ../cmake/config.cmake ../../../vtar/VTAR.cmake .
		echo "set(USE_LIBBACKTRACE OFF)" >>config.cmake
		echo "set(THREADS_PREFER_PTHREAD_FLAG TRUE)" >>config.cmake
		echo "set(USE_VTA_FPGA ON)" >>config.cmake
		cmake ..
		cmake --build . --target runtime vta --parallel
	)
)
