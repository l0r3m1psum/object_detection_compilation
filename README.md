# Notes on TVM

Under the `include` directory of the TVM project we have:

  * `tvm/runtime/c_runtime_api.h`
  * `tvm/runtime/c_backend_api.h`

implemetaions can be found in the `src` directory at:

  * `runtime/c_runtime_api.h`
  * `runtime/cpu_device_api.h`
  * `runtime/cuda/cuda_device_api.h`

Compilation can be triggered by calling `tvm.runtime.Module.export_lib` which
internally visits and save all submodules and calls `tvm.runtime.Module.save`
that internally just calls `tvm.runtime._ffi_api.ModuleSaveToFile`, this is
just a wrapper for `tvm::runtime::ModuleNode::SaveToFile`. Assuming that we are
compiling tp C the implementation is
`tvm::codegen::CSourceModuleNode::SaveToFile` in
`runtime/target/source/source_module.cc` which just writes the `this->code_`
member initialized at module creation.

`tvm.compile` internally calls `tvm.relax.build` or `tvm.tir.build`.

L'ultima versione di TVM in cui è disponibile documentrasione per
[VTA](https://tvm.apache.org/docs/v0.16.0/topic/vta/index.html)
e
[microTVM](https://tvm.apache.org/docs/v0.16.0/topic/microtvm/index.html)
è la 0.16, ma nella repository il codice rimane fino alla versione
[0.18](https://github.com/apache/tvm/tree/v0.18.0).

`tvmc` si trova fino alla versione 0.19

# Resources

## Visdrone

https://aiskyeye.com/home/ this seems to be the home of the VisDrone
challenge/dataset.
https://github.com/VisDrone/VisDrone-Dataset
https://github.com/VisDrone/VisDrone2018-DET-toolkit constains the description
of the labels.
https://datasetninja.com/vis-drone-2019-det a good exploration of the dataset.
https://paperswithcode.com/dataset/visdrone
https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset
https://docs.ultralytics.com/it/datasets/detect/visdrone/ a tutorial


A list of relevant models
https://github.com/facebookresearch/Detectron#introduction

## Metrics

[A Survey on Performance Metrics for Object-Detection
Algorithms](https://ieeexplore.ieee.org/document/9145130)
[Comparative Analysis of Object Detection Metrics with a Companion Open-Source
Toolkit](https://doi.org/10.3390/electronics10030279)

## COCO

https://cocodataset.org/#detection-eval
https://cocodataset.org/#format-data

## PASCAL VOC

http://host.robots.ox.ac.uk/pascal/VOC/

# Outdoor configuration

Install Visual Studio

Download Visdrone

Download and models from torch hub

After doing the build with `build.bat` in `PCbuild`

```
pushd %installdir%\Programs
    curl -O "https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_576.02_windows.exe"
popd
git submodule update --init --recursive
.\python.exe -m ensurepip
.\python.exe -m pip download -r projreq.txt -d "%installdir%\Programs\wheelhouse"
.\python.exe -m pip download torch torchvision --index-url https://download.pytorch.org/whl/cu118 -d "%installdir%\Programs\wheelhouse"
REM On the offline PC
.\python.exe -m venv myvenv
REM Activate virtual environment
pip install --no-index --find-links "%installdir%\Programs\wheelhouse" -r projreq.txt
pip install --no-index --find-links "%installdir%\Programs\wheelhouse" torch torchvision
pip install --no-index --find-links "%installdir%\Programs\wheelhouse" pytest
```

note that somehow the default sys.path is set to the module in cpython's source
tree. To build a windows installer you have to go to `Tools\msi\build.bat`
