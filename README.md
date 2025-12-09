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

Relay had a predecessor called [NNVM](https://discuss.tvm.apache.org/t/difference-relay-vs-nnvm/7549#:~:text=NNVM%20was%20the%20precursor%20to%20Relay).

[Halide IRNode](https://halide-lang.org/docs/struct_halide_1_1_internal_1_1_i_r_node.html)

[TVM stmt](https://tvm.apache.org/docs/reference/api/doxygen/stmt_8h.html)

[Hyper AI's mirror of the old VTA documentation](https://tvm.hyper.ai/docs/0.12.0/topic/vta)

## TIR

```
BufferStore(tir::Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices, ffi::Optional<PrimExpr> predicate) // Leaf
Evaluate(PrimExpr value) // Leaf
LetStmt(tir::Var var, PrimExpr value, tir::Stmt body)
AttrStmt(ffi::Any node, ffi::String attr_key, PrimExpr value, tir::Stmt body)
AssertStmt(PrimExpr condition, PrimExpr message, tir::Stmt body)
BufferRealize(tir::Buffer buffer, ffi::Array<Range> bounds, PrimExpr condition, tir::Stmt body)
Allocate(tir::Var buffer_var, DataType dtype, ffi::Array<PrimExpr> extents, PrimExpr condition, tir::Stmt body, ffi::Map<ffi::String, ffi::Any> annotations)
AllocateConst(tir::Var buffer_var, DataType dtype, ffi::Array<PrimExpr> extents, ObjectRef data_or_idx, tir::Stmt body, ffi::Map<ffi::String, ffi::Any> annotations)
DeclBuffer(tir::Buffer, tir::Stmt body)
SeqStmt(ffi::Array<tir::Stmt> seq)
IfThenElse(PrimExpr condition, tir::Stmt then_case, ffi::Optional<tir::Stmt> else_case)
For(tir::Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, tir::Stmt body, ffi::Map<ffi::String, ffi::Any>annotations)
While(PrimExpr condition, tir::Stmt body)
Block(
  ffi::Array<IterVar> iter_vars,
  ffi::Array<BufferRegion> reads,
  ffi::Array<BufferRegion> writes,
  ffi::String name_hint,
  tir::Stmt body,
  ffi::Optional<tir::Stmt> init,
  ffi::Array<tir::Buffer> alloc_buffers,
  ffi::Array<MatchBufferRegion> match_buffers,
  ffi::Map<ffi::String, ffi::Any> annotations
)
BlockRealize(ffi::Array<PrimExpr> iter_values, PrimExpr predicate, Block block)
```

[Loop Transformations](https://www.emmtrix.com/wiki/Loop_Transformations)

```
https://llvm.org/doxygen/LoopFlatten_8cpp_source.html
https://www.cs.cornell.edu/courses/cs6120/2020fa/blog/loop-flatten/#implementation

for (int i = 0; i < M; i++)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < O; k++)
      for (int l = 0; l < P; l++)
        f(a[i*M*N*O + j*N*O + k*O + l]); /* f(a[i][j][k][l]); */

for (int i = 0; i < M*N*O*P; i++) // Assuming no overflow
  f(a[i]); // Address calculation is much easier
```

# Resources

## Compilers

  * [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794)
  * [Relax: Composable Abstractions for End-to-End Dynamic Machine Learning](https://arxiv.org/abs/2311.02103)
  * [Destination-Passing Style for Efficient Memory Management](https://simon.peytonjones.org/assets/pdfs/destination-passing-style.pdf)
  * [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/abs/2207.04296)

  * [Dive into Deep Learning Compiler](https://tvm.d2l.ai)
  * [Machine Learning Compiler](https://mlc.ai)

## Quantization

  * [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)
  * [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
  * [Speed up integer-arithmetic-only inference via bit-shifting](https://www.nature.com/articles/s41598-025-02544-4)

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

```
pushd %installdir%\Programs
    curl -O "https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_576.02_windows.exe"
popd
git submodule update --init --recursive
pushd submodules\tvm
  git submodule deinit -f 3rdparty\flashinfer
popd
.\python.exe -m ensurepip
.\python.exe -m pip download -r projreq.txt -d "%installdir%\Programs\wheelhouse"
.\python.exe -m pip download torch torchvision --index-url https://download.pytorch.org/whl/cu118 -d "%installdir%\Programs\wheelhouse"
REM On the offline PC
pip install --no-index --find-links "%installdir%\Programs\wheelhouse" -r projreq.txt
pip install --no-index --find-links "%installdir%\Programs\wheelhouse" torch torchvision
pip install --no-index --find-links "%installdir%\Programs\wheelhouse" pytest
```

note that somehow the default sys.path is set to the module in cpython's source
tree. To build a windows installer you have to go to `Tools\msi\build.bat`

# VTA

`vta/python/vta/top` stands for tensor operators which mirrors
`python/tvm/topi` tensor operators invetory.

Other open source hardware accellerators for deep learning:

  * [NVDLA](https://nvdla.org/)
  * [OpenDLA](https://github.com/OpenDLA/OpenDLA)
  * [PipeCNN](https://github.com/doonny/PipeCNN)
  * [Coral NPU](https://github.com/google-coral/coralnpu)

# Convolution

[CUDA Tensor Layouts for Convolution](https://leimao.github.io/blog/CUDA-Convolution-Tensor-Layouts/)
where the layout I'm interested in is `N[C/x]HW[x]`.

https://arxiv.org/pdf/2009.11224

https://www.alcf.anl.gov/files/Gouicem_ALCF_SDL_2018_MKLDNN_presentation.pdf#page=8

https://discuss.tvm.apache.org/t/couldnt-find-vta-codegen-file/8079/3#:~:text=of%20NCHW%20to-,NCHW16n16c,-%2C%20and%20print%20Relay

https://tvm.apache.org/docs/v0.16.0/topic/vta/tutorials/optimize/convolution_opt.html

https://dl.acm.org/doi/pdf/10.1145/3625004

https://tvm.d2l.ai/chapter_cpu_schedules/packed_conv.html
