#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/vm.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/container/shape_tuple.h>

#include <safetensors.hh>

std::ostream& operator<<(std::ostream& os, const tvm::Array<tvm::String>& strings) {
  os << '[';
  for (int i = 0, e = strings.size(); i != e; ++i) {
    if (i != 0) os << ',';
    os << static_cast<std::string>(strings[i]);
  }
  os << ']';
  return os;
}

// tvm::runtime::LoadVMModule seems to be brocken (for some MSVC bug?) so here
// there is a reimplementation.
static tvm::runtime::Module
LoadVMModule(const std::string& path, tvm::Device device) {
  using namespace tvm::runtime;
  Module dso_mod = Module::LoadFromFile(path, "");
  PackedFunc vm_load_executable = dso_mod.GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_load_executable` does not exist";
  Module mod = vm_load_executable();
  PackedFunc vm_initialization = mod.GetFunction("vm_initialization");
  CHECK(vm_initialization != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_initialization` does not exist";
  vm_initialization(static_cast<int>(device.device_type), static_cast<int>(device.device_id),
                    static_cast<int>(AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
                    static_cast<int>(AllocatorType::kPooled));
  return mod;
}

static void
manually_load_module() {
  // runtime.module.loadbinary_cuda
  const char *loader_name = "runtime.module.loadfile_so";
  const tvm::runtime::PackedFunc *pf = tvm::runtime::Registry::Get(loader_name);
  if (!pf) {
    std::cerr << "cannot find " << loader_name << '\n';
    abort();
  }
  const char *file_so = "build/mlp.dll";
  tvm::runtime::TVMRetValue rv = (*pf)(file_so);
  if (rv.type_code() != kTVMModuleHandle) {
    std::cerr << "cannot load " << file_so << '\n';
    abort();
  }
  tvm::runtime::Module mod = rv.operator tvm::runtime::Module();

  std::cout << mod->type_key() << '\n';

  tvm::runtime::ModuleNode *mod_node = mod.operator->();

  tvm::runtime::relax_vm::VMExecutable *vm = static_cast<tvm::runtime::relax_vm::VMExecutable *>(mod_node);
  // std::cout << vm->AsText() << '\n';

  std::vector<tvm::runtime::relax_vm::VMFuncInfo> func_table = vm->func_table;

  std::cout << func_table.size() << '\n';
  for (const auto& func_info : func_table) {
    std::cout << func_info.name << '\n';
  }

  // Il modulo esportato espone solo le "sigole" funzioni, non la forward, per
  // ottenere quelle funzioni il modulo va inizializzato con vm_load_executable
  // e vm_initialization.
  // >dumpbin /exports build\mlp.dll
  const char *function_name = "add";
  tvm::runtime::PackedFunc func = (*mod_node).GetFunction(function_name, true);
  if (!func.defined()) {
    std::cerr << "cannot get " << function_name << '\n';
  }
  else {
    static float my_a[256], my_b[256], my_c[256];

    for (int i = 0; i < 256; i++) {
        my_a[i] = 1.0f;
        my_b[i] = 2.0f;
    }
    tvm::runtime::NDArray a = tvm::runtime::NDArray::Empty(
        { 1, 256 }, // outpur of matmul has two dimensions
        DLDataType{ kDLFloat, 32, 1 },
        DLDevice{ kDLCPU, 0 }
    );
    tvm::runtime::NDArray b = tvm::runtime::NDArray::Empty(
        { 256 },
        DLDataType{ kDLFloat, 32, 1 },
        DLDevice{ kDLCPU, 0 }
    );
    tvm::runtime::NDArray c = tvm::runtime::NDArray::Empty(
        { 1, 256 },
        DLDataType{ kDLFloat, 32, 1 },
        DLDevice{ kDLCPU, 0 }
    );

    a.CopyFromBytes(my_a, sizeof my_a);
    b.CopyFromBytes(my_b, sizeof my_b);
    func(a, b, c); // DPS
    c.CopyToBytes(my_c, sizeof my_c);

    std::cout << my_c[0] << '\n';
  }
}

static bool
ends_with(std::string const& value, std::string const& ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.crbegin(), ending.crend(), value.crbegin());
}

#include <windows.h>
#include <debugapi.h>

int main() {
  std::string file_so = "build\\resnet18.dll";
  tvm::runtime::Module vm = LoadVMModule(file_so, tvm::Device{kDLCPU, 0});
  std::cout << vm->type_key() << '\n';
  tvm::runtime::PackedFunc forward = vm.GetFunction("main", false);
  CHECK(forward != nullptr) << "cannot get forward";

  std::string file_safetensors = "build\\resnet18.safetensors";
  safetensors::safetensors_t weights{};
  {
    std::string warn{}, err{};
    bool ok = safetensors::mmap_from_file(file_safetensors, &weights, &warn, &err);
    LOG_IF(WARNING, warn != "") << warn;
    CHECK(ok) << "cannot load weights: " << err;
  }

  {
    safetensors::tensor_t tensor{};
    for (size_t i = 0; i < weights.tensors.size(); i++) {
      bool found = weights.tensors.at(i, &tensor);
      if (!found) abort();
    }
  }

  int num_args = weights.tensors.size() + 1;
  TVMValue *values = new TVMValue[num_args];
  int *type_codes = new int[num_args];
  tvm::runtime::TVMArgsSetter setter(values, type_codes);

  {
    DLDevice device{ kDLCUDA, 0 };
    DLDataType datatype{ kDLFloat, 32, 1 };

    // __debugbreak();
    int correction = 0;
    std::vector<tvm::runtime::ShapeTuple::index_type> shape_vec{1, 3, 224, 224};
    safetensors::tensor_t tmp_tensor{};
    tvm::runtime::NDArray tmp_ndarray = tvm::runtime::NDArray::Empty(shape_vec, datatype, device);
    setter(0, std::move(tmp_ndarray));
    for (int i = 1; i < num_args; i++) {
      bool found = weights.tensors.at(i-1, &tmp_tensor);
      CHECK(found);
      const std::string& key = weights.tensors.keys()[i-1];
      if (ends_with(key, "num_batches_tracked")
        || ends_with(key, "running_mean")
        || ends_with(key, "running_var")) {
        correction++;
        continue;
      }
      CHECK(tmp_tensor.dtype == safetensors::dtype::kFLOAT32);
      shape_vec.resize(tmp_tensor.shape.size());
      for (size_t i = 0; i < shape_vec.size(); i++) shape_vec[i] = tmp_tensor.shape[i];
      tvm::runtime::NDArray tmp_ndarray = tvm::runtime::NDArray::Empty(shape_vec, datatype, device); // I hate C++
      tmp_ndarray.CopyFromBytes(
        weights.databuffer_addr + tmp_tensor.data_offsets[0],
        tmp_tensor.data_offsets[1] - tmp_tensor.data_offsets[0]
      );
      setter(i-correction, std::move(tmp_ndarray));
    }
    num_args -= correction;
  }

  tvm::runtime::TVMArgs args(values, type_codes, num_args);
  tvm::runtime::TVMRetValue rv;
  forward.CallPacked(args, &rv);
  CHECK(rv.type_code() == kTVMNDArrayHandle);

  if (false) {
    DLDevice device{ kDLCUDA, 0 };
    DLDataType datatype{ kDLFloat, 32, 1 };
    tvm::runtime::NDArray x  = tvm::runtime::NDArray::Empty({1, 784}, datatype, device);
    tvm::runtime::NDArray w1 = tvm::runtime::NDArray::Empty({256, 784}, datatype, device);
    tvm::runtime::NDArray b1 = tvm::runtime::NDArray::Empty({256}, datatype, device);
    tvm::runtime::NDArray w2 = tvm::runtime::NDArray::Empty({10, 256}, datatype, device);
    tvm::runtime::NDArray y  = tvm::runtime::NDArray::Empty({1, 10}, datatype, device);

    tvm::runtime::TVMRetValue rv = forward(x, w1, b1, w2);
    if (rv.type_code() != kTVMNDArrayHandle) {
      std::cerr << "not an array\n";
      return 1;
    }
    tvm::runtime::NDArray res = rv.operator tvm::runtime::NDArray();
  }
  // *(std::string *)((unsigned char *)&key - 8)
  return 0;
}
