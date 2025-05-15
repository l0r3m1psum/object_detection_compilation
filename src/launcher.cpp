#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/vm.h>
#include <tvm/runtime/disco/builtin.h>

std::ostream& operator<<(std::ostream& os, const tvm::Array<tvm::String>& strings) {
  os << '[';
  for (int i = 0, e = strings.size(); i != e; ++i) {
    if (i != 0) os << ',';
    os << static_cast<std::string>(strings[i]);
  }
  os << ']';
  return os;
}

int main() {
  // runtime.module.loadbinary_cuda
  const char *loader_name = "runtime.module.loadfile_so";
  const tvm::runtime::PackedFunc *pf = tvm::runtime::Registry::Get(loader_name);
  if (!pf) {
    std::cerr << "cannot find " << loader_name << '\n';
    return 1;
  }
  const char *file_so = "build/mlp.dll";
  tvm::runtime::TVMRetValue rv = (*pf)(file_so);
  // __debugbreak();
  if (rv.type_code() != kTVMModuleHandle) {
    std::cerr << "cannot load " << file_so << '\n';
    return 1;
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

  // Il modulo esportato espone solo le "sigole" funzioni, non la forward
  // dumpbin /exports build\mlp.dll
  const char *function_name = "add";
  tvm::runtime::PackedFunc func = (*mod_node).GetFunction(function_name, true);
  if (!func.defined()) {
    std::cerr << "cannot get " << function_name << '\n';
    return 1;
  }

  static float my_a[256], my_b[256], my_c[256];

  for (int i = 0; i < 256; i++) {
    my_a[i] = 1.0f;
    my_b[i] = 2.0f;
  }
  tvm::runtime::NDArray a = tvm::runtime::NDArray::Empty(
    {1, 256}, // outpur of matmul has two dimensions
    DLDataType{kDLFloat, 32, 1},
    DLDevice{kDLCPU, 0}
  );
  tvm::runtime::NDArray b = tvm::runtime::NDArray::Empty(
    {256},
    DLDataType{kDLFloat, 32, 1},
    DLDevice{kDLCPU, 0}
  );
  tvm::runtime::NDArray c = tvm::runtime::NDArray::Empty(
    {1, 256},
    DLDataType{kDLFloat, 32, 1},
    DLDevice{kDLCPU, 0}
  );

  a.CopyFromBytes(my_a, sizeof my_a);
  b.CopyFromBytes(my_b, sizeof my_b);
  func(a, b, c); // DPS
  c.CopyToBytes(my_c, sizeof my_c);

  std::cout << my_c[0] << '\n';

  if (false) {
    // FIXME: brokewn for some reason...
    tvm::runtime::Module mod2 = tvm::runtime::LoadVMModule(file_so, DLDevice{kDLCPU, 0});
    std::cout << mod2->type_key() << '\n';
  }

  {
    tvm::runtime::Module vm = (mod.GetFunction("vm_load_executable")()).operator tvm::runtime::Module();
    std::cout << vm->type_key() << '\n';

    tvm::runtime::PackedFunc vm_initialization = vm.GetFunction("vm_initialization");
    vm_initialization(tvm::runtime::Int{1}, tvm::runtime::Int{0}, tvm::runtime::Int{2});
    tvm::runtime::PackedFunc forward = vm.GetFunction("forward", true);
    if (!forward.defined()) {
      std::cerr << "cannot get " << "forward" << '\n';
      return 1;
    }

    std::cout << (forward == nullptr) << '\n';

    tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty(
      {1, 784},
      DLDataType{kDLFloat, 32, 1},
      DLDevice{kDLCPU, 0}
    );
    tvm::runtime::NDArray w1 = tvm::runtime::NDArray::Empty(
      {784, 256},
      DLDataType{kDLFloat, 32, 1},
      DLDevice{kDLCPU, 0}
    );
    tvm::runtime::NDArray b1 = tvm::runtime::NDArray::Empty(
      {1, 256},
      DLDataType{kDLFloat, 32, 1},
      DLDevice{kDLCPU, 0}
    );
    tvm::runtime::NDArray w2 = tvm::runtime::NDArray::Empty(
      {256, 10},
      DLDataType{kDLFloat, 32, 1},
      DLDevice{kDLCPU, 0}
    );
    tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty(
      {1, 10},
      DLDataType{kDLFloat, 32, 1},
      DLDevice{kDLCPU, 0}
    );

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
