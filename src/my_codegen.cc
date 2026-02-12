#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

#define PRINT  std::cout << __func__ << '\n';

static void
my_func(TVMArgs args, TVMRetValue *rv) {
  PRINT
}

static void
invoke(TVMArgs args, TVMRetValue *rv) {
  PRINT
}

static void
set_input(TVMArgs args, TVMRetValue* rv) {
  PRINT
}

static void
get_output(TVMArgs args, TVMRetValue* rv) {
  PRINT

  std::vector<int64_t> shape = {16};
  DataType dtype = DataType::Float(32);
  Device cpu_dev = { kDLCPU, 0 };
  NDArray ret = NDArray::Empty(shape, dtype, cpu_dev);
  // size_t size = 16 * sizeof (float32);
  // ret.CopyFromBytes(src.dataPointer, size);
  *rv = ret;
}

static void
get_num_outputs(TVMArgs args, TVMRetValue* rv) {
  PRINT
  *rv = 1;
}

static void
get_symbol(TVMArgs args, TVMRetValue* rv) {
  PRINT
  *rv = std::string("ksdnfòjna");
}

static void
get_const_vars(TVMArgs args, TVMRetValue* rv) {
  PRINT
  *rv = Array<String>();
}

static void
fused_relax_add_my_target(TVMArgs args, TVMRetValue* rv) {
  PRINT

  DLTensor* x = args[0];
  DLTensor* y = args[1];

  float* x_data = static_cast<float*>(x->data);
  float* y_data = static_cast<float*>(y->data);

  int64_t num_elements = 1;
  for (int i = 0; i < x->ndim; ++i) {
    num_elements *= x->shape[i];
  }

  for (int64_t i = 0; i < num_elements; ++i) {
    y_data[i] = x_data[i] + 1.0f;
  }

}

class MyTargetRuntime : public ModuleNode {
 public:
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
    std::cout << "Getting: " << name << '\n';
    if (name == "invoke" || name == "run") {
      return PackedFunc(invoke);
    } else if (name == "set_input") {
      return PackedFunc(set_input);
    } else if (name == "get_output") {
      return PackedFunc(get_output);
    } else if (name == "get_num_outputs") {
      return PackedFunc(get_num_outputs);
    } else if (name == "get_symbol") {
      return PackedFunc(get_symbol);
    } else if (name == "get_const_vars") {
      return PackedFunc(get_const_vars);
    } else if (name == "fused_relax_add_my_target") {
      return PackedFunc(fused_relax_add_my_target);
    } else {
      return PackedFunc(); // NOTE: I guess that this means use the default implementation...
    }
  }
  int GetPropertyMask() const final { return ModulePropertyMask::kRunnable | ModulePropertyMask::kBinarySerializable; }
  const char* type_key() const { return "my_target"; }

  void SaveToBinary(dmlc::Stream *stream) {
    PRINT
    std::string my_symbol = "custom_algo_v1";
    stream->Write(my_symbol);
  }

  static Module LoadFromBinary(void* strm) {
    PRINT
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string symbol;
    stream->Read(&symbol);
    auto exec = make_object<MyTargetRuntime>();
    return Module(exec);
  }
};

Module MyTargetRuntimeCreate(const std::string& symbol, const std::string& model_path) {
  auto exec = make_object<MyTargetRuntime>();
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.my_target_runtime.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = MyTargetRuntimeCreate(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_my_target")
.set_body_typed(MyTargetRuntime::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
