#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>

#include <cstdlib>

#define PRINT std::cout << __func__ << '\n';
namespace tvm {
namespace runtime {

class MyDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { PRINT }

  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    PRINT
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void *AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint) final {
    PRINT
    // return std::aligned_alloc(size, alignment);
    return _aligned_malloc(size, alignment);
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    PRINT
    // std::free(ptr);
    _aligned_free(ptr);
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final { PRINT }

  void CopyDataFromTo(
    const void* from, size_t from_offset,
    void* to, size_t to_offset, size_t size,
    Device dev_from, Device dev_to,
    DLDataType type_hint, TVMStreamHandle stream
  ) final {
    PRINT
    std::memcpy(
      static_cast<unsigned char*>(to) + to_offset,
      static_cast<const unsigned char*>(from) + from_offset,
      size
    );
  }

  static MyDeviceAPI *Global() {
    PRINT
    static MyDeviceAPI *inst = new MyDeviceAPI();
    return inst;
  }
};

#define TVM_REGISTER_GLOBAL_OVERRIDE(OpName) \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) = ::tvm::runtime::Registry::Register((OpName), true)

TVM_REGISTER_GLOBAL_OVERRIDE("device_api.ext_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = MyDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace tvm

extern "C" {
TVM_DLL void my_func(double *x, double *y, size_t len);

void my_func(double *x, double *y, size_t len) {
  for (size_t i = 0; i < len; i++) {
    y[i] = x[i]+1;
  }
}
}
