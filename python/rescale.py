import tvm
from tvm import relax, ir, tir
from tvm.script import ir as I, relax as R

import vtar

@I.ir_module
class CascadedAddsModule:
    @R.function
    def main(
        x: R.Tensor((1, 16, 32, 32), "int8"),
        w1: R.Tensor((16, 16, 3, 3), "int8"),
        w2: R.Tensor((16, 16, 3, 3), "int8"),
        w3: R.Tensor((16, 16, 3, 3), "int8")
    ):
        with R.dataflow():
            c1 = R.nn.conv2d(x, w1, padding=(1, 1))
            c2 = R.nn.conv2d(x, w2, padding=(1, 1))
            c3 = R.nn.conv2d(x, w3, padding=(1, 1))
            add1 = R.nn.relu(
                R.qnn.add(
                    c2, R.const(1.0), R.const(0),
                    c3, R.const(1.0), R.const(0),
                    R.const(1.0), R.const(0),
                )
            )
            add2 = R.qnn.add(
                c1, R.const(1.0), R.const(0),
                add1, R.const(1.0), R.const(0),
                R.const(1.0), R.const(0),
            )
            R.output(add2)
        return add2

if __name__ == "__main__":
    mod = CascadedAddsModule
    mod.show()
    mod = vtar.relax.transform.ReScale()(mod)
    mod.show()
