from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

# Note that when printing this function in the REPL all shapes of the variables
# are infered.
@R.function
def main(
    x: R.Tensor((1, 784), dtype="float32"),
    weight: R.Tensor((784, 256), dtype="float32"),
    bias: R.Tensor((256,), dtype="float32"),
) -> R.Tensor((1, 256), dtype="float32"):
    with R.dataflow():
        lv0 = R.matmul(x, weight)
        lv1 = R.add(lv0, bias)
        gv = R.nn.relu(lv1)
        R.output(gv)
    return gv

import copy
orig_main = copy.deepcopy(main)

@R.function
def relax_mlp(
    data: R.Tensor(("n", 784), dtype="float32"),
    w0: R.Tensor((784, 128), dtype="float32"),
    b0: R.Tensor((128,), dtype="float32"),
    w1: R.Tensor((128, 10), dtype="float32"),
    b1: R.Tensor((10,), dtype="float32"),
) -> R.Tensor(("n", 10), dtype="float32"):
    with R.dataflow():
        lv0 = R.matmul(data, w0) + b0
        lv1 = R.nn.relu(lv0)
        lv2 = R.matmul(lv1, w1) + b1
        R.output(lv2)
    return lv2

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def linear(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
        M, N, K = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (M, K), "float32")
        W = T.match_buffer(w, (K, N), "float32")
        B = T.match_buffer(b, (N,), "float32")
        Z = T.match_buffer(z, (M, N), "float32")
        Y = T.alloc_buffer((M, N), "float32")
        for i, j, k in T.grid(M, N, K):
            with T.block("Y"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[v_i, v_j] = T.float32(0.0)
                Y[v_i, v_j] += X[v_i, v_k] * W[v_k, v_j]
        for i, j in T.grid(M, N):
            with T.block("Z"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                Z[v_i, v_j] = Y[v_i, v_j] + B[v_j]

    @T.prim_func(private=True)
    def relu(x: T.handle, y: T.handle):
        M, N = T.int64(), T.int64()
        X = T.match_buffer(x, (M, N), "float32")
        Y = T.match_buffer(y, (M, N), "float32")
        for i, j in T.grid(M, N):
            with T.block("Y"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                Y[v_i, v_j] = T.max(X[v_i, v_j], T.float32(0.0))

    @R.function
    def main(
        x: R.Tensor(("n", 784), dtype="float32"), # symbolic shape 'n'
        w0: R.Tensor((784, 256), dtype="float32"),
        b0: R.Tensor((256,), dtype="float32"),
        w1: R.Tensor((256, 10), dtype="float32"),
        b1: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor(("n", 10), dtype="float32"):
        cls = Module
        n = T.int64()
        with R.dataflow():
            # lv0 is implicitly passed as the last argument to cls.linear (DPS)
            lv0 = R.call_tir(cls.linear, (x, w0, b0), out_sinfo=R.Tensor((n, 256), dtype="float32"))
            lv1 = R.call_tir(cls.relu, (lv0,), out_sinfo=R.Tensor((n, 256), dtype="float32"))
            lv2 = R.call_tir(cls.linear, (lv1, w1, b1), out_sinfo=R.Tensor((n, 10), dtype="float32"))
            R.output(lv2)
        return lv2

# Structure Info represent the type of Relax expression. TensorStructInfo
# (R.Tensor in TVMScript) represent the shape and dtype of a tensor expressoion.

# relax.block_builder
# relax.frontend.nn
import tvm
import numpy

@R.function
def main(
    x: R.Tensor((1, 784), dtype="float32"),
    weight: R.Tensor((784, 256), dtype="float32"),
    bias: R.Tensor((256,), dtype="float32"),
) -> R.Tensor((1, 256), dtype="float32"):
# TODO: make a dynamic shape quantize function to avoid generating two.
    with R.dataflow():
        # The tvm.relax.op API needs to be used with tvm.relax.BlockBuilder.
        # The functions in tvm.relax.op are exposed in TVMScript script with the
        # same name e.g. tvm.relax.op.quantize becomes R.quantize.
        # R.const needs to be passed as an argument as is and cannot be saved in
        # a variable!!!
        xq = R.quantize(x, scale=R.const(127, 'float32'), zero_point=R.const(0, 'float16'))
        wq = R.quantize(weight, scale=R.const(127, 'float32'), zero_point=R.const(0, 'float16'))
        lv0q = R.matmul(xq, wq)
        lv0 = R.dequantize(lv0q, scale=R.const(127, 'float32'), zero_point=R.const(0, 'float16'))
        lv1 = R.add(lv0, bias)
        gv = R.nn.relu(lv1)
        R.output(gv)
    return gv

Main = tvm.IRModule({'main': main})
zero_pipeline = tvm.relax.get_pipeline('zero')
mod_op = zero_pipeline(Main)
ex = tvm.compile(Main, tvm.target.Target('llvm'))
vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
res = vm['main'](
    tvm.nd.array(numpy.ones((1, 784), dtype='float32')),
    tvm.nd.array(numpy.ones((784, 256), dtype='float32')),
    tvm.nd.array(numpy.ones((256,), dtype='float32')),
)
print(res)


# A Relax IR mutator
@tvm.relax.expr_functor.mutator
class ReluAndMatmulRewriter(tvm.relax.expr_functor.PyExprMutator):
    def __init__(self, mod: tvm.IRModule):
        super().__init__(mod)

    def visit_call_(self, call: tvm.relax.Call) -> tvm.relax.Expr:
        if call.op.name == "relax.nn.relu":
            return tvm.relax.op.nn.gelu(call.args[0])

        if call.op.name == "relax.matmul":
            lv0 = tvm.relax.op.quantize(call.args[0], scale=R.const(127, 'float32'), zero_point=R.const(0, 'float16'))
            lv1 = tvm.relax.op.quantize(call.args[1], scale=R.const(127, 'float32'), zero_point=R.const(0, 'float16'))
            lv2 = tvm.relax.op.matmul(lv0, lv1)
            lv3 = tvm.relax.op.dequantize(lv2, scale=R.const(127, 'float32'), zero_point=R.const(0, 'float16'))
            return lv3

        return super().visit_call_(call)

# A TVM pass
@tvm.transform.module_pass(opt_level=0, name="ReluToGeluAndQuantizeMatmul")
class ReluToGeluAndQuantizeMatmul:
    def transform_module(self, mod: tvm.IRModule, _ctx: tvm.transform.PassContext) -> tvm.IRModule:
        """IRModule-level transformation"""
        rewriter = ReluAndMatmulRewriter(mod)
        for g_var, func in mod.functions_items():
            if isinstance(func, tvm.relax.Function):
                func = rewriter.visit_expr(func)
                rewriter.builder_.update_func(g_var, func)
        return rewriter.builder_.get()

mod = ReluToGeluAndQuantizeMatmul()(tvm.IRModule({'main': orig_main}))
mod.show()
