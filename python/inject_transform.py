import tvm
from tvm import relax
from tvm import ir
from tvm.script import relax as R
from tvm.script import ir as I

@relax.expr_functor.mutator
class FuncInjectRewriter(relax.PyExprMutator):
    def __init__(self, mod: ir.IRModule, param_name: str, prep_gv: ir.GlobalVar) -> None:
        super().__init__(mod)
        self.param_name = param_name
        self.prep_gv = prep_gv

    def visit_function_(self, func: relax.Function) -> relax.Expr:
        target_var = next((p for p in func.params if p.name_hint == self.param_name), None)
        if not target_var:
            return super().visit_function_(func)

        with self.builder_.function("new_main", func.params, dict(func.attrs)):
            with self.builder_.dataflow():
                prep_res = self.builder_.emit(relax.Call(self.prep_gv, (target_var,)))
                # This makes it so that visit_expr replaces the instances of
                # target_var with prep_res
                self.set_var_remap(target_var.vid, prep_res)
                output = self.visit_expr(func.body)
                gv = self.builder_.emit_output(output)
            _ = self.builder_.emit_func_output(gv)
        res = self.builder_.get()["new_main"]
        return res

@ir.transform.module_pass(opt_level=0)
class InjectFunctionTransform:
    def __init__(self, target_func_name: str, param_name: str, prep_func_name: str) -> None:
        self.target_func_name = target_func_name
        self.param_name = param_name
        self.prep_func_name = prep_func_name

    def transform_module(self, mod: ir.IRModule, ctx: ir.transform.PassContext) -> ir.IRModule:
        prep_gv = mod.get_global_var(self.prep_func_name)

        rewriter = FuncInjectRewriter(mod, self.param_name, prep_gv)
        for gv, func in mod.functions.items():
            if gv.name_hint == self.target_func_name and isinstance(func, relax.Function):
                updated_func = rewriter.visit_expr(func)
                # updated_func = relax.analysis.remove_all_unused(updated_func)
                rewriter.builder_.update_func(gv, updated_func)

        return rewriter.builder_.get()

@I.ir_module
class Module:
    @R.function
    def my_preprocessing(img: R.Tensor((1, 3, 224, 224), "float32")):
        with R.dataflow():
            scale = R.multiply(img, R.const(0.0392))
            out = R.subtract(scale, R.const(0.5))
            R.output(out)
        return out

    @R.function
    def main(data: R.Tensor((1, 3, 224, 224), "float32"), weight: R.Tensor((32, 3, 3, 3), "float32")):
        with R.dataflow():
            conv = R.nn.conv2d(data, weight, strides=1, padding=1, dilation=1)
            relu = R.nn.relu(conv)
            R.output(relu)
        return relu

mod = InjectFunctionTransform("main", "data", "my_preprocessing")(Module)
mod.show()
