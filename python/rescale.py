import tvm
from tvm import relax, ir
from tvm.script import ir as I
from tvm.script import relax as R
import vtar

qnn_add = ir.Op.get("relax.qnn.add")

import tvm
from tvm import relax
from tvm.script import ir as I, relax as R
import numpy

from typing import Dict, Set, List, Tuple

def optimize_scales(s_y: float, s_i: numpy.ndarray) -> numpy.ndarray:
    """
    s_y: output scale of the "summation"
    s_i: input scales of every addition the "summation"
    """
    if len(s_i.shape) != 1: raise ValueError("s_i must be a vector")
    K = s_i.size
    best_error = numpy.inf
    best_result = None

    exact_n = numpy.log2(s_i/s_y)
    floor_choice = numpy.floor(exact_n)
    ceil_choice = numpy.ceil(exact_n)
    choices = numpy.stack((floor_choice, ceil_choice))
    iota = numpy.arange(K, dtype="int64")
    mask = numpy.ones(K, dtype="int64") << iota

    for i in range(2**K):
        # Consider that a number written with bits that counts from 0 to 2**K-1
        # enumerates all possible {0,1}**K strings. Hence using a mask to detect
        # if a bit is set or not we can use it to select from one row or another
        # of the choices matrix.
        row_indices = ((i & mask) > 0).astype(int)
        n = choices[row_indices, iota]
        sum_num = numpy.sum(2**n * s_i)
        sum_den = numpy.sum(2**(2*n))
        delta_s_y = (sum_num - s_y * sum_den) / (1 + sum_den)
        delta_s_i = (2**n) * (s_y + delta_s_y) - s_i
        delta_s = numpy.concatenate(((delta_s_y,), delta_s_i))
        error = numpy.linalg.norm(delta_s)
        if error < best_error:
            best_error = error
            best_result = n

    return best_result.astype(int)

def find_connected_linear_ops(
    root_var: relax.Var,
    bindings: Dict[relax.Var, relax.Expr],
) -> Tuple[List[relax.Var], List[float]]:
    connected_linear_ops = []
    leaf_scales = []

    def _trace(var):
        expr = bindings[var]

        if isinstance(expr, relax.Call) and hasattr(expr.op, "name"):
            print(expr.op.name)
            if expr.op.name == "relax.qnn.add": # Linear ops
                connected_linear_ops.append(var)
                (
                    a, s_a, z_a,
                    b, s_b, z_b,
                    s_c, z_c,
                ) = expr.args

                if (
                    (s_a.data.numpy() <= 0).any()
                    or (s_b.data.numpy() <= 0).any()
                    or (s_c.data.numpy() <= 0).any()
                ):
                    raise ValueError("All scales of quantized addition must be"
                        " strictly positive.")
                if (z_c.data.numpy() != 0).all():
                    raise ValueError("Quantized addition with output zero point"
                        " different form zero is not supported.")

                if isinstance(a, relax.Var):
                    expr = bindings[a]
                    if (
                        isinstance(expr, relax.Call)
                        and hasattr(expr.op, "name")
                        and expr.op.name != "relax.qnn.add"
                        and expr.op.name != "relax.nn.relu"
                    ):
                        leaf_scales.append(s_a.data.numpy())
                    else:
                        _trace(a)
                if isinstance(b, relax.Var):
                    expr = bindings[b]
                    if (
                        isinstance(expr, relax.Call)
                        and hasattr(expr.op, "name")
                        and expr.op.name != "relax.qnn.add"
                        and expr.op.name != "relax.nn.relu"
                    ):
                        leaf_scales.append(s_b.data.numpy())
                    else:
                        _trace(b)

                # for arg in expr.args:
                #     if isinstance(arg, relax.Var):
                #         _trace(arg)
            elif expr.op.name == "relax.nn.relu": # homogeneous functions
                _trace(expr.args[0])
                # for arg in expr.args:
                #     if isinstance(arg, relax.Var):
                #         _trace(arg)

    _trace(root_var)
    return connected_linear_ops, leaf_scales

@relax.expr_functor.mutator
class ReScaleMutator(relax.PyExprMutator):
    def visit_function_(self, func: relax.Function):
        self.roots_and_pots_leaves: Dict[relax.Var, List[int]] = {}
        already_connected_linear_ops: Set[relax.Var] = set()
        var2val: Dict[relax.Var, relax.Expr] = relax.analysis.get_var2val(func)

        for var, expr in reversed(var2val.items()):
            if isinstance(expr, relax.Call) and getattr(expr.op, "name", "") == "relax.qnn.add":

                if var in already_connected_linear_ops:
                    continue

                connected_linear_ops, leaf_scales = find_connected_linear_ops(var, var2val)

                already_connected_linear_ops.update(connected_linear_ops)

                # Given reversed topological order the first element of the
                # list is always the output qnn.add.
                (
                    _, _, _,
                    _, _, _,
                    s_y, _,
                ) = var2val[connected_linear_ops[0]].args

                pots = optimize_scales(s_y.data.numpy().item(), numpy.array(leaf_scales))
                print(pots)

                self.roots_and_pots_leaves[var] = leaf_scales

        # FIXME: why there are two roots?
        print(already_connected_linear_ops)
        print(self.roots_and_pots_leaves)

        return super().visit_function_(func)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        new_value = self.visit_expr(binding.value)
        print(binding.var in self.roots_and_pots_leaves)
        # TODO: when a root is found rebuild the expression tree lowering
        # qnn.add to IOA operations.
        return

        # Apply multiplier adjustments if this variable was optimized
        if binding.var in self.scale_adjustments:
            factor = self.scale_adjustments[binding.var]

            # Emit raw unscaled node
            unscaled_var = self.builder_.emit(new_value, name_hint=binding.var.name_hint + "_unscaled")

            # Multiply by scale adjustment
            factor_const = relax.const(factor, dtype="float32")
            scaled_expr = relax.multiply(unscaled_var, factor_const)

            # Bind back to original variable ID
            new_var = self.builder_.emit(scaled_expr, name_hint=binding.var.name_hint)
            self.set_var_remap(binding.var.vid, new_var)
        else:
            new_var = self.builder_.emit(new_value, name_hint=binding.var.name_hint)
            self.set_var_remap(binding.var.vid, new_var)

@ir.transform.module_pass(opt_level=0)
class ReScale:
    def transform_module(self, mod, ctx):
        rewriter = ReScaleMutator(mod)

        for global_var, func in mod.functions.items():
            if isinstance(func, relax.Function):
                updated_func = rewriter.visit_expr(func)
                updated_func = relax.analysis.remove_all_unused(updated_func)
                rewriter.builder_.update_func(global_var, updated_func)

        return rewriter.builder_.get()

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
                relax.Call(qnn_add, (
                    c2, R.const(1.0), R.const(0),
                    c3, R.const(1.0), R.const(0),
                    R.const(1.0), R.const(0),
                ))
            )
            add2 = relax.Call(qnn_add, (
                c1, R.const(1.0), R.const(0),
                add1, R.const(1.0), R.const(0),
                R.const(1.0), R.const(0),
            ))
            R.output(add2)
        return add2

if __name__ == "__main__":
    mod = CascadedAddsModule
    mod.show()
    mod = ReScale()(mod)
    mod.show()
