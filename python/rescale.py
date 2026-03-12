import tvm
from tvm import relax, ir, tir
from tvm.script import ir as I, relax as R

import vtar
import numpy

from typing import Dict, Set, List, Tuple

qnn_add = ir.Op.get("relax.qnn.add")

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
            elif expr.op.name == "relax.nn.relu": # Homogeneous functions
                _trace(expr.args[0])

    _trace(root_var)
    return connected_linear_ops, leaf_scales

def rebuild_tree(bb: relax.BlockBuilder, root_var: relax.Var, var2val: Dict, pots: List[int]) -> relax.Expr:
    # Use an iterator so we pop the exact n_i corresponding to the leaves in DFS
    # order as the find_connected_linear_ops function does.
    pot_iter = iter(pots)

    def _build_leaf(arg, zero_point, n: int) -> relax.Expr:
        """Emits (a - z_a) <> s_a"""
        val_i32 = bb.emit(relax.op.astype(arg, "int32"))
        zp_i32 = bb.emit(relax.op.astype(zero_point, "int32"))

        diff = bb.emit(relax.op.subtract(val_i32, zp_i32))

        n_val = int(n)
        if n_val > 0:
            shift_const = relax.const(n_val, "int32")
            return bb.emit(relax.op.left_shift(diff, shift_const))
        elif n_val < 0:
            shift_const = relax.const(-n_val, "int32")
            return bb.emit(relax.op.right_shift(diff, shift_const))
        else:
            return diff

    def _trace(var: relax.Var) -> relax.Expr:
        expr = var2val[var]

        if isinstance(expr, relax.Call) and hasattr(expr.op, "name"):
            if expr.op.name == "relax.qnn.add":
                a, s_a, z_a, b, s_b, z_b, s_c, z_c = expr.args

                # Checks if leaves needs to be produced for LHS or RHS.
                if isinstance(a, relax.Var):
                    expr_a = var2val.get(a)
                    if isinstance(expr_a, relax.Call) and getattr(expr_a.op, "name", "") in ["relax.qnn.add", "relax.nn.relu"]:
                        lhs = _trace(a)
                    else:
                        n_a = next(pot_iter)
                        lhs = _build_leaf(a, z_a, n_a)
                else:
                    n_a = next(pot_iter)
                    lhs = _build_leaf(a, z_a, n_a)

                if isinstance(b, relax.Var):
                    expr_b = var2val.get(b)
                    if isinstance(expr_b, relax.Call) and getattr(expr_b.op, "name", "") in ["relax.qnn.add", "relax.nn.relu"]:
                        rhs = _trace(b)
                    else:
                        n_b = next(pot_iter)
                        rhs = _build_leaf(b, z_b, n_b)
                else:
                    n_b = next(pot_iter)
                    rhs = _build_leaf(b, z_b, n_b)

                return bb.emit(relax.op.add(lhs, rhs))

            elif expr.op.name == "relax.nn.relu":
                inner_expr = _trace(expr.args[0])
                return bb.emit(relax.op.nn.relu(inner_expr))

        return var

    res = _trace(root_var)

    # Cast back to the original datatype of the root (e.g., int8 / uint8)
    out_dtype = root_var.struct_info.dtype
    res = bb.emit(relax.op.maximum(res, relax.const(tir.max_value(out_dtype).value)))
    res = bb.emit(relax.op.maximum(relax.const(tir.min_value(out_dtype).value), res))
    res = bb.emit(relax.op.astype(res, out_dtype))

    return res

@relax.expr_functor.mutator
class ReScaleMutator(relax.PyExprMutator):

    def visit_function_(self, func: relax.Function) -> relax.Function:
        self.roots_and_pots_leaves: Dict[relax.Var, List[int]] = {}
        self.already_connected_linear_ops: Set[relax.Var] = set()
        self.var2val: Dict[relax.Var, relax.Expr] = relax.analysis.get_var2val(func)

        return super().visit_function_(func)

    def visit_dataflow_block_(self, block: relax.DataflowBlock) -> relax.DataflowBlock:
        for binding in reversed(block.bindings):
            var, expr = binding.var, binding.value
            if isinstance(expr, relax.Call) and getattr(expr.op, "name", "") == "relax.qnn.add":

                if var in self.already_connected_linear_ops:
                    continue

                connected_linear_ops, leaf_scales = find_connected_linear_ops(var, self.var2val)

                self.already_connected_linear_ops.update(connected_linear_ops)

                # Given reversed topological order the first element of the
                # list is always the output qnn.add.
                (
                    _, _, _,
                    _, _, _,
                    s_y, _,
                ) = self.var2val[connected_linear_ops[0]].args

                pots = optimize_scales(s_y.data.numpy().item(), numpy.array(leaf_scales))

                self.roots_and_pots_leaves[var] = pots

        return super().visit_dataflow_block_(block)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if binding.var in self.roots_and_pots_leaves:
            pots = self.roots_and_pots_leaves[binding.var]

            new_expr = rebuild_tree(self.builder_, binding.var, self.var2val, pots)

            # Rebind the original variable ID to the AST
            new_var = self.builder_.emit(new_expr, name_hint=binding.var.name_hint)
            self.set_var_remap(binding.var.vid, new_var)
            return

        super().visit_var_binding_(binding)

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
