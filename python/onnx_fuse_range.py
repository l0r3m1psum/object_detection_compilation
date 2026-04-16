import copy
from typing import Dict, Tuple, Set

import onnx

def rewrite_loop_to_arange(
    node: onnx.NodeProto,
    graph: onnx.GraphProto,
    bwd_node_map: Dict[str, onnx.NodeProto],
    init_map: Dict[str, onnx.NodeProto],
) -> None:
    """implements the following rule

        Loop(
            Cast<to: int>(Ceil(Div(Cast<to: float>(Sub(limit, start)), delta))),
            true,
            0
        ) -> Range(start, limit, delta)

    where loop also must have a certain internal structure.
    """

    loop_node = node
    if loop_node.op_type != "Loop": return

    if len(loop_node.input) != 3: return
    trip_cnt_name, cond_name, start_name = loop_node.input

    if cond_name not in init_map or start_name not in init_map: return
    cond: bool = onnx.numpy_helper.to_array(init_map[cond_name]).item()
    start: int = onnx.numpy_helper.to_array(init_map[start_name]).item()
    if not cond or start != 0: return

    if trip_cnt_name not in bwd_node_map: return
    cast_2_node = bwd_node_map[trip_cnt_name]
    if cast_2_node.op_type != "Cast": return

    if cast_2_node.input[0] not in bwd_node_map: return
    ceil_node = bwd_node_map[cast_2_node.input[0]]
    if ceil_node.op_type != "Ceil": return

    if ceil_node.input[0] not in bwd_node_map: return
    div_node = bwd_node_map[ceil_node.input[0]]
    if div_node.op_type != "Div": return

    if div_node.input[0] not in bwd_node_map: return
    cast_1_node = bwd_node_map[div_node.input[0]]
    if cast_1_node.op_type != "Cast": return

    if cast_1_node.input[0] not in bwd_node_map: return
    sub_node = bwd_node_map[cast_1_node.input[0]]
    if sub_node.op_type != "Sub": return

    loop_body = next((attr.g for attr in loop_node.attribute if attr.name == "body"), None)
    if loop_body is None: return

    # TODO: Verify body signature: (int64 i, bool cond, int32 prev) => (bool cond_out, int32 current, int32 range)
    if len(loop_body.input) != 3 or len(loop_body.output) != 3: return

    # TODO: verify the graph structure.
    op_types = [n.op_type for n in loop_body.node]
    if len(op_types) != 3 or "Add" not in op_types or op_types.count("Identity") < 2: return

    limit_name = sub_node.input[0]
    start_name = sub_node.input[1]
    delta_name = div_node.input[1]

    if limit_name == start_name or limit_name == delta_name or start_name == delta_name: return

    if len(loop_node.output) != 2: return
        
    # final_state_output = loop_node.output[0] unused
    range_output_name = loop_node.output[1]


    # At this point we have found an occurrence of the rewrite rule (an
    # homomorphishm). We just need to add the new graph and remove the
    # connection to the old root.

    range_node = onnx.helper.make_node(
        "Range",
        inputs=[start_name, limit_name, delta_name],
        outputs=[range_output_name],
        name=loop_node.name + "_fused_range"
    )

    # To keep the topological order
    i = next(i for i, node in enumerate(graph.node) if node == loop_node)
    graph.node.insert(i, range_node)

    del loop_node.output[1]
    # loop_node.output[1] += "_detached"
    bwd_node_map[range_output_name] = range_node

def reachability(
    name: str,
    bwd_node_map: Dict[str, onnx.NodeProto],
    init_map: Dict[str, onnx.TensorProto],
    unreachable_node: Set[str],
    unreachable_initializer: Set[str],
    visited: Set[str],
) -> None:

    # To avoid exponential path explosion. E.g. consider the following graph:
    #   *   *   *   *   *   *   *   *   *
    #  / \ / \ / \ / \ / \ / \ / \ / \ / \
    # *   *   *   *   *   *   *   *   *   *
    #  \ / \ / \ / \ / \ / \ / \ / \ / \ /
    #   *   *   *   *   *   *   *   *   *
    if name in visited: return
    visited.add(name)

    if name in bwd_node_map:
        unreachable_node.discard(name)
        node = bwd_node_map[name]
        for input in node.input:
            reachability(input, bwd_node_map, init_map, unreachable_node, unreachable_initializer, visited)
        return
    if name in init_map:
        unreachable_initializer.discard(name)
        return

    print("This should be the input", name)


def onnx_term_graph_rewrite(
    model: onnx.ModelProto, # g_0
    graph_rewrite_rule, # R = (g,n,n')
) -> onnx.ModelProto:
    """An implementation of Term Graph Rewriting by H. P. Barendregt et al.

    Given a graph rewrite rule R it finds all the homomorphism (or occurrences
    of the rewrite rule) f_i from g|n to g_0 and apply all the ∆_i = (R, f_i) in
    parallel. The core assumption here is that all redexes are disjoint.

    Consider the rule S(S(X)) -> B(X) and the term graph S(S(S(A))) here
    application of the rule can result in either B(S(A)) or S(B(A)) hence the
    rule can't be applied in parallel.

    To avoid this problem lab(n) shall be different from all the other labels in g.

    Parallel rewrite makes it easy to use a single for loop pas to rewrite the
    graph instead of reaching a fixed point with a while.
    """

    new_model = copy.deepcopy(model)
    new_graph: onnx.GraphProto = new_model.graph


    bwd_node_map: Dict[str, onnx.NodeProto] = {
        output_name: node
        for node in new_graph.node
        for output_name in node.output
    }
    init_map: Dict[str, onnx.TensorProto] = {
        init.name: init for init in new_graph.initializer
    }

    # We use list because we modify the container while iterating
    for node in list(new_graph.node):
        graph_rewrite_rule(node, new_graph, bwd_node_map, init_map)


    unreachable_node = set(bwd_node_map.keys())
    unreachable_initializer = set(init_map.keys())
    # TODO: if a path from the output does not reach the input or an initializer
    # all nodes along that path can be deleted.
    visited: Set[str] = set()
    for output in new_graph.output:
        reachability(output.name, bwd_node_map, init_map, unreachable_node, unreachable_initializer, visited)

    print(unreachable_node)

    to_delete: List[int] = []
    for i, node in enumerate(new_graph.node):
        # TODO: understand the difference between node.name and Tensor Names
        if node.output and all(out_tensor in unreachable_node for out_tensor in node.output):
            to_delete.append(i)

    for i in reversed(to_delete): del new_graph.node[i]

    to_delete = []
    for i, initializer in enumerate(new_graph.initializer):
        if initializer.name in unreachable_initializer:
            to_delete.append(i)

    for i in reversed(to_delete): del new_graph.initializer[i]

    onnx.checker.check_model(new_model)
    return new_model

model = onnx.load(r"C:\Users\Diego\Downloads\yolov3-12-int8.onnx")
new_model = onnx_term_graph_rewrite(model, rewrite_loop_to_arange)
onnx.save(new_model, r"C:\Users\Diego\Downloads\yolov3-range-12-int8.onnx")

import vtar.relax.frontend.onnx
mod = vtar.relax.frontend.onnx.from_onnx(new_model)
mod.show()
