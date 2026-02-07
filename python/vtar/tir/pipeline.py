import tvm
from . import transform
from .. import dlight

def get_vtar_tir_transform() -> tvm.ir.transform.Pass:
    # Some documentation for old transformations is still available here
    # https://mlc.ai/docs/reference/api/tir/transform.html
    return tvm.transform.Sequential([
        tvm.tir.transform.LowerInitBlock(), # Needed for TVM functions that won't go on VTA
        # Taken from tvm.tir.get_default_tir_pipeline in pipeline.py ###########
        tvm.tir.transform.PlanAndUpdateBufferAllocationLocation(),
        tvm.tir.transform.ConvertBlocksToOpaque(),
        tvm.tir.transform.CompactBufferAllocation(),
        tvm.tir.transform.LowerMatchBuffer(),
        tvm.tir.transform.Simplify(),
        tvm.tir.transform.LowerOpaqueBlock(), # This together with ConvertBlocksToOpaque removes Block and BlockRealize nodes.
        tvm.tir.transform.FlattenBuffer(),
        tvm.tir.transform.InjectVirtualThread(),
        tvm.tir.transform.Simplify(),
        transform.LoopFission(),
        ########################################################################
        # transform.InjectConv2DTransposeSkip(), # TODO
        transform.InjectDMAIntrin2(),
        # transform.InjectSkipCopy(), # TODO: Just for debug, requires adding remove nop transform tvm.tir.transform.RemoveNoOp(),
        transform.AnnotateALUCoProcScope(),
        transform.LiftAttrScope("coproc_uop_scope"),
        # transform.LiftAllocToScopeBegin(), # NOTE: this "messes up" the allocated nodes and StorageRewrite fails because it can't find them
        transform.LiftAttrScope("coproc_scope"),
        transform.LiftAttrScope("extern_scope"),
        transform.InjectCoProcSync(), # NOTE: this was probably just used for development
        transform.ReplaceVTAVar(),
        transform.CoProcSync(), # This inserts the coproc_(dep_push|dep_pop|sync|read_barrier|write_barrier)
        tvm.tir.transform.StorageRewrite(),
        transform.InjectDebug(),
        transform.InjectALUIntrin(),
        tvm.tir.transform.LowerDeviceStorageAccessInfo(),
        transform.FoldUopLoop(),
        transform.CPUAccessRewrite(),
        tvm.tir.transform.AnnotateEntryFunc(),
        tvm.ir.transform.PrintIR(),
        tvm.tir.transform.MakePackedAPI(),
    ])

def get_actual_pipeline():
    return tvm.transform.Sequential([
        tvm.tir.transform.ForceNarrowIndexToInt32(),
        tvm.dlight.ApplyDefaultSchedule(
            dlight.Conv2D(),
        ),
        get_vtar_tir_transform(),
    ])
