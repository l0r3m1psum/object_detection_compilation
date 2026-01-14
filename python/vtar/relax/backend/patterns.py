from tvm import relax

def _get_patterns():
    # There are patterns for Integer Only Arithmetic (IOA)
    # There are patterns for Simulated Quantization (SQ)

    # TODO: find a way to use SQ patterns also for graph rewriting using extract_matched_expr

    qbias = relax.dpl.is_op("relax.dequantize")(
        relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
    )
    sq_conv2d = relax.dpl.is_op("relax.nn.conv2d")(
        relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
        ),
        relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
        )
    )
    sq_conv2d = relax.dpl.is_op("relax.add")(
        sq_conv2d,
        relax.dpl.is_op("relax.reshape")(qbias, relax.dpl.wildcard()) | qbias # reshape is optional
    )
    sq_conv2d  = relax.dpl.is_op("relax.nn.relu")(sq_conv2d) | sq_conv2d # ReLU is optional
    sq_conv2d = relax.dpl.is_op("relax.quantize")(
        sq_conv2d , relax.dpl.wildcard(), relax.dpl.wildcard()
    )
    sq_conv2d = relax.dpl.is_op("relax.astype")(sq_conv2d).has_dtype("int8")

    sq_useless = relax.dpl.is_op("relax.dequantize")(
        relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
    )
    sq_useless = (relax.dpl.is_op("relax.reshape")
        | relax.dpl.is_op("relax.nn.max_pool2d"))(sq_useless)
    sq_useless = relax.dpl.is_op("relax.dequantize")(
        sq_useless, relax.dpl.wildcard(), relax.dpl.wildcard()
    )

    sq_add = relax.dpl.is_op("relax.add")(
        relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
        ),
        relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
        )
    )
    sq_add = relax.dpl.is_op("relax.nn.relu")(sq_add) | sq_add
    sq_add = relax.dpl.is_op("relax.dequantize")(
        sq_add, relax.dpl.wildcard(), relax.dpl.wildcard()
    )
    sq_add = relax.dpl.is_op("relax.astype")(sq_add).has_dtype("int8")

    sq_linear = relax.dpl.is_op("relax.matmul")(
        relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
        ),
        relax.dpl.is_op("relax.permute_dims")(
            relax.dpl.is_op("relax.dequantize")(
                relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
            )
        )
    )
    sq_linear = relax.dpl.is_op("relax.add")(
        sq_linear,
        relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
        )
    ) | sq_linear
    # TODO: add support for relax.nn.linear
    sq_linear = relax.dpl.is_op("relax.nn.relu")(sq_linear) | sq_linear
    sq_linear = relax.dpl.is_op("relax.quantize")(
        sq_linear, relax.dpl.wildcard(), relax.dpl.wildcard()
    )
    sq_linear = relax.dpl.is_op("relax.astype")(sq_linear)

    sq_avg_pool2d = relax.dpl.is_op("relax.dequantize")(
        relax.dpl.wildcard(), relax.dpl.wildcard(), relax.dpl.wildcard()
    )
    sq_avg_pool2d = relax.dpl.is_op("relax.mean")(sq_avg_pool2d)
    sq_avg_pool2d = relax.dpl.is_op("relax.quantize")(
        sq_avg_pool2d, relax.dpl.wildcard(), relax.dpl.wildcard()
    )
    sq_avg_pool2d = relax.dpl.is_op("relax.astype")(sq_avg_pool2d).has_dtype("int8")
    # TODO: add support for relax.nn.avg_pool2d

    ioa_qconv2d = relax.dpl.is_op("relax.nn.conv2d")(
        relax.dpl.wildcard().has_dtype("int8"),
        relax.dpl.wildcard().has_dtype("int8"),
    ).has_dtype("int32")
    ioa_qconv2d = relax.dpl.is_op("relax.add")(ioa_qconv2d, relax.dpl.wildcard()) | ioa_qconv2d
    ioa_qconv2d = relax.dpl.is_op("relax.right_shift")(ioa_qconv2d, relax.dpl.is_const()) | ioa_qconv2d
    ioa_qconv2d = relax.dpl.is_op("relax.add")(ioa_qconv2d, relax.dpl.is_const()) | ioa_qconv2d
    ioa_qconv2d = relax.dpl.is_op("relax.minimum")(ioa_qconv2d, relax.dpl.is_const()) | ioa_qconv2d
    ioa_qconv2d = relax.dpl.is_op("relax.maximum")(relax.dpl.is_const(), ioa_qconv2d) | ioa_qconv2d # not commutative for some reason...
    ioa_qconv2d = relax.dpl.is_op("relax.astype")(ioa_qconv2d).has_dtype("int8")

    # NOTE: probably VTA can load int8 and cast them to int32 for acc memory
    qadd_lhs = relax.dpl.is_op("relax.subtract")(
        relax.dpl.is_op("relax.astype")(relax.dpl.wildcard().has_dtype("int8")).has_dtype("int32"),
        relax.dpl.wildcard().has_dtype("int32")
    )
    qadd_lhs = relax.dpl.is_op("relax.astype")(qadd_lhs).has_dtype("float32")
    qadd_lhs = relax.dpl.is_op("relax.multiply")(relax.dpl.is_const(), qadd_lhs) # not commutative for some reason...
    qadd_rhs = relax.dpl.is_op("relax.subtract")(
        relax.dpl.is_op("relax.astype")(relax.dpl.wildcard().has_dtype("int8")).has_dtype("int32"),
        relax.dpl.wildcard().has_dtype("int32")
    )
    qadd_rhs = relax.dpl.is_op("relax.astype")(qadd_rhs).has_dtype("float32")
    qadd_rhs = relax.dpl.is_op("relax.multiply")(relax.dpl.is_const(), qadd_rhs) # not commutative for some reason...
    qadd = relax.dpl.is_op("relax.add")(qadd_lhs, qadd_rhs) # Why can't I use qadd_lhs.dup()?
    qadd = relax.dpl.is_op("relax.add")(qadd, relax.dpl.is_const())
    qadd = relax.dpl.is_op("relax.round")(qadd)
    qadd = relax.dpl.is_op("relax.minimum")(qadd, relax.dpl.is_const())
    qadd = relax.dpl.is_op("relax.maximum")(relax.dpl.is_const(), qadd) # not commutative for some reason...
    qadd = relax.dpl.is_op("relax.astype")(qadd).has_dtype("int8")

    # TODO: avg_pool
    # TODO: linear

    patterns = (
        relax.transform.FusionPattern("vtar.ioa_qconv2d", ioa_qconv2d),
        relax.transform.FusionPattern("vtar.qadd", qadd),
        # relax.transform.FusionPattern("vtar.qlinear", matcher.quant_linear_pattern),
        # relax.transform.FusionPattern("vtar.qavg_pool", matcher.quant_avg_pool_pattern),
        relax.transform.FusionPattern("vtar.sq_conv2d", sq_conv2d),
        relax.transform.FusionPattern("vtar.sq_useless", sq_useless),
        relax.transform.FusionPattern("vtar.sq_add", sq_add),
        relax.transform.FusionPattern("vtar.sq_linear", sq_linear),
        relax.transform.FusionPattern("vtar.sq_avg_pool2d", sq_avg_pool2d),
    )
    return patterns

relax.backend.pattern_registry.register_patterns(_get_patterns())
