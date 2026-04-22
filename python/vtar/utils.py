import numpy
from typing import Tuple

def internal_scale_zero_point(
    clip_min: numpy.ndarray,
    clip_max: numpy.ndarray,
    dtype_min: int,
    dtype_max: int,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    scale = (clip_max - clip_min)/(dtype_max - dtype_min)
    zero_point = numpy.round(dtype_min - clip_min/scale)
    zero_point = numpy.clip(zero_point, dtype_min, dtype_max)
    return scale, zero_point

def asymmetric_scale_zero_point(
    clip_min: numpy.ndarray,
    clip_max: numpy.ndarray,
    dtype: str,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    iinfo = numpy.iinfo(dtype)
    dtype_min, dtype_max = iinfo.min, iinfo.max
    return internal_scale_zero_point(clip_min, clip_max, dtype_min, dtype_max)

def symmetric_scale_zero_point(
    clip_min: numpy.ndarray,
    clip_max: numpy.ndarray,
    dtype: str,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    clip_val = numpy.maximum(numpy.abs(clip_min), numpy.abs(clip_max))
    clip_min, clip_max = -clip_val, clip_val
    iinfo = numpy.iinfo(dtype)
    is_signed = dtype[0] == "i"
    # TODO: verify if ONNX runtime does the same thing.
    offset = 1 if is_signed else 0
    dtype_min, dtype_max = iinfo.min + offset, iinfo.max
    return internal_scale_zero_point(clip_min, clip_max, dtype_min, dtype_max)

# TODO: write test for this function.
def get_strictly_power_of_two(arr) -> Tuple[numpy.ndarray, numpy.ndarray]:
    x = numpy.asanyarray(arr, dtype=numpy.float32)
    x_bits = x.view(numpy.int32)

    SIGN_MASK     = numpy.asarray(-2147483648, dtype='int32') # 0x80000000
    EXPONENT_MASK = 0x7F800000
    MANTISSA_MASK = 0x007FFFFF
    EXPONENT_SIZE_MASK = 0xFF
    MANTISSA_SIZE = 23
    BIAS = 127

    # https://it.wikipedia.org/wiki/IEEE_754#Precisione_singola_(32_bit)
    not_inf_or_nan = (x_bits < EXPONENT_MASK)
    not_neg_inf_or_nan = (x_bits > 0) & not_inf_or_nan
    has_zero_mantissa = (x_bits & (MANTISSA_MASK | SIGN_MASK)) == 0
    is_pow2 = not_neg_inf_or_nan & has_zero_mantissa

    powers = ((x_bits >> MANTISSA_SIZE) & EXPONENT_SIZE_MASK) - BIAS

    return is_pow2, powers
    # return numpy.where(is_pow2, powers, numpy.nan)

def re_quantize(
    s_x: numpy.ndarray, s_w: numpy.ndarray, s_y: numpy.ndarray,
    q_w: numpy.ndarray, z_w: numpy.ndarray, q_b: numpy.ndarray,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    M = s_w*s_x/s_y
    n = numpy.ceil(numpy.log2(M))
    M_star = numpy.exp2(n)
    assert ((1 <= M_star/M) & (M_star/M < 2)).all()
    s_star_w = M_star/M*s_w
    assert (s_star_w >= s_w).all()
    mm = s_w/s_star_w
    q_star_w = mm * (q_w - z_w) + z_w
    q_star_b = mm * q_b

    int8, int32 = numpy.int8, numpy.int32
    ii8, ii32 = numpy.iinfo(int8), numpy.iinfo(int32)
    ii8_min, ii8_max = -128, 127
    ii32_min, ii32_max = -2**31, 2**31-1

    res_n = n.astype(int32)
    res_w = numpy.clip(numpy.round(q_star_w), ii8_min, ii8_max).astype(int8)
    res_b = numpy.clip(numpy.round(q_star_b), ii32_min, ii32_max).astype(int32)

    return res_n, res_w, res_b

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
