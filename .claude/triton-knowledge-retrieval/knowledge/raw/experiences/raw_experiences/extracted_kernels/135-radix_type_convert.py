@triton.jit
def radix_type_convert(k):
    if tl.constexpr(k.dtype == tl.int8):
        ik = k.to(tl.int8, bitcast=True)
        mask = (ik >> 7) & 0x1
        o = tl.where(mask, ik & 0x7F, ik | 0x80)
    elif tl.constexpr(k.dtype == tl.int16):
        ik = k.to(tl.int16, bitcast=True)
        mask = (ik >> 15) & 0x1
        o = tl.where(mask, ik & 0x7FFF, ik | 0x8000)
    elif tl.constexpr(k.dtype == tl.int32):
        ik = k.to(tl.int32, bitcast=True)
        mask = (ik >> 31) & 0x1
        o = tl.where(mask, ik & 0x7FFFFFFF, ik | 0x80000000)
    elif tl.constexpr(k.dtype == tl.int64):
        ik = k.to(tl.int64, bitcast=True)
        mask = (ik >> 63) & 0x1
        o = tl.where(mask, ik & 0x7FFFFFFFFFFFFFFF, ik | 0x8000000000000000)
    else:
        o = k
    return o