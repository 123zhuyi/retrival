#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
  	 	
import numpy as np
  	 	
__golden__ = {
   	"kernel": {
   	    "cast": "cast_golden"
   	}
}
  	 	
_DATA_TYPE_INT_TO_STR = {
    0: 'float32',
    1: 'float16',
    2: 'int8',
    3: 'int32',
    4: 'uint8',
    6: 'int16',
    7: 'uint16',
    8: 'uint32',
    9: 'int64',
    10: 'uint64',
    11: 'double',
    12: 'bool',
    16: 'complex64',
    17: 'complex128',
    27: 'bfloat16',
    29: 'int4',
    30: 'uint1',
    33: 'complex32',
    34: 'hifloat8',
    35: 'float8_e5m2',
    36: 'float8_e4m3fn',
    40: 'float4_e2m1',
    41: 'float4_e1m2',
}
  	 	
_SPECIAL_DTYPES = ("bfloat16", "int4",
                   "float8_e5m2", "float8_e4m3fn",
                   "float4_e2m1", "float4_e1m2",
                   "hifloat8")
  	 	
def _resolve_custom_numpy_dtype(dtype_str):
    if dtype_str == "bfloat16":
        from ml_dtypes import bfloat16
        return bfloat16
    elif dtype_str == "int4":
        from ml_dtypes import int4
        return int4
    elif dtype_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2
        return float8_e5m2
    elif dtype_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    elif dtype_str == "hifloat8":
        from en_dtypes import hifloat8
        return hifloat8
    elif dtype_str == "float4_e2m1":
        from ml_dtypes import float4_e2m1
        return float4_e2m1
    elif dtype_str == "float4_e1m2":
        from ml_dtypes import float4_e1m2
        return float4_e1m2
    return None
  	 	
def cast_golden(x,
                dst_type: int,
                **kwargs):
    '''
    Kernel golden for cast.
    All the parameters follow @cast_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    dst_type_str = _DATA_TYPE_INT_TO_STR.get(dst_type, str(dst_type))
    if (x.dtype.name == "bfloat16" and dst_type_str == "hifloat8") or \
            (x.dtype.name == "hifloat8" and dst_type_str == "bfloat16"):
        np_dtype = _resolve_custom_numpy_dtype(dst_type_str)
        return x.astype(np.float32).astype(np_dtype)
    elif dst_type_str in _SPECIAL_DTYPES:
        np_dtype = _resolve_custom_numpy_dtype(dst_type_str)
        return x.astype(np_dtype)
    elif dst_type_str == "complex32":
        _shape = list(x.shape)
        x = x.reshape(_shape + [1])
        imag = np.zeros(_shape + [1], dtype=np.float16)
        res = np.concatenate((x, imag), axis=-1)
        return res
    elif dst_type_str == "bool":
        return x.astype(np.bool_)
    else:
        return x.astype(getattr(np, dst_type_str))
