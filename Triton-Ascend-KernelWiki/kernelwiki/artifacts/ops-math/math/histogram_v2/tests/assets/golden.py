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
import numpy

__golden__ = {
  	"kernel": {
  	    "histogram_v2": "histogram_v2_golden"
  	}
}

def histogram_v2_golden(x, min, max, *, bins=100, y_dtype=3, **kwargs):
    '''
    Kernel golden for histogram_v2.
    All the parameters follow @histogram_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
	    input_formats, output_formats, input_ori_formats, output_ori_formats,
	    input_dtypes, output_dtypes.
    '''
    output_dtypes = kwargs.get("output_dtypes", ["int32"])
    out_dtype = output_dtypes[0] if output_dtypes else "int32"

    if out_dtype not in ("float32", "int32"):
        print("out dtype error")

    min_data = min[0]
    max_data = max[0]
    x_dtype = x.dtype.name

    if x_dtype in ("float16", "float32"):
        min_data = numpy.float32(min_data)
        max_data = numpy.float32(max_data)
        x = x.astype(numpy.float32)
        bins_val = numpy.float32(bins)
    elif x_dtype in ("int8", "uint8", "int16"):
        min_data = numpy.int32(min_data)
        max_data = numpy.int32(max_data)
        x = x.astype(numpy.int32)
        bins_val = numpy.int32(bins)
    else:
        min_data = numpy.int64(min_data)
        max_data = numpy.int64(max_data)
        x = x.astype(numpy.int64)
        bins_val = numpy.int32(bins)

    res = numpy.zeros((bins,), dtype=out_dtype)
    for value in x.reshape(-1):
        if min_data <= value <= max_data:
            index = int((value - min_data) * bins_val / (max_data - min_data))
            if index == bins:
                index -= 1
            res[index] += 1

    return res
