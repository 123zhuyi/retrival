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
        "tril": "tril_golden"
    }
}


def tril_golden(x, diagonal: int = 0, **kwargs):
    '''
    Kernel golden for tril.
    All the parameters follow @tril_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    if kwargs.get('input_dtypes') and kwargs['input_dtypes'][0] == "complex32":
        real, imag = np.split(x, 2, axis=-1)
        real = np.squeeze(real, axis=-1)
        imag = np.squeeze(imag, axis=-1)
        real = np.tril(real, diagonal)
        imag = np.tril(imag, diagonal)
        golden = np.stack((real, imag), axis=-1)
    else:
        golden = np.tril(x, diagonal)
    return golden