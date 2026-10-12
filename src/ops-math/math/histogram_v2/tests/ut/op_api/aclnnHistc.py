#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import numpy as np
from ut_golden_common import numpy_to_torch_tensor


def gen_golden(case_info: dict):
    input_desc = case_info["input_desc"]
    input0 = input_desc[0]
    bins = input_desc[1]["value"]
    minVal = input_desc[2]["value"]
    maxVal = input_desc[3]["value"]

    output_desc = case_info["output_desc"][0]
    result = torch.from_numpy(np.empty(output_desc["view_shape"], dtype=output_desc["dtype"]))

    input0_t = numpy_to_torch_tensor(input0["value"])

    if input0["view_shape"] != input0["storage_shape"]:
        input0_t = torch.from_numpy(np.copy(input0["value"]).transpose((0,1))).transpose(1,0)
    else:
        input0_t = torch.from_numpy(input0["value"])

    torch.histc(input0_t, bins=bins, min=minVal, max=maxVal)
    return result