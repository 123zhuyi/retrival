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

def gen_golden(case_info: dict):

  output_desc = case_info["output_desc"][0]
  result = torch.from_numpy(np.empty(output_desc["view_shape"], dtype=output_desc["dtype"]))

  input_desc = case_info["input_desc"]
  start = input_desc[0]["value"]
  end = input_desc[1]["value"]
  steps = input_desc[2]["value"]
  torch.linspace(start, end, steps, out=result)

  return result.numpy()