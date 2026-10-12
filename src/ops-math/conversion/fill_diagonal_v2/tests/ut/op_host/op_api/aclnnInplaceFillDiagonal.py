#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import numpy
from ut_golden_common import numpy_to_torch_tensor

def gen_golden(case_info: dict):
  input_desc = case_info["input_desc"]
  self_tensor = numpy_to_torch_tensor(input_desc[0]["value"])
  scalar = input_desc[1]["value"]
  wrap = bool(input_desc[2]["value"])

  if input_desc[0]["view_shape"] != input_desc[0]["storage_shape"]:
    self_tensor = self_tensor.transpose(0, 1)

  self_tensor.fill_diagonal_(scalar, wrap)
  
  if input_desc[0]["view_shape"] != input_desc[0]["storage_shape"]:
    self_tensor = self_tensor.transpose(0, 1)

  return self_tensor.numpy()