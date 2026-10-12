#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import glob
import numpy as np
from ml_dtypes import bfloat16


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)

    # 从 JSON 第一个 case 获取参数
    d_type = "float16"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]

    # 生成输入数据
    input_x = np.ones((3, 5)).astype(d_type_dict["float16"])
    input_target = np.ones((3, 5)).astype(d_type_dict["float16"])
    attr_reduction = "mean"
    attr_log_target = False

    # 计算 golden 数据 (KL divergence: target * (log(target) - x))
    x_f = input_x.astype(np.float64)
    t_f = input_target.astype(np.float64)
    if attr_log_target:
        pointwise = np.exp(t_f) * (t_f - x_f)
    else:
        pointwise = np.where(t_f > 0, t_f * (np.log(t_f) - x_f), 0.0)
    if attr_reduction == "mean":
        golden = np.array([pointwise.mean()])
    elif attr_reduction == "sum":
        golden = np.array([pointwise.sum()])
    elif attr_reduction == "batchmean":
        batch = t_f.shape[0] if t_f.ndim > 0 else 1
        golden = np.array([pointwise.sum() / batch])
    else:
        golden = pointwise

    # 保存数据到文件
    input_x.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_kl_div_v2_x.bin")
    input_target.astype(d_type_dict["float16"]).tofile(
        f"{d_type}_input_kl_div_v2_target.bin"
    )
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float16_golden_kl_div_v2.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float16"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float16"]).tofile("float16_golden_kl_div_v2.bin")

    print(f"生成完成: dtype={d_type}")
