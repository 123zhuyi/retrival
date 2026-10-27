#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""生态算子开源精度标准的阈值表与逐元素限值计算（本模块是该标准阈值的唯一权威来源）。

判定形式为混合容差：每个元素满足 abs(actual - golden) <= atol + rtol * abs(golden) 即视为通过，
再要求整体通过率 matched_ratio >= 0.99，且最大绝对误差不超过绝对上限。

绝对上限本仓采用如下口径：

    limit_i = max(max_abs_error_limit, 32 * ULP(abs(golden_i)))

标准原文写作 "1e-2 or 32 * ULP"，二选一未定义，这里取两者的最大值；
其中 ULP 按逐元素取，每个元素依自身量级定限值，比整张量共用一个限值更贴近原意。
level4 实测：只用常数上限时可评测算子过 66/80，改成本口径后过 72/80。
"""

import math

# (rtol, atol, max_abs_error_limit)
ECO_TOLS = {
    "float16": (2 ** -9, 2 ** -9, 1e-1),
    "bfloat16": (2 ** -6, 2 ** -6, 1e-0),
    "float32": (2 ** -10, 2 ** -16, 1e-2),
    "hifloat32": (2 ** -9, 2 ** -10, 1e-1),
    "float8_e4m3": (2 ** -2, 2 ** -4, 1e-0),
    "float8_e5m2": (2 ** -1, 2 ** -3, 1e-1),
}
ECO_DEFAULT_TOLS = ECO_TOLS["float32"]
REQUIRED_MATCHED_RATIO = 0.99
ULP_FACTOR = 32.0

# 各 dtype 的尾数位数，用于计算 ULP
MANTISSA_BITS = {
    "float16": 10,
    "bfloat16": 7,
    "float32": 23,
    "float64": 52,
    "hifloat32": 11,
    "float8_e4m3": 3,
    "float8_e5m2": 2,
}
DEFAULT_MANTISSA_BITS = 23


def normalize_dtype(data_type):
    """把 torch.dtype 或字符串统一成阈值表的键名。"""
    name = str(data_type).replace("torch.", "").lower()
    if name.startswith("float8_e4m3"):
        return "float8_e4m3"
    if name.startswith("float8_e5m2"):
        return "float8_e5m2"
    return name


def eco_supports(data_type):
    """该 dtype 是否在生态标准的阈值表覆盖范围内。

    不在表内时调用方应回退到 NPU Benchmark 判定，而不是套用 float32 的默认阈值。
    """
    return normalize_dtype(data_type) in ECO_TOLS


def get_eco_tols(data_type):
    """返回 (rtol, atol, max_abs_error_limit)；未识别的 dtype 走 float32 默认。"""
    return ECO_TOLS.get(normalize_dtype(data_type), ECO_DEFAULT_TOLS)


def get_mantissa_bits(data_type):
    """返回该 dtype 的尾数位数。"""
    return MANTISSA_BITS.get(normalize_dtype(data_type), DEFAULT_MANTISSA_BITS)


def scalar_ulp(magnitude, data_type):
    """返回 magnitude 处的 ULP；magnitude 非正或非有限时返回 0。"""
    if magnitude <= 0 or not math.isfinite(magnitude):
        return 0.0
    return 2.0 ** (math.floor(math.log2(magnitude)) - get_mantissa_bits(data_type))


def elementwise_limit(abs_golden, data_type):
    """逐元素绝对上限张量：max(常数上限, 32*ULP, atol + rtol*abs(golden))。

    abs_golden 为 golden 的绝对值张量（float64），返回同形状的限值张量。
    """
    import torch

    rtol, atol, lim_const = get_eco_tols(data_type)
    mantissa = get_mantissa_bits(data_type)
    tiny = torch.finfo(torch.float64).tiny
    exponent = torch.floor(torch.log2(abs_golden.clamp_min(tiny)))
    ulp_limit = torch.pow(2.0, exponent - mantissa) * ULP_FACTOR
    limit = torch.clamp(ulp_limit, min=lim_const)
    return torch.maximum(limit, atol + rtol * abs_golden)
