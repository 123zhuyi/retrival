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
"""TTK TestSpec for SquareSumAll kernel and GEIR tests.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/arch35/square_sum_all.cpp 有 arch35 实现 |
| geir   | ✅   | op_graph/square_sum_all_proto.h 有 REG_OP(SquareSumAll) |
| aclnn  | ❌   | CMakeLists.txt 配置 ACLNNTYPE aclnn_exclude，无 op_api 目录，无 aclnn 文档 |
| e2e    | ❌   | libtorch_npu.so 中 `aclnnSquareSumAll` 符号数为 0；对照组 `aclnnAdd*` 有命中，证明该判定方法有效 |

kernel 与 geir 共用一个注册键（算子蛇形名），geir 不另写 Spec 类。

判据（tolerance）为什么是 cross_check：
    cross_check 是 TTK 里唯一会去取 third_party 输出来判精度的标准。配任何别的标准，
    third_party 照样会被执行、xpu_metrics 照样有 device_us，但三方数据不进判定——
    "三方精度"那一栏实际上只是两方（NPU vs golden）。本算子交付三方精度腿，
    因此 Spec 默认判据必须是 cross_check。

    代价：cross_check 拿不到三方输出时判 GOLDEN_FAILURE。泛化档（1000 例）与性能档
    （200 例）不接远端 XPU，跑这两档时必须在命令行显式覆盖判据：
        python3 -m ttk kernel ... --compare close
    优先级是 CLI > Spec.tolerance > 默认（见 ttk/core_modules/comparison/resolve.py
    的 _float_choice），所以 CLI 覆盖是受支持的正规用法，不是绕过。

    为什么这两档选 close(isclose) 而不是 stat_rel_err：判据完全由用例 CSV 驱动——
    `precision_tolerances` 给 (rtol=2**-10, ptol=0.01)、`absolute_precision` 给
    2**-16，isclose 按 |o-g| <= atol + rtol*|g| 逐元素判、再按 ptol 卡匹配率；且它是
    少数会把**匹配率百分比**写进结果 CSV 的标准（stat_rel_err / cross_check 只写
    PASS/FAIL），泛化档要逐例呈现精度百分比，靠的就是这条。

    与原先自写 compare 的差异（不是等价替换，如实记）：混合容差、99% 匹配率、
    NaN/同号 Inf 三条一致；原先另有一条与量级无关的绝对上限 max_abs_error <= 1e-2，
    isclose 没有对应项。该上限并不是更好的判据——它不随量级缩放，当 |y| 超过约
    8.4e4 时 1 个 fp32 ulp 已大于 1e-2，等于要求逐位相等；平方和的量级随 N 线性增长，
    这条迟早会在合法用例上失效。改用纯相对判据是有意为之，但对大量级输出确实比原先松。

本 Spec 不实现 compare 钩子：
    TTK 的 try_custom_compare 只要拿到非 None 返回值就直接采信，**跳过标准判据**
    （ttk/core_modules/comparison/custom.py）。自写 compare 会把 cross_check 一起短路掉，
    三方数据永远进不了判定。特殊值（NaN / 同号 Inf）的语义比对 stat_rel_err 与
    cross_check 自身都已实现，不需要算子再包一层。
"""

import numpy as np
import torch

__spec__ = {
    # Kernel and GEIR share the snake-case registration and the same TestSpec.
    "square_sum_all": "SquareSumAllTestSpec",
}

# Retain the repository-facing legacy entry while consumers migrate to TestSpec.
__golden__ = {
    "kernel": {"square_sum_all": "square_sum_all_golden"},
}


def _reduce_one(x):
    """Reduce a single input with torch competitor interfaces (R3: no hand-rolled formula).

    torch.square + torch.sum are used instead of a numpy expression so the golden and the
    kernel cannot share the same misreading of the semantics. Accumulating in float64 is a
    deliberate tier choice for this large-reduction operator rather than a precision cast:
    the kernel accumulates in float32, so a float32 golden would sit on the same error
    floor and could not flag a real regression.
    """
    # Widening to float64 in numpy always yields a fresh writable buffer, so from_numpy
    # never has to warn about a read-only input array.
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    reduced = torch.sum(torch.square(tensor))
    # Keep the high-precision reference in FP64.  In cross_check mode TTK promotes
    # FP32 inputs to FP64 before invoking the golden; casting the result back to
    # FP32 would quantize the reference onto the same grid as the NPU/GPU outputs
    # and make scalar error ratios collapse to integer ULP ratios.
    return reduced.numpy().reshape(1)


def square_sum_all_golden(x1, x2, **kwargs):
    """Compute and return both independent reductions in float64."""
    del kwargs
    return [_reduce_one(x1), _reduce_one(x2)]


class _Compose:
    """Competitor baseline built from a torch interface independent of the golden.

    torch.dot is a different high-level entry point than the golden's
    square-then-sum, so a misreading in one path does not reproduce in the other.
    Parameter names follow the REG_OP registration (x1, x2).

    ⚠️ 累加档位必须与【内核实现】一致，即 float32——不是与 golden 一致。
    内核在 float32 上做核内累加与跨核合并；竞品若像 golden 那样升 float64，它对
    golden 的相对误差会趋近 0。f419709 后 cross_check 按 dtype 的 small_value
    给比值分母夹底（FP32 为 2**-14）；这能消除旧版固定 1e-7 的一部分假红，但
    标量输出仍会把误差比量化成少数 ULP 的比例。因此三方 compose 仍须和内核
    处于同一精度档位。输出 dtype 同样 cast 回 NPU 的 float32。
    """

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, x1, x2, **kwargs):
        del kwargs
        outputs = []
        for tensor in (x1, x2):
            flat = tensor.reshape(-1).to(torch.float32)
            outputs.append(torch.dot(flat, flat).to(torch.float32).reshape(1))
        return outputs


class SquareSumAllTestSpec:
    """SquareSumAll test contract shared by the TTK kernel and GEIR paths."""

    @staticmethod
    def golden(x1, x2, **kwargs):
        return square_sum_all_golden(x1, x2, **kwargs)

    third_party = {"torch": _Compose}
    # cross_check L1：mare<=5 / mere<=1.5 / rmse<=1.5（见 resolve.LEVEL_PRESETS）。
    # 取 L1 而非 L2：内核按分核结果合并，累加顺序与竞品的单核归约不同，比值天然有
    # 抖动；平方和全为非负项、无对消，条件数好，L1 足以拦住真实劣化。
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}


# 【不存在】aclnn 通路：CMakeLists.txt 配置 ACLNNTYPE aclnn_exclude，算子无 op_api 目录，
# 也不交付 docs/aclnnSquareSumAll.md，因此不注册 SquareSumAllAclnnSpec。
# 【不存在】e2e 通路：strings libtorch_npu.so | grep -c aclnnSquareSumAll = 0
# （对照组 aclnnAdd* 有命中，证明该判定方法有效；legacy SquareSumAll 动态实现于 2021-07-05 入仓，
# 当前 torch_npu 安装于 2026-07-30，版本晚于算子既有接口），
# torch 侧没有任何接口会执行本算子，因此不注册 SquareSumAllTorchSpec。
# 上述两条不是漏写，勿反复重查。
