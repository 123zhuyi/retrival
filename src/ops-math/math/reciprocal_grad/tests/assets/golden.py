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

__spec__ = {"reciprocal_grad": "ReciprocalGradTestSpec"}


class ReciprocalGradTestSpec:
    """reciprocal_grad 算子测试规范（kernel 流程，双标杆 cross_check）。

    算子公式: z = -y * y * dy
    golden / third_party 均使用 TensorFlow 实现。

    精度管理（对齐 NPU kernel 的 CastNoSatVF + CAST_RINT 行为）:
      - golden (Promote): cross_check 强制 Promote，框架已把 fp16/bf16→fp32、
        fp32→fp64 提升后调 golden；原始 dtype 仅能从 testcase_name 推断。
        输出必须降回原始 dtype —— cross_check Path A 要求 NPU 与 golden 的
        NaN/±Inf 逐位一致；且不降精度会走 __normalize_goldens 的 >65504→Inf
        阈值，RINT 边界值(65504~65520)会被误判为 Inf，与 NPU 的 65504 不符。
      - golden (非 Promote): 输入为原始 dtype，直接读 y.dtype 升精度计算后降回。
      - TfImpl: XPU 在 golden 之后执行（Promote 作用域已退出），收到的始终是
        原始 dtype 输入，可直接读 y.dtype 做升降精度。
    """

    def golden(y, dy, **kwargs):
        from tensorflow.python.ops import gen_math_ops
        import tensorflow as tf

        if kwargs.get("golden_mode") == "Promote":
            result = gen_math_ops.reciprocal_grad(y=y, dy=dy).numpy()
            name = kwargs.get("testcase_name", "")
            if "_bf16_" in name or "_bfloat16_" in name:
                result = result.astype(tf.bfloat16.as_numpy_dtype, copy=False)
            elif "_float16_" in name or "_fp16_" in name:
                result = result.astype(numpy.float16, copy=False)
            elif "_float32_" in name or "_fp32_" in name:
                result = result.astype(numpy.float32, copy=False)
            return [result]

        ori_dtype = y.dtype
        if str(ori_dtype) in ("float16", "bfloat16"):
            y = y.astype(numpy.float32)
            dy = dy.astype(numpy.float32)
        result = gen_math_ops.reciprocal_grad(y=y, dy=dy).numpy()
        return [result.astype(ori_dtype, copy=False)]

    class TfImpl:
        """tf impl — 在 XPU server 上执行，入参为 tf.Tensor（原始 dtype）。"""

        def __call__(self, y, dy, **kwargs):
            from tensorflow.python.ops import gen_math_ops
            import tensorflow as tf

            ori_dtype = y.dtype
            if ori_dtype in (tf.bfloat16, tf.float16):
                y = tf.cast(y, tf.float32)
                dy = tf.cast(dy, tf.float32)
            result = gen_math_ops.reciprocal_grad(y=y, dy=dy)
            return [tf.cast(result, ori_dtype)]

    third_party = {"tf": TfImpl}

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L0"},
        "float32": {"standard": "cross_check", "level": "L0"},
        "bfloat16": {"standard": "cross_check", "level": "L0"},
    }
