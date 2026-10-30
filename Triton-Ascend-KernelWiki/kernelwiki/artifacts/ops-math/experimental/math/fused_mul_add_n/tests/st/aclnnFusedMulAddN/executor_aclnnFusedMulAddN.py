#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------
"""ATK golden executor for FusedMulAddN.

Operator semantics (truth source = AscendC kernel):
    y_i = x1_i * x3[0] + x2_i
  - x1 / x2 / y : same shape, same dtype, ND
  - x3          : single-element scalar tensor (ShapeSize == 1), broadcast as a scalar
  - dtypes      : float32 / float16 / bfloat16 / int32 / int16

This file registers ONLY the golden (cpu reference). The device-under-test (pyaclnn) side is
resolved by the built-in AclnnBaseApi via the case json's ``aclnn_name: FusedMulAddN``
(-> aclnnFusedMulAddNGetWorkspaceSize / aclnnFusedMulAddN), with the trailing ``y`` output
auto-allocated from this golden's output shape/dtype.

Golden policy:
  - float dtypes  -> compute in fp32 (higher-precision reference); single_bm compares the
                     low-precision NPU output against this fp32 truth with tolerance.
  - integer dtypes-> compute in the native int dtype with two-step wraparound
                     (tmp = x1*x3[0] [wrap]; y = tmp + x2 [wrap]), matching the kernel's
                     native Muls->Add truncation; int outputs are compared binary-equal.
"""

import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


@register("aclnn_fused_mul_add_n")
class AclnnFusedMulAddNApi(BaseApi):
    """Golden: y = x1 * x3[0] + x2 (x3 single-element scalar broadcast)."""

    @staticmethod
    def _inputs(input_data):
        kw = input_data.kwargs
        if kw and "x1" in kw:
            return kw["x1"], kw["x2"], kw["x3"]
        args = input_data.args
        return args[0], args[1], args[2]

    def __call__(self, input_data, with_output: bool = False):
        x1, x2, x3 = self._inputs(input_data)
        scalar0 = x3.flatten()[0]

        if x1.dtype in _INT_DTYPES:
            # Native two-step wraparound (torch integer ops wrap on overflow, like the kernel).
            return x1 * scalar0 + x2

        # Float path: compute the FMA in fp32 (matches the kernel's fp32-intermediate path), then
        # cast back to x1's dtype. The output dtype MUST equal x1's dtype: the op enforces
        # y.dtype == x1.dtype, and ATK allocates the device `y` output from this golden's dtype.
        y = x1.float() * x3.float().flatten()[0] + x2.float()
        return y.to(x1.dtype)
