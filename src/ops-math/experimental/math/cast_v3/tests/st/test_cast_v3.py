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
import torch
import torch_npu  # noqa: F401


def test_cast_v3_basic():
    shapes = [(128,), (1024, 1024), (32, 64, 128), (1, 3, 224, 224)]
    dtypes = [
        (torch.float32, torch.float16),
        (torch.float16, torch.float32),
        (torch.float32, torch.int32),
        (torch.int32, torch.float32),
        (torch.int8, torch.float16),
        (torch.uint8, torch.float16),
        (torch.bool, torch.float16),
        (torch.int16, torch.float32),
        (torch.int64, torch.float32),
    ]

    for shape in shapes:
        for src_dtype, dst_dtype in dtypes:
            if src_dtype == torch.bool:
                x_cpu = torch.randint(0, 2, shape, dtype=src_dtype)
            elif src_dtype.is_floating_point:
                x_cpu = torch.randn(shape, dtype=src_dtype)
            elif src_dtype == torch.uint8:
                x_cpu = torch.randint(0, 255, shape, dtype=src_dtype)
            else:
                x_cpu = torch.randint(-128, 127, shape, dtype=src_dtype)

            x_npu = x_cpu.npu()
            y_cpu = x_cpu.to(dst_dtype)
            y_npu = x_npu.to(dst_dtype)

            if dst_dtype.is_floating_point:
                atol = 1e-3 if dst_dtype == torch.float16 else 1e-4
                rtol = 1e-3 if dst_dtype == torch.float16 else 1e-4
                ok = torch.allclose(y_npu.cpu(), y_cpu, atol=atol, rtol=rtol)
            else:
                ok = torch.equal(y_npu.cpu(), y_cpu)

            print(f"shape={shape}, {src_dtype} -> {dst_dtype}: {ok}")
            assert ok, f"Failed for shape={shape}, {src_dtype} -> {dst_dtype}"


def test_cast_v3_bf16():
    shape = (1024, 1024)
    x_fp32 = torch.randn(shape, dtype=torch.float32)
    x_npu = x_fp32.npu()

    y_bf16_npu = x_npu.to(torch.bfloat16)
    y_bf16_cpu = x_fp32.to(torch.bfloat16)
    print(
        f"fp32 -> bf16: {torch.allclose(y_bf16_npu.cpu().float(), y_bf16_cpu.float(), atol=1e-2, rtol=1e-2)}"
    )

    y_back_fp32_npu = y_bf16_npu.to(torch.float32)
    y_back_fp32_cpu = y_bf16_cpu.to(torch.float32)
    print(
        f"bf16 -> fp32: {torch.allclose(y_back_fp32_npu.cpu(), y_back_fp32_cpu, atol=1e-2, rtol=1e-2)}"
    )


def test_cast_v3_non_contiguous():
    x = torch.randn(128, 128).npu()
    x_t = x.t()
    assert not x_t.is_contiguous()
    y = x_t.to(torch.float16)
    y_cpu = x_t.cpu().to(torch.float16)
    print(
        f"non-contiguous cast: {torch.allclose(y.cpu(), y_cpu, atol=1e-3, rtol=1e-3)}"
    )


if __name__ == "__main__":
    test_cast_v3_basic()
    test_cast_v3_bf16()
    test_cast_v3_non_contiguous()
    print("All tests passed!")
