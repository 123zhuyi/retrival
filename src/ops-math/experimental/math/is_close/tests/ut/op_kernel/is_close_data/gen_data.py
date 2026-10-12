#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np


def write_case(dtype, case_name, x1, x2, rtol, atol, equal_nan):
    prefix = f"{dtype}_{case_name}"
    golden = np.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan).astype(np.int8)
    np.asarray(x1).tofile(f"{prefix}_input_x1_is_close.bin")
    np.asarray(x2).tofile(f"{prefix}_input_x2_is_close.bin")
    golden.reshape(-1).tofile(f"{prefix}_golden_y_is_close.bin")


def gen_float32_equal():
    x1 = np.array([
        1.0, 1.00001, 1.2, np.nan, np.nan, np.inf, -np.inf, 0.0,
        -1.0, 100.0, 3.0, -3.0, 5.0, 5.2, -7.0, 8.0,
    ], dtype=np.float32)
    x2 = np.array([
        1.0, 1.0, 1.0, np.nan, 0.0, np.inf, np.inf, 1e-9,
        -1.000001, 100.01, 2.0, -3.2, 5.5, 5.0, -6.99999, 9.0,
    ], dtype=np.float32)
    write_case("float32", "equal", np.tile(x1, 4), np.tile(x2, 4), 1e-5, 1e-8, True)


def gen_float16_equal():
    x1 = np.array([
        0.0, 1.0, 1.01, 2.0, -3.0, 10.0, 100.0, -100.0,
        5.0, 6.0, 7.0, 8.0, -9.0, -10.0, 11.0, 12.0,
    ], dtype=np.float16)
    x2 = np.array([
        0.0, 1.0, 1.0, 2.5, -3.0, 10.01, 100.5, -101.0,
        4.99, 6.5, 7.0, 8.2, -8.9, -10.0, 13.0, 12.0,
    ], dtype=np.float16)
    write_case("float16", "equal", np.tile(x1, 4), np.tile(x2, 4), 1e-3, 1e-3, False)


def gen_float32_broadcast():
    x1 = np.array([[1.0, np.nan, np.inf, -2.0]], dtype=np.float32)
    x2 = np.array([
        [1.0, np.nan, np.inf, -2.00001],
        [1.1, 0.0, -np.inf, -1.0],
    ], dtype=np.float32)
    write_case("float32", "broadcast", x1, x2, 1e-5, 1e-8, True)


def gen_float32_broadcast_cross4d():
    x1 = np.linspace(-4.0, 4.0, num=5 * 1 * 7 * 1, dtype=np.float32).reshape(5, 1, 7, 1)
    x2 = np.linspace(-4.0, 4.0, num=1 * 3 * 1 * 9, dtype=np.float32).reshape(1, 3, 1, 9)
    write_case("float32", "broadcast_cross4d", x1, x2, 1e-5, 1e-8, False)


def gen_float32_packed_broadcast5d():
    x1 = np.linspace(-6.0, 6.0, num=2 * 1 * 3 * 1 * 8, dtype=np.float32).reshape(2, 1, 3, 1, 8)
    x2 = np.linspace(-6.0, 6.0, num=1 * 2 * 1 * 3 * 8, dtype=np.float32).reshape(1, 2, 1, 3, 8)
    write_case("float32", "packed_broadcast5d", x1, x2, 1e-5, 1e-8, False)


def gen_int32_x2_scalar():
    x1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    x2 = np.array([3], dtype=np.int32)
    write_case("int32", "x2_scalar", x1, x2, 0.0, 0.0, False)


def main(dtype, case_name):
    for file_name in os.listdir("."):
        if file_name.endswith(".bin"):
            os.remove(file_name)

    cases = {
        ("float32", "equal"): gen_float32_equal,
        ("float16", "equal"): gen_float16_equal,
        ("float32", "broadcast"): gen_float32_broadcast,
        ("float32", "broadcast_cross4d"): gen_float32_broadcast_cross4d,
        ("float32", "packed_broadcast5d"): gen_float32_packed_broadcast5d,
        ("int32", "x2_scalar"): gen_int32_x2_scalar,
    }
    generator = cases.get((dtype, case_name))
    if generator is None:
        raise ValueError(f"unsupported dtype/case: {dtype}/{case_name}")
    generator()


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: gen_data.py <dtype> [case_name]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "equal")
