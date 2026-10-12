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

import sys

import numpy as np


def compare_data(dtype, case_name):
    prefix = f"{dtype}_{case_name}"
    output = np.fromfile(f"{prefix}_output_y_is_close.bin", np.int8)
    golden = np.fromfile(f"{prefix}_golden_y_is_close.bin", np.int8)
    diff_idx = np.where(output != golden)[0]
    if len(diff_idx) == 0:
        print("PASSED!")
        return True

    print("FAILED!")
    for idx in diff_idx[:5]:
        print(f"index: {idx}, output: {output[idx]}, golden: {golden[idx]}")
    return False


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: compare_data.py <dtype> [case_name]")
        sys.exit(1)
    sys.exit(0 if compare_data(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "equal") else 1)
