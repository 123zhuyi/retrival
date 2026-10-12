#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import sys
import numpy as np
from ml_dtypes import bfloat16
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def get_threshold_by_dtype(dtype):
    """
    根据数据类型获取通过阈值（社区标准）

    精度标准:
    - FLOAT16: threshold = 2^-10 ≈ 0.000977
    - BFLOAT16: threshold = 2^-7 ≈ 0.00781
    - FLOAT32: threshold = 2^-13 ≈ 0.000122
    - HiFLOAT32: threshold = 2^-11 ≈ 0.000488
    - FLOAT8 E4M3: threshold = 2^-3 ≈ 0.125
    - FLOAT8 E5M2: threshold = 2^-2 ≈ 0.25
    """
    dtype_str = str(dtype).lower().replace(" ", "").replace("_", "")

    thresholds = {
        "float16": 2 ** (-10),
        "bfloat16": 2 ** (-7),
        "float32": 2 ** (-13),
        "float64": 2 ** (-13),
        "hifloat32": 2 ** (-11),
    }

    if "float8e4m3" in dtype_str or "fp8e4m3" in dtype_str:
        return 2 ** (-3)
    elif "float8e5m2" in dtype_str or "fp8e5m2" in dtype_str:
        return 2 ** (-2)

    return thresholds.get(dtype_str, 2 ** (-13))


def calculate_mare(actual, golden):
    """计算最大相对误差(MARE)"""
    relative_errors = np.abs(actual - golden) / np.maximum(np.abs(golden), 1e-6)
    return np.max(relative_errors)


def calculate_mere(actual, golden):
    """计算平均相对误差(MERE)"""
    relative_errors = np.abs(actual - golden) / np.maximum(np.abs(golden), 1e-6)
    return np.mean(relative_errors)


def compare_data_float(golden_file_lists, output_file_lists, d_type):
    """
    浮点类型精度比对

    通过标准:
    - MERE < threshold
    - MARE < 10 * threshold
    """
    if d_type == "float16":
        np_dtype = np.float16
    elif d_type == "float32":
        np_dtype = np.float32
    elif d_type == "float64":
        np_dtype = np.float64
    elif d_type == "bfloat16":
        np_dtype = bfloat16
    elif d_type == "hifloat32":
        np_dtype = np.float32
    elif d_type in ("fp8_e4m3fn", "fp8_e5m2"):
        np_dtype = np.uint8
    else:
        np_dtype = np.float32

    threshold = get_threshold_by_dtype(d_type)
    mare_threshold = 10 * threshold

    def _read_bin(path):
        raw = np.fromfile(path, np_dtype)
        if d_type == "fp8_e4m3fn":
            from ml_dtypes import float8_e4m3fn

            return raw.view(float8_e4m3fn).astype(np.float32)
        elif d_type == "fp8_e5m2":
            from ml_dtypes import float8_e5m2

            return raw.view(float8_e5m2).astype(np.float32)
        return raw

    data_same = True
    # 当 golden 文件数 < output 文件数时，将多个 output 拼接后与单个 golden 比对
    if len(golden_file_lists) == 1 and len(output_file_lists) > 1:
        tmp_gold = _read_bin(golden_file_lists[0])
        tmp_out = np.concatenate([_read_bin(f) for f in output_file_lists])
        mere = calculate_mere(tmp_out, tmp_gold)
        mare = calculate_mare(tmp_out, tmp_gold)

        mere_pass = mere < threshold
        mare_pass = mare < mare_threshold
        is_pass = mere_pass and mare_pass

        if is_pass:
            print(f"PASSED! MERE={mere:.6f}, MARE={mare:.6f}")
        else:
            print(
                f"FAILED! MERE={mere:.6f} (threshold={threshold:.6f}), MARE={mare:.6f} (threshold={mare_threshold:.6f})"
            )
            diff = np.abs(tmp_out - tmp_gold)
            diff_idx = np.argsort(diff)[-5:][::-1]
            for idx in diff_idx:
                print(
                    f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}"
                )
            data_same = False
    else:
        for gold, out in zip(golden_file_lists, output_file_lists):
            tmp_out = _read_bin(out)
            tmp_gold = _read_bin(gold)

            mere = calculate_mere(tmp_out, tmp_gold)
            mare = calculate_mare(tmp_out, tmp_gold)

            mere_pass = mere < threshold
            mare_pass = mare < mare_threshold
            is_pass = mere_pass and mare_pass

            if is_pass:
                print(f"PASSED! MERE={mere:.6f}, MARE={mare:.6f}")
            else:
                print(
                    f"FAILED! MERE={mere:.6f} (threshold={threshold:.6f}), MARE={mare:.6f} (threshold={mare_threshold:.6f})"
                )
                diff = np.abs(tmp_out - tmp_gold)
                diff_idx = np.argsort(diff)[-5:][::-1]
                for idx in diff_idx:
                    print(
                        f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}"
                    )
                data_same = False
    return data_same


def compare_data_integer(golden_file_lists, output_file_lists, d_type):
    """
    整数类型精度比对

    通过标准: 二进制一致 或 绝对误差为0
    """
    if d_type == "int32":
        np_dtype = np.int32
    elif d_type == "int8":
        np_dtype = np.int8
    elif d_type == "int16":
        np_dtype = np.int16
    elif d_type == "int64":
        np_dtype = np.int64
    elif d_type == "uint8":
        np_dtype = np.uint8
    elif d_type == "uint16":
        np_dtype = np.uint16
    elif d_type == "uint32":
        np_dtype = np.uint32
    elif d_type == "uint64":
        np_dtype = np.uint64
    elif d_type == "bool":
        np_dtype = np.bool_
    else:
        np_dtype = np.int32

    data_same = True
    if len(golden_file_lists) == 1 and len(output_file_lists) > 1:
        tmp_gold = np.fromfile(golden_file_lists[0], np_dtype)
        tmp_out = np.concatenate([np.fromfile(f, np_dtype) for f in output_file_lists])
        bitwise_match = np.array_equal(tmp_out, tmp_gold)
        abs_error_zero = np.all(
            np.abs(tmp_out.astype(np.int64) - tmp_gold.astype(np.int64)) == 0
        )
        is_pass = bitwise_match or abs_error_zero
        if is_pass:
            print(
                f"PASSED! bitwise_match={bitwise_match}, abs_error_zero={abs_error_zero}"
            )
        else:
            print("FAILED!")
            data_same = False
    else:
        for gold, out in zip(golden_file_lists, output_file_lists):
            tmp_out = np.fromfile(out, np_dtype)
            tmp_gold = np.fromfile(gold, np_dtype)

            # 检查二进制一致
            bitwise_match = np.array_equal(tmp_out, tmp_gold)
            # 检查绝对误差为0
            abs_error_zero = np.all(
                np.abs(tmp_out.astype(np.int64) - tmp_gold.astype(np.int64)) == 0
            )

            is_pass = bitwise_match or abs_error_zero

            if is_pass:
                print(
                    f"PASSED! bitwise_match={bitwise_match}, abs_error_zero={abs_error_zero}"
                )
            else:
                print("FAILED!")
                diff_idx = np.where(tmp_out != tmp_gold)[0][:5]
                for idx in diff_idx:
                    print(
                        f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}"
                    )
                data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def infer_dtype_from_filename():
    """从 golden 文件名推断 dtype"""
    golden_files = glob.glob(curr_dir + "/*golden*.bin")
    if not golden_files:
        return "float32"

    filename = os.path.basename(golden_files[0])
    for dtype in [
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "fp8_e4m3fn",
        "fp8_e5m2",
        "hifloat32",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
    ]:
        if filename.startswith(dtype + "_"):
            return dtype
    return "float32"


def infer_dtype_from_single_filename(filename):
    """从单个文件名推断 dtype"""
    basename = os.path.basename(filename)
    for dt in [
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "fp8_e4m3fn",
        "fp8_e5m2",
        "hifloat32",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
    ]:
        if basename.startswith(dt + "_"):
            return dt
    return "float32"


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)

    if not golden_file_lists and not output_file_lists:
        print(
            "No golden or output files found (no-output operator), skipping comparison"
        )
        return True

    if not golden_file_lists or not output_file_lists:
        print("ERROR: No golden or output files found")
        return False

    # 单 golden 多 output：拼接比对
    if len(golden_file_lists) == 1 and len(output_file_lists) > 1:
        if d_type in [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bool",
        ]:
            result = compare_data_integer(golden_file_lists, output_file_lists, d_type)
        else:
            result = compare_data_float(golden_file_lists, output_file_lists, d_type)
        print("compare result:", result)
        return result

    if len(golden_file_lists) != len(output_file_lists):
        print(
            f"ERROR: file count mismatch: golden={len(golden_file_lists)}, output={len(output_file_lists)}"
        )
        return False

    # 逐对比对，每个 golden 文件独立推断 dtype
    all_pass = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        file_dtype = infer_dtype_from_single_filename(gold)
        print(
            f"Comparing: {os.path.basename(gold)} vs {os.path.basename(out)} (dtype={file_dtype})"
        )
        if file_dtype in [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bool",
        ]:
            pair_result = compare_data_integer([gold], [out], file_dtype)
        else:
            pair_result = compare_data_float([gold], [out], file_dtype)
        if not pair_result:
            all_pass = False

    print("compare result:", all_pass)
    return all_pass


if __name__ == "__main__":
    # 从文件名推断 dtype，或使用命令行参数
    if len(sys.argv) >= 2:
        d_type = sys.argv[1]
    else:
        d_type = infer_dtype_from_filename()
        print(f"从文件名推断 dtype: {d_type}")

    ret = process(d_type)
    exit(0 if ret else 1)
