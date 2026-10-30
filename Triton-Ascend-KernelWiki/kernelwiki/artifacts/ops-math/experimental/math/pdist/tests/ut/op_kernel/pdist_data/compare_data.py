#!/usr/bin/env python3
import sys
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file_lists, output_file_lists, d_type):
    dtype_map = {"float16": np.float16, "float32": np.float32}
    np_dtype = dtype_map.get(d_type)
    if np_dtype is None:
        raise ValueError("d_type must be float16 or float32")

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        if np_dtype == np.float16:
            rtol, atol = 1e-3, 1e-3
        else:
            rtol, atol = 1e-4, 1e-5
        diff_res = np.isclose(tmp_out, tmp_gold, rtol=rtol, atol=atol, equal_nan=True)
        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                print(f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(os.path.join(curr_dir, "*golden*.bin")))
    output_file_lists = sorted(glob.glob(os.path.join(curr_dir, "*output*.bin")))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("compare result:", result)
    return result


if __name__ == '__main__':
    ret = process(sys.argv[1])
    exit(0 if ret else 1)
