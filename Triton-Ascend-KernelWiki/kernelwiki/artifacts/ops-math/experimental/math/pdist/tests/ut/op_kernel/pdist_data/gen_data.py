#!/usr/bin/env python3
import sys
import os
import numpy as np


def parse_shape(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    return tuple(int(x) for x in shape_str.split(","))


def gen_data_and_golden(shape_str, p_str, d_type="float32"):
    dtype_map = {"float32": np.float32, "float16": np.float16}
    np_type = dtype_map[d_type]
    shape = parse_shape(shape_str)
    p = float(p_str)

    np.random.seed(42)
    x = np.random.randn(*shape).astype(np_type)
    x.tofile(f"{d_type}_input_x.bin")

    N, M = shape
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            diff = x[i].astype(np.float64) - x[j].astype(np.float64)
            if p == 0.0:
                d = np.sum(diff != 0).astype(np.float64)
            elif np.isinf(p):
                d = np.max(np.abs(diff))
            else:
                d = np.sum(np.abs(diff) ** p) ** (1.0 / p)
            dists.append(d)

    golden = np.array(dists, dtype=np_type)
    golden.tofile(f"{d_type}_golden_pdist.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: gen_data.py 'shape' 'p' 'dtype'")
        print("  e.g.: gen_data.py '(4,3)' '2.0' 'float32'")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])
