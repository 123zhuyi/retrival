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
"""
PopulationCount kernel-direct golden.

Uses TensorFlow's direct competitor tf.raw_ops.PopulationCount as the golden
reference. TensorFlow runs in a subprocess to avoid loading it into the TTK
process together with Torch/Torch-NPU.
Formula: y = sum(bit_i(x)), i in [0, 15]
  x : int16 or uint16 input
  y : uint8 population count in [0, 16]
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


__golden__ = {"kernel": {"population_count": "population_count_golden"}}


_GOLDEN_PYTHON_ENV = "POPULATION_COUNT_GOLDEN_PYTHON"
_TF_SCRIPT = r"""
import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

with np.load(sys.argv[1]) as input_data:
    x = input_data["x"]

result = tf.raw_ops.PopulationCount(x=tf.convert_to_tensor(x))
np.save(sys.argv[2], result.numpy())
"""


def population_count_golden(x, **kwargs):
    """
    Kernel golden for population_count.
    All the parameters follow @population_count_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
             input_formats, output_formats, input_ori_formats, output_ori_formats,
             input_dtypes, output_dtypes.
    """
    del kwargs
    with tempfile.TemporaryDirectory(prefix="population_count_golden_") as temp_dir:
        input_path = Path(temp_dir) / "input.npz"
        output_path = Path(temp_dir) / "output.npy"
        np.savez(input_path, x=np.asarray(x))

        worker_env = os.environ.copy()
        worker_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        worker_env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        python_executable = worker_env.get(_GOLDEN_PYTHON_ENV, sys.executable)
        subprocess.run(
            [python_executable, "-c", _TF_SCRIPT, str(input_path), str(output_path)],
            check=True,
            env=worker_env,
            timeout=180,
        )
        return np.load(output_path).astype(np.uint8, copy=False)
