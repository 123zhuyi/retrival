#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np
import tensorflow as tf
DATA_MAX = 10000
DATA_MIN = -10000

bfp16 = tf.bfloat16.as_numpy_dtype

def get_cpu_data(feeds, feeds_repeat_times, output_feeds_size, ori_shape):
    data_cpu = np.repeat(feeds, feeds_repeat_times, axis = 0)
    total = sum(feeds_repeat_times)
    pad_diff = output_feeds_size - total
    pad_shape = [(0, 0) for _ in range(len(ori_shape))]
    pad_shape[0] = (0, pad_diff)
    data_cpu = np.pad(data_cpu, pad_shape, 'constant', constant_values = (0))
    return data_cpu

def gen_data_and_golden(feeds_repeat_times, output_feeds_size, feeds_dtype, repeat_times_dtype, row_shape):
    input_size = len(feeds_repeat_times)
    ori_shape = np.array([input_size] + row_shape)
    ori_data = np.random.uniform(DATA_MIN, DATA_MAX, ori_shape).astype(feeds_dtype)
    np_repeat_times = np.array(feeds_repeat_times).astype(repeat_times_dtype)
    cpu_out = get_cpu_data(ori_data, np_repeat_times, output_feeds_size, ori_shape)
    ori_data.tofile("./input_feeds.bin")
    np_repeat_times.tofile("./input_feeds_repeat_times.bin")
    cpu_out.tofile("./output_y.bin")

if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])