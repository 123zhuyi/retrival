# /**
#  * This program is free software, you can redistribute it and/or modify.
#  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  * This file is a part of the CANN Open Software.
#  * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
#  * Please refer to the License for details. You may not use this file except in compliance with the License.
#  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
#  * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
#  * the software repository for the full text of the License.
#  */

import sys
import os
import numpy as np
import re
import torch
import tensorflow as tf

def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list


def gen_data_and_golden(qkv_shape_str, qkv_bias_shape_str, output_size_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    qkv_shape, _ = parse_str_to_shape_list(qkv_shape_str)
    qkv_bias_shape, _ = parse_str_to_shape_list(qkv_bias_shape_str)
    _, output_size = parse_str_to_shape_list(output_size_str)

    qkv_size = np.prod(qkv_shape)
    tmp_qkv = np.random.random(qkv_size).reshape(qkv_shape).astype(np_type)
    qkv_tensor = torch.tensor(tmp_qkv.astype(np.float32), dtype=torch.float32)

    qkv_bias_size = np.prod(qkv_bias_shape)
    tmp_qkv_bias = np.random.random(qkv_bias_size).reshape(qkv_bias_shape).astype(np_type)
    qkv_bias_tensor = torch.tensor(tmp_qkv_bias.astype(np.float32), dtype=torch.float32)

    q_k_v_golden = torch._transform_bias_rescale_qkv(qkv_tensor, qkv_bias_tensor, 3)

    q_golden = np.array(q_k_v_golden[0]).astype(np_type)
    k_golden = np.array(q_k_v_golden[1]).astype(np_type)
    v_golden = np.array(q_k_v_golden[2]).astype(np_type)

    qkv_tensor.numpy().astype(np_type).tofile(f"{d_type}_qkv_transform_bias_rescale_qkv.bin")
    qkv_bias_tensor.numpy().astype(np_type).tofile(f"{d_type}_qkv_bias_transform_bias_rescale_qkv.bin")

    q_golden.astype(np_type).tofile(f"{d_type}_q_golden_transform_bias_rescale_qkv.bin")
    qkv_bias_tensor.numpy().astype(np_type).tofile(f"{d_type}_k_golden_transform_bias_rescale_qkv.bin")
    qkv_bias_tensor.numpy().astype(np_type).tofile(f"{d_type}_v_golden_transform_bias_rescale_qkv.bin")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Param num must be 5, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])