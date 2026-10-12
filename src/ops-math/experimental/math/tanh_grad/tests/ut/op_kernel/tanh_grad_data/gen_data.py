# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np

def parse_str_to_shape_list(shape_str):
    """将形状字符串转换为列表"""
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32"):
    """生成 tanh_grad 测试数据"""
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)

    # 生成 dy: 输入梯度，在合理范围内随机生成
    input_dy = np.random.uniform(-5.0, 5.0, size=size).reshape(shape).astype(np_type)
    
    # 生成 y: tanh的输出，范围应该在 (-1, 1) 之间
    input_y = np.random.uniform(-0.999, 0.999, size=size).reshape(shape).astype(np_type)

    # 计算基准输出
    golden_dx = input_dy * (1.0 - input_y * input_y)

    # 保存文件
    input_dy.tofile(f"{d_type}_input_dy_tanh_grad.bin")
    input_y.tofile(f"{d_type}_input_y_tanh_grad.bin")
    golden_dx.tofile(f"{d_type}_golden_tanh_grad.bin")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 gen_data.py '(shape)' 'dtype'")
        exit(1)
    
    # 清理旧的bin文件
    os.system("rm -rf ./*.bin")
    
    # 生成数据
    shape_str = sys.argv[1]
    d_type = sys.argv[2]
    gen_data_and_golden(shape_str, d_type)