# /**
#  * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
#  * and is contributed to the CANN Open Software.
#  *
#  * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
#  * All Rights Reserved.
#  *
#  * Authors (accounts):
#  * - Liang Yanglin <@liang-yanglin>
#  * - Su Tonghua <@sutonghua>
#  *
#  * This program is free software: you can redistribute it and/or modify it.
#  * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
#  * You may not use this file except in compliance with the License.
#  * See the LICENSE file at the root of the repository for the full text of the License.
#  *
#  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
#  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  */

import sys
import os
import numpy as np
import re
import tensorflow as tf


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def get_np_dtype(d_type):
    # 映射数据类型字符串到numpy对应类型，扩展Cast常用类型
    d_type_dict = {
        # 浮点类型
        "float": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype,
        # 整数类型
        "uint8": np.uint8,
        "int8": np.int8,
        "int32": np.int32,
        "int64": np.int64,
        "int16": np.int16
    }
    if d_type not in d_type_dict:
        raise ValueError(f"不支持的数据类型：{d_type}，支持的类型：{list(d_type_dict.keys())}")
    return d_type_dict[d_type]


def gen_data_and_golden(shape_str, input_d_type, output_d_type):
    # 获取numpy数据类型
    np_input_type = get_np_dtype(input_d_type)
    np_output_type = get_np_dtype(output_d_type)

    # 解析形状并计算总元素数
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape) if shape.size > 0 else 1  # 标量时size=1

    # ========== 1. 生成输入数据：根据数据类型选择合适的随机数范围 ==========
    if np.issubdtype(np_input_type, np.floating):
        # 浮点类型：保留原随机值（0、0.5、1、65504、nan、inf），覆盖常见场景
        tmp_input = np.random.choice(
            [0, 0.5, 1, 65504, np.nan, np.inf],
            size=size,
            replace=True  # 允许重复选择
        )
    elif np.issubdtype(np_input_type, np.integer):
        # 整数类型：生成对应范围的随机整数，避免超出类型范围
        type_info = np.iinfo(np_input_type)
        # 缩小范围（如int32取-10000~10000），避免极端值转换异常
        tmp_input = np.random.randint(
            max(type_info.min, -10000),
            min(type_info.max, 10000),
            size=size
        )
    else:
        # 其他类型默认生成0
        tmp_input = np.zeros(size, dtype=np_input_type)

    tmp_input = tmp_input.reshape(shape).astype(np_input_type)
    tmp_golden = tmp_input.astype(np_output_type)

    tmp_input.astype(np_input_type).tofile(f"{np_input_type}_output_t_cast_v2.bin")
    tmp_golden.astype(np_output_type).tofile(f"{np_output_type}_golden_t_cast_v2.bin")


if __name__ == "__main__":
    # 校验命令行参数数量：脚本名 + 形状字符串 + 输入类型 + 输出类型
    if len(sys.argv) != 4:
        print("参数数量错误！")
        sys.exit(1)
    # 清理旧的bin文件
    os.system("rm -rf *.bin")
    # 解析命令行参数（转为小写，兼容大小写输入）
    shape_str = sys.argv[1]
    input_d_type = sys.argv[2].lower()
    output_d_type = sys.argv[3].lower()
    # 生成数据和黄金文件
    try:
        gen_data_and_golden(shape_str, input_d_type, output_d_type)
    except Exception as e:
        print(f"生成数据失败：{str(e)}")
        sys.exit(1)
    sys.exit(0)



