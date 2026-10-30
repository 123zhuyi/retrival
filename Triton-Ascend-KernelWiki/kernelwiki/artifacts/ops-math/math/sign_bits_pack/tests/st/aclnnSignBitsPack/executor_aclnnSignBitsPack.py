#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawe                                                                                                                                                                                                                     i Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        aa
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import numpy as np

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

@register("ascend_aclnn_signbitspack")
class AclnnSigbBitsPack(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        selfTensor = input_data.kwargs["self"]
        size = input_data.kwargs["size"]
        # 将float列表转换为numpy数组以便进行位操作等处理
        float_array = selfTensor.numpy()
        # 获取符号位
        sign_bits = np.sign(float_array).astype(np.int8)
        # 这里简单地将除符号位外的其他位都设为0（假设这就是你想要的1位Adam表示方式，实际可能需根据具体需求调整）
        one_bit_adam = np.where(sign_bits >= 0, 1, 0)
        num_elements = len(one_bit_adam)
        remainder = num_elements % 8
        if remainder!= 0:
            # 需要填充的元素个数
            padding_size = 8 - remainder
            # 创建填充的1位Adam值，这里对于 -1按照之前假设为1
            padding_values = np.full(padding_size, 0, dtype=np.uint8)
            one_bit_adam = np.concatenate((one_bit_adam, padding_values), axis=0)
        # 将1位Adam值重新整形为二维数组，每行8个元素
        one_bit_adam_reshaped = one_bit_adam.reshape(-1, 8)
        # 用于存储打包后的uint8值的列表
        packed_uint8_list = []
        for row in one_bit_adam_reshaped:
            # 创建一个8位的二进制数，初始值为0
            binary_value = 0
            for i, bit in enumerate(row):
                # 将每个1位Adam值按照从左到右（对应二进制低位到高位）的顺序设置到二进制数中
                binary_value += bit << i
            # 将二进制数转换为uint8类型并添加到列表中
            packed_uint8_list.append(binary_value)
        # 将列表转换为numpy数组并返回
        packed_uint8_array = np.array(packed_uint8_list, dtype=np.uint8)
        num_packed = len(packed_uint8_array)
        reshaped_size = num_packed // size
        # 将打包后的uint8数组转化为二维Tensor
        tensor_result = torch.from_numpy(packed_uint8_array).reshape(size, reshaped_size)
        return tensor_result
