#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import torch
import logging
import math
import numpy as np
import copy
import time
import random
import ctypes
import copy
import tensorflow as tf
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray,pointer

# ifa
def get_np_dtype(type_str):
    type_dict = {
        'fp64': np.float64, 'fp32': np.float32, 'fp16': np.float16,
        'int64': np.int64, 'int32': np.int32, 'int16': np.int16, 'int8': np.int8,
        'uint64': np.uint64, 'uint32': np.uint32, 'uint16': np.uint16, 'uint8': np.uint8,
        'bool': np.bool_, 'complex64': np.complex64, 'complex128': np.complex128,
        'complex32': np.float16,
        'bf16': tf.bfloat16.as_numpy_dtype,
        'bfloat16': tf.bfloat16.as_numpy_dtype,
        'float4_e2m1': np.uint8,
        'float4_e1m2': np.uint8,
        'float8_e8m0': np.uint8,
        'hifloat8': np.uint8,
        'fp4_e1m2': np.uint8,
        "fp4_e2m1": np.uint8,
        "fp8_e8m0": np.uint8,
        "float32": np.float32,
        "float16": np.float16,
        "qint8": np.int8,
        "qint32": np.int32,
        "quint8": np.uint8,
        "qint16": np.int16,
        "uint1": np.uint8,
        "quint16": np.uint16,
        "fp4_e2m1_as_fp32": np.float32
    }
    if type_str == "int4":
        from ml_dtypes import int4
        return int4
    elif type_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2
        return float8_e5m2
    elif type_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    else:
        return type_dict[type_str]

def get_pt_dtype(type_str):
    type_dict = {
        'fp32': torch.float32, 'fp16': torch.float16, 'fp64': torch.float64,
        'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64,
        'uint8': torch.uint8, 'bool': torch.bool, 'complex64': torch.complex64,
        'complex128': torch.complex128, 'bf16': torch.bfloat16, 'uint1': torch.uint8
    }
    if type_str == 'hifloat8':
        import torch_npu
        return torch_npu.hifloat8
    return type_dict[type_str]

def cvt_hifuint8_to_float(x, over_mode=True):
    x = int(x)
    if x == 0:
        return float(0)
    elif x == 128:
        if over_mode:
            return np.nan
        else:
            return float(0)
    elif x == 239:
        if over_mode:
            return -np.inf
        else:
            return -32768
    elif x == 111:
        if over_mode:
            return np.inf
        else:
            return 32768
    else:
        if x >= 128:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & 120 #b01111000 = 120
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= 12:
            #b1100 =12 D4
            exponet = x & 30 #b00011110 = 30
            exponet_int = exponet >> 1
            if exponet_int >= 8:
                #b1000 = 8
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 8

            fra_int = x & 1 #b00000001
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= 8:
            #b1000 =8 D3
            exponet = x & 28 #b00011100 = 28
            exponet_int = exponet >> 2
            if exponet_int >= 4:
                #b100 = 4
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 4
            fra_int = x & 3 #b00000011
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= 4:
            #b0100 =8 D2
            exponet = x & 24  # b00011000 = 24
            exponet_int = exponet >> 3
            if exponet_int >= 2:
                # b10 = 2
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 2
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= 2:
            #b0010 =2 D1
            exponet = x & 8 # b00001000 = 8
            exponet_sign = exponet >> 3
            if exponet_sign >= 1:
                # b10 = 2
                exponet_value = -1
            else:
                exponet_value = 1
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 1:
            #d0
            exponet_value = 0
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 0:
            #dml
            m_value = 1
            exponet_value = (x & 7) - 23  # b00000111 = 7
        else:
            print("error,dot error")
            m_value = 0.0
            exponet_value = 0
        return sign*pow(2.0, exponet_value)*m_value

def cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {'0': 0.0, '1': 0.25, '2': 0.5, '3': 0.75,
                     '4': 1.0, '5': 1.25, '6': 1.5, '7': 1.75,
                     '8': -0.0, '9': -0.25, '10': -0.5, '11': -0.75,
                     '12': -1.0, '13': -1.25, '14': -1.5, '15': -1.75
                     }
    x = int(x)
    first_fp4val = x & 0x0f
    first_fp4str = str(first_fp4val)
    return Fp4e1m2ToBf16[first_fp4str]

def new_trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        bfloat16_tensor[i] = cvt_fp4_e2m1_to_bfloat16(in_tensor[i])
    return bfloat16_tensor.reshape(shape_tensor)

def new_trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        bfloat16_tensor[i] = cvt_fp4_e1m2_to_bfloat16(in_tensor[i])
    return bfloat16_tensor.reshape(shape_tensor)

def trans_np_hifuint8_tensor_to_float32(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.float32)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_hifuint8_to_float(in_tensor[i])
    out_tensor = out_tensor.reshape(shape_tensor).astype(np.float32)
    return out_tensor

def find_output_value(input_list, output_value_list):
    temp_list = []
    is_list_or_tuple = False
    for item in input_list:
        if not isinstance(item, list) and not isinstance(item, tuple):
            output_value_list.append(item)
        else:
            for elment in item:
                temp_list.append(elment)
    if len(temp_list):
        is_list_or_tuple = True
    return is_list_or_tuple, temp_list


def get_output_list(is_list_or_tuple, input_list, output_value_list):
    while True:
        if not is_list_or_tuple:
            return output_value_list
        else:
            is_list_or_tuple, input_list = find_output_value(input_list, output_value_list)


def dequant(x, deqscale, relu_weight):
    deqscale = np.uint64(deqscale)
    deqscale = np.frombuffer(deqscale, dtype=np.float32)
    deqscale = deqscale[: 1][0]
    if relu_weight is None:
        relu_weight = deqscale
    else:
        relu_weight = np.full(deqscale.shape, relu_weight)
    res_cal = np.zeros(x.shape, np.float16)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    x[n, c, h, w] = x[n, c, h, w].astype(np.float32)
                    if x[n, c, h, w] >= 0:
                        res_cal[n, c, h, w] = x[n, c, h, w] * deqscale
                    else:
                        res_cal[n, c, h, w] = x[n, c, h, w] * relu_weight
    return res_cal


def s8_saturation(inputdata):
    if inputdata > 127:
        inputdata = 127
    elif inputdata < -128:
        inputdata = -128
    outdata = np.int8(inputdata)
    return outdata


def s9_saturation(inputdata):
    if inputdata > 255:
        inputdata = 255
    elif inputdata < -256:
        inputdata = -256
    return inputdata


def quant(x, qscale, qoffset):
    s8_res_cal = np.zeros(x.shape, np.int8)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    s8_res_cal[n, c, h, w] = s8_saturation(
                        np.round(s9_saturation(np.half(x[n, c, h, w]) * np.half(qscale)) + np.half(qoffset)))
    return s8_res_cal


def quant_pc(x, qscale, qoffset):
    s8_res_cal = np.zeros(x.shape, np.int8)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    s8_res_cal[n, c, h, w] = s8_saturation(
                        np.round(s9_saturation(np.half(x[n, c, h, w]) * np.half(qscale[0, c, 0, w])) + np.half(
                            qoffset[0, c, 0, w])))
    return s8_res_cal


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    """ Convert binary data into the floating point value """
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2 ** exponent) * mant_mult


def trans_tensor_fp8_e8m0_to_bf16(tensor):
    # import pdb;pdb.set_trace()
    from ml_dtypes import bfloat16
    # 创建一个与输入tensor形状相同的新tensor
    new_tensor = np.zeros_like(tensor).astype(bfloat16)
    # 使用nditer遍历tensor中的每个元素
    with np.nditer(tensor, flags=['multi_index'], op_flags=['readwrite']) as it:
        for x in it:
            new_value = ieee_754_conversion(0, int(x), 0)
            # 将处理后的结果回填到新tensor中
            new_tensor[it.multi_index] = new_value
    return new_tensor

def msd(a, b): 
    #step1: preprocess
    a = a.astype(np.float32)  #keep
    amax = np.max(np.abs(a), axis=1) * 1.001
    amodamax = (a / amax)
    a1 = np.floor(128 * amodamax)
    a2 = np.floor(128 ** 2 * amodamax - a1 * 128)
    # step2: matmul
    c1 = np.dot(a1.astype(np.float32), b.astype(np.float32))
    c2 = np.dot(a2.astype(np.float32), b.astype(np.float32))
    # step3: postPress
    c = amax * (c1 / 128 + c2 / 128**2)
    return c

def antiquant(k, v, scale, offset, pto_flag, lowprecision_flag, kv_dtype="int8", q_dtype="fp16"):
    N = k.shape[1]
    S = v.shape[2]
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1", "float4_e1m2", "float4_e2m1"]:
        if lowprecision_flag:
            if q_dtype == "float16":
                scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
                scale = scale_bf16.astype(np.float16)
            elif q_dtype == "bfloat16":
                scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
                scale = scale_bf16
            else:
                exit(1)
        else:
            scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
            scale = scale_bf16.astype(np.float16)
        scale1 = scale[0]
        scale2 = scale[1]
        if kv_dtype in ["fp4_e1m2", "float4_e1m2"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v)
                else:
                    exit(1)
            else:
                key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)     
        else:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v)
                else:
                    exit(1)
            else:
                key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)
        g = scale.shape[4]
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    key_cur = key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx * grp_size:(gIdx + 1) * grp_size]
                    # broardcast
                    scale_k = scale1[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx:(gIdx + 1)]
                    scale_k = np.full((1, grp_size), scale_k)
                    # mul
                    key_cur = key_cur * scale_k
                    # copy to
                    key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1),
                    gIdx * grp_size:(gIdx + 1) * grp_size] = key_cur

                for gIdx in range(g):
                    # copy from
                    value_cur = value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx * grp_size:(gIdx + 1) * grp_size]
                    # broardcast
                    scale_v = scale2[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx:(gIdx + 1)]
                    scale_v = np.full((1, grp_size), scale_v)
                    # mul
                    value_cur = value_cur * scale_v
                    # copy to
                    value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1),
                    gIdx * grp_size:(gIdx + 1) * grp_size] = value_cur
    else:
        if lowprecision_flag:
            if q_dtype == "float16":
                scale = scale.astype(np.float16)
                offset = offset.astype(np.float16)
            elif q_dtype == "bfloat16":
                scale = scale.astype(tf.bfloat16.as_numpy_dtype)
                offset = offset.astype(tf.bfloat16.as_numpy_dtype)
            else:
                print(f"Invalid input q dtype 4: {q_dtype}")
        else:
            scale = scale.astype(np.float32)
            offset = offset.astype(np.float32)
        if kv_dtype == "int8":
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"Invalid input q dtype 5: {q_dtype}")
            else:
                key = k.astype(np.float32)
                value = v.astype(np.float32)
        elif kv_dtype in ["int4", "int32"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    exit(1)
            else:
                scale = scale.astype(np.float16)
                offset = offset.astype(np.float16)
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        elif kv_dtype == "hifloat8":
            # uint8->fp16，精度转换
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(tf.bfloat16.as_numpy_dtype)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    exit(1)
            else:
                key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
        elif kv_dtype in ["float8_e5m2", "float8_e4m3fn"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    exit(1)
            else:
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        else:
            exit(1)
            
        if q_dtype == "bfloat16":
            key = k.astype(tf.bfloat16.as_numpy_dtype)
            value = v.astype(tf.bfloat16.as_numpy_dtype)
        else:
            key = k.astype(np.float16)
            value = v.astype(np.float16)

        if pto_flag:
            scale1 = scale[0:1, :, :]
            scale2 = scale[1:2, :, :]
            offset1 = offset[0:1, :, :]
            offset2 = offset[1:2, :, :]
            for nIdx in range(N):
                key_cur = key[:, nIdx:(nIdx + 1), :, :]
                scale1_cur = np.expand_dims(scale1, axis=-1)
                offset1_cur = np.expand_dims(offset1, axis=-1)
                key_cur = np.add(key_cur, offset1_cur)
                key_cur = np.multiply(key_cur, scale1_cur)
                key[:, nIdx:(nIdx + 1), :, :] = key_cur

                value_cur = value[:, nIdx:(nIdx + 1), :, :]
                scale2_cur = np.expand_dims(scale2, axis=-1)
                offset2_cur = np.expand_dims(offset2, axis=-1)
                value_cur = np.add(value_cur, offset2_cur)
                value_cur = np.multiply(value_cur, scale2_cur)
                value[:, nIdx:(nIdx + 1), :, :] = value_cur
        else:
            # 2,n,1,d -> 1,n,1,d
            scale1 = scale[0:1, :, :, :]
            scale2 = scale[1:2, :, :, :]
            offset1 = offset[0:1, :, :, :]
            offset2 = offset[1:2, :, :, :]
            for nIdx in range(N):
                for sIdx in range(S):
                    key_cur = key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :]
                    key_cur = key_cur + offset1[:, nIdx:(nIdx + 1), :, :]
                    key_cur = key_cur * scale1[:, nIdx:(nIdx + 1), :, :]
                    key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :] = key_cur

                    value_cur = value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :]
                    value_cur = value_cur + offset2[:, nIdx:(nIdx + 1), :, :]
                    value_cur = value_cur * scale2[:, nIdx:(nIdx + 1), :, :]
                    value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :] = value_cur
    return key, value


def kv_split_antiquant(k, v, k_scale, k_offset, v_scale, v_offset, k_anti_mode, v_anti_mode, lowprecision_flag, kv_dtype="int8", q_dtype="float16"):
    N = k.shape[1]
    S = k.shape[2]
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1", "float4_e1m2", "float4_e2m1"]:
        k_scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(k_scale)
        v_scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(v_scale)
        if lowprecision_flag:
            if q_dtype == "float16":
                k_scale = k_scale_bf16.astype(np.float16)
                v_scale = v_scale_bf16.astype(np.float16)
            elif q_dtype == "bfloat16":
                k_scale = k_scale_bf16
                v_scale = v_scale_bf16
            else:
                exit(1)
        else:
            k_scale = k_scale_bf16.astype(np.float16)
            v_scale = v_scale_bf16.astype(np.float16)

        if kv_dtype in ["fp4_e1m2", "float4_e1m2"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v)
                else:
                    exit(1)
            else:
                key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
        else:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v)
                else:
                    exit(1)
            else:
                key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)

    else:
        if lowprecision_flag:
            if q_dtype == "float16":
                k_scale = k_scale.astype(np.float16)
                k_offset = k_offset.astype(np.float16)
                v_scale = v_scale.astype(np.float16)
                v_offset = v_offset.astype(np.float16)
            elif q_dtype == "bfloat16":
                k_scale = k_scale.astype(tf.bfloat16.as_numpy_dtype)
                k_offset = k_offset.astype(tf.bfloat16.as_numpy_dtype)
                v_scale = v_scale.astype(tf.bfloat16.as_numpy_dtype)
                v_offset = v_offset.astype(tf.bfloat16.as_numpy_dtype)
            else:
                exit(1)
        else:
            k_scale = k_scale.astype(np.float32)
            k_offset = k_offset.astype(np.float32)
            v_scale = v_scale.astype(np.float32)
            v_offset = v_offset.astype(np.float32)
        if kv_dtype == "int8":
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(tf.bfloat16.as_numpy_dtype)
                else:
                    exit(1)
            else:
                key = k.astype(np.float32)
                value = v.astype(np.float32)
        elif kv_dtype in ["int4", "int32"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    exit(1)
            else:
                k_scale = k_scale.astype(np.float16)
                k_offset = k_offset.astype(np.float16)
                v_scale = v_scale.astype(np.float16)
                v_offset = v_offset.astype(np.float16)
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        elif kv_dtype == "hifloat8":
            # uint8->fp16，精度转换
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(tf.bfloat16.as_numpy_dtype)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype( tf.bfloat16.as_numpy_dtype)
                else:
                    exit(1)
            else:
                key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
        elif kv_dtype in ["float8_e5m2", "float8_e4m3fn"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    exit(1)
            else:
                key = k.astype(np.float16)
                value = v.astype(np.float16) 
        else:
            exit(1)

    if k_anti_mode in ['1', '4']:
        scale1 = k_scale[0:1, :, :]  # 11S
        offset1 = k_offset[0:1, :, :]
        for nIdx in range(N):
            key_cur = key[:, nIdx:(nIdx + 1), :, :]
            # scale: 11s -> 11s1
            scale1_cur = np.expand_dims(scale1, axis=-1)
            offset1_cur = np.expand_dims(offset1, axis=-1)
            key_cur = np.add(key_cur, offset1_cur)
            key_cur = np.multiply(key_cur, scale1_cur)
            key[:, nIdx:(nIdx + 1), :, :] = key_cur
    elif k_anti_mode in ['0', '2']:
        # 2,n,1,d -> 1,n,1,d
        scale1 = k_scale[0:1, :, :, :]
        offset1 = k_offset[0:1, :, :, :]
        for nIdx in range(N):
            for sIdx in range(S):
                key_cur = key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :]
                key_cur = key_cur + offset1[:, nIdx:(nIdx + 1), :, :]
                key_cur = key_cur * scale1[:, nIdx:(nIdx + 1), :, :]
                key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :] = key_cur
    elif k_anti_mode in ['3', '5']:
        # 计算, scale为1NS, key为1nsd
        scale1 = k_scale[0:1, :, :]
        offset1 = k_offset[0:1, :, :]
        for nIdx in range(N):
            # key_cur：11sd
            key_cur = key[:, nIdx:(nIdx + 1), :, :]
            # scale/offset：11s
            scale1_cur = scale1[:, nIdx:(nIdx + 1), :]
            offset1_cur = offset1[:, nIdx:(nIdx + 1), :]
            # scale/offset：11s1
            scale1_cur = np.expand_dims(scale1_cur, axis=-1)
            offset1_cur = np.expand_dims(offset1_cur, axis=-1)
            # (11sd + 11s1) * 11s1
            key_cur = np.add(key_cur, offset1_cur)
            key_cur = np.multiply(key_cur, scale1_cur)
            key[:, nIdx:(nIdx + 1), :, :] = key_cur
    elif k_anti_mode == '6':
        g = k_scale.shape[4]
        k_scale = k_scale[0]
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    key_cur = key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx * grp_size:(gIdx + 1) * grp_size]
                    # broardcast
                    scale_k = k_scale[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx:(gIdx + 1)]
                    scale_k = np.full((1, grp_size), scale_k)
                    # mul
                    key_cur = key_cur * scale_k
                    # copy to
                    key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1),
                    gIdx * grp_size:(gIdx + 1) * grp_size] = key_cur
    if v_anti_mode in ['1', '4']:
        scale2 = v_scale[0:1, :, :]
        offset2 = v_offset[0:1, :, :]
        for nIdx in range(N):
            value_cur = value[:, nIdx:(nIdx + 1), :, :]
            scale2_cur = np.expand_dims(scale2, axis=-1)
            offset2_cur = np.expand_dims(offset2, axis=-1)
            value_cur = np.add(value_cur, offset2_cur)
            value_cur = np.multiply(value_cur, scale2_cur)
            value[:, nIdx:(nIdx + 1), :, :] = value_cur
    elif v_anti_mode in ['0', '2']:
        # 2,n,1,d -> 1,n,1,d
        scale2 = v_scale[0:1, :, :, :]
        offset2 = v_offset[0:1, :, :, :]
        for nIdx in range(N):
            for sIdx in range(S):
                value_cur = value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :]
                value_cur = value_cur + offset2[:, nIdx:(nIdx + 1), :, :]
                value_cur = value_cur * scale2[:, nIdx:(nIdx + 1), :, :]
                value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :] = value_cur
    elif v_anti_mode in ['3', '5']:
        # 计算, scale为1SN, value为bnsd
        scale1 = v_scale[0:1, :, :]
        offset1 = v_offset[0:1, :, :]
        for nIdx in range(N):
            # value_cur：11sd
            value_cur = value[:, nIdx:(nIdx + 1), :, :]
            # scale/offset：11s
            scale1_cur = scale1[:, nIdx:(nIdx + 1), :]
            offset1_cur = offset1[:, nIdx:(nIdx + 1), :]
            # scale/offset：11s1
            scale1_cur = np.expand_dims(scale1_cur, axis=-1)
            offset1_cur = np.expand_dims(offset1_cur, axis=-1)
            # (11sd + 11s1) * 11s1
            value_cur = np.add(value_cur, offset1_cur)
            value_cur = np.multiply(value_cur, scale1_cur)
            value[:, nIdx:(nIdx + 1), :, :] = value_cur
    elif v_anti_mode == '6':
        g = v_scale.shape[4]
        v_scale = v_scale[0]
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    value_cur = value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx * grp_size:(gIdx + 1) * grp_size]
                    # broardcast
                    scale_v = v_scale[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx:(gIdx + 1)]
                    scale_v = np.full((1, grp_size), scale_v)
                    # mul
                    value_cur = value_cur * scale_v
                    # copy to
                    value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1),
                    gIdx * grp_size:(gIdx + 1) * grp_size] = value_cur
    return key, value


def _trans_antiparam_to_1bs(tensor):
    if len(tensor.shape) == 2:
        tensor_new = tensor.reshape(1, tensor.shape[0], tensor.shape[1])
    else:
        tensor_new = tensor

    return tensor_new


def softmax(x):
    # this func is only used by quant_dequant
    # x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    # ans = ans.astype(np.float16)
    return ans, x_sum, x_max


def softmax_flash(x, max_front=None, sum_front=None, axis=None, update=False):
    if update == False:
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y / x_sum
        ans = ans.astype(np.float16)
        x_max = x_max.astype(np.float16)
        x_sum = x_sum.astype(np.float16)
        exp_max = None
        return x_max, x_sum, ans, exp_max
    else:
        x = x.astype(np.float16)
        max_front = max_front.astype(np.float16)
        sum_front = sum_front.astype(np.float16)
        x_max = np.max(np.concatenate((max_front, x), axis=-1), axis=-1, keepdims=True)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        exp_max = np.exp(max_front - x_max)
        x_sum = exp_max * sum_front + x_sum
        exp_max = exp_max * sum_front / x_sum
        out = x_exp / x_sum
        out = out.astype(np.float16)
        exp_max = exp_max.astype(np.float16)
        return x_max, x_sum, out, exp_max


def sum_ceil(lst, a):
    res = 0
    for num in lst:
        res += math.ceil(num / a)
    return res


def get_h(shape, layout):
    if layout == "BNSD":
        h = shape[1] * shape[3]
    elif layout == "BSND":
        h = shape[2] * shape[3]
    elif layout == "BSH":
        h = shape[2]
    else:
        h = 0
    return h


def get_sinner(ifa_param):
    from tbe.common.utils.op_tiling import do_op_tiling
    from tbe.common.platform import set_current_compile_soc_info
    set_current_compile_soc_info("Ascend910B1")
    q_shape = ifa_param['q_shape']
    q_dtype = trans_input_dtype(ifa_param['q_dtype'])
    m_shape = ifa_param['m_shape']
    m_dtype = ifa_param['m_dtype']
    p_shape = ifa_param['p_shape']
    p_dtype = ifa_param['p_dtype']
    k_shape = ifa_param['k_shape_list'][0]
    v_shape = ifa_param['v_shape_list'][0]
    k_shape[0] = v_shape[0] = q_shape[0]
    out_shape = ifa_param['out_shape']
    out_dtype = ifa_param['out_dtype']

    op_type = "FusedInferAttentionScore"
    compile_info = {}
    # qkv
    inputs = [{'shape': q_shape, 'ori_shape': q_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': q_dtype,
               'range': [(0, None)], 'param_name': 'query'},
              {'shape': k_shape, 'ori_shape': k_shape, 'format': 'ND', 'ori_format': 'ND',
               'dtype': 'int8', 'range': [(0, None)], 'param_name': 'key'},
              {'shape': v_shape, 'ori_shape': v_shape, 'format': 'ND', 'ori_format': 'ND',
               'dtype': 'int8', 'range': [(0, None)], 'param_name': 'value'}]
    # pse_shift
    if ifa_param['p_flag']:
        inputs.extend([{'shape': p_shape, 'ori_shape': p_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': p_dtype,
                        'range': [(0, None)], 'param_name': 'pse_shift'}])
    else:
        inputs.extend([None])
    # atten_mask
    if ifa_param['m_flag']:
        inputs.extend([{'shape': m_shape, 'ori_shape': m_shape, 'format': 'ND', 'ori_format': 'ND',
                        'dtype': m_dtype, 'range': [(0, None)], 'param_name': 'atten_mask'}])
    else:
        inputs.extend([None])
    # actual_seq_lengths
    inputs.extend([None])
    # actual_seq_lengths_kv
    if ifa_param['as_flag']:
        act_seq = ifa_param['actualSeqLengths_raw']
        inputs.extend([{'shape': [len(act_seq)], 'ori_shape': [len(act_seq)], 'format': 'ND', 'ori_format': 'ND',
                        'range': [(0, None)], 'const_value': act_seq, 'dtype': 'int64',
                        'param_name': 'actual_seq_lengths_kv'}])
    else:
        inputs.extend([None])
    # dequant_scale1 quant_scale1 dequant_scale2
    if ifa_param['in_quant_flag']:
        inputs.extend([{'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'uint64', 'range': [(0, None)], 'param_name': 'dequant_scale_1'},
                       {'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'float32', 'range': [(0, None)], 'param_name': 'quant_scale_1'},
                       {'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'uint64', 'range': [(0, None)], 'param_name': 'dequant_scale_2'}])
    else:
        inputs.extend([None, None, None])
    # quant_scale2 quant_offset2
    if ifa_param['out_quant_flag']:
        quantScale2_shape = ifa_param['quantScale2_shape']
        quantOffset2_shape = ifa_param['quantOffset2_shape']
        quantScale2_dtype = ifa_param['quantScale2_dtype']
        quantOffset2_dtype = ifa_param['quantOffset2_dtype']
        inputs.extend([{'shape': quantScale2_shape, 'ori_shape': quantScale2_shape, 'format': 'ND',
                        'ori_format': 'ND', 'dtype': quantScale2_dtype, 'range': [(0, None)],
                        'param_name': 'quant_scale_2'}])
        if ifa_param['out_quant_offset_flag']:
            inputs.extend([{'shape': quantOffset2_shape, 'ori_shape': quantOffset2_shape, 'format': 'ND',
                            'ori_format': 'ND', 'dtype': quantOffset2_dtype, 'range': [(0, None)],
                            'param_name': 'quant_offset_2'}])
        else:
            inputs.extend([None])
    else:
        inputs.extend([None, None])
    # antiquant_scale antiquant_offset
    if ifa_param['kv_quant_flag']:
        anti_quant_shape = ifa_param['antiquantscale_shape_raw']
        anti_quant_dtype = ifa_param['antiquantscale_dtype']
        inputs.extend([{'shape': anti_quant_shape, 'ori_shape': anti_quant_shape, 'format': 'ND',
                        'ori_format': 'ND', 'dtype': anti_quant_dtype, 'range': [(0, None)],
                        'param_name': 'antiquant_scale'}])
        if ifa_param['kv_quant_offset_flag']:
            inputs.extend([{'shape': anti_quant_shape, 'ori_shape': anti_quant_shape, 'format': 'ND',
                            'ori_format': 'ND', 'dtype': anti_quant_dtype, 'range': [(0, None)],
                            'param_name': 'antiquant_offset'}])
        else:
            inputs.extend([None])
    else:
        inputs.extend([None, None])

    # block_table
    if ifa_param['pa_flag']:
        bt_shape = ifa_param['bt_shape']
        bt_dtype = ifa_param['bt_dtype']
        inputs.extend([{'shape': bt_shape, 'ori_shape': bt_shape, 'format': 'ND',
                        'ori_format': 'ND', 'dtype': bt_dtype, 'range': [(0, None)], 'param_name': 'block_table'}])
        bs = int(ifa_param['blocksize'])
        act_seq = ifa_param['actualSeqLengths']
        kvcache_shape_1 = sum_ceil(act_seq, bs)
        kvcache_shape_2 = bs
        kvcache_shape_3 = get_h(k_shape, ifa_param['inputLayout'])
        inputs[1]['shape'] = inputs[1]['ori_shape'] = [kvcache_shape_1, kvcache_shape_2, kvcache_shape_3]
        inputs[2]['shape'] = inputs[2]['ori_shape'] = [kvcache_shape_1, kvcache_shape_2, kvcache_shape_3]
    else:
        inputs.extend([None])
    # query_padding_size
    inputs.extend([None])
    # kv_padding_size
    if ifa_param['padding_flag']:
        inputs.extend([{'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'int64', 'range': [(0, None)], 'param_name': 'kv_padding_size'}])
    else:
        inputs.extend([None])

    outputs = [{'shape': out_shape, 'ori_shape': out_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': out_dtype,
                'range': [(0, None)], 'param_name': 'attention_out'}]

    attrs = [{'name': 'num_heads', 'dtype': 'int', 'value': ifa_param['numHeads']},
             {'name': 'scale_value', 'dtype': 'float', 'value': ifa_param['scaleValue']},
             {'name': 'pre_tokens', 'dtype': 'int', 'value': ifa_param['pre_tokens']},
             {'name': 'next_tokens', 'dtype': 'int', 'value': ifa_param['next_tokens']},
             {'name': 'input_layout', 'dtype': 'str', 'value': ifa_param['inputLayout']},
             {'name': 'num_key_value_heads', 'dtype': 'int', 'value': ifa_param['numKeyValueHeads']},
             {'name': 'sparse_mode', 'dtype': 'int', 'value': ifa_param['sparse_mode']},
             {'name': 'inner_precise', 'dtype': 'int', 'value': ifa_param['innerprecise']},
             {'name': 'block_size', 'dtype': 'int', 'value': ifa_param['blocksize']},
             {'name': 'antiquant_mode', 'dtype': 'int', 'value': int(ifa_param['antiquant_mode'])},
             {'name': 'softmax_lse_flag', 'dtype': 'bool', 'value': False}]

    ret = do_op_tiling(op_type, compile_info, inputs, outputs, attrs=attrs)
    tiling_key = ret.get("tiling_key")
    cache_tiling = ret.get("tiling_data")
    tiling_data_pr = np.frombuffer(cache_tiling, dtype=np.int32)
    # 20240118前:245
    idx_singleProcessSInnerSize = 254
    singleProcessSInnerSize = tiling_data_pr[idx_singleProcessSInnerSize]

    return singleProcessSInnerSize


def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, input_dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D])
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape


def trans_uint64_2_float32(input):
    output = np.uint64(input)
    output = np.frombuffer(output, dtype=np.float32)
    output = output[: 1][0]

    return output


def torch_to_numpy(tensor):
    n_tensor = None

    if type(tensor) == "<class 'torch.Tensor'>":
        if tensor.dtype in ['torch.float16', 'torch.bool']:
            n_tensor = tensor.numpy()
        elif tensor.dtype in ['torch.bfloat16']:
            n_tensor = tensor.numpy()
    return n_tensor


def _n_trans_shape_to_bnsd(tensor, shape, layout, headnums=None):
    if layout == "BSH":
        if headnums is None:
            return tensor, shape
        B = shape[0]
        S = shape[1]
        H = shape[2]
        N = headnums
        D = H // N
        tensor = tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3)
        return tensor, [B, N, S, D]
    elif layout == "BSND":
        B = shape[0]
        S = shape[1]
        N = shape[2]
        D = shape[3]
        tensor = tensor.transpose(0, 2, 1, 3)
        return tensor, [B, N, S, D]
    else:
        return tensor, shape


def _t_bsh_to_bnsd(tensor, bsh_shape, headnums, batch=1):
    if len(bsh_shape) == 2:
        B = batch
        S = bsh_shape[0] // B
        H = bsh_shape[1]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    elif len(bsh_shape) == 3:
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    else:
        return tensor, bsh_shape


def trans_bnsd_to_layout(tensor, shape, layout):
    if layout == "BSH":
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout == "BSND":
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    else:
        return tensor


def trans_bnsd_to_bsh(tensor, shape):
    if len(shape) == 4:
        B = shape[0]
        N = shape[1]
        S = shape[2]
        D = shape[3]
        H = N * D
        return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)
    else:
        return tensor


def _t_broadcast_mask_n(m_tensor, m_shape, N, B):
    if len(m_shape) == 4:
        # b11s
        m_res = torch.zeros([B, N, 1, m_shape[3]])
        for i in range(B):
            for j in range(N):
                m_res[i:i + 1, j:j + 1, :, :] = m_tensor[i:i + 1, :, :, :]
        return m_res
    elif len(m_shape) == 3:
        # b1s
        m_res = torch.zeros([B, N, 1, m_shape[2]])
        for i in range(B):
            for j in range(N):
                m_res[i:i + 1, j:j + 1, :, :] = m_tensor[i:i + 1, :, :]
        return m_res
    elif len(m_shape) == 2:
        # bs
        m_res = torch.zeros([B, N, 1, m_shape[1]])
        for i in range(B):
            for j in range(N):
                m_res[i:i + 1, j:j + 1, 0, :] = m_tensor[i:i + 1, :]
        return m_res
    else:
        exit(1)


def _trans_antiparam_to_2bnsg(shape, tensor, layout, numKeyValueHeads):
    # bsh: (2, B, S, H/32) -> (2, B, N, S, D/32)
    if layout == "BSH":
        B = shape[1]
        S = shape[2]
        G = shape[3]
        N = numKeyValueHeads
        G2 = int(G / N)
        new_tensor = tensor.reshape(2, B, S, N, G2).transpose(0, 1, 3, 2, 4)
        return new_tensor
    # bsh: (2, B, S, N, D/32) -> (2, B, N, S, D/32)
    elif layout == "BSND":
        new_tensor = tensor.transpose(0, 1, 3, 2, 4)
        return new_tensor
    else:
        return tensor


def _trans_antiparam_to_1bnsg(shape, tensor, layout, numKeyValueHeads):
    # bsh: (1, B, S, H/32) -> (1, B, N, S, D/32)
    if layout == "BSH":
        B = shape[1]
        S = shape[2]
        G = shape[3]
        N = numKeyValueHeads
        G2 = int(G / N)
        new_tensor = tensor.reshape(1, B, S, N, G2).transpose(0, 1, 3, 2, 4)
        return new_tensor
    # bsh: (1, B, S, N, D/32) -> (1, B, N, S, D/32)
    elif layout == "BSND":
        new_tensor = tensor.transpose(0, 1, 3, 2, 4)
        return new_tensor
    else:
        return tensor


def broadcast_kv_dequant_tensor_fp8e8m0(tensor, numKeyValueHeads, numheads):
    B = tensor.shape[1]
    S = tensor.shape[3]
    G = tensor.shape[4]
    if numKeyValueHeads == numheads:
        tensor.astype(np.uint8)
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        tensor_new = np.zeros([2, B, numheads, S, G], dtype=np.uint8)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, :, i:i + 1, :, :] = tensor[:, :, j:j + 1, :, :]
        return tensor_new


def broadcast_kv_split_antiquant_tensor_fp8e8m0(tensor, numKeyValueHeads, numheads):
    B = tensor.shape[1]
    S = tensor.shape[3]
    G = tensor.shape[4]
    if numKeyValueHeads == numheads:
        tensor.astype(np.uint8)
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        tensor_new = np.zeros([1, B, numheads, S, G], dtype=np.uint8)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, :, i:i + 1, :, :] = tensor[:, :, j:j + 1, :, :]
        return tensor_new


def _t_softmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def _t_ifaattention_act(ifa_param):
    # 低精度标记位
    lowprecision_flag = False

    if ifa_param['action_type'] == "one" or (ifa_param['action_type'] == "bm" and ifa_param['innerprecise'] == 1):
        lowprecision_flag = True

    q_tensor = ifa_param['q_tensor_cur']
    k_tensor = ifa_param['k_sub_tensor']
    v_tensor = ifa_param['v_sub_tensor']
    m_tensor = ifa_param['mask_cur']
    m_dtype = ifa_param['m_dtype']
    p_tensor = ifa_param['pse_cur']
    scalar = ifa_param['scaleValue']
    act_seq = ifa_param['act_seq']
    sinner = ifa_param['sinner']
    lse_flag = ifa_param['softmax_lse_flag']
    if ifa_param['padding_flag']:
        s_begin = k_tensor.shape[2] - act_seq - ifa_param['padding_size']
        s_end = k_tensor.shape[2] - ifa_param['padding_size']

    # matmul_dtype 赋值
    matmul_dtype = np.float32
    if ifa_param['in_quant_flag']:
        # 输入int8 场景，mm使用int32
        matmul_dtype = np.int32
    # elif ifa_param['out_quant_flag']:
    # 输出int8场景，mm使用float16
    #    matmul_dtype = np.float16
    else:
        # 非量化场景，mm使用float32
        matmul_dtype = np.float32
    
    if not ifa_param["is_benchmark_task"]:
        matmul_dtype = np.float16

    kv_smax = k_tensor.shape[2]
    if act_seq is None or act_seq == -1:
        k_cur = k_tensor
        v_cur = v_tensor
    else:
        kv_smax = act_seq
        if ifa_param['padding_flag']:
            k_cur = k_tensor[:, :, s_begin:s_end, :]
            v_cur = v_tensor[:, :, s_begin:s_end, :]
            if ifa_param['kv_quant_pto_flag']:
                ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset_cur'][:, :, s_begin:s_end]
            if ifa_param['k_antiquant_pto_flag'] or ifa_param['k_antiquant_ptopa_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, s_begin:s_end]
            if ifa_param['v_antiquant_pto_flag'] or ifa_param['v_antiquant_ptopa_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, s_begin:s_end]
            if ifa_param['k_antiquant_ptoh_flag'] or ifa_param['k_antiquant_ptohpa_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, s_begin:s_end]
            if ifa_param['v_antiquant_ptoh_flag'] or ifa_param['v_antiquant_ptohpa_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, s_begin:s_end]
            if ifa_param['kv_quant_ptog_flag']:
                ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale_cur'][:, :, :, s_begin:s_end, :]
                ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset_cur'][:, :, :, s_begin:s_end, :]
            if ifa_param['k_antiquant_ptog_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, :, s_begin:s_end, :]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, :, s_begin:s_end, :]
            if ifa_param['v_antiquant_ptog_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, :, s_begin:s_end, :]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, :, s_begin:s_end, :]
        else:
            k_cur = k_tensor[:, :, :act_seq, :]
            v_cur = v_tensor[:, :, :act_seq, :]
            if ifa_param['kv_quant_pto_flag']:
                ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale_cur'][:, :, :act_seq]
                ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset_cur'][:, :, :act_seq]
            if ifa_param['k_antiquant_pto_flag'] or ifa_param['k_antiquant_ptopa_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, :act_seq]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, :act_seq]
            if ifa_param['v_antiquant_pto_flag'] or ifa_param['v_antiquant_ptopa_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, :act_seq]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, :act_seq]
            if ifa_param['k_antiquant_ptoh_flag'] or ifa_param['k_antiquant_ptohpa_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, :act_seq]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, :act_seq]
            if ifa_param['v_antiquant_ptoh_flag'] or ifa_param['v_antiquant_ptohpa_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, :act_seq]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, :act_seq]
            if ifa_param['kv_quant_ptog_flag']:
                ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale_cur'][:, :, :, :act_seq, :]
                ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset_cur'][:, :, :, :act_seq, :]
            if ifa_param['k_antiquant_ptog_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, :, :act_seq, :]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, :, :act_seq, :]
            if ifa_param['v_antiquant_ptog_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, :, :act_seq, :]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, :, :act_seq, :]

    if ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
        k_cur, v_cur = kv_split_antiquant(k_cur, v_cur, ifa_param['k_antiquantScale_cur'],
                                          ifa_param['k_antiquantOffset_cur'],
                                          ifa_param['v_antiquantScale_cur'], ifa_param['v_antiquantOffset_cur'],
                                          ifa_param['k_antiquant_mode'], ifa_param['v_antiquant_mode'], 
                                          lowprecision_flag, ifa_param['k_dtype'], ifa_param['q_dtype'])
    elif ifa_param['kv_quant_flag']:
        k_cur, v_cur = antiquant(k_cur, v_cur, ifa_param['antiquantScale_cur'], ifa_param['antiquantOffset_cur'],
                                 ifa_param['kv_quant_pto_flag'], lowprecision_flag, ifa_param['k_dtype'], ifa_param['q_dtype'])

    qkBmmRes = np.matmul(matmul_dtype(q_tensor), matmul_dtype(k_cur.transpose(0, 1, 3, 2)), dtype=matmul_dtype)

    k_cur_msd = k_cur.transpose(0, 1, 3, 2)
    if not ifa_param["is_benchmark_task"] :
        if ifa_param['kv_quant_flag'] or ifa_param['v_antiquant_flag']:
            qkBmmRes = np.int32(qkBmmRes)
            k_cur_msd = np.int32(k_cur_msd)
        for b_index in range(q_tensor.shape[0]):
            for head_index in range(q_tensor.shape[1]):
                qkBmmRes[b_index, head_index, :, :] = msd(q_tensor[b_index, head_index, :, :], k_cur_msd[b_index, head_index, :, :])
    
    if not ifa_param["is_benchmark_task"]:
        qkBmmRes = np.float16(qkBmmRes)
        scalar = np.float16(scalar)
    else:
        scalar = np.float32(scalar)
    if ifa_param['in_quant_flag']:
        qkBmmRes = dequant(qkBmmRes, ifa_param['dequantScale1'], None)

    qkEleRes = qkBmmRes * scalar

    if ifa_param['p_flag']:
        if act_seq is None or act_seq == -1:
            pse_cur = ifa_param['pse_cur']
        else:
            if ifa_param['padding_flag']:
                pse_cur = ifa_param['pse_cur'][:, :, :, s_begin:s_end]
            else:
                pse_cur = ifa_param['pse_cur'][:, :, :, : act_seq]
        qkEleRes = qkEleRes + pse_cur.to(torch.float32).numpy()
    
    if ifa_param['m_flag']:
        mask_tensor = ifa_param['mask_cur'].numpy()
        if act_seq is None:
            mask_cur = mask_tensor
        else:
            if ifa_param['padding_flag']:
                mask_cur = mask_tensor[:, :, :, s_begin:s_end]
            else:
                mask_cur = mask_tensor[:, :, :, : act_seq]

        if m_dtype == 'float16':
            qkEleRes = qkEleRes + mask_cur * -10000
        else:
            qkEleRes[mask_cur.astype(np.bool_)] = -1000000000000

    # 处理输入量化（带s切分）
    if ifa_param['in_quant_flag']:

        bmm2Res = np.zeros(q_tensor.shape, dtype=np.float16)  # B, S, N, D
        s_loop_times = int((kv_smax + sinner - 1) / sinner)
        if s_loop_times > 1:
            N = q_tensor.shape[1]
            for nIdx in range(N):
                for sinner_loop_times in range(s_loop_times):
                    sinner_tail = sinner
                    # 最后一次loop
                    if sinner_loop_times == (kv_smax - 1) // sinner:
                        if kv_smax == sinner:
                            sinner_tail = sinner
                        else:
                            sinner_tail = kv_smax % sinner
                    qk_cur_loop = qkEleRes[:, nIdx:(nIdx + 1), :, (sinner_loop_times * sinner):(
                            (sinner_loop_times) * sinner + sinner_tail)]  # 1, 1, 1, Sinner
                    v_cur_loop = v_cur[:, nIdx:(nIdx + 1),
                                 (sinner_loop_times * sinner):((sinner_loop_times) * sinner + sinner_tail),
                                 :]  # 1, 1, Sinner, 1
                    # 处理第一次loop
                    if sinner_loop_times == 0:
                        softmax_max, softmax_sum, softmax_res, exp_max_res = softmax_flash(qk_cur_loop)
                        softmax_res = quant(softmax_res, ifa_param['quantScale1'], 0)
                        bmm2Res_1 = np.matmul(softmax_res, v_cur_loop, dtype=matmul_dtype)
                        bmm2Res_1 = dequant(bmm2Res_1, ifa_param['dequantScale2'], None)
                        max_front = softmax_max
                        sum_front = softmax_sum
                        o_front = bmm2Res_1
                    # 处理后续场景
                    else:
                        softmax_max, softmax_sum, softmax_res, exp_max_res = softmax_flash(qk_cur_loop,
                                                                                           max_front=max_front,
                                                                                           sum_front=sum_front,
                                                                                           update=True)
                        softmax_res = quant(softmax_res, ifa_param['quantScale1'], 0)
                        o_front_update = o_front.astype(np.float32) * exp_max_res.astype(np.float32)  # Mul
                        o_front_update = o_front_update.astype(np.float16)
                        bmm2Res_1 = np.matmul(softmax_res, v_cur_loop, dtype=matmul_dtype)
                        bmm2Res_1 = dequant(bmm2Res_1, ifa_param['dequantScale2'], None)
                        bmm2Res_1 = o_front_update.astype(np.float32) + bmm2Res_1.astype(np.float32)
                        max_front = softmax_max
                        sum_front = softmax_sum
                        o_front = bmm2Res_1.astype(np.float16)
                # 按n拼接
                bmm2Res[:, nIdx:(nIdx + 1), :, :] = o_front
        # 量化非切分场景
        else:
            softmax_res = softmax(qkEleRes)
            softmax_res = quant(softmax_res, ifa_param['quantScale1'], 0)
            bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype)
            bmm2Res = dequant(bmm2Res, ifa_param['dequantScale2'], None)
    # 非输入量化场景
    else:
        if not ifa_param["is_benchmark_task"]:
            qkEleRes = np.float16(qkEleRes)
        softmax_res, softmax_sum, softmax_max = softmax(qkEleRes)
        if lowprecision_flag and ifa_param['action_type'] == "one":
            if ifa_param['q_dtype'] == "float16":
                bmm2Res = np.matmul(softmax_res.astype(tf.float16.as_numpy_dtype).astype(np.float32), v_cur, dtype=matmul_dtype) / softmax_sum
            else:
                bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_cur, dtype=matmul_dtype) / softmax_sum
        else:
            bmm2Res = np.matmul(matmul_dtype(softmax_res), matmul_dtype(v_cur), dtype=matmul_dtype) / softmax_sum
            if not ifa_param["is_benchmark_task"]:
                if ifa_param['kv_quant_flag'] or ifa_param['v_antiquant_flag']:
                    v_cur = np.int32(v_cur)
                for b_index in range(softmax_res.shape[0]):
                    for head_index in range(softmax_res.shape[1]):
                        bmm2Res[b_index, head_index, :, :] = msd(softmax_res[b_index, head_index, :, :], v_cur[b_index, head_index, :, :])
        if lse_flag:
            lse = np.log(softmax_sum) + softmax_max
        else:
            lse = None

    if not ifa_param["is_benchmark_task"]:
        if ifa_param['q_dtype'] == "float16":
            bmm2Res = np.float16(bmm2Res).astype(np.float32)
        elif ifa_param['q_dtype'] == "bfloat16":
            bmm2Res = bmm2Res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32)
    
    return bmm2Res, lse


def str_to_bool_list(s: str):
    bool_list = []
    for item in s:
        if item == 0:
            bool_list.append(False)
        else:
            bool_list.append(True)
    return bool_list


def trans_input_dtype(input_dtype):
    if input_dtype == 'fp16':
        return 'float16'
    elif input_dtype == 'int8':
        return 'int8'
    elif input_dtype == 'uint8':
        return 'uint8'
    elif input_dtype == 'bf16':
        return 'bfloat16'
    elif input_dtype == 'bool':
        return 'bool'
    elif input_dtype == 'int32':
        return 'int32'
    elif input_dtype == 'fp32':
        return 'float32'
    elif input_dtype == 'int4':
        return 'int4'
    else:
        return input_dtype


def split_tensor_shape_by_b(input_list):
    # [[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    list_len = input_list[0][0]
    list_new = []
    for i in range(0, list_len):
        list_new_item = []
        list_new_item.append(1)
        list_new_item.append(input_list[0][1])
        list_new_item.append(input_list[0][2])
        list_new_item.append(input_list[0][3])
        list_new.append(list_new_item)
    return list_new


def split_tensor_by_b(input_list):
    # tensor:[[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    input_tensor = input_list[0]
    split_data = np.split(input_tensor, input_tensor.shape[0])
    return split_data


def _trans_antiparam_to_2n1d(shape, tensor, layout, numKeyValueHeads, d):
    # pertensor
    if shape == [2]:
        new_tensor = np.zeros((2, numKeyValueHeads, 1, d))
        new_tensor[0, :, :, :] = tensor[0]
        new_tensor[1, :, :, :] = tensor[1]
    # per channel
    elif len(shape) == 2:
        d_num = int(shape[1] / numKeyValueHeads)
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
    elif len(shape) == 3:
        new_tensor = tensor.reshape(2, 1, shape[1], shape[2]).transpose(0, 2, 1, 3)
    elif len(shape) == 4:
        if shape[1] == 1:
            new_tensor = np.transpose(tensor, (0, 2, 1, 3))
        else:
            new_tensor = tensor
    else:
        new_tensor = tensor
    return new_tensor

def _trans_antiparam_to_1n1d(shape, tensor, layout, numKeyValueHeads, d, mode):
    if mode == '0':
        # per tensor
        if shape == [1]:
            new_tensor = np.zeros((1, numKeyValueHeads, 1, d))
            new_tensor[0, :, :, :] = tensor[0]
        # per channel
        elif len(shape) == 1:
            h = shape[0]
            d_num = int(h / numKeyValueHeads)
            new_tensor = tensor.reshape(1, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
        elif len(shape) == 2:
            if shape[0] == 1:
                h = shape[1]
                d_num = int(h / numKeyValueHeads)
                new_tensor = tensor.reshape(1, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
            else:
                new_tensor = tensor.reshape(1, 1, shape[0], shape[1]).transpose(0, 2, 1, 3)
        elif len(shape) == 3:
            if shape[0] == 1:
                new_tensor = tensor.reshape(1, 1, shape[1], shape[2]).transpose(0, 2, 1, 3)
            else:
                new_tensor = tensor.reshape(1, 1, shape[0], shape[2]).transpose(0, 2, 1, 3)
        elif len(shape) == 4:
            if shape[1] == 1:
                new_tensor = tensor.transpose(0, 2, 1, 3)
            else:
                new_tensor = tensor
        else:
            new_tensor = tensor
        return new_tensor
    elif mode == '2':
        # pertensor-head
        new_tensor = np.zeros((1, numKeyValueHeads, 1, d))
        for i in range(numKeyValueHeads):
            new_tensor[:, i, :, :] = tensor[i]
        return new_tensor
    else:
        return tensor

# trans bns(kvn) -> bns(qn)
def broadcast_kv_split_dequant_tensor_ptoh(tensor, numKeyValueHeads, numheads, dtype=np.float32):
    if numKeyValueHeads == numheads:
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        B = shape[0]
        G = shape[1]
        S = shape[2]
        tensor_new = np.zeros([B, numheads, S], dtype=dtype)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, i:i + 1, :] = tensor[:, j:j + 1, :]
        return tensor_new


def _t_increattention_bnsd(ifa_param):
    batch_value = ifa_param['q_shape_bnsd'][0]
    n = ifa_param['q_shape_bnsd'][1]
    k_tensor_list = ifa_param['k_tensor_list_bnsd']
    v_tensor_list = ifa_param['v_tensor_list_bnsd']
    k_shape_list = ifa_param['k_shape_list_bnsd']
    v_shape_list = ifa_param['v_shape_list_bnsd']
    actseqlens = ifa_param['actualSeqLengths']
    actseqlens_size = len(actseqlens)
    y = np.zeros(ifa_param['q_shape_bnsd'], dtype=np.float32)
    lse = np.full([batch_value, n, 1, 1], -339999995214436424907732413799364296704)
    # 连续场景预处理
    if ifa_param['continue_flag']:
        # 处理连续场景：将单个tensor shape依据B值拆成列表
        k_shape_list = split_tensor_shape_by_b(k_shape_list)
        v_shape_list = split_tensor_shape_by_b(v_shape_list)
        # 处理连续场景：将单个tensor依据B值拆成列表
        k_tensor_list = split_tensor_by_b(k_tensor_list)
        v_tensor_list = split_tensor_by_b(v_tensor_list)

    
    for b_index in range(batch_value):

        prefix_len = 0
        if ifa_param['prefix_flag']:
            prefix_len = ifa_param['prefix_act_lens']
        # 处理实际参与计算的kv s
        if ifa_param['as_flag']:
            ifa_param['act_seq'] = act_seq = actseqlens[b_index] + prefix_len
        else:
            ifa_param['act_seq'] = act_seq = k_shape_list[b_index][2] + prefix_len

        ifa_param['k_sub_shape'] = k_sub_shape = k_shape_list[b_index]
        ifa_param['v_sub_shape'] = v_sub_shape = v_shape_list[b_index]
        if act_seq == 0 or 0 in k_sub_shape or 0 in v_sub_shape:
            continue
        ifa_param['k_sub_tensor'] = k_tensor_list[b_index]
        ifa_param['v_sub_tensor'] = v_tensor_list[b_index]
        ifa_param['q_tensor_cur'] = ifa_param['q_tensor_bnsd'][b_index:(b_index + 1), :, :, :]
        # prefix拼接
        if ifa_param['prefix_flag']:
            k_sub_tensor_temp = np.concatenate((ifa_param['k_prefix_tensor'], ifa_param['k_sub_tensor']), axis=2)
            v_sub_tensor_temp = np.concatenate((ifa_param['v_prefix_tensor'], ifa_param['v_sub_tensor']), axis=2)
            ifa_param['k_sub_tensor'] = k_sub_tensor_temp
            ifa_param['v_sub_tensor'] = v_sub_tensor_temp

        # 判断attenmask是否为空
        if not ifa_param['m_flag']:
            ifa_param['mask_cur'] = None
        else:
            ifa_param['mask_cur'] = ifa_param['m_tensor'][b_index:(b_index + 1), :, :, :]
        # 判断pse是否为空,如果非空,检查pse第一维是否为1：如果格式为1n1s,则直接传入下层计算;如果格式为bn1s,则按B拆分后进入下层。
        if not ifa_param['p_flag']:
            pse_cur = None
        elif ifa_param['p_tensor'].shape[0] == 1:
            pse_cur = ifa_param['p_tensor']
        else:
            pse_cur = ifa_param['p_tensor'][b_index:(b_index + 1), :, :, :]
        ifa_param['pse_cur'] = pse_cur
        # 伪量化Per token处理
        if ifa_param['kv_quant_pto_flag']:
            ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale'][:, b_index:(b_index + 1), :]
            ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset'][:, b_index:(b_index + 1), :]
        # ptog伪量化参数切分
        elif ifa_param['kv_quant_ptog_flag']:
            ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale'][:, b_index:(b_index + 1), :, :, :]
            ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset'][:, b_index:(b_index + 1), :, :, :]
        elif ifa_param['kv_quant_flag']:
            ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale']
            ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset']
        else:
            pass
        # kv分离伪量化处理
        if ifa_param['k_antiquant_pto_flag'] or ifa_param['k_antiquant_ptopa_flag']:
            ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale'][:, b_index:(b_index + 1), :]
            ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset'][:, b_index:(b_index + 1), :]
        elif ifa_param['k_antiquant_ptoh_flag'] or ifa_param['k_antiquant_ptohpa_flag']:
            ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale'][b_index:(b_index + 1), :, :]
            ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset'][b_index:(b_index + 1), :, :]
        elif ifa_param['k_antiquant_ptog_flag'] :
            ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale'][:, b_index:(b_index + 1), :, :, :]
            ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset'][:, b_index:(b_index + 1), :, :, :]
        elif ifa_param['k_antiquant_flag']:
            ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale']
            ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset']
        else:
            pass
        if ifa_param['v_antiquant_pto_flag'] or ifa_param['v_antiquant_ptopa_flag']:
            ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale'][:, b_index:(b_index + 1), :]
            ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset'][:, b_index:(b_index + 1), :]
        elif ifa_param['v_antiquant_ptoh_flag'] or ifa_param['v_antiquant_ptohpa_flag']:
            ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale'][b_index:(b_index + 1), :, :]
            ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset'][b_index:(b_index + 1), :, :]
        elif ifa_param['v_antiquant_ptog_flag']:
            ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale'][:, b_index:(b_index + 1), :, :, :]
            ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset'][:, b_index:(b_index + 1), :, :, :]
        elif ifa_param['v_antiquant_flag']:
            ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale']
            ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset']
        else:
            pass
        y[b_index:(b_index + 1), :, :, :], lse[b_index:(b_index + 1), :, :, :] = _t_ifaattention_act(ifa_param)

    if ifa_param['out_quant_flag']:
        if ifa_param['out_quant_pc_flag']:
            y = quant_pc(y, ifa_param['quantScale2'], ifa_param['quantOffset2'])
        else:
            y = quant(y, ifa_param['quantScale2'][0], ifa_param['quantOffset2'][0])
    return y, lse

def get_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m


def is_0_tensor(tensor_list):
    for i in tensor_list:
        if 0 in i:
            return True
    return False


def trans_to_1n1d(tensor, shape, d):
    if len(shape) == 4:
        # 1n1d->1n1d
        return tensor
    elif len(shape) == 2:
        # nd->1n1d
        n = shape[0]
        d = shape[1]
        new_tensor = tensor.reshape(1, n, 1, d)
        return new_tensor
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        if h % d != 0:
            return None
        n = int(h / d)
        new_tensor = tensor.reshape(1, n, 1, d)
        return new_tensor
    else:
        return None


def get_kvs_list(shape_list, layout):
    kvs_list = []
    for shape in shape_list:
        if layout == "BNSD":
            kvs = shape[2]
        else:
            kvs = shape[1]
        kvs_list.append(kvs)
    return kvs_list


def cut_padding_size(tensor, list, padding_size):
    shape = tensor.shape
    ms = shape[-1]
    for i in range(len(list)):
        cut_len = ms - padding_size - list[i]
        tensor[i:(i + 1), ..., :cut_len] = 1
    return tensor


def get_param(torch_tensor_list, params):
    ifa_param = {}
    ifa_param['normal_flag'] = True

    # ===参数获取===
    ifa_param['flag_list'] = str_to_bool_list(params['flaglist'])

    # >> q info
    ifa_param['q_shape'] = q_shape = params['shape_input'][0]
    ifa_param['q_dtype'] = q_dtype = trans_input_dtype(params['dtype_input'][0])
    ifa_param['q_tensor'] = torch_tensor_list[0]
    ifa_param['b'] = b = ifa_param['q_shape'][0]
    # >> kv info
    # k和v的位置通过b计算
    k1_shape = params['shape_input'][1]
    kb1 = k1_shape[0]
    # 如果第一个K的B=1,则默认kv列表长度为b;如果第一个K的B!=1,则默认kv列表长度为1
    if kb1 == 1:
        k_shape_num = v_shape_num = b
    else:
        k_shape_num = v_shape_num = 1

    ifa_param['k_start_index'] = k_start_index = 1
    ifa_param['k_end_index'] = k_end_index = int(k_shape_num)
    ifa_param['v_start_index'] = v_start_index = int(k_shape_num) + 1
    ifa_param['v_end_index'] = v_end_index = int(k_shape_num + v_shape_num)

    ifa_param['k_shape_list'] = k_ori_shape_list = params['shape_input'][k_start_index:k_end_index + 1]
    ifa_param['v_shape_list'] = v_ori_shape_list = params['shape_input'][v_start_index:v_end_index + 1]
    ifa_param['k_dtype'] = k_dtype = trans_input_dtype(params['dtype_input'][1])
    ifa_param['v_dtype'] = v_dtype = trans_input_dtype(params['dtype_input'][v_start_index])
    ifa_param['k_tensor_list'] = torch_tensor_list[k_start_index:k_end_index + 1]
    ifa_param['v_tensor_list'] = torch_tensor_list[v_start_index:v_end_index + 1]

    # >> out info
    ifa_param['out_shape'] = out_shape = q_shape
    ifa_param['out_dtype'] = trans_input_dtype(params['dtype_output'][0])

    # >> m info
    ifa_param['m_shape'] = m_shape = params['shape_input'][v_end_index + 2]
    ifa_param['m_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 2])
    ifa_param['m_tensor'] = m_tensor = torch_tensor_list[v_end_index + 2]

    # >> p info
    ifa_param['p_shape'] = p_shape = params['shape_input'][v_end_index + 1]
    ifa_param['p_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 1])
    ifa_param['p_tensor'] = p_tensor = torch_tensor_list[v_end_index + 1]
    # kv perfix

    ifa_param['k_prefix_shape'] = k_prefix_shape = params['shape_input'][v_end_index + 19]
    ifa_param['k_prefix_dtype'] = k_prefix_dtype = trans_input_dtype(params['dtype_input'][v_end_index + 19])
    ifa_param['k_prefix_tensor'] = k_prefix_tensor = torch_tensor_list[v_end_index + 19]

    ifa_param['v_prefix_shape'] = v_prefix_shape = params['shape_input'][v_end_index + 20]
    ifa_param['v_prefix_dtype'] = v_prefix_dtype = trans_input_dtype(params['dtype_input'][v_end_index + 20])

    ifa_param['v_prefix_tensor'] = v_prefix_tensor = torch_tensor_list[v_end_index + 20]
    # >> attr info
    ifa_param['actualSeqLengths_raw'] = actualSeqLengths = params['actualseqlengths']
    ifa_param['numHeads'] = numHeads = params['numheads']
    ifa_param['scaleValue'] = scaleValue = params['scalevalue']
    ifa_param['inputLayout'] = inputLayout = params['inputlayout']
    # 当numKeyValueHeads传入0时，处理为与numHeads相等
    numKeyValueHeads = params['numkeyvalueheads'] if params['numkeyvalueheads'] != 0 else numHeads
    ifa_param['numKeyValueHeads'] = numKeyValueHeads
    ifa_param['blocksize'] = params['blocksize']
    ifa_param['innerprecise'] = params['innerprecise']
    ifa_param['antiquant_mode'] = str(params['antiquant_mode'])
    # kv antiquant分离
    ifa_param['k_antiquant_mode'] = str(params['k_antiquant_mode'])
    ifa_param['v_antiquant_mode'] = str(params['v_antiquant_mode'])
    if "pre_tokens" in params.keys():
        ifa_param['pre_tokens'] = params['pre_tokens']
    else:
        ifa_param['pre_tokens'] = params['pretokens']
    if "next_tokens" in params.keys():
        ifa_param['next_tokens'] = params['next_tokens']
    else:
        ifa_param['next_tokens'] = params['nexttokens']
    if "sparse_mode" in params.keys():
        ifa_param['sparse_mode'] = params['sparse_mode']
    else:
        ifa_param['sparse_mode'] = params['sparsemode']
    ifa_param['softmax_lse_flag'] = params['softmax_lse_flag']

    # >> quant info
    ifa_param['dequantScale1'] = params['range_input'][v_end_index + 3][0]
    ifa_param['quantScale1'] = params['range_input'][v_end_index + 4][0]
    ifa_param['dequantScale2'] = params['range_input'][v_end_index + 5][0]

    ifa_param['quantScale2_shape'] = params['shape_input'][v_end_index + 6]
    ifa_param['quantScale2'] = torch_tensor_list[v_end_index + 6]

    ifa_param['quantOffset2_shape'] = params['shape_input'][v_end_index + 7]
    ifa_param['quantOffset2'] = torch_tensor_list[v_end_index + 7]

    # >> kv_s list
    ifa_param['kv_s_list'] = get_kvs_list(ifa_param['k_shape_list'], ifa_param['inputLayout'])

    # ===flag_list判断===
    ifa_param['p_flag'] = ifa_param['m_flag'] = ifa_param['as_flag'] = ifa_param['pa_flag'] = False
    ifa_param['in_quant_flag'] = ifa_param['out_quant_flag'] = ifa_param['kv_quant_flag'] = False
    ifa_param['out_quant_offset_flag'] = ifa_param['kv_quant_offset_flag'] = ifa_param['kv_quant_pto_flag'] = False
    ifa_param['k_antiquant_flag'] = ifa_param['v_antiquant_flag'] = False
    ifa_param['k_antiquant_offset_flag'] = ifa_param['v_antiquant_offset_flag'] = False
    ifa_param['k_antiquant_pc_flag'] = ifa_param['v_antiquant_pc_flag'] = False
    ifa_param['k_antiquant_pto_flag'] = ifa_param['v_antiquant_pto_flag'] = False
    ifa_param['k_antiquant_pth_flag'] = ifa_param['v_antiquant_pth_flag'] = False
    ifa_param['k_antiquant_ptoh_flag'] = ifa_param['v_antiquant_ptoh_flag'] = False
    ifa_param['k_antiquant_ptopa_flag'] = ifa_param['v_antiquant_ptopa_flag'] = False
    ifa_param['k_antiquant_ptohpa_flag'] = ifa_param['v_antiquant_ptohpa_flag'] = False
    ifa_param['k_antiquant_ptog_flag'] = ifa_param['v_antiquant_ptog_flag'] = False
    ifa_param['kv_quant_ptog_flag'] = False
    ifa_param['continue_flag'] = False
    ifa_param['padding_flag'] = False
    ifa_param['out_quant_pc_flag'] = False
    ifa_param['prefix_flag'] = ifa_param['prefix_act_flag'] = False
    flag_list = str_to_bool_list(params['flaglist'])
    kv_batch_value = ifa_param['k_shape_list'][0][0]
    if ifa_param['k_dtype'] in ['fp4_e2m1', 'fp4_e1m2', 'float4_e1m2', 'float4_e2m1']:
        ifa_param['kv_quant_ptog_flag'] = True
    # 判断是否是连续场景
    if kv_batch_value >= 1 and len(ifa_param['k_shape_list']) == 1:
        ifa_param['continue_flag'] = True
    if not flag_list[0] or not flag_list[1] or not flag_list[2] or not flag_list[-1]:
        ifa_param['normal_flag'] = False
    if 0 in q_shape:
        ifa_param['normal_flag'] = False
    if 0 in k_ori_shape_list or len(k_ori_shape_list) == 0 or is_0_tensor(k_ori_shape_list):
        ifa_param['normal_flag'] = False
    if 0 in v_ori_shape_list or len(v_ori_shape_list) == 0 or is_0_tensor(v_ori_shape_list):
        ifa_param['normal_flag'] = False
    if flag_list[3]:
        ifa_param['p_flag'] = True
    if flag_list[4]:
        ifa_param['m_flag'] = True
    if flag_list[5]:
        ifa_param['as_flag'] = True
    if flag_list[6] and flag_list[7] and flag_list[8]:
        ifa_param['in_quant_flag'] = True
    if flag_list[9]:
        ifa_param['out_quant_flag'] = True
        if flag_list[10]:
            ifa_param['out_quant_offset_flag'] = True
        else:
            # 当不传入quantOffset2时，也需要生成一个和scale一样大小的全0 Offset，用于后续计算
            ifa_param['quantOffset2'] = np.zeros(ifa_param['quantScale2_shape'], np.float32)
            ifa_param['quantOffset2_shape'] = ifa_param['quantScale2_shape']
        if ifa_param['quantScale2_shape'] != [1]:
            ifa_param['out_quant_pc_flag'] = True

    if k_dtype != q_dtype:
        # 判断kv 分离的antiquant是否存在
        if flag_list[16]:
            ifa_param["k_antiquant_flag"] = True
            if flag_list[17]:
                ifa_param['k_antiquant_offset_flag'] = True
            if ifa_param['k_antiquant_mode'] == '0':
                ifa_param['k_antiquant_pc_flag'] = True
            if ifa_param['k_antiquant_mode'] == '1':
                ifa_param['k_antiquant_pto_flag'] = True
            if ifa_param['k_antiquant_mode'] == '2':
                ifa_param['k_antiquant_pth_flag'] = True
            if ifa_param['k_antiquant_mode'] == '3':
                ifa_param['k_antiquant_ptoh_flag'] = True
            if ifa_param['k_antiquant_mode'] == '4':
                ifa_param['k_antiquant_ptopa_flag'] = True
            if ifa_param['k_antiquant_mode'] == '5':
                ifa_param['k_antiquant_ptohpa_flag'] = True
            if ifa_param['k_antiquant_mode'] == '6':
                ifa_param['k_antiquant_ptog_flag'] = True
                ifa_param['kv_quant_ptog_flag'] = False
        if flag_list[18]:
            ifa_param["v_antiquant_flag"] = True
            if flag_list[19]:
                ifa_param['v_antiquant_offset_flag'] = True
            if ifa_param['v_antiquant_mode'] == '0':
                ifa_param['v_antiquant_pc_flag'] = True
            if ifa_param['v_antiquant_mode'] == '1':
                ifa_param['v_antiquant_pto_flag'] = True
            if ifa_param['v_antiquant_mode'] == '2':
                ifa_param['v_antiquant_pth_flag'] = True
            if ifa_param['v_antiquant_mode'] == '3':
                ifa_param['v_antiquant_ptoh_flag'] = True
            if ifa_param['v_antiquant_mode'] == '4':
                ifa_param['v_antiquant_ptopa_flag'] = True
            if ifa_param['v_antiquant_mode'] == '5':
                ifa_param['v_antiquant_ptohpa_flag'] = True
            if ifa_param['v_antiquant_mode'] == '6':
                ifa_param['v_antiquant_ptog_flag'] = True
                ifa_param['kv_quant_ptog_flag'] = False
        if flag_list[11] and not flag_list[16] and not flag_list[18]:
            ifa_param['kv_quant_flag'] = True
            if flag_list[12]:
                ifa_param['kv_quant_offset_flag'] = True
            if ifa_param['antiquant_mode'] == '1':
                ifa_param['kv_quant_pto_flag'] = True
    if flag_list[13]:
        ifa_param['pa_flag'] = True
        ifa_param['bt_shape'] = params['shape_input'][12]
        ifa_param['bt_dtype'] = trans_input_dtype(params['dtype_input'][12])
        ifa_param['block_table'] = torch_tensor_list[12]
        ifa_param['k_cache_shape'] = params['shape_input'][15]
        ifa_param['v_cache_shape'] = params['shape_input'][16]

    if flag_list[14] and ifa_param['as_flag'] and ifa_param['continue_flag'] and not ifa_param['pa_flag']:
        ifa_param['padding_flag'] = True
        ifa_param['padding_size'] = params['range_input'][v_end_index + 11][0]
        if ifa_param['padding_size'] < 0:
            ifa_param['padding_size'] = 0
    if flag_list[20] and flag_list[21]:
        ifa_param['prefix_flag'] = True
    elif not flag_list[20] and not flag_list[21]:
        ifa_param['prefix_flag'] = False
    else:
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常--> prefix未成对出现！')
    if flag_list[22]:
        ifa_param['prefix_act_flag'] = True
        ifa_param['prefix_act_lens'] = params['prefix_act_lens'][0]

    # 获取kv量化参数
    if ifa_param['kv_quant_flag']:
        ifa_param['antiquantscale_shape_raw'] = antiquantscale_shape_raw = params['shape_input'][v_end_index + 8]
        if 0 in ifa_param['antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
            print('[WARNING]:异常-->  antiquantscale为空场景！')
        ifa_param['antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 8])
        ifa_param['antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 8]
        if ifa_param['kv_quant_offset_flag']:
            ifa_param['antiquantoffset_shape_raw'] = antiquantoffset_shape_raw = params['shape_input'][v_end_index + 9]
            if 0 in antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  antiquantoffset为空场景！')
            ifa_param['antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 9]
        else:
            ifa_param['antiquantoffset_shape_raw'] = antiquantscale_shape_raw
            ifa_param['antiquantoffset_tensor_raw'] = torch.zeros(antiquantscale_shape_raw).numpy()
    # 获取kv分离的量化参数
    if ifa_param['k_antiquant_flag']:
        ifa_param['k_antiquantscale_shape_raw'] = k_antiquantscale_shape_raw = params['shape_input'][v_end_index + 15]
        if 0 in ifa_param['k_antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
            print('[WARNING]:异常-->  k_antiquantscale为空场景！')
        ifa_param['k_antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 15])
        ifa_param['k_antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 15]
        if ifa_param['k_antiquant_offset_flag']:
            ifa_param['k_antiquantoffset_shape_raw'] = k_antiquantoffset_shape_raw = params['shape_input'][
                v_end_index + 16]
            ifa_param['k_antiquantoffset_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 16])
            if 0 in k_antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  k_antiquantoffset为空场景！')
            ifa_param['k_antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 16]
        else:
            ifa_param['k_antiquantoffset_shape_raw'] = k_antiquantscale_shape_raw
            ifa_param['k_antiquantoffset_dtype'] = ifa_param['k_antiquantscale_dtype']
            ifa_param['k_antiquantoffset_tensor_raw'] = torch.zeros(k_antiquantscale_shape_raw).numpy()
    if ifa_param['v_antiquant_flag']:
        ifa_param['v_antiquantscale_shape_raw'] = v_antiquantscale_shape_raw = params['shape_input'][v_end_index + 17]
        if 0 in ifa_param['v_antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
            print('[WARNING]:异常-->  v_antiquantscale为空场景！')
        ifa_param['v_antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 17])
        ifa_param['v_antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 17]
        if ifa_param['v_antiquant_offset_flag']:
            ifa_param['v_antiquantoffset_shape_raw'] = v_antiquantoffset_shape_raw = params['shape_input'][
                v_end_index + 18]
            ifa_param['v_antiquantoffset_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 18])
            if 0 in v_antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  v_antiquantoffset为空场景！')
            ifa_param['v_antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 18]
        else:
            ifa_param['v_antiquantoffset_shape_raw'] = v_antiquantscale_shape_raw
            ifa_param['v_antiquantoffset_dtype'] = ifa_param['v_antiquantscale_dtype']
            ifa_param['v_antiquantoffset_tensor_raw'] = torch.zeros(v_antiquantscale_shape_raw).numpy()

    return ifa_param


# 融合表格参数获取函数
def get_param_fus(torch_tensor_list, params):
    ifa_param = {}
    ifa_param['normal_flag'] = True

    # ===参数获取===
    ifa_param['flag_list'] = str_to_bool_list(params['flaglist'])
    # >> q info
    ifa_param['q_shape'] = q_shape = params['shape_input'][0]
    ifa_param['q_dtype'] = q_dtype = trans_input_dtype(params['dtype_input'][0])
    ifa_param['q_tensor'] = torch_tensor_list[0]
    ifa_param['b'] = b = ifa_param['q_shape'][0]
    # >> kv info
    # k和v的位置通过b计算
    k1_shape = params['shape_input'][1]
    kb1 = k1_shape[0]
    # 如果第一个K的B=1,则默认kv列表长度为b;如果第一个K的B!=1,则默认kv列表长度为1
    if kb1 == 1:
        k_shape_num = v_shape_num = b
    else:
        k_shape_num = v_shape_num = 1

    ifa_param['k_start_index'] = k_start_index = 1
    ifa_param['k_end_index'] = k_end_index = int(k_shape_num)
    ifa_param['v_start_index'] = v_start_index = int(k_shape_num) + 1
    ifa_param['v_end_index'] = v_end_index = int(k_shape_num + v_shape_num)

    ifa_param['k_shape_list'] = k_ori_shape_list = params['shape_input'][k_start_index:k_end_index + 1]
    ifa_param['v_shape_list'] = v_ori_shape_list = params['shape_input'][v_start_index:v_end_index + 1]
    ifa_param['k_dtype'] = k_dtype = trans_input_dtype(params['dtype_input'][1])
    ifa_param['v_dtype'] = v_dtype = trans_input_dtype(params['dtype_input'][v_start_index])
    ifa_param['k_tensor_list'] = torch_tensor_list[k_start_index:k_end_index + 1]
    ifa_param['v_tensor_list'] = torch_tensor_list[v_start_index:v_end_index + 1]

    # >> out info
    ifa_param['out_shape'] = out_shape = q_shape
    ifa_param['out_dtype'] = trans_input_dtype(params['dtype_output'][0])

    # >> p info
    ifa_param['p_shape'] = p_shape = params['shape_input'][v_end_index + 1]
    ifa_param['p_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 1])
    ifa_param['p_tensor'] = p_tensor = torch_tensor_list[v_end_index + 1]

    # >> m info
    ifa_param['m_shape'] = m_shape = params['shape_input'][v_end_index + 2]
    ifa_param['m_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 2])
    ifa_param['m_tensor'] = m_tensor = torch_tensor_list[v_end_index + 2]

    # kv prefix
    ifa_param['k_prefix_shape'] = k_prefix_shape = params['shape_input'][v_end_index + 17]
    ifa_param['k_prefix_dtype'] = k_prefix_dtype = trans_input_dtype(params['dtype_input'][v_end_index + 17])
    ifa_param['k_prefix_tensor'] = k_prefix_tensor = torch_tensor_list[v_end_index + 17]

    ifa_param['v_prefix_shape'] = v_prefix_shape = params['shape_input'][v_end_index + 18]
    ifa_param['v_prefix_dtype'] = v_prefix_dtype = trans_input_dtype(params['dtype_input'][v_end_index + 18])

    ifa_param['v_prefix_tensor'] = v_prefix_tensor = torch_tensor_list[v_end_index + 18]
    # >> attr info
    ifa_param['actualSeqLengths_raw'] = actualSeqLengths = params['actualseqlengthskv']
    ifa_param['numHeads'] = numHeads = params['numheads']
    ifa_param['scaleValue'] = scaleValue = params['scalevalue']
    ifa_param['inputLayout'] = inputLayout = params['inputlayout']
    # 当numKeyValueHeads传入0时，处理为与numHeads相等
    numKeyValueHeads = params['numkeyvalueheads'] if params['numkeyvalueheads'] != 0 else numHeads
    ifa_param['numKeyValueHeads'] = numKeyValueHeads
    ifa_param['blocksize'] = params['blocksize']
    ifa_param['innerprecise'] = params['innerprecise']
    ifa_param['antiquant_mode'] = str(params['antiquant_mode'])
    # kv antiquant分离
    ifa_param['k_antiquant_mode'] = str(params['k_antiquant_mode'])
    ifa_param['v_antiquant_mode'] = str(params['v_antiquant_mode'])

    ifa_param['pre_tokens'] = params['pretokens']
    ifa_param['next_tokens'] = params['nexttokens']
    ifa_param['sparse_mode'] = params['sparsemode']
    ifa_param['softmax_lse_flag'] = params['softmax_lse_flag']

    # >> quant info
    ifa_param['dequantScale1'] = params['range_input'][v_end_index + 3][0]
    ifa_param['quantScale1'] = params['range_input'][v_end_index + 4][0]
    ifa_param['dequantScale2'] = params['range_input'][v_end_index + 5][0]

    ifa_param['quantScale2_shape'] = params['shape_input'][v_end_index + 6]
    ifa_param['quantScale2'] = torch_tensor_list[v_end_index + 6]

    ifa_param['quantOffset2_shape'] = params['shape_input'][v_end_index + 7]
    ifa_param['quantOffset2'] = torch_tensor_list[v_end_index + 7]

    # >> kv_s list
    ifa_param['kv_s_list'] = get_kvs_list(ifa_param['k_shape_list'], ifa_param['inputLayout'])

    # ===flag_list判断===
    ifa_param['p_flag'] = ifa_param['m_flag'] = ifa_param['as_flag'] = ifa_param['pa_flag'] = False
    ifa_param['in_quant_flag'] = ifa_param['out_quant_flag'] = ifa_param['kv_quant_flag'] = False
    ifa_param['out_quant_offset_flag'] = ifa_param['kv_quant_offset_flag'] = ifa_param['kv_quant_pto_flag'] = False
    ifa_param['k_antiquant_flag'] = ifa_param['v_antiquant_flag'] = False
    ifa_param['k_antiquant_offset_flag'] = ifa_param['v_antiquant_offset_flag'] = False
    ifa_param['k_antiquant_pc_flag'] = ifa_param['v_antiquant_pc_flag'] = False
    ifa_param['k_antiquant_pto_flag'] = ifa_param['v_antiquant_pto_flag'] = False
    ifa_param['k_antiquant_pth_flag'] = ifa_param['v_antiquant_pth_flag'] = False
    ifa_param['k_antiquant_ptoh_flag'] = ifa_param['v_antiquant_ptoh_flag'] = False
    ifa_param['k_antiquant_ptopa_flag'] = ifa_param['v_antiquant_ptopa_flag'] = False
    ifa_param['k_antiquant_ptohpa_flag'] = ifa_param['v_antiquant_ptohpa_flag'] = False
    ifa_param['k_antiquant_ptog_flag'] = ifa_param['v_antiquant_ptog_flag'] = False
    ifa_param['kv_quant_ptog_flag'] = False
    ifa_param['continue_flag'] = False
    ifa_param['padding_flag'] = False
    ifa_param['out_quant_pc_flag'] = False
    ifa_param['prefix_flag'] = ifa_param['prefix_act_flag'] = False
    flag_list = str_to_bool_list(params['flaglist'])
    kv_batch_value = ifa_param['k_shape_list'][0][0]
    if ifa_param['k_dtype'] in ['fp4_e2m1', 'fp4_e1m2', 'float4_e1m2', 'float4_e2m1']:
        ifa_param['kv_quant_ptog_flag'] = True
    # 判断是否是连续场景
    if kv_batch_value >= 1 and len(ifa_param['k_shape_list']) == 1:
        ifa_param['continue_flag'] = True
    if not flag_list[0] or not flag_list[1] or not flag_list[2] or not flag_list[24]:
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常-->  q/k/v/out不输入场景！')
    if 0 in q_shape:
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常-->  q为空场景！')
    if 0 in k_ori_shape_list or len(k_ori_shape_list) == 0 or is_0_tensor(k_ori_shape_list):
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常-->  k为空场景！')
    if 0 in v_ori_shape_list or len(v_ori_shape_list) == 0 or is_0_tensor(v_ori_shape_list):
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常-->  v为空场景！')
    if flag_list[3]:
        ifa_param['p_flag'] = True
    if flag_list[4]:
        ifa_param['m_flag'] = True
    if flag_list[6]:
        ifa_param['as_flag'] = True
    if flag_list[7] and flag_list[8] and flag_list[9]:
        ifa_param['in_quant_flag'] = True
    if flag_list[10]:
        ifa_param['out_quant_flag'] = True
        if flag_list[11]:
            ifa_param['out_quant_offset_flag'] = True
        else:
            # 当不传入quantOffset2时，也需要生成一个和scale一样大小的全0 Offset，用于后续计算
            ifa_param['quantOffset2'] = np.zeros(ifa_param['quantScale2_shape'], np.float32)
            ifa_param['quantOffset2_shape'] = ifa_param['quantScale2_shape']
        if ifa_param['quantScale2_shape'] != [1]:
            ifa_param['out_quant_pc_flag'] = True

    if k_dtype != q_dtype:
        # 判断kv 分离的antiquant是否存在
        if flag_list[17]:
            ifa_param["k_antiquant_flag"] = True
            if flag_list[18]:
                ifa_param['k_antiquant_offset_flag'] = True
            if ifa_param['k_antiquant_mode'] == '0':
                ifa_param['k_antiquant_pc_flag'] = True
            if ifa_param['k_antiquant_mode'] == '1':
                ifa_param['k_antiquant_pto_flag'] = True
            if ifa_param['k_antiquant_mode'] == '2':
                ifa_param['k_antiquant_pth_flag'] = True
            if ifa_param['k_antiquant_mode'] == '3':
                ifa_param['k_antiquant_ptoh_flag'] = True
            if ifa_param['k_antiquant_mode'] == '4':
                ifa_param['k_antiquant_ptopa_flag'] = True
            if ifa_param['k_antiquant_mode'] == '5':
                ifa_param['k_antiquant_ptohpa_flag'] = True
            if ifa_param['k_antiquant_mode'] == '6':
                ifa_param['k_antiquant_ptog_flag'] = True
                ifa_param['kv_quant_ptog_flag'] = False
        if flag_list[19]:
            ifa_param["v_antiquant_flag"] = True
            if flag_list[20]:
                ifa_param['v_antiquant_offset_flag'] = True
            if ifa_param['v_antiquant_mode'] == '0':
                ifa_param['v_antiquant_pc_flag'] = True
            if ifa_param['v_antiquant_mode'] == '1':
                ifa_param['v_antiquant_pto_flag'] = True
            if ifa_param['v_antiquant_mode'] == '2':
                ifa_param['v_antiquant_pth_flag'] = True
            if ifa_param['v_antiquant_mode'] == '3':
                ifa_param['v_antiquant_ptoh_flag'] = True
            if ifa_param['v_antiquant_mode'] == '4':
                ifa_param['v_antiquant_ptopa_flag'] = True
            if ifa_param['v_antiquant_mode'] == '5':
                ifa_param['v_antiquant_ptohpa_flag'] = True
            if ifa_param['v_antiquant_mode'] == '6':
                ifa_param['v_antiquant_ptog_flag'] = True
                ifa_param['kv_quant_ptog_flag'] = False
        if flag_list[12] and not flag_list[17] and not flag_list[19]:

            ifa_param['kv_quant_flag'] = True
            if flag_list[13]:
                ifa_param['kv_quant_offset_flag'] = True
            if ifa_param['antiquant_mode'] == '1':
                ifa_param['kv_quant_pto_flag'] = True
    if flag_list[14]:
        ifa_param['pa_flag'] = True
        ifa_param['bt_shape'] = params['shape_input'][12]
        ifa_param['bt_dtype'] = trans_input_dtype(params['dtype_input'][12])
        ifa_param['block_table'] = torch_tensor_list[12]
        ifa_param['k_cache_shape'] = params['shape_input'][21]
        ifa_param['v_cache_shape'] = params['shape_input'][22]

    if flag_list[16] and ifa_param['as_flag'] and ifa_param['continue_flag'] and not ifa_param['pa_flag']:
        ifa_param['padding_flag'] = True
        ifa_param['padding_size'] = params['range_input'][v_end_index + 12][0]
        if ifa_param['padding_size'] < 0:
            ifa_param['padding_size'] = 0
    if flag_list[21] and flag_list[22]:
        ifa_param['prefix_flag'] = True
    elif not flag_list[21] and not flag_list[22]:
        ifa_param['prefix_flag'] = False
    else:
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常--> prefix未成对出现！')
    if flag_list[23]:
        ifa_param['prefix_act_flag'] = True
        ifa_param['prefix_act_lens'] = params['prefix_act_lens'][0]

    # 获取kv量化参数
    if ifa_param['kv_quant_flag']:
        ifa_param['antiquantscale_shape_raw'] = antiquantscale_shape_raw = params['shape_input'][v_end_index + 8]
        if 0 in ifa_param['antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
            print('[WARNING]:异常-->  antiquantscale为空场景！')
        ifa_param['antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 8])
        ifa_param['antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 8]
        if ifa_param['kv_quant_offset_flag']:
            ifa_param['antiquantoffset_shape_raw'] = antiquantoffset_shape_raw = params['shape_input'][v_end_index + 9]
            if 0 in antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  antiquantoffset为空场景！')
            ifa_param['antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 9]
        else:
            ifa_param['antiquantoffset_shape_raw'] = antiquantscale_shape_raw
            ifa_param['antiquantoffset_tensor_raw'] = torch.zeros(antiquantscale_shape_raw).numpy()
    # 获取kv分离的量化参数
    if ifa_param['k_antiquant_flag']:
        ifa_param['k_antiquantscale_shape_raw'] = k_antiquantscale_shape_raw = params['shape_input'][v_end_index + 13]
        if 0 in ifa_param['k_antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
            print('[WARNING]:异常-->  k_antiquantscale为空场景！')
        ifa_param['k_antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 13])
        ifa_param['k_antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 13]
        if ifa_param['k_antiquant_offset_flag']:
            ifa_param['k_antiquantoffset_shape_raw'] = k_antiquantoffset_shape_raw = params['shape_input'][
                v_end_index + 14]
            ifa_param['k_antiquantoffset_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 14])
            if 0 in k_antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  k_antiquantoffset为空场景！')
            ifa_param['k_antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 14]
        else:
            ifa_param['k_antiquantoffset_shape_raw'] = k_antiquantscale_shape_raw
            ifa_param['k_antiquantoffset_dtype'] = ifa_param['k_antiquantscale_dtype']
            ifa_param['k_antiquantoffset_tensor_raw'] = torch.zeros(k_antiquantscale_shape_raw).numpy()
    if ifa_param['v_antiquant_flag']:
        ifa_param['v_antiquantscale_shape_raw'] = v_antiquantscale_shape_raw = params['shape_input'][v_end_index + 15]
        if 0 in ifa_param['v_antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
            print('[WARNING]:异常-->  v_antiquantscale为空场景！')
        ifa_param['v_antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 15])
        ifa_param['v_antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 15]
        if ifa_param['v_antiquant_offset_flag']:
            ifa_param['v_antiquantoffset_shape_raw'] = v_antiquantoffset_shape_raw = params['shape_input'][
                v_end_index + 16]
            ifa_param['v_antiquantoffset_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 16])
            if 0 in v_antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  v_antiquantoffset为空场景！')
            ifa_param['v_antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 16]
        else:
            ifa_param['v_antiquantoffset_shape_raw'] = v_antiquantscale_shape_raw
            ifa_param['v_antiquantoffset_dtype'] = ifa_param['v_antiquantscale_dtype']
            ifa_param['v_antiquantoffset_tensor_raw'] = torch.zeros(v_antiquantscale_shape_raw).numpy()

    return ifa_param


def left_padding_handler(tensor, act_list, padding_size):
    new_tensor = tensor.clone()
    b = tensor.shape[0]
    smax = tensor.shape[1]
    for bindex in range(b):
        act_s = act_list[bindex]
        start_index = smax - padding_size - act_s
        end_index = smax - padding_size
        new_tensor[bindex:bindex + 1, 0:act_s, :, :] = tensor[bindex:bindex + 1, start_index:end_index, :, :]
    return new_tensor


def left_padding_handler_mask(tensor, act_list, padding_size):
    new_tensor = tensor.clone()
    b = tensor.shape[0]
    smax = tensor.shape[1]
    for bindex in range(b):
        act_s = act_list[bindex]
        start_index = smax - padding_size - act_s
        end_index = smax - padding_size
        new_tensor[bindex:bindex + 1, 0:act_s] = tensor[bindex:bindex + 1, start_index:end_index]
        new_tensor[bindex:bindex + 1, act_s:] = 1
    return new_tensor


# 将int32类型的数据,拆成Int4,存进int8
def trans_int32_2_int4(input_int32):
    # 将Int32类型的数据按bit位平均拆成8份，每份长度为4bit
    parts = [(input_int32 >> i) & 0xf for i in range(0, 32, 4)]
    output_int4 = []
    for part in parts:
        # 将每份数据构造成一个Int8类型的数据
        sign = (part >> 3) & 0x1
        # 剩余的值作为int8最后3个bit的值
        value = part & 0x7
        if sign == 1:
            # 如果符号位为1，则需要将value转换成负数
            value = -((value ^ 0x7) + 1)
        output_int4.append(value)
    return output_int4


# 将int32的tensor拆分，存入Int8类型的tensor
def trans_int32_2_int4_tensor_bnsd(tensor_int32, shape_int32):
    shape_int4 = [shape_int32[0], shape_int32[1], shape_int32[2], shape_int32[3] * 8]
    tensor_int4 = np.zeros(shape_int4).astype(np.int8)
    for Bid in range(shape_int32[0]):
        for Nid in range(shape_int32[1]):
            for Sid in range(shape_int32[2]):
                for Did in range(shape_int32[3]):
                    int4_data_list = trans_int32_2_int4(
                        int(tensor_int32[Bid:Bid + 1, Nid:Nid + 1, Sid:Sid + 1, Did:Did + 1]))
                    for i in range(8):
                        tensor_int4[Bid:Bid + 1, Nid:Nid + 1, Sid:Sid + 1, 8 * Did + i:8 * Did + i + 1] = \
                            int4_data_list[i]
    return tensor_int4, shape_int4


# pertoken_pa 场景，将scale/offset从BB转为1BS
def trans_bb_2_1bs(input_tensor, tables, batch):
    block_size = input_tensor.shape[1]
    max_seq = tables.shape[1] * block_size
    max_block_per_seq = math.ceil(max_seq / block_size)
    output_tensor = np.zeros((1, batch, max_seq), dtype=np.float32)
    for bindex in range(batch):
        table = tables[bindex]
        for block_base_index in range(max_block_per_seq):
            block_index = table[block_base_index]
            if block_index == -1:
                break
            if (block_base_index + 1) * block_size > max_seq:
                lens = max_seq - block_base_index * block_size
                output_tensor[0, bindex, block_base_index * block_size:max_seq] = input_tensor[block_index][0:lens]
            else:
                output_tensor[0, bindex, block_base_index * block_size:(block_base_index + 1) * block_size] = \
                    input_tensor[block_index]
    return output_tensor

# perheadtoken_pa 场景，将scale/offset从BnB转为1BnS
def trans_bb_2_1bns(input_tensor, tables, batch):
    num_kv_head = input_tensor.shape[1]
    block_size = input_tensor.shape[2]
    max_seq = tables.shape[1] * block_size
    max_block_per_seq = math.ceil(max_seq / block_size)
    output_tensor = np.zeros((batch, num_kv_head, max_seq), dtype=np.float32)

    for bindex in range(batch):
        table = tables[bindex]
        for block_base_index in range(max_block_per_seq):
            block_index = table[block_base_index]
            if block_index == -1:
                break
            if (block_base_index + 1) * block_size > max_seq:
                lens = max_seq - block_base_index * block_size
                output_tensor[bindex, :, block_base_index * block_size:max_seq] = input_tensor[block_index][:][0:lens]
            else:
                output_tensor[bindex, :, block_base_index * block_size:(block_base_index + 1) * block_size] = \
                input_tensor[block_index]
    return output_tensor

def aclnn_op_func_ifa_cpu(torch_tensor_list, params):
    fused_flag = False
    # ==判断是否是融合表格==
    if 'fused_flag' in params.keys():
        ifa_param = get_param_fus(torch_tensor_list, params)
        fused_flag = True
    else:
        ifa_param = get_param(torch_tensor_list, params)

    if not ifa_param['normal_flag']:
        return torch.zeros(ifa_param['out_shape'])
    ifa_param["action_type"] = params["action_type"]

    # ===参数预处理===
    actualSeqLengths = ifa_param['actualSeqLengths_raw']
    q_shape = ifa_param['q_shape']
    inputLayout = ifa_param['inputLayout']
    numHeads = ifa_param['numHeads']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    p_shape = ifa_param['p_shape']
    k_start_index = ifa_param['k_start_index']
    k_end_index = ifa_param['k_end_index']
    v_start_index = ifa_param['v_start_index']
    v_end_index = ifa_param['v_end_index']
    out_shape = ifa_param['out_shape']
    k_ori_shape_list = ifa_param['k_shape_list']
    v_ori_shape_list = ifa_param['v_shape_list']
    k_dtype = ifa_param['k_dtype']
    v_dtype = ifa_param['v_dtype']
    m_tensor = ifa_param['m_tensor']
    m_shape = ifa_param['m_shape']

    # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    if len(actualSeqLengths) == 1 and len(actualSeqLengths) != q_shape[0]:
        actualSeqLengths_item = actualSeqLengths[0]
        for b_count in range(q_shape[0] - 1):
            actualSeqLengths.append(actualSeqLengths_item)
    if len(actualSeqLengths) > q_shape[0]:
        actualSeqLengths = actualSeqLengths[:q_shape[0]]

    ifa_param['actualSeqLengths'] = actualSeqLengths

    # >> q 预处理：将q转换为bnsd
    q_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(torch_tensor_list[0], q_shape, inputLayout, numHeads)
    ifa_param['q_tensor_bnsd'] = q_tensor
    ifa_param['q_shape_bnsd'] = q_bnsd_shape
    ifa_param['q_d'] = q_bnsd_shape[3]
    ifa_param['q_b'] = q_bnsd_shape[0]

    # >> kv预处理：1、将kv list 转换为bnsd 2、GQA场景，将kvn扩展为qn
    k_tensor_list = []
    v_tensor_list = []
    k_shape_list = []
    v_shape_list = []
    k_tensor_bnsd_list = []
    v_tensor_bnsd_list = []
    k_shape_bnsd_list = []
    v_shape_bnsd_list = []
    x = 0
    ifa_param['ks_max'] = 0
    for i in range(k_start_index, k_end_index + 1):
        k_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(torch_tensor_list[i], k_ori_shape_list[x], inputLayout,
                                                        numKeyValueHeads)
        x = x + 1
        k_tensor_bnsd_list.append(k_tensor)
        k_shape_bnsd_list.append(k_bnsd_shape)
        # k int32预处理
        if ifa_param['k_dtype'] == "int32":
            k_tensor, k_bnsd_shape = trans_int32_2_int4_tensor_bnsd(k_tensor, k_bnsd_shape)

        if numKeyValueHeads != numHeads:
            k_tensor, k_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_dtype)
        if k_bnsd_shape[2] > ifa_param['ks_max']:
            ifa_param['ks_max'] = k_bnsd_shape[2]
        k_tensor_list.append(k_tensor)
        k_shape_list.append(k_bnsd_shape)

    x = 0
    for i in range(v_start_index, v_end_index + 1):
        v_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(torch_tensor_list[i], v_ori_shape_list[x], inputLayout,
                                                        numKeyValueHeads)
        x = x + 1
        # k int32预处理
        v_tensor_bnsd_list.append(v_tensor)
        v_shape_bnsd_list.append(v_bnsd_shape)
        if ifa_param['v_dtype'] == "int32":
            v_tensor, v_bnsd_shape = trans_int32_2_int4_tensor_bnsd(v_tensor, v_bnsd_shape)
        if numKeyValueHeads != numHeads:
            # print("broadcast v")
            v_tensor, v_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor, v_dtype)
        v_tensor_list.append(v_tensor)
        v_shape_list.append(v_bnsd_shape)

    ifa_param['k_tensor_list_bnsd'] = k_tensor_list
    ifa_param['k_shape_list_bnsd'] = k_shape_list
    ifa_param['v_tensor_list_bnsd'] = v_tensor_list
    ifa_param['v_shape_list_bnsd'] = v_shape_list

    # >> left padding 异常场景处理：起点/终点异常，返回全0
    if ifa_param['padding_flag']:
        max_act_seq = max(actualSeqLengths)
        kv_s = ifa_param['ks_max']
        if kv_s - ifa_param['padding_size'] - max_act_seq < 0:
            print('[WARNING]:paddingsize 溢出，输出空tensor！')
            return torch.zeros(out_shape)

    
    if ifa_param['p_flag']:
        p_tensor = torch_tensor_list[3]
    # p_tensor = torch_tensor_list[3]
    
    # >> m预处理：1、将m扩展为BN1S  2、padding场景下，偏移部分设置为1 3、针对FP16格式，将tensor转成0/1
    m_dtype = get_pt_dtype(params['dtype_input'][v_end_index + 2])
    m_tensor = torch.tensor(m_tensor, dtype=m_dtype)
    if ifa_param['m_flag']:
        m_rewrite_flag = False
        if ifa_param['padding_flag']:
            m_tensor = cut_padding_size(m_tensor, actualSeqLengths, ifa_param['padding_size'])
            m_rewrite_flag = True
        if ifa_param['m_dtype'] == "float16":
            m_tensor = torch.where(m_tensor < 0.8, torch.tensor(0, dtype=torch.float16),
                                   torch.tensor(1, dtype=torch.float16))
            m_rewrite_flag = True
        if m_rewrite_flag:
            print(f"[INFO]rewrite attenmask")
            tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite',
                                          tensors=m_tensor, params=params, data_range=[0, 2])
        ifa_param['m_tensor'] = _t_broadcast_mask_n(m_tensor, m_shape, numHeads, m_shape[0])

    # >> 后量化参数处理：1、给dtype赋值 2、如果是perchannel模式，要将后量化参数统一转换成1n1d格式
    if ifa_param['out_quant_flag']:
        ifa_param['quantScale2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 6])
        ifa_param['quantOffset2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 7])
    if ifa_param['out_quant_pc_flag']:
        ifa_param['quantScale2'] = trans_to_1n1d(ifa_param['quantScale2'], ifa_param['quantScale2_shape'],
                                                 ifa_param['q_shape_bnsd'][3])
        if ifa_param['quantScale2'] is None:
            return torch.zeros(out_shape)
        ifa_param['quantOffset2'] = trans_to_1n1d(ifa_param['quantOffset2'], ifa_param['quantOffset2_shape'],
                                                  ifa_param['q_shape_bnsd'][3])
        if ifa_param['quantOffset2'] is None:
            return torch.zeros(out_shape)
        ifa_param['quantScale2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 6])
        ifa_param['quantOffset2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 7])

    if ifa_param['pa_flag'] and "output" not in params["action_type"]:
        block_table = params['block_table']
        k_cache = torch_tensor_list[21]
        v_cache = torch_tensor_list[22]
       
    if ifa_param['kv_quant_flag']:
        antiquantscale_shape_raw = ifa_param['antiquantscale_shape_raw']
        antiquantscale_tensor_raw = ifa_param['antiquantscale_tensor_raw']
        antiquantoffset_shape_raw = ifa_param['antiquantoffset_shape_raw']
        antiquantoffset_tensor_raw = ifa_param['antiquantoffset_tensor_raw']

        if ifa_param['kv_quant_ptog_flag']:
            antiquantscale_tensor = broadcast_kv_dequant_tensor_fp8e8m0(antiquantscale_tensor_raw, numKeyValueHeads,
                                                                        numHeads)
            antiquantoffset_tensor = broadcast_kv_dequant_tensor_fp8e8m0(antiquantoffset_tensor_raw, numKeyValueHeads,
                                                                         numHeads)

            ifa_param['antiquantScale'] = antiquantscale_tensor
            ifa_param['antiquantOffset'] = antiquantoffset_tensor
        # pertensor/perchannel ->1、将kv量化参数统一转换成2n1d 2、GQA场景下，扩展kv量化参数
        elif not ifa_param['kv_quant_pto_flag']:
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_tensor = _trans_antiparam_to_2n1d(antiquantscale_shape_raw, antiquantscale_tensor_raw,
                                                             inputLayout, numKeyValueHeads, ifa_param['q_d'])
            antiquantoffset_tensor = _trans_antiparam_to_2n1d(antiquantoffset_shape_raw, antiquantoffset_tensor_raw,
                                                              inputLayout, numKeyValueHeads, ifa_param['q_d'])
            # GQA场景，扩展kv反量化参数
            q_dtype_np = get_np_dtype(params['dtype_input'][0])
            antiquantscale_tensor = broadcast_kv_dequant_tensor(antiquantscale_tensor, numKeyValueHeads, numHeads,
                                                                q_dtype_np)
            antiquantoffset_tensor = broadcast_kv_dequant_tensor(antiquantoffset_tensor, numKeyValueHeads, numHeads,
                                                                 q_dtype_np)

            ifa_param['antiquantScale'] = antiquantscale_tensor
            ifa_param['antiquantOffset'] = antiquantoffset_tensor
        else:
            ifa_param['antiquantScale'] = antiquantscale_tensor_raw
            ifa_param['antiquantOffset'] = antiquantoffset_tensor_raw
    # 判断kv分离的伪量化
    if ifa_param["k_antiquant_flag"] and not ifa_param["v_antiquant_flag"]:
        print("[ERROR] k_antiquant exit, but v_antiquant not exit")
        exit(-1)
    elif not ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
        print("[ERROR] v_antiquant exit, but k_antiquant not exit")
        exit(-1)
    else:
        if ifa_param["k_antiquant_flag"]:
            k_antiquantscale_shape_raw = ifa_param['k_antiquantscale_shape_raw']
            k_antiquantscale_tensor_raw = ifa_param['k_antiquantscale_tensor_raw']
            k_antiquantoffset_shape_raw = ifa_param['k_antiquantoffset_shape_raw']
            k_antiquantoffset_tensor_raw = ifa_param['k_antiquantoffset_tensor_raw']
            print(f"k_antiquant_pto_flag: {ifa_param['k_antiquant_pto_flag']}")
            print(f"k_antiquant_ptoh_flag: {ifa_param['k_antiquant_ptoh_flag']}")
            print(f"k_antiquant_ptopa_flag: {ifa_param['k_antiquant_ptopa_flag']}")
            print(f"k_antiquant_ptohpa_flag: {ifa_param['k_antiquant_ptohpa_flag']}")
            # pertoken_head
            if ifa_param['k_antiquant_ptoh_flag']:
                # GQA场景，扩展kv反量化参数
                k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantscale_tensor_raw,
                                                                                 numKeyValueHeads,
                                                                                 numHeads)
                k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantoffset_tensor_raw,
                                                                                  numKeyValueHeads,
                                                                                  numHeads)
                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
            # pertensor/perchannel ->1、将kv量化参数统一转换成1n1d 2、GQA场景下，扩展kv量化参数
            elif ifa_param['k_antiquant_pc_flag'] or ifa_param['k_antiquant_pth_flag']:
                k_antiquantscale_tensor = _trans_antiparam_to_1n1d(k_antiquantscale_shape_raw,
                                                                   k_antiquantscale_tensor_raw,
                                                                   inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                   ifa_param['k_antiquant_mode'])
                k_antiquantoffset_tensor = _trans_antiparam_to_1n1d(k_antiquantoffset_shape_raw,
                                                                    k_antiquantoffset_tensor_raw,
                                                                    inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                    ifa_param['k_antiquant_mode'])
                # GQA场景，扩展kv反量化参数
                q_dtype_np = get_np_dtype(params['dtype_input'][0])
                k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor(k_antiquantscale_tensor, numKeyValueHeads,
                                                                            numHeads, q_dtype_np)
                k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor(k_antiquantoffset_tensor, numKeyValueHeads,
                                                                             numHeads, q_dtype_np)
                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
            # pertoken_pa -> 1、将bb转成1bs
            elif ifa_param['k_antiquant_ptopa_flag']:
                if ifa_param['pa_flag']:
                    ifa_param['k_antiquantScale'] = trans_bb_2_1bs(k_antiquantscale_tensor_raw,
                                                                   ifa_param['block_table'], ifa_param['q_b'])
                    ifa_param['k_antiquantOffset'] = trans_bb_2_1bs(k_antiquantoffset_tensor_raw,
                                                                    ifa_param['block_table'], ifa_param['q_b'])
                else:
                    exit(1)
            # pertoken_head_pa -> 1、将bb转成1bns
            elif ifa_param['k_antiquant_ptohpa_flag']:
                if ifa_param['pa_flag']:
                    k_antiquantscale_tensor = trans_bb_2_1bns(k_antiquantscale_tensor_raw, ifa_param['block_table'], ifa_param['q_b'])
                    k_antiquantoffset_tensor = trans_bb_2_1bns(k_antiquantoffset_tensor_raw, ifa_param['block_table'], ifa_param['q_b'])
                    k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantscale_tensor, numKeyValueHeads, numHeads)
                    k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantoffset_tensor, numKeyValueHeads, numHeads)
                    ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                    ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
                else:
                    exit(1)
            elif ifa_param['k_antiquant_ptog_flag']:
                k_antiquantscale_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(k_antiquantscale_tensor_raw, numKeyValueHeads, numHeads)
                k_antiquantoffset_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(k_antiquantoffset_tensor_raw, numKeyValueHeads, numHeads)
                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor

            # pertoken -> 如果是BS,转成1BS
            else:
                ifa_param['k_antiquantScale'] = _trans_antiparam_to_1bs(k_antiquantscale_tensor_raw)
                ifa_param['k_antiquantOffset'] = _trans_antiparam_to_1bs(k_antiquantoffset_tensor_raw)

        if ifa_param["v_antiquant_flag"]:
            v_antiquantscale_shape_raw = ifa_param['v_antiquantscale_shape_raw']
            v_antiquantscale_tensor_raw = ifa_param['v_antiquantscale_tensor_raw']
            v_antiquantoffset_shape_raw = ifa_param['v_antiquantoffset_shape_raw']
            v_antiquantoffset_tensor_raw = ifa_param['v_antiquantoffset_tensor_raw']
            # pertoken_head
            if ifa_param['v_antiquant_ptoh_flag']:
                # GQA场景，扩展kv反量化参数
                v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantscale_tensor_raw,
                                                                                 numKeyValueHeads,
                                                                                 numHeads)
                v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantoffset_tensor_raw,
                                                                                  numKeyValueHeads,
                                                                                  numHeads)
                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertensor/perchannel ->1、将kv量化参数统一转换成1n1d 2、GQA场景下，扩展kv量化参数
            elif ifa_param['v_antiquant_pc_flag'] or ifa_param['v_antiquant_pth_flag']:
                v_antiquantscale_tensor = _trans_antiparam_to_1n1d(v_antiquantscale_shape_raw,
                                                                   v_antiquantscale_tensor_raw,
                                                                   inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                   ifa_param['v_antiquant_mode'])
                v_antiquantoffset_tensor = _trans_antiparam_to_1n1d(v_antiquantoffset_shape_raw,
                                                                    v_antiquantoffset_tensor_raw,
                                                                    inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                    ifa_param['v_antiquant_mode'])
                # GQA场景，扩展kv反量化参数
                q_dtype_np = get_np_dtype(params['dtype_input'][0])
                v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor(v_antiquantscale_tensor, numKeyValueHeads,
                                                                            numHeads, q_dtype_np)
                v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor(v_antiquantoffset_tensor, numKeyValueHeads,
                                                                             numHeads, q_dtype_np)
                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertoken_pa -> 1、将bb转成1bs
            elif ifa_param['v_antiquant_ptopa_flag']:
                if ifa_param['pa_flag']:
                    ifa_param['v_antiquantScale'] = trans_bb_2_1bs(v_antiquantscale_tensor_raw, ifa_param['block_table'], ifa_param['q_b'])
                    ifa_param['v_antiquantOffset'] = trans_bb_2_1bs(v_antiquantoffset_tensor_raw, ifa_param['block_table'], ifa_param['q_b'])
                else:
                    exit(1)
             # pertoken_head_pa -> 1、将bnb转成1bns
            elif ifa_param['v_antiquant_ptohpa_flag']:
                if ifa_param['pa_flag']:
                    v_antiquantscale_tensor = trans_bb_2_1bns(v_antiquantscale_tensor_raw, ifa_param['block_table'], ifa_param['q_b'])
                    v_antiquantoffset_tensor = trans_bb_2_1bns(v_antiquantoffset_tensor_raw, ifa_param['block_table'], ifa_param['q_b'])
                    v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantscale_tensor, numKeyValueHeads, numHeads)
                    v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantoffset_tensor, numKeyValueHeads, numHeads)
                    ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                    ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
                else:
                    exit(1)
            # pertoken_group
            elif ifa_param['v_antiquant_ptog_flag']:
                v_antiquantscale_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(v_antiquantscale_tensor_raw, numKeyValueHeads, numHeads)
                v_antiquantoffset_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(v_antiquantoffset_tensor_raw, numKeyValueHeads, numHeads)
                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertoken -> 如果是BS,转成1BS
            else:
                ifa_param['v_antiquantScale'] = _trans_antiparam_to_1bs(v_antiquantscale_tensor_raw)
                ifa_param['v_antiquantOffset'] = _trans_antiparam_to_1bs(v_antiquantoffset_tensor_raw)

    # >> 输入量化预处理：获取切分使用的sinner
    ifa_param['sinner'] = 0
    if ifa_param['in_quant_flag']:
        sinner = get_sinner(ifa_param)
        ifa_param['sinner'] = sinner

    # psefix 预处理：1.转成1nsd 2.GQA场景扩展N; 3.按act_prefix裁剪 4.获取perfix_act_lens
    if ifa_param['prefix_flag']:
        k_prefix_tensor = ifa_param['k_prefix_tensor']
        k_prefix_shape = ifa_param['k_prefix_shape']
        v_prefix_tensor = ifa_param['v_prefix_tensor']
        v_prefix_shape = ifa_param['v_prefix_shape']
        k_prefix_tensor, k_prefix_shape = _n_trans_shape_to_bnsd(k_prefix_tensor, k_prefix_shape, inputLayout,
                                                                 numKeyValueHeads)
        v_prefix_tensor, v_prefix_shape = _n_trans_shape_to_bnsd(v_prefix_tensor, v_prefix_shape, inputLayout,
                                                                 numKeyValueHeads)

        # kv prefix int32预处理
        if ifa_param['k_prefix_dtype'] == "int32":
            k_prefix_tensor, k_prefix_shape = trans_int32_2_int4_tensor_bnsd(k_prefix_tensor, k_prefix_shape)
        if ifa_param['v_prefix_dtype'] == "int32":
            v_prefix_tensor, v_prefix_shape = trans_int32_2_int4_tensor_bnsd(v_prefix_tensor, v_prefix_shape)

        if numKeyValueHeads != numHeads:
            k_prefix_tensor, k_prefix_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_prefix_tensor,
                                                                   ifa_param['k_prefix_dtype'])
            v_prefix_tensor, v_prefix_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_prefix_tensor,
                                                                   ifa_param['v_prefix_dtype'])

        if ifa_param['prefix_act_flag']:
            k_prefix_tensor = k_prefix_tensor[:, :, :ifa_param['prefix_act_lens'], :]
            v_prefix_tensor = v_prefix_tensor[:, :, :ifa_param['prefix_act_lens'], :]
        else:
            ifa_param['prefix_act_lens'] = k_prefix_shape[2]

        ifa_param['k_prefix_tensor'] = k_prefix_tensor
        ifa_param['v_prefix_tensor'] = v_prefix_tensor

    ifa_param["is_benchmark_task"] = params["is_benchmark_task"]
    # ===计算入口===
    out_shape = q_shape
    y_all, lse = _t_increattention_bnsd(ifa_param)
    y_all = torch.from_numpy(y_all)
    lse = lse.astype('float64')
    lse = torch.from_numpy(lse)
    y_all = trans_bnsd_to_layout(y_all, out_shape, inputLayout)
    if ifa_param["softmax_lse_flag"]:
        return y_all, lse
    else:
        return y_all

# pfa
def _random_fill_tensor(tensor, shape, random_number, value=0):
    for i in range(0, random_number):
        point = []
        for k in range(0, len(shape)):
            point.append(random.randint(1, shape[k]) - 1)
        tensor[point[0], point[1]] = value
    return tensor

def _np_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor,dtype):

    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape

def broadcast_kv_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    factor = numheads // numKeyValueHeads
    shape = tensor.shape
    D = shape[3]
    if 'bfloat16' in str(dtype):
        tensor_new = np.zeros([2, numheads, 1, D], dtype=tf.bfloat16.as_numpy_dtype)
    else:
        tensor_new = np.zeros([2, numheads, 1, D], dtype=dtype)
    for i in range(numheads):
        j = i // factor
        tensor_new[:, i:i + 1, :, :] = tensor[:, j:j + 1, :, :]
    return tensor_new

def broadcast_kv_split_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    factor = numheads // numKeyValueHeads
    shape = tensor.shape
    D = shape[2]
    if 'bfloat16' in str(dtype):
        tensor_new = np.zeros([numheads, 1, D], dtype=tf.bfloat16.as_numpy_dtype)
    else:
        tensor_new = np.zeros([numheads, 1, D], dtype=dtype)
    for i in range(numheads):
        j = i // factor
        tensor_new[i:i + 1, :, :] = tensor[j:j + 1, :, :]
    return tensor_new

def _create_mask_right_down(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, actualprefixKV,prefix_kvs, batch, numheads, kvs_list,m_dtype):

    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) != 0:
            S2 = actualSeqLengthsKV[i]+actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i]+actualprefixKV
        else:
            S2 = mask_s_kv-prefix_kvs+actualprefixKV
        next_tokens = S2 - S1
        next_tokens_list.append(next_tokens)
        atten_masks = _create_mask(m_shape, pre_tokens, next_tokens)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, next_tokens_list

def _create_mask(m_shape, pre_tokens, next_tokens):
    next_masks = np.triu(np.ones(m_shape, dtype='uint8'), k=1 + int(next_tokens))  # 生成下三角全是0的矩阵
    pre_mask = np.tril(np.ones(m_shape, dtype='uint8'), k=-1 - int(pre_tokens))  # 生成上三角全是0的矩阵
    atten_masks = pre_mask + next_masks

    return atten_masks
def _create_mask_no_sparse(m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype, random_ones=0):
    re_mask_batch = []
    re_mask_npu_batch = []

    pad_flag = False
    npu_mask = None
    if m_shape[0] !=npu_m_shape[0] or m_shape[1]!=npu_m_shape[1]:
        pad_flag = True
        npu_mask = np.ones(npu_m_shape, dtype='uint8')
    cpu_mask = _create_mask(m_shape, pre_tokens, next_tokens)
    if pad_flag:
        if batch == None:
            cpu_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            npu_mask[:cpu_mask.shape[0], :cpu_mask.shape[1]] = cpu_mask
            return cpu_mask, npu_mask
        for i in range(batch):
            re_mask_num = []
            re_mask_npu_num = []
            re_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)

            npu_mask[:re_mask.shape[0], :re_mask.shape[1]] = re_mask
            if numheads:
                # cpu
                re_mask_num.append(re_mask)
                re_mask_batch.append(re_mask_num)
                # npu

                re_mask_npu_num.append(npu_mask)
                re_mask_npu_batch.append(re_mask_npu_num)
            else:
                re_mask_batch.append(re_mask)
                re_mask_npu_batch.append(npu_mask)
        cpu_mask = np.array(re_mask_batch).astype(m_dtype)
        npu_mask = np.array(re_mask_npu_batch).astype(m_dtype)
        return cpu_mask, npu_mask
    else:
        if batch ==None:
            cpu_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            return cpu_mask,cpu_mask
        for i in range(batch):
            re_mask_num = []
            re_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            if numheads:
                re_mask_num.append(re_mask)
                re_mask_batch.append(re_mask_num)
            else:
                re_mask_batch.append(re_mask)

        cpu_mask = np.array(re_mask_batch).astype(m_dtype)
        return cpu_mask,cpu_mask
def _create_mask_left_up(m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype, random_ones=0):
    re_mask_batch = []
    attentionmask = _create_mask(m_shape, pre_tokens, next_tokens)
    for i in range(batch):
        re_mask_batch.append(attentionmask)
    return re_mask_batch

def _create_mask_band(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV,actualprefixKV, prefix_kvs,batch, numheads, kvs_list, m_dtype):

    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    pre_tokens_list = []
    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) != 0:
            S2 = actualSeqLengthsKV[i]+actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i]+actualprefixKV
        else:
            S2 = mask_s_kv-prefix_kvs+actualprefixKV
        pre_tokens_new = S1-S2+pre_tokens
        pre_tokens_list.append(pre_tokens_new)

        next_tokens_new = S2-S1+next_tokens
        next_tokens_list.append(next_tokens_new)
        atten_masks = _create_mask(m_shape, pre_tokens_new, next_tokens_new)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, pre_tokens_list, next_tokens_list

def _create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, actualprefixKV,prefix_kvs,kvs_list,batch=1, numheads=1, sp_mode=0, random_ones=0):
    # mask shape [sq,skv]  #mshape  npu  fshape cpu
    if sp_mode == 0:
        cpu_mask, npu_mask = _create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads,
                                                    m_dtype, random_ones)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 1:
        pre_tokens = 214748647
        next_tokens = 214748647
        cpu_mask, npu_mask = _create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads,
                                                    m_dtype, random_ones)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens

    if sp_mode == 2:
        pre_tokens = 214748647
        next_tokens = 0
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 3:  # rightdown
        pre_tokens = 214748647
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, next_tokens_new = _create_mask_right_down(cpu_m_shape, pre_tokens, next_tokens, actualSeqLengths,
                                                            actualSeqLengthsKV, actualprefixKV, prefix_kvs, batch, numheads, kvs_list, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens_new
    if sp_mode == 4:
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, pre_tokens_new, next_tokens_new = _create_mask_band(cpu_m_shape, pre_tokens, next_tokens,
                                                                      actualSeqLengths, actualSeqLengthsKV, actualprefixKV, prefix_kvs,batch,
                                                                      numheads, kvs_list, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens_new, next_tokens_new

def t_bsh_to_bsnd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
        G-golden  need  bsnd
    """
    if inputLayout == "SH":  # SH格式
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        if B == 1:
            return tensor.reshape(B, S, N, D), [B, S, N, D]
        else:
            tmp = torch.zeros((B, S, N, D))
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp, [B, S, N, D]
    elif inputLayout == "BSH":
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, D), [B, S, N, D]
    elif inputLayout == "NSD":
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D).permute(0, 2, 1, 3), [B, S, N, D]
    else:
        return tensor.permute(0, 2, 1, 3), bsh_shape

def np_bsh_to_bnsd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
    cpu golden  need  bnsd
    """
    if inputLayout == "SH":  # SH格式
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        if B == 1:
            return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
        else:
            tmp = np.zeros((B, S, N, D), dtype=tensor.dtype)
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSH":
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "NSD":
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D), [B, N, S, D]
    elif inputLayout == "BSND":
        B = bsh_shape[0]
        S = bsh_shape[1]
        N = bsh_shape[2]
        D = bsh_shape[3]
        return tensor.transpose(0, 2, 1, 3), [B, N, S, D]
    else:
        return tensor, bsh_shape

def alibi_biases(pre_shift_shape):
    # [0,  1, 2]
    # [-1, 0, 1]
    # [-2,-1, 0]
    q_seq = pre_shift_shape[2]
    kv_seq = pre_shift_shape[3]
    alibi_biases = np.zeros([q_seq, kv_seq])
    for x in range(0, kv_seq):
        for y in range(0, q_seq):
            alibi_biases[y, x] = abs(x - y - (kv_seq - q_seq))*(-1)
    return alibi_biases

def get_all_alibi(numHeads,pre_shift_shape,pse_dtype=np.float32): #输入是bnss或者是1nss B, Nq, Sq, Skv, pse_layout, pse_type, pse_dtype,pre_shift_shape
    m = get_slopes(numHeads)  #m  系数
    m = m.numpy()
    alibi_biase = alibi_biases(pre_shift_shape)
    pse_shift = np.zeros(pre_shift_shape,dtype=np.float32)

    for n in range(numHeads):
        pse_shift[:, n:n + 1, :, :] = alibi_biase * m[n]
    return pse_shift.astype(pse_dtype)

def _np_broadcast_pseShift_n(pse_shift_tensor, pse_shift_shape, q_batch):#pre_shift, pre_shift_shape, q_bnsd_shape[0]
    #1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        # pse_res = np.zeros([B, pse_shift_shape[1],pse_shift_shape[2], pse_shift_shape[3]])
        pse_res = np.zeros([B, pse_shift_tensor.shape[1], pse_shift_tensor.shape[2], pse_shift_tensor.shape[3]])
        for i in range(B):
            pse_res[i:i+1] = pse_shift_tensor[0]
        return pse_res

def _t_broadcast_pseShift_n(pse_shift_tensor, pse_shift_shape, q_batch):#pre_shift, pre_shift_shape, q_bnsd_shape[0]
    #1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        pse_res = torch.zeros([B, pse_shift_shape[1],pse_shift_shape[2], pse_shift_shape[3]])
        for i in range(B):
            pse_res[i:i+1] = pse_shift_tensor[0]
        return pse_res

def _np_broadcast_mask_n(m_tensor, m_shape,cpu_m_shape, numheads, q_batch):
    mask_cur_shape = cpu_m_shape
    if len(m_shape) == 4:
        # b1ss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[2], m_shape[3]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = []
        for i in range(B):
            m_res.append(m_tensor)
        return m_res
    else:
        return m_tensor

def _t_broadcast_mask_n_pfa(m_tensor, m_shape, numheads, q_batch):
    if len(m_shape) == 4:
        # b1ss
        B = m_shape[0]

        m_res = torch.zeros([B, numheads, m_shape[2], m_shape[3]])
        for i in range(B):
            mask_cur_shape = [m_shape[2], m_shape[3]]
            mask_cur = m_tensor[i:i + 1, :, :, :].reshape(mask_cur_shape)
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = mask_cur
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = torch.zeros([B, numheads, m_shape[1], m_shape[2]])
        for i in range(B):
            mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :].reshape(mask_cur_shape)
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = mask_cur
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = torch.zeros([B, numheads, m_shape[0], m_shape[1]])
        for i in range(B):
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = m_tensor
        return m_res
    else:
        return m_tensor

def npSoftmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum

def npSoftmax_new(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_max, x_sum

def softmax_flashv2(x, max_front=None, sum_front=None, update=None, is_fp16=False):
    """
    Compute the softmax function for each channel of the input x.
    """
    if update == None:
        if is_fp16:
            x = x.astype(np.float32)
        x_max = np.max(x, axis=-1, keepdims=True)
        x_sub = x - x_max  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        out = x_exp
        # out = x_exp / x_sum
        exp_max = None
        if is_fp16:
            out = out.astype(np.float16)

        return out, x_max, x_sum, exp_max
    else:
        if is_fp16:
            x = x.astype(np.float32)
            max_front = max_front.astype(np.float32)
            sum_front = sum_front.astype(np.float32)
        x_max_tmp = np.max(x, axis=-1, keepdims=True)  # tmp
        x_sub = x - x_max_tmp  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)  # tmp
        x_max = np.max(np.concatenate((max_front, x_max_tmp), axis=-1), axis=-1, keepdims=True)  # ->x_max
        x_exp_new = np.exp(x_max_tmp - x_max)  # -> x_max_tmp
        exp_max = np.exp(max_front - x_max)  # -> exp_max
        # update sum
        exp_max = exp_max * sum_front
        reduce_tmp = x_exp_new * x_sum
        x_sum = exp_max + reduce_tmp  # x_sum

        # exp_max = exp_max/x_sum
        exp_max = np.exp(max_front - x_max)
        out = x_exp * x_exp_new
        # out = x_exp*x_exp_new / x_sum

        # ### softmax ratio
        # softma_ratio = x_sum * x_exp_new / x_sum
        # out_new = out * softma_ratio
        # pround("out_new===================", out_new)
        if is_fp16:
            out = out.astype(np.float16)
            # x_max = x_max.astype(np.float16)
            # x_sum = x_sum.astype(np.float16)
            exp_max = exp_max.astype(np.float16)

        return out, x_max, x_sum, exp_max

def _np_pfaattention_act_int8(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param,
                             dequant_scale1=None,
                             dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                             out_dtype="<class 'numpy.int8'>"):
    S = None
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']

    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]
    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]

    S = kvs_end-kvs_begin

    if mask_tensor is not None:
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            mask_tensor = mask_tensor[:, :, :(qs_end - qs_begin), :(kvs_end - kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    one_loop_size = 1024
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xffffe000)
    dequant_scale1.dtype = np.float32
    
    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype

        qk_i = np.matmul(q_tensor.astype(np.int32), ki.transpose(0, 1, 3, 2).astype(np.int32))
        qk_i = qk_i.astype(np.float32) * dequant_scale1

        # 转成fp16防止溢出
        qk_i = np.where(qk_i > 65504, 65504, qk_i)
        qk_i = np.where(qk_i < -65504, -65504, qk_i)
        qk_i = qk_i.astype(np.float16)
        qk_i = qk_i* np.float16(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=True)
            py_s_i = py_s_i * quant_scale1.astype(np.float16)
            py_s_i = np.around(py_s_i)

            py_s_i = np.where(py_s_i > 127, 127, py_s_i)
            py_s_i = np.where(py_s_i < -128, -128, py_s_i)
            py_s_i = py_s_i.astype(np.int8)

            o_i = np.matmul(py_s_i.astype(np.int32), vi.astype(np.int32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, max_front, sum_front, update=True, is_fp16=True)
            py_s_i = py_s_i * quant_scale1.astype(np.float16)
            py_s_i = np.around(py_s_i)

            # 转成int8防止溢出
            py_s_i = np.where(py_s_i > 127, 127, py_s_i)
            py_s_i = np.where(py_s_i < -128, -128, py_s_i)
            py_s_i = py_s_i.astype(np.int8)

            o_i = np.matmul(py_s_i.astype(np.int32), vi.astype(np.int32))
            o_i = o_i.astype(np.float32) * dequant_scale2

            # 转成fp16防止溢出
            o_i = np.where(o_i > 65504, 65504, o_i)
            o_i = np.where(o_i < -65504, -65504, o_i)
            o_i = o_i.astype(np.float16)

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param['lseflag']:
        lse = np.log(sum_front) + max_front


    o_front = o_front / sum_front.astype(np.float16)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)

    return o_front,lse

def _t_pfaattention_act_innerprecise(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, dequant_scale1=None,
                        dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None):
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]

    q_t = q_tensor.to(torch.float32)
    k_t = k_tensor.to(torch.float32)
    v_t = v_tensor.to(torch.float32)

    q_tensor = q_t.cuda().requires_grad_(False)
    k_tensor = k_t.cuda().requires_grad_(False)
    v_tensor = v_t.cuda().requires_grad_(False)

    qkBmmRes = torch.matmul(q_tensor, k_tensor.permute([0, 1, 3, 2]))

    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes += pse_tensor

    if mask_tensor is not None:
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes[mask_tensor.to(bool)] = -1.7 * 10 ** 38
    softmax_res, x_max, x_sum = _t_softmax(qkEleRes)
    if q_tensor.shape[2] > k_tensor.shape[2]+preTokens:
        ss0 = k_tensor.shape[2]+preTokens
        softmax_res[:,:,ss0:,:] = 0
    if nextTokens < 0:
        ss1 = -nextTokens
        softmax_res[:, :, :ss1, :] = 0

    bmm2Res = torch.matmul(softmax_res.to(torch.float16).to(torch.float32), v_tensor).cpu()
    return bmm2Res

def _t_pfaattention_act(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, dequant_scale1=None,
                        dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None):
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]

    q_t = q_tensor.to(torch.float32)
    k_t = k_tensor.to(torch.float32)
    v_t = v_tensor.to(torch.float32)

    q_tensor = q_t.cuda().requires_grad_(False)
    k_tensor = k_t.cuda().requires_grad_(False)
    v_tensor = v_t.cuda().requires_grad_(False)

    qkBmmRes = torch.matmul(q_tensor, k_tensor.permute([0, 1, 3, 2]))
    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes += pse_tensor

    if mask_tensor is not None:
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes[mask_tensor.to(bool)] = -1.7 * 10 ** 38
    softmax_res, x_max, x_sum = _t_softmax(qkEleRes)
    if q_tensor.shape[2] > k_tensor.shape[2] + preTokens:
        ss0 = k_tensor.shape[2] + preTokens
        softmax_res[:, :, ss0:, :] = 0
    if nextTokens < 0:
        ss1 = -nextTokens
        softmax_res[:, :, :ss1, :] = 0
    bmm2Res = torch.matmul(softmax_res.to(torch.bfloat16).to(torch.float32), v_tensor).cpu()
    return bmm2Res

def _np_pfaattention_act(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param, quant_scale2=None, quant_offset2=None, antiquant_scale_k=None,
                            antiquant_scale_v=None, antiquant_offset_k=None, antiquant_offset_v=None, out_dtype="<class 'numpy.int8'>"):
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']

    matmul_dtype = np.float32
    if not pfa_param["is_benchmark_task"]:
        matmul_dtype = np.float16

    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    qDtype = q_tensor.dtype

    #KVcahce反量化  这块逻辑需要修改
    if k_tensor.dtype==np.int8:
        k_tensor = k_tensor.astype(np.float16)
        v_tensor = v_tensor.astype(np.float16)
        if antiquant_offset_k is not None:
            if pfa_param['anti_or_kvanti'] ==2 and pfa_param['k_antiquant_mode']==1:
                antiquant_offset_k = np.expand_dims(antiquant_offset_k[:,kvs_begin:kvs_end], axis=-1)
                antiquant_offset_v = np.expand_dims(antiquant_offset_v[:,kvs_begin:kvs_end], axis=-1)
            k_tensor += antiquant_offset_k
            v_tensor += antiquant_offset_v
        if pfa_param['anti_or_kvanti'] == 2 and pfa_param['k_antiquant_mode']==1:
            antiquant_scale_k = np.expand_dims(antiquant_scale_k[:, kvs_begin:kvs_end], axis=-1)
            antiquant_scale_v = np.expand_dims(antiquant_scale_v[:, kvs_begin:kvs_end], axis=-1)
        k_tensor *= antiquant_scale_k
        v_tensor *= antiquant_scale_v

    qkBmmRes = np.matmul(q_tensor, k_tensor.transpose([0, 1, 3, 2]), dtype=matmul_dtype)

    k_cur_msd = k_tensor.transpose(0, 1, 3, 2)
    if not pfa_param["is_benchmark_task"]:
        for b_index in range(q_tensor.shape[0]):
            for head_index in range(q_tensor.shape[1]):
                qkBmmRes[b_index, head_index, :, :] = msd(q_tensor[b_index, head_index, :, :], k_cur_msd[b_index, head_index, :, :])

    if not pfa_param["is_benchmark_task"]:
        qkBmmRes = np.float16(qkBmmRes)
        scalar = np.float16(scalar)
    else:
        scalar = np.float32(scalar)
    qkEleRes = qkBmmRes * scalar

    #这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
        qkEleRes += pse_tensor.astype(np.float32)

    if mask_tensor is not None:

        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            mask_tensor = mask_tensor[:, :, :(qs_end-qs_begin),:(kvs_end-kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
        qkEleRes[mask_tensor.astype(np.bool_)] = -1.7 * 10 ** 38
    if not pfa_param["is_benchmark_task"]:
            qkEleRes = np.float16(qkEleRes)
    softmax_res, x_max, x_sum = npSoftmax_new(qkEleRes)

    lse = None
    if 'lseflag' in pfa_param and pfa_param['lseflag']:
        lse = np.log(x_sum) + x_max

    bmm2Res = np.matmul(softmax_res, v_tensor, dtype=matmul_dtype)

    if not pfa_param["is_benchmark_task"]:
                for b_index in range(softmax_res.shape[0]):
                    for head_index in range(softmax_res.shape[1]):
                        bmm2Res[b_index, head_index, :, :] = msd(softmax_res[b_index, head_index, :, :], v_tensor[b_index, head_index, :, :])
    #/sum
    bmm2Res = bmm2Res/x_sum
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                bmm2Res[:, :, i, :] = 0
                if lse is not None:
                    lse[:,:,i,:] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        bmm2Res = bmm2Res * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            bmm2Res += quant_offset2.astype(np.float16)
        bmm2Res = np.around(bmm2Res)
        # 转成int8防止溢出
        bmm2Res = np.where(bmm2Res > 127, 127, bmm2Res)
        bmm2Res = np.where(bmm2Res < -128, -128, bmm2Res)
        bmm2Res = bmm2Res.astype(np.int8)
    return bmm2Res, lse

def _np_promtattention_bnsd(q_tensor, q_shape, k_tensor_list, k_shape_list, v_tensor_list, v_shape_list, pse_bnsd_tensor, mask_tensor, scale, actseqlens,
                           actkvseqlens, preTokens_list, nextTokens_list, sparse,pfa_param, dequant_scale1=None, dequant_scale2=None, quant_scale1=None, quant_scale2=None,
                           quant_offset2=None, antiquant_scale=None, antiquant_offset=None, out_dtype="<class 'numpy.int8'>"):
    batch_value = q_shape[0]
    numhead_value = q_shape[1]
    actseqlens_size = len(actseqlens)
    y = np.zeros(q_shape, dtype=np.float32)
    lse = np.full(pfa_param['lseshape'], np.inf)

    preTokens = preTokens_list
    nextTokens = nextTokens_list
    tensor_list_flag = len(k_tensor_list)
    if mask_tensor is not None:
        mask_cur = np.zeros([1, 1, q_shape[2], len(mask_tensor[0][0])], dtype='uint8')
    if tensor_list_flag==1:
        k_tensor = k_tensor_list[0]
        k_shape = k_shape_list[0]
        v_tensor = v_tensor_list[0]
        v_shape = v_shape_list[0]

    dequant_scale1_copy = copy.deepcopy(dequant_scale1)
    dequant_scale2_copy = copy.deepcopy(dequant_scale2)

    for b_index in range(batch_value):
        if tensor_list_flag!=1:
            k_tensor = k_tensor_list[b_index]
            k_shape = k_shape_list[b_index]
            v_tensor = v_tensor_list[b_index]
            v_shape = v_shape_list[b_index]


        if len(actseqlens) == 0:
            act_seq = None
        else:
            act_seq = int(actseqlens[b_index])

        if len(actkvseqlens) == 0:
            act_kv_seq = None
        else:
            act_kv_seq = int(actkvseqlens[b_index])
        if act_seq == 0 or act_kv_seq == 0 or 0 in k_shape or 0 in v_shape:
            continue

        qs_begin = 0
        qs_end = q_tensor.shape[2]
        kvs_begin = 0
        kvs_end = k_tensor.shape[2]
        if 'queryPaddingSize' in pfa_param:
            qs_begin = int(q_tensor.shape[2] - act_seq - pfa_param['queryPaddingSize'][0])
            qs_end = int(q_tensor.shape[2] - pfa_param['queryPaddingSize'][0])
        else:
            if act_seq is not None:
                qs_end = act_seq
        if 'kvPaddingSize' in pfa_param:
            kvs_begin = int(k_tensor.shape[2] - act_kv_seq - pfa_param['kvPaddingSize'][0])
            kvs_end = int(k_tensor.shape[2] - pfa_param['kvPaddingSize'][0])
        else:
            if act_kv_seq is not None:
                kvs_end = act_kv_seq
        pfa_param['qs_begin'] = qs_begin
        pfa_param['qs_end'] = qs_end
        pfa_param['kvs_begin'] = kvs_begin
        pfa_param['kvs_end'] = kvs_end
        if 'actualprefixKV' in pfa_param:
            if pfa_param['actualprefixKV']==0:
                pfa_param['kvs_end']+=pfa_param["shared_prefix_k"].shape[2]
            else:
                pfa_param['kvs_end']+=pfa_param['actualprefixKV']
        pfa_param['sparse'] = sparse
        # bnsd
        if sparse == 4:
            preTokens = preTokens_list[b_index]
        if sparse == 3 or sparse == 4:
            nextTokens = nextTokens_list[b_index]
    
        for n_index in range(numhead_value):
            if len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4:
                q_tensor_cur = q_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                if tensor_list_flag==1:
                    k_tensor_cur = k_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                    v_tensor_cur = v_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                else:
                    k_tensor_cur = k_tensor[:, n_index:(n_index + 1), :, :]
                    v_tensor_cur = v_tensor[:, n_index:(n_index + 1), :, :]
                if "shared_prefix_k" in pfa_param:
                    actual_prefix_len = pfa_param['actualprefixKV']
                    if actual_prefix_len==0:
                        shared_prefix_k = pfa_param["shared_prefix_k"][:,n_index:(n_index + 1),:,:]
                        shared_prefix_v = pfa_param["shared_prefix_v"][:,n_index:(n_index + 1),:,:]
                    else:
                        shared_prefix_k = pfa_param["shared_prefix_k"][:, n_index:(n_index + 1), :actual_prefix_len, :]
                        shared_prefix_v = pfa_param["shared_prefix_v"][:, n_index:(n_index + 1), :actual_prefix_len, :]
                    k_tensor_cur = np.append(shared_prefix_k, k_tensor_cur, axis=2)
                    v_tensor_cur = np.append(shared_prefix_v, v_tensor_cur, axis=2)

                if mask_tensor is None:
                    mask_cur = None
                else:
                    mask_cur[0, 0, :, :] = mask_tensor[b_index]
                    # mask_cur = mask_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if pse_bnsd_tensor is None:
                    pse_cur = None
                else:
                    pse_cur = pse_bnsd_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                quant_scale2_cur = None
                quant_offset2_cur = None
                if quant_scale2 is not None:
                    if len(quant_scale2.shape) == 1:
                        quant_scale2_cur = quant_scale2
                    else:
                        quant_scale2_cur = quant_scale2[:, n_index:(n_index + 1), :, :]
                    if quant_offset2 is not None:
                        if len(quant_offset2.shape) == 1:
                            quant_offset2_cur = quant_offset2
                        else:
                            quant_offset2_cur = quant_offset2[:, n_index:(n_index + 1), :, :]

                antiquant_scale_k = None
                antiquant_scale_v = None
                antiquant_offset_k = None
                antiquant_offset_v = None
                if k_tensor_cur.dtype==np.int8 and q_tensor_cur.dtype!=np.int8:
                    if pfa_param['anti_or_kvanti']==1:
                        if len(antiquant_scale.shape) == 1:
                            antiquant_scale_k = antiquant_scale[0]
                            antiquant_scale_v = antiquant_scale[1]
                        else:
                            antiquant_scale_k = antiquant_scale[0,n_index:(n_index + 1),:,:]
                            antiquant_scale_v = antiquant_scale[1,n_index:(n_index + 1),:,:]
                        if antiquant_offset is not None:
                            if len(antiquant_offset.shape) ==1:
                                antiquant_offset_k = antiquant_offset[0]
                                antiquant_offset_v = antiquant_offset[1]
                            else:
                                antiquant_offset_k = antiquant_offset[0,n_index:(n_index + 1),:,:]
                                antiquant_offset_v = antiquant_offset[1,n_index:(n_index + 1),:,:]
                    if pfa_param['anti_or_kvanti']==2:#分离量化参数
                        if pfa_param['k_antiquant_mode'] ==0 :
                            if len(pfa_param['k_antiquant_scale'].shape) == 1:
                                antiquant_scale_k = pfa_param['k_antiquant_scale']
                                antiquant_scale_v = pfa_param['v_antiquant_scale']
                            else:
                                antiquant_scale_k = pfa_param['k_antiquant_scale'][n_index:(n_index + 1),:,:]
                                antiquant_scale_v = pfa_param['v_antiquant_scale'][n_index:(n_index + 1),:,:]
                            if pfa_param['k_antiquant_offset'] is not None:
                                if len(pfa_param['k_antiquant_offset'].shape) ==1:#petensor
                                    antiquant_offset_k = pfa_param['k_antiquant_offset']
                                    antiquant_offset_v = pfa_param['v_antiquant_offset']
                                else:
                                    antiquant_offset_k = pfa_param['k_antiquant_offset'][n_index:(n_index + 1),:,:]
                                    antiquant_offset_v = pfa_param['v_antiquant_offset'][n_index:(n_index + 1),:,:]
                        if pfa_param['k_antiquant_mode'] == 1:
                            antiquant_scale_k = pfa_param['k_antiquant_scale'][b_index:(b_index + 1),  :]
                            antiquant_scale_v = pfa_param['v_antiquant_scale'][b_index:(b_index + 1),  :]
                            if pfa_param['k_antiquant_offset'] is not None:
                                antiquant_offset_k = pfa_param['k_antiquant_offset'][b_index:(b_index + 1), :]
                                antiquant_offset_v = pfa_param['v_antiquant_offset'][b_index:(b_index + 1), :]
                
                if q_tensor.dtype == "int8":
                    y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :],lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act_int8(q_tensor_cur,
                                                                                                      k_tensor_cur,
                                                                                                      v_tensor_cur,
                                                                                                      pse_cur,
                                                                                                      mask_cur,
                                                                                                      scale,
                                                                                                      act_seq,
                                                                                                      act_kv_seq,
                                                                                                      preTokens,
                                                                                                      nextTokens,
                                                                                                      pfa_param,
                                                                                                      dequant_scale1_copy,
                                                                                                      dequant_scale2_copy,
                                                                                                      quant_scale1,
                                                                                                      quant_scale2_cur,
                                                                                                      quant_offset2_cur,
                                                                                                      out_dtype)
                else:
                    y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :], lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act(q_tensor_cur,
                                                                                                 k_tensor_cur,
                                                                                                 v_tensor_cur,
                                                                                                 pse_cur,
                                                                                                 mask_cur, scale,
                                                                                                 act_seq,
                                                                                                 act_kv_seq,
                                                                                                 preTokens,
                                                                                                 nextTokens,
                                                                                                 pfa_param,
                                                                                                 quant_scale2_cur,
                                                                                                 quant_offset2_cur,
                                                                                                 antiquant_scale_k,
                                                                                                 antiquant_scale_v,
                                                                                                 antiquant_offset_k,
                                                                                                 antiquant_offset_v,
                                                                                                 out_dtype)

    return y,lse

def conv_float_to_u32(data_f):
    fp = ctypes.pointer(ctypes.c_float(data_f))
    cp = ctypes.cast(fp, ctypes.POINTER(ctypes.c_uint))
    data_hex = cp.contents.value
    result = (data_hex // 8192) * 8192
    return result

def trans_19bit(deqscale):
    res_19bit = np.zeros(deqscale.shape[0], dtype=np.uint64)

    for idx, scale in enumerate(deqscale):
        val = np.bitwise_and(((conv_float_to_u32(scale) >> 13) << 13), 0xffffffff)  # [31:13], M
        val = val.astype(np.uint64)
        res_19bit[idx] = val

    return res_19bit

def qtensor_seqlength(q_shape, inputLayout):
    if inputLayout == "SH":  # SH格式
        return q_shape[0]
    elif inputLayout == "BSH":
        return q_shape[1]
    elif inputLayout == "NSD":
        return q_shape[1]
    elif inputLayout == "BSND":
        return q_shape[1]
    else: #BNSD
        return q_shape[2]

def kvantiquantmode(flagList,torch_tensor_list,params,v_end_index,numKeyValueHeads):
    antiquant_scale = None
    antiquant_offset = None

    k_antiquant_scale = None
    v_antiquant_scale = None

    k_antiquant_offset = None
    v_antiquant_offset = None

    anti_or_kvanti = 0
    # 获取kv量化参数
    flagtensor = 0
    if flagList[12]:
        anti_or_kvanti = 1
        antiquantscale_shape = params['shape_input'][v_end_index + 8]
        antiquantscale_tensor = torch_tensor_list[v_end_index + 8]
        if len(antiquantscale_shape) == 1:  # pertensor
            antiquant_scale = antiquantscale_tensor
        else:  # perchanel
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_2n1d_tensor = _t_trans_2h_to_2n1d(antiquantscale_shape, antiquantscale_tensor,
                                                           numKeyValueHeads)

            antiquant_scale = antiquantscale_2n1d_tensor.permute(0, 2, 1, 3)
    if flagList[13]:
        antiquantoffset_shape = params['shape_input'][v_end_index + 9]
        antiquantoffset_tensor = torch_tensor_list[v_end_index + 9]
        if len(antiquantoffset_shape) == 1:
            antiquant_offset = antiquantoffset_tensor
        else:
            antiquantoffset_2n1d_tensor = _t_trans_2h_to_2n1d(antiquantoffset_shape, antiquantoffset_tensor,
                                                            numKeyValueHeads)

            antiquant_offset = antiquantoffset_2n1d_tensor.permute(0, 2, 1, 3)
    if flagList[12]:
        return antiquant_scale[:1],antiquant_scale[1:2],antiquant_offset[:1],antiquant_offset[1:2]
    # 判断kv分离的伪量化
    if flagList[17] or flagList[19]:
        anti_or_kvanti = 2
        q_dtype_np = get_np_dtype(params['dtype_input'][0])
        k_antiquantscale_shape = params['shape_input'][v_end_index + 13]
        k_antiquantscale_tensor = torch_tensor_list[v_end_index + 13]
        v_antiquantscale_shape = params['shape_input'][v_end_index + 15]
        v_antiquantscale_tensor = torch_tensor_list[v_end_index + 15]

        if params['k_antiquant_mode'] == 0:
            if len(k_antiquantscale_shape) == 1 and k_antiquantscale_shape[0] == 1:  # pertensor
                k_antiquant_scale = k_antiquantscale_tensor
                v_antiquant_scale = v_antiquantscale_tensor
            else:  # perchanel
                # 将kv量化参数转换为n1d(匹配bnsd)
                k_antiquantscale_n1d_tensor = _t_trans_h_to_n1d(k_antiquantscale_shape, k_antiquantscale_tensor,
                                                              numKeyValueHeads)
                v_antiquantscale_n1d_tensor = _t_trans_h_to_n1d(v_antiquantscale_shape, v_antiquantscale_tensor,
                                                              numKeyValueHeads)
                k_antiquant_scale = k_antiquantscale_n1d_tensor.permute(0, 2, 1, 3)
                v_antiquant_scale = v_antiquantscale_n1d_tensor.permute(0, 2, 1, 3)
        if params['k_antiquant_mode'] == 1:  # pertoken
            # pertoken -> BS
            k_antiquant_scale = k_antiquantscale_tensor.reshape(k_antiquantscale_tensor.shape[0],k_antiquantscale_tensor.shape[1],1,1)
            v_antiquant_scale = v_antiquantscale_tensor.reshape(v_antiquantscale_tensor.shape[0],v_antiquantscale_tensor.shape[1],1,1)

    if flagList[18] or flagList[20]:
        q_dtype_np = get_np_dtype(params['dtype_input'][0])
        k_antiquantoffset_shape = params['shape_input'][v_end_index + 14]
        k_antiquantoffset_tensor = torch_tensor_list[v_end_index + 14]
        v_antiquantoffset_shape = params['shape_input'][v_end_index + 16]
        v_antiquantoffset_tensor = torch_tensor_list[v_end_index + 16]

        if params['k_antiquant_mode'] == 0:
            if len(k_antiquantoffset_shape) == 1 and k_antiquantoffset_shape[0] == 1:  # pertensor
                k_antiquant_offset = k_antiquantoffset_tensor
                v_antiquant_offset = v_antiquantoffset_tensor
            else:  # perchanel
                # 将kv量化参数转换为2n1d(匹配bnsd)
                k_antiquantoffset_n1d_tensor = _t_trans_h_to_n1d(k_antiquantoffset_shape,
                                                               k_antiquantoffset_tensor, numKeyValueHeads)
                v_antiquantoffset_n1d_tensor = _t_trans_h_to_n1d(v_antiquantoffset_shape,
                                                               v_antiquantoffset_tensor, numKeyValueHeads)


                k_antiquant_offset = k_antiquantoffset_n1d_tensor.permute(0, 2, 1, 3)
                v_antiquant_offset = v_antiquantoffset_n1d_tensor.permute(0, 2, 1, 3)
        if params['k_antiquant_mode'] == 1:  # pertoken
            # pertoken -> BS
            k_antiquant_offset = k_antiquantoffset_tensor.reshape(k_antiquantoffset_tensor.shape[0],k_antiquantoffset_tensor.shape[1],1,1)
            v_antiquant_offset = v_antiquantoffset_tensor.reshape(v_antiquantoffset_tensor.shape[0],v_antiquantoffset_tensor.shape[1],1,1)
    if flagList[17]:
        return k_antiquant_scale,v_antiquant_scale,k_antiquant_offset,v_antiquant_offset
    return k_antiquant_scale,v_antiquant_scale,k_antiquant_offset,v_antiquant_offset

def get_attention_mask_batch_num(npu_m_shape,q_bnsd_shape):
    batch, numhead = None, None
    if len(npu_m_shape)==2:
        s1 = npu_m_shape[0]
        s2 = npu_m_shape[1]
        return batch, numhead,s1,s2
    if len(npu_m_shape)==3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead,s1,s2
    if len(npu_m_shape)==4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead,s1,s2

def _trans_2h_to_2n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 4:#2N1D
        return tensor
    elif len(shape) == 2:#2H
        d_num = shape[1] // numKeyValueHeads
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
        return new_tensor
    elif len(shape) == 3:#2ND
        N = shape[1]
        D = shape[2]
        new_tensor = tensor.reshape(2,1,N,D).transpose(0, 2, 1, 3)
        return new_tensor
    else:
        return tensor

def _t_trans_2h_to_2n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 4:#2N1D
        return tensor
    elif len(shape) == 2:#2H
        d_num = shape[1] // numKeyValueHeads
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).permute(0, 2, 1, 3)
        return new_tensor
    elif len(shape) == 3:#2ND
        N = shape[1]
        D = shape[2]
        new_tensor = tensor.reshape(2,1,N,D).permute(0, 2, 1, 3)
        return new_tensor
    else:
        return tensor

def _trans_h_to_n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 3:#N1D
        return tensor
    elif len(shape) == 1:#H
        d_num = shape[0] // numKeyValueHeads
        new_tensor = tensor.reshape(1, numKeyValueHeads, d_num).transpose(1, 0, 2)
        return new_tensor
    elif len(shape) == 2:#ND
        N = shape[0]
        D = shape[1]
        new_tensor = tensor.reshape(1,N,D).transpose(1, 0, 2)
        return new_tensor
    else:
        return tensor

def _t_trans_h_to_n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 3:#N1D
        return tensor
    elif len(shape) == 1:#H
        d_num = shape[0] // numKeyValueHeads
        new_tensor = tensor.reshape(1, numKeyValueHeads, d_num).permute(1, 0, 2)
        return new_tensor
    elif len(shape) == 2:#ND
        N = shape[0]
        D = shape[1]
        new_tensor = tensor.reshape(1,N,D).permute(1, 0, 2)
        return new_tensor
    else:
        return tensor

def _t_trans_1h_to_1n1d(shape, tensor, numHeads,inputLayout):
    if len(shape) == 4:
        # 1n1d->1n1d
        if inputLayout=="BSND": #1,1,n,d
            return tensor
        else:   #BNSD 1N1D
            return tensor.transpose(1,2)
    elif len(shape) == 3:
        if inputLayout =="BSND": # 1,n,d ->1,n,1,d
            n = shape[1]
            d = shape[2]
            return tensor.reshape(1,1,n,d)
        elif inputLayout == "BNSD" or inputLayout == "NSD": #n,1,d -> 1,n,1,d
            n = shape[0]
            d = shape[2]
            return tensor.reshape(1,n,1,d).transpose(1,2)
        else: #bsh 11h->1n1d
            h = shape[2]
            d = h//numHeads
            return  tensor.reshape(1,1,numHeads,d)
    elif len(shape) == 2:
        if inputLayout=="BSH":#1,H->1,n,1,d
            d = shape[1] // numHeads
            return tensor.reshape(1, 1, numHeads, d)
        else: #BSND ND ->1N1D
            n = shape[0]
            d = shape[1]
            return tensor.reshape(1, 1, n, d)
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        d = h//numHeads
        return tensor.reshape(1,1,numHeads,d)
    else:
        exit(1)

def _trans_1h_to_1n1d(shape, tensor, numHeads,inputLayout):
    if len(shape) == 4:
        # 1n1d->1n1d
        if inputLayout=="BSND": #1,1,n,d
            return tensor.transpose(0,2,1,3)
        else:   #BNSD 1N1D
            return tensor
    elif len(shape) == 3:
        if inputLayout =="BSND": # 1,n,d ->1,n,1,d
            n = shape[1]
            d = shape[2]
            return tensor.reshape(1,1,n,d).transpose(0,2,1,3)
        elif inputLayout == "BNSD" or inputLayout == "NSD" or inputLayout == "BNSD_BSND": #n,1,d -> 1,n,1,d
            n = shape[0]
            d = shape[2]
            return tensor.reshape(1,n,1,d)
        else: #bsh 11h->1n1d
            h = shape[2]
            d = h//numHeads
            return  tensor.reshape(1,1,numHeads,d).transpose(0,2,1,3)
    elif len(shape) == 2:
        if inputLayout=="BSH":#1,H->1,n,1,d
            d = shape[1] // numHeads
            return tensor.reshape(1, 1, numHeads, d).transpose(0,2,1,3)
        else: #BSND ND ->1N1D
            n = shape[0]
            d = shape[1]
            return tensor.reshape(1, 1, n, d).transpose(0,2,1,3)
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        d = h//numHeads
        return tensor.reshape(1,1,numHeads,d).transpose(0,2,1,3)
    else:
        exit(1)

def aclnnPromptFlashAttention_unification(torch_tensor_list, params):

    # torch_tensor_list ---tensor  params--attr参数
    pfa_param = {}
    action_type = params["action_type"]
    gold = ["bm_output", "bm_output_gold", "bm_gold"]

    flagList = params['flaglist']
    actualSeqLengths = params['actualseqlengths']
    actualSeqLengthsKV = None
    numHeads = int(params['numheads'])
   
    if len(flagList) > 25 and flagList[25] == 0:
        inputLayout = 'BSH'
        #所有的参数的默认值都给写在这里
        numKeyValueHeads = 0
        scaleValue = 1.0
        preTokens = 214748647
        nextTokens = 214748647
        sp_mode = 0
        block_size =0
    else:
        scaleValue = params['scalevalue'] if 'scalevalue' in params else 1
        preTokens = params['pretokens'] if 'pretokens' in params else 214748647
        nextTokens = params['nexttokens'] if 'nexttokens' in params else 214748647
        inputLayout = params['inputlayout'] if 'inputlayout' in params else "BSH"
        numKeyValueHeads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
        sp_mode = params['sparsemode'] if 'sparsemode' in params else 0

    if numKeyValueHeads == 0:
        numKeyValueHeads = numHeads

    q_shape = params['shape_input'][0]
    out_dtype = get_np_dtype(params['dtype_output'][0])  # numpy类型

    if inputLayout == "SH":
        actualSeqLengthsKV = actualSeqLengths
    else:
        actualSeqLengthsKV = params['actualseqlengthskv']

    if 0 in q_shape or len(q_shape) == 0:
        return torch.from_numpy(torch_tensor_list[0])
    q_tensor, q_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[0], q_shape, numHeads, actualSeqLengths, inputLayout)
    out_shape = q_shape

    # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    if actualSeqLengths !=None:
        if len(actualSeqLengths) == 1 and len(actualSeqLengths) != q_shape[0]:
            actualSeqLengths_item = actualSeqLengths[0]
            for b_count in range(q_bnsd_shape[0] - 1):
                actualSeqLengths.append(actualSeqLengths_item)
        # >> actualSeqLengths预处理：actualSeqLengths长度超过
        if len(actualSeqLengths) > q_bnsd_shape[0]:
            actualSeqLengths = actualSeqLengths[:q_bnsd_shape[0]]
    if actualSeqLengthsKV != None:
        if len(actualSeqLengthsKV) == 1 and len(actualSeqLengthsKV) != q_shape[0]:
            actualSeqLengthsKV_item = actualSeqLengthsKV[0]
            for b_count in range(q_bnsd_shape[0] - 1):
                actualSeqLengthsKV.append(actualSeqLengthsKV_item)
        # >> actualSeqLengths预处理：actualSeqLengths长度超过
        if len(actualSeqLengthsKV) > q_bnsd_shape[0]:
            actualSeqLengthsKV = actualSeqLengthsKV[:q_bnsd_shape[0]]

    # >> kv info
    k_tensor_list = []
    v_tensor_list = []
    k_shape_bnsd_list = []
    v_shape_bnsd_list = []
    k_tensor_bnsd_list = []
    v_tensor_bnsd_list = []
    k_shape_bnsd_raw_list = []
    v_shape_bnsd_raw_list = []
    # k和v的shape_list只能通过抛去已知tensor个数，除2计算
    num_input = len(params['shape_input'])
    tensornum = 21
    k_shape_num = v_shape_num = int((num_input - tensornum) / 2)

    k_start_index = 1
    k_end_index = int(k_shape_num)
    v_start_index = int(k_shape_num) + 1
    v_end_index = int(k_shape_num + v_shape_num)
    kvs_max = 0
    kvs_list = []

    for i in range(k_start_index, k_end_index + 1):

        k_shape = params['shape_input'][i]
        if 0 in k_shape or len(k_shape) == 0:
            return torch.zeros(out_shape)
        k_tensor, k_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[i], k_shape, numKeyValueHeads, actualSeqLengthsKV,
                                                inputLayout)
        k_tensor_bnsd_list.append(k_tensor)
        k_shape_bnsd_raw_list.append(k_bnsd_shape)
        kvs_list.append(k_bnsd_shape[2])
        kvs_max = max(kvs_max,k_bnsd_shape[2])
        if numKeyValueHeads != numHeads:
            k_tensor, k_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_tensor.dtype )
        k_tensor_list.append(k_tensor)
        k_shape_bnsd_list.append(k_bnsd_shape)


    for i in range(v_start_index, v_end_index + 1):
        v_shape = params['shape_input'][i]
        v_tensor, v_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[i], v_shape, numKeyValueHeads, actualSeqLengthsKV,
                                                inputLayout)
        v_tensor_bnsd_list.append(v_tensor)
        v_shape_bnsd_raw_list.append(v_bnsd_shape)
        if numKeyValueHeads != numHeads:
            v_tensor, v_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor,v_tensor.dtype)
        v_tensor_list.append(v_tensor)
        v_shape_bnsd_list.append(v_bnsd_shape)
    kv_empty_flag = False
    k_shape = params['shape_input'][1]
    kvs = kvs_max
    actualprefixKV = 0
    prefix_k_tensor = None
    prefix_v_tensor = None
    prefix_kvs = 0
    if flagList[21]:
        prefix_k_shape = params['shape_input'][v_end_index+17]
        prefix_v_shape = params['shape_input'][v_end_index+18]
        if len(params['prefix_act_lens'])!=0:
            actualprefixKV = params['prefix_act_lens'][0]

        prefix_k_tensor,prefix_k_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[v_end_index+17], prefix_k_shape, numKeyValueHeads, actualprefixKV,
                                                inputLayout)

        prefix_v_tensor,prefix_v_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[v_end_index + 18], prefix_v_shape, numKeyValueHeads,
                                         actualprefixKV,
                                         inputLayout)
        if numKeyValueHeads != numHeads:
            prefix_k_tensor, prefix_k_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, prefix_k_tensor, prefix_k_tensor.dtype)
            prefix_v_tensor, prefix_v_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, prefix_v_tensor, prefix_v_tensor.dtype)
        pfa_param ["actualprefixKV"] = actualprefixKV
        pfa_param ["shared_prefix_k"] = prefix_k_tensor
        pfa_param ["shared_prefix_v"] = prefix_v_tensor
        prefix_kvs = prefix_k_bnsd_shape[2]
        if actualprefixKV==0:
            actualprefixKV = prefix_k_bnsd_shape[2]
        kvs += prefix_kvs

    pse_bnsd_tensor = None
    pse_shift_shape = params['shape_input'][v_end_index+1]  ##1nss or bnss
    qs = q_bnsd_shape[2]
    if flagList[3] == 0 or 0 in pse_shift_shape:
        pse_bnsd_tensor = None
    else:
        pse_bnsd_tensor = torch_tensor_list[3].to(torch.float32).numpy()
          
    m_bnsd_tensor = None
    npu_m_shape = params['shape_input'][v_end_index+2]
    m_dtype = get_np_dtype(params['dtype_input'][v_end_index+2])

    randoms = 0
    mrandom_type = "NORMAL"
    if 'mrandomtype' in params:
        mrandom_type = params['mrandomtype']
        if mrandom_type == 'ones':
            randoms = int(params['mrandom'])

    if flagList[4] == 0 or 0 in npu_m_shape:
        preTokens = 214748647
        nextTokens = 214748647
    else:
        batch = q_bnsd_shape[0]
        numheads = q_bnsd_shape[1]
        npu_m_shape_s = npu_m_shape
        if sp_mode == 0 or sp_mode == 1:
            batch, numheads, ns1, ns2 = get_attention_mask_batch_num(npu_m_shape,q_bnsd_shape)  # 获取输入attentionmask的batch 和numhead
            npu_m_shape_s = [ns1, ns2]
        cpu_m_shape = [qs, kvs]  # cpu

        cpu_m_tensor, npu_m_tensor, preTokens, nextTokens = _create_random_mask_by_spars(cpu_m_shape, npu_m_shape_s,
                                                                                         m_dtype, preTokens, nextTokens,
                                                                                         actualSeqLengths,
                                                                                         actualSeqLengthsKV, actualprefixKV,prefix_kvs,kvs_list, batch,
                                                                                         numheads, sp_mode,
                                                                                         random_ones=randoms)
        if mrandom_type == 'invalid' or mrandom_type == 'invaild':
            randoms = int(params['mrandom'])
            cpu_m_tensor[..., :randoms] = 1
            npu_m_tensor[..., :randoms] = 1

        npu_m_tensor = torch.from_numpy(npu_m_tensor)
     
        if sp_mode == 0 or sp_mode == 1:
            m_bnsd_tensor = _np_broadcast_mask_n(cpu_m_tensor, npu_m_shape, cpu_m_shape, numHeads, q_bnsd_shape[0])
        else:
            m_bnsd_tensor = cpu_m_tensor

    dequant_scale1, dequant_scale2, quant_scale1, quant_scale2, quant_offset2 = None, None, None, None, None

    if flagList[7]:
        dequant_scale1 = torch_tensor_list[v_end_index+3]
        dequant_scale1 = np.frombuffer(dequant_scale1, dtype=np.float32)
        dequant_scale1 = dequant_scale1[: 1]

    if flagList[8]:
        quant_scale1 = torch_tensor_list[v_end_index+4]
    if flagList[9]:
        dequant_scale2 = torch_tensor_list[v_end_index+5]
        dequant_scale2 = np.frombuffer(dequant_scale2, dtype=np.float32)
        dequant_scale2 = dequant_scale2[: 1]
    if flagList[10]:
        quant_scale2 = torch_tensor_list[v_end_index+6]
        quant_scale2_shape = params['shape_input'][v_end_index+6]
        per_channel = True
        if len(quant_scale2_shape) == 1:
            if quant_scale2_shape[0] == 1:
                per_channel = False
        if per_channel:
            quant_scale2 = _trans_1h_to_1n1d(quant_scale2_shape, quant_scale2,
                                             numHeads, inputLayout)
    if flagList[11]:
        quant_offset2 = torch_tensor_list[v_end_index+7]
        quant_offset2_shape = params['shape_input'][v_end_index+7]
        per_channel = True
        if len(quant_offset2_shape) == 1:
            if quant_offset2_shape[0] == 1:
                per_channel = False
        if per_channel:
            quant_offset2 = _trans_1h_to_1n1d(quant_offset2_shape, quant_offset2,
                                              numHeads, inputLayout)

    antiquant_scale = None
    antiquant_offset = None

    k_antiquant_scale = None
    v_antiquant_scale = None

    k_antiquant_offset = None
    v_antiquant_offset = None

    anti_or_kvanti = 0
    if len(flagList)>20:
        # 获取kv量化参数
        if flagList[12]:
            anti_or_kvanti = 1
            antiquantscale_shape = params['shape_input'][v_end_index+8]
            antiquantscale_tensor = torch_tensor_list[v_end_index+8]
            if len(antiquantscale_shape) == 1:  #pertensor
                antiquant_scale = antiquantscale_tensor
            else:  #perchanel
                # 将kv量化参数转换为2n1d(匹配bnsd)
                antiquantscale_2n1d_tensor = _trans_2h_to_2n1d(antiquantscale_shape, antiquantscale_tensor, numKeyValueHeads)

                # GQA场景，扩展kv反量化参数
                if numKeyValueHeads != numHeads:
                    antiquant_scale = broadcast_kv_dequant_tensor(antiquantscale_2n1d_tensor, numKeyValueHeads, numHeads)
                else:
                    antiquant_scale = antiquantscale_2n1d_tensor
        if flagList[13]:
            antiquantoffset_shape = params['shape_input'][v_end_index+9]
            antiquantoffset_tensor = torch_tensor_list[v_end_index+9]
            if len(antiquantoffset_shape) == 1:
                antiquant_offset = antiquantoffset_tensor
            else:
                antiquantoffset_2n1d_tensor = _trans_2h_to_2n1d(antiquantoffset_shape, antiquantoffset_tensor,numKeyValueHeads)

                # GQA场景，扩展kv反量化参数
                if numKeyValueHeads!=numHeads:
                    antiquant_offset = broadcast_kv_dequant_tensor(antiquantoffset_2n1d_tensor, numKeyValueHeads, numHeads)
                else:
                    antiquant_offset = antiquantoffset_2n1d_tensor
        # 判断kv分离的伪量化

        if flagList[17] or flagList[19]:
            anti_or_kvanti = 2
            q_dtype_np = get_np_dtype(params['dtype_input'][0])
            k_antiquantscale_shape = params['shape_input'][v_end_index + 13]
            k_antiquantscale_tensor = torch_tensor_list[v_end_index + 13]
            v_antiquantscale_shape = params['shape_input'][v_end_index + 15]
            v_antiquantscale_tensor = torch_tensor_list[v_end_index + 15]

            if params['k_antiquant_mode'] ==0:
                if len(k_antiquantscale_shape) == 1 and k_antiquantscale_shape[0] == 1:  # pertensor
                    k_antiquant_scale = k_antiquantscale_tensor
                    v_antiquant_scale = v_antiquantscale_tensor
                else:  # perchanel
                    # 将kv量化参数转换为n1d(匹配bnsd)
                    k_antiquantscale_n1d_tensor = _trans_h_to_n1d(k_antiquantscale_shape, k_antiquantscale_tensor,numKeyValueHeads)
                    v_antiquantscale_n1d_tensor = _trans_h_to_n1d(v_antiquantscale_shape, v_antiquantscale_tensor,numKeyValueHeads)
                    # GQA场景，扩展kv反量化参数
                    if numKeyValueHeads != numHeads:
                        k_antiquant_scale = broadcast_kv_split_dequant_tensor(k_antiquantscale_n1d_tensor, numKeyValueHeads,numHeads,q_dtype_np)
                        v_antiquant_scale = broadcast_kv_split_dequant_tensor(v_antiquantscale_n1d_tensor,numKeyValueHeads,numHeads,q_dtype_np)
                    else:
                        k_antiquant_scale = k_antiquantscale_n1d_tensor
                        v_antiquant_scale = v_antiquantscale_n1d_tensor
            if params['k_antiquant_mode'] ==1:#pertoken
                # pertoken -> BS
                k_antiquant_scale = k_antiquantscale_tensor
                v_antiquant_scale = v_antiquantscale_tensor
        pfa_param['k_antiquant_scale'] = k_antiquant_scale
        pfa_param['v_antiquant_scale'] = v_antiquant_scale

        if flagList[18] or flagList[20]:
            q_dtype_np = get_np_dtype(params['dtype_input'][0])
            k_antiquantoffset_shape = params['shape_input'][v_end_index + 14]
            k_antiquantoffset_tensor = torch_tensor_list[v_end_index + 14]
            v_antiquantoffset_shape = params['shape_input'][v_end_index + 16]
            v_antiquantoffset_tensor = torch_tensor_list[v_end_index + 16]

            if params['k_antiquant_mode'] ==0:
                if len(k_antiquantoffset_shape) == 1 and k_antiquantoffset_shape[0] == 1:  # pertensor
                    k_antiquant_offset = k_antiquantoffset_tensor
                    v_antiquant_offset = v_antiquantoffset_tensor
                else:  # perchanel
                    # 将kv量化参数转换为2n1d(匹配bnsd)
                    k_antiquantoffset_n1d_tensor = _trans_h_to_n1d(k_antiquantoffset_shape, k_antiquantoffset_tensor,numKeyValueHeads)
                    v_antiquantoffset_n1d_tensor = _trans_h_to_n1d(v_antiquantoffset_shape, v_antiquantoffset_tensor,numKeyValueHeads)

                    # GQA场景，扩展kv反量化参数
                    if numKeyValueHeads != numHeads:
                        k_antiquant_offset = broadcast_kv_split_dequant_tensor(k_antiquantoffset_n1d_tensor, numKeyValueHeads,numHeads,q_dtype_np)
                        v_antiquant_offset = broadcast_kv_split_dequant_tensor(v_antiquantoffset_n1d_tensor,numKeyValueHeads,numHeads,q_dtype_np)
                    else:
                        k_antiquant_offset = k_antiquantoffset_n1d_tensor
                        v_antiquant_offset = v_antiquantoffset_n1d_tensor
            if params['k_antiquant_mode'] ==1:#pertoken
                # pertoken -> BS
                k_antiquant_offset = k_antiquantoffset_tensor
                v_antiquant_offset = v_antiquantoffset_tensor
        pfa_param['k_antiquant_offset'] = k_antiquant_offset
        pfa_param['v_antiquant_offset'] = v_antiquant_offset


        pfa_param['anti_or_kvanti'] = anti_or_kvanti
        pfa_param['k_antiquant_mode'] = params['k_antiquant_mode']
        if flagList[14]:
            block_table = params['block_table']
            k_cache = torch_tensor_list[21]
            v_cache = torch_tensor_list[22]
    if flagList[15]:
        pfa_param['queryPaddingSize'] = torch_tensor_list[v_end_index+11]
    if flagList[16]:
        pfa_param['kvPaddingSize'] = torch_tensor_list[v_end_index+12]
    pfa_param['lseflag'] = params["softmax_lse_flag"]
    pfa_param['lseshape'] = [q_bnsd_shape[0],q_bnsd_shape[1],q_bnsd_shape[2],1]
    pfa_param["is_benchmark_task"] = params["is_benchmark_task"]


    y_all,lse = _np_promtattention_bnsd(q_tensor, q_bnsd_shape, k_tensor_list, k_shape_bnsd_list, v_tensor_list, v_shape_bnsd_list,
                                    pse_bnsd_tensor,
                                    m_bnsd_tensor, scaleValue, actualSeqLengths, actualSeqLengthsKV, preTokens,
                                    nextTokens, sp_mode, pfa_param, dequant_scale1,
                                    dequant_scale2, quant_scale1, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, out_dtype)
    if inputLayout == "BSH":
        y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape)
    elif inputLayout == "NSD":
        y_all = y_all.reshape(out_shape)
    elif inputLayout == "BSND" or inputLayout == "BNSD_BSND":
        y_all = y_all.transpose(0, 2, 1, 3)
    elif len(out_shape) == 2:
        if q_bnsd_shape[0] == 1:
            y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape).astype(np.float32)
        else:
            sums = 0
            B = q_bnsd_shape[0]
            N = q_bnsd_shape[1]
            S = q_bnsd_shape[2]
            D = q_bnsd_shape[3]
            y_all = y_all.transpose(0, 2, 1, 3)
            yNewAll = np.zeros((S, N * D))
            for i in range(B):
                for j in range(actualSeqLengths[i]):
                    yNewAll[sums + j, :] = y_all[i:i + 1, j].reshape(1, N * D)
                sums += actualSeqLengths[i]
            y_all = yNewAll.astype(np.float32)
    else:
        y_all = y_all
    y_all = torch.from_numpy(y_all)
    if ("softmaxlseflag" in params  and params["softmaxlseflag"]==True) or ("softmax_lse_flag" in params and params["softmax_lse_flag"]==True):
        lse = torch.from_numpy(lse.astype(np.float32))
        return y_all,lse

    return y_all

# ATK 处理逻辑
dtype_map = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.int8: "int8",
    torch.bool: "bool",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uin8",
    torch.float64: "fp64"
}

def overwrite_structured_mask(input_data):
    """
    根据 sparseMode 强制修改 attenMask 的数值结构：
    Mode 2/3: 下三角保留 (Causal)，上三角遮蔽。
    Mode 4:   Band 结构。
    """
    mode = input_data.kwargs.get('sparseMode', 0)
    mask_tensor = input_data.kwargs.get('attenMaskOptional', None)
    
    # print("input_data is: ", input_data)
    # 如果没有 mask 或者 mode 是 0/1 (Default/All)，通常保持随机或全零即可，不做强制修改
    # (或者根据需求，Mode 0 也可以重写，这里主要处理 2,3,4)
    if mask_tensor is None or mode not in [2, 3, 4]:
        return

    # 获取 Mask 的 Shape (预期是 2048x2048 或 [B, 1, 2048, 2048])
    shape = mask_tensor.shape
    # 获取最后两个维度
    S = shape[-2]
    KVS = shape[-1]
    
    # 构造标准的结构化 Mask (numpy)
    # PFA 定义: 1 代表遮蔽(Masked), 0 代表保留(Keep)
    new_mask = np.zeros((S, KVS), dtype=np.int8)
    
    if mode == 2 or mode == 3: 
        # Mode 2 (LeftUpCausal) / Mode 3 (RightDownCausal)
        # 构造上三角 Mask (k=1 表示对角线往上一格开始全是 1)
        new_mask = np.triu(np.ones((S, KVS), dtype=np.int8), k=1)
        
    elif mode == 4:
        # Mode 4 (Band)
        # 需要读取 preTokens 和 nextTokens
        # 注意：Generator生成的可能是 Tensor 或 int，要做兼容处理
        pre_t = input_data.kwargs.get('preTokens', 2147483647)
        next_t = input_data.kwargs.get('nextTokens', 2147483647)
        
        # 如果是 Tensor/List 取第一个值简化处理 (因为 Mode 4 Mask 是固定的 2048x2048)
        if hasattr(pre_t, 'item'): pre_t = pre_t.item()
        if isinstance(pre_t, (list, tuple)): pre_t = pre_t[0]
        if hasattr(next_t, 'item'): next_t = next_t.item()
        if isinstance(next_t, (list, tuple)): next_t = next_t[0]
        
        # 利用广播机制生成 Band
        rows = np.arange(S)[:, None]
        cols = np.arange(KVS)[None, :]
        # 遮蔽条件: j > i + next  OR  j < i - pre
        mask_condition = (cols > rows + next_t) | (cols < rows - pre_t)
        new_mask[mask_condition] = 1

    # --- 将构造好的 2D Mask 广播回原始 Shape ---
    
    # 1. 转回 Tensor
    # 保持和原 Mask 相同的 dtype (通常是 bool 或 int8)
    orig_dtype = mask_tensor.dtype
    new_mask_tensor = torch.from_numpy(new_mask).to(orig_dtype)
    
    # 2. 恢复维度 (Broadcast)
    # 如果原 Mask 是 [B, 1, 2048, 2048]，需要把 2D 扩展回去
    if len(shape) == 4:
        # [2048, 2048] -> [1, 1, 2048, 2048] -> [B, 1, 2048, 2048]
        new_mask_tensor = new_mask_tensor.unsqueeze(0).unsqueeze(0)
        new_mask_tensor = new_mask_tensor.expand(shape[0], shape[1], -1, -1).contiguous()
    elif len(shape) == 3:
        new_mask_tensor = new_mask_tensor.unsqueeze(0)
        new_mask_tensor = new_mask_tensor.expand(shape[0], -1, -1).contiguous()
        
    # 3. 覆盖回 input_data
    if mode in[2, 3, 4]:
        new_mask_tensor = new_mask_tensor.reshape(2048, 2048)
    input_data.kwargs['attenMaskOptional'] = new_mask_tensor
    return input_data
    # print(f"[DEBUG] Overwrite Mask for Mode {mode}: Shape {new_mask_tensor.shape}")

def load_kv_cache(input_data: InputDataset, case_id):
    key_shape = list(input_data.kwargs["key"][0].shape)
    if len(key_shape) == 3:
        H = key_shape[2]
    else:
        H = input_data.kwargs["numKeyValueHeads"] * key_shape[3]
    blocktable_shape = list(input_data.kwargs["blockTableOptional"].shape)
    
    total_blocks = blocktable_shape[0] * blocktable_shape[1]
    cache_shape = [total_blocks, input_data.kwargs["blockSize"], H]

    base_path = "./block_table/"
    files_to_clean = [
        os.path.join(base_path, f"k_cache_{case_id}.npy"),
        os.path.join(base_path, f"v_cache_{case_id}.npy"),
        os.path.join(base_path, f"block_table_{case_id}.npy"),
    ]
    try:
        # --- 文件加载和处理（正常逻辑） ---
        k_cache = np.load(files_to_clean[0])
        v_cache = np.load(files_to_clean[1])
        block_table = np.load(files_to_clean[2])
    finally:
        for f_path in files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
    
    k_cache_tensor = torch.tensor(k_cache, dtype=torch.float32).to(dtype=input_data.kwargs["key"][0].dtype).reshape(cache_shape).npu()
    v_cache_tensor = torch.tensor(v_cache, dtype=torch.float32).to(dtype=input_data.kwargs["value"][0].dtype).reshape(cache_shape).npu()
    
    input_data.kwargs["key"][0] = k_cache_tensor
    input_data.kwargs["value"][0] = v_cache_tensor

    block_table_tensor = torch.tensor(block_table, dtype=torch.int32)
    input_data.kwargs["blockTableOptional"] = torch.tensor(block_table_tensor, dtype=torch.int32).reshape(blocktable_shape).npu()
    
    return

def create_kv_cache(tensor_list, param, case_id):

    blockSize = param['blocksize']
    blockTableShape = param['shape_input'][12]
    blockNum = param['shape_input'][21][0]
    if 0 in blockTableShape:
        print('[WARNING]:block_table为空场景，输出空tensor！')
        return torch.zeros(out_shape)
    blockNumPerBlock = []
    block_num_min = 0
    actualSeqLengths = param["actualseqlengthskv"]
    for actual_seq in actualSeqLengths:
        blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
        block_num_min += math.ceil(actual_seq / blockSize)
    if block_num_min > blockNum:
        print(
            f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}")
        exit(1)
    block_idx_list = np.arange(0, blockNum, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
    block_idx = 0
    block_table = [-1] * blockTableShape[1]
    block_table = np.tile(block_table, (blockTableShape[0], 1)).astype(np.int32)
    block_table_batch_idx = 0
    for idx in blockNumPerBlock:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1
    param['block_table'] = block_table
    if param["is_benchmark_task"]:
        np.save(f"./block_table/block_table_{case_id}.npy", block_table)

    k_cache = np.zeros(param['shape_input'][21]).astype(np.float32)
    v_cache = np.zeros(param['shape_input'][22]).astype(np.float32)
    
    q_bnsd_shape = param["shape_input"][0]
    if param["inputlayout"] == "BSH":
        q_bnsd_shape = [q_bnsd_shape[0], param["numheads"], q_bnsd_shape[1], q_bnsd_shape[2] / param["numheads"]]
    elif param["inputlayout"] == "BSND":
        q_bnsd_shape = [q_bnsd_shape[0], q_bnsd_shape[2], q_bnsd_shape[1], q_bnsd_shape[3]]
    # gen kv cache
    if len(param['shape_input'][21]) == 3:
        # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
        k_tensor_bsh_raw = trans_bnsd_to_bsh(tensor_list[1], param['shape_input'][1])
        v_tensor_bsh_raw = trans_bnsd_to_bsh(tensor_list[2], param['shape_input'][2])
        # kv paddIng
        kvh = param["numkeyvalueheads"] * q_bnsd_shape[3]
        k_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)).astype(np.float32)
        v_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)).astype(np.float32)
        k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
        v_tensor_bsh[:, :v_tensor_bsh_raw.shape[1], :] = v_tensor_bsh_raw[:, :, :]
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_cache[kv_cache_blk_id, 0:blockSize, :] = k_tensor_bsh[b,
                                                                block_offset:(block_offset + blockSize), :]
                    v_cache[kv_cache_blk_id, 0:blockSize, :] = v_tensor_bsh[b,
                                                                block_offset:(block_offset + blockSize), :]
        if param["is_benchmark_task"]:
            np.save(f"./block_table/k_cache_{case_id}.npy", k_cache)
            np.save(f"./block_table/v_cache_{case_id}.npy", v_cache)
        tensor_list[21] = k_cache
        tensor_list[22] = v_cache
    return tensor_list, param

def trans_input_to_params(input_data : InputDataset, case_id, is_benchmark_task=False):

    tensor_list = [None] * 23
    params = {
        'range_input': [[-1, 1], [-1, 1], [-1, 1], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null']],
        'dtype_input': ['fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'uint64', 'fp32', 'uint64', 'fp16', 'fp16', 'fp16', 'fp16', 'int32', 'int64', 'int64', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16'], 'format_input': ['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND'],
        'dtype_output': ['fp16'],
        'type_input': ['tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor'],
        'distribution_input': 'uniform', 
        'attr_1': 'actualseqlengths', 'actualseqlengths': [], 'required_actualseqlengths': 1, 
        'attr_2': 'actualseqlengthskv', 'actualseqlengthskv': [], 'required_actualseqlengthskv': 1, 
        'attr_3': 'prefix_act_lens', 'prefix_act_lens': [], 'required_prefix_act_lens': 1, 
        'attr_4': 'numheads', 'numheads': 8, 'required_numheads': 1, 
        'attr_5': 'scalevalue', 'scalevalue': 0.08838834764831843, 'required_scalevalue': 1, 
        'attr_6': 'pretokens', 'pretokens': 2147483647, 'required_pretokens': 1, 
        'attr_7': 'nexttokens', 'nexttokens': 2147483647, 'required_nexttokens': 1, 
        'attr_8': 'inputlayout', 'inputlayout': 'BNSD', 'required_inputlayout': 1, 
        'attr_9': 'numkeyvalueheads', 'numkeyvalueheads': 1, 'required_numkeyvalueheads': 1, 
        'attr_10': 'sparsemode', 'sparsemode': 0, 'required_sparsemode': 1, 
        'attr_11': 'innerprecise', 'innerprecise': 0, 'required_innerprecise': 1, 
        'attr_12': 'blocksize', 'blocksize': 0, 'required_blocksize': 1, 
        'attr_13': 'antiquant_mode', 'antiquant_mode': 0, 'required_antiquant_mode': 1, 
        'attr_14': 'softmax_lse_flag', 'softmax_lse_flag': False, 'required_softmax_lse_flag': 1, 
        'attr_15': 'k_antiquant_mode', 'k_antiquant_mode': 0, 'required_k_antiquant_mode': 1, 
        'attr_16': 'v_antiquant_mode', 'v_antiquant_mode': 0, 'required_v_antiquant_mode': 1, 
        'attr_17': 'fused_flag', 'fused_flag': 0, 'required_fused_flag': 1, 
        'attr_18': 'mrandomtype', 'mrandomtype': 'Normal', 'required_mrandomtype': 1, 
        'attr_19': 'mrandom', 'mrandom': 0, 'required_mrandom': 1, 
        'attr_20': 'prandom', 'prandom': 0, 'required_prandom': 1, 
        'attr_21': 'enablegpu', 'enablegpu': 'True', 'required_enablegpu': 1, 
        'attr_22': 'flaglist', 'flaglist': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 'required_flaglist': 1, 
        'test_type': 'training', 'action_type': 'bm', 'framework_type': 'aclnn', 'precision_method': 0, 'op_name': 'aclnnFusedInferAttentionScoreV2'}
    
    params["is_benchmark_task"] = is_benchmark_task
    params["case_id"] = case_id

    if input_data.kwargs["query"].dtype == torch.int8:
        tensor_list[0] = query = input_data.kwargs["query"].to(dtype=torch.int8).numpy()
        tensor_list[1] = key = input_data.kwargs["key"][0].to(dtype=torch.int8).numpy()
        tensor_list[2] = value = input_data.kwargs["value"][0].to(dtype=torch.int8).numpy()
    elif input_data.kwargs["antiquantScaleOptional"] != None:
        tensor_list[0] = query = input_data.kwargs["query"].to(dtype=torch.float32).numpy()
        tensor_list[1] = key = input_data.kwargs["key"][0].to(dtype=torch.int8).numpy()
        tensor_list[2] = value = input_data.kwargs["value"][0].to(dtype=torch.int8).numpy()
    else:
        tensor_list[0] = query = input_data.kwargs["query"].to(dtype=torch.float32).numpy()
        tensor_list[1] = key = input_data.kwargs["key"][0].to(dtype=torch.float32).numpy()
        tensor_list[2] = value = input_data.kwargs["value"][0].to(dtype=torch.float32).numpy()
        
    tensor_list[3] = pse = input_data.kwargs["pseShiftOptional"] if input_data.kwargs["pseShiftOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[4] = attenmask = input_data.kwargs["attenMaskOptional"] if input_data.kwargs["attenMaskOptional"] != None else np.array([], dtype=np.float32)

    tensor_list[5] = dequantscale1 = input_data.kwargs["deqScale1Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["deqScale1Optional"] != None else np.array([], dtype=np.float32)
    tensor_list[6] = quantscale1 = input_data.kwargs["quantScale1Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantScale1Optional"] != None else np.array([], dtype=np.float32)
    tensor_list[7] = dequantscale2 = input_data.kwargs["deqScale2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["deqScale2Optional"] != None else np.array([], dtype=np.float32)

    tensor_list[8] = quantscale2 = input_data.kwargs["quantScale2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantScale2Optional"] != None else np.array([], dtype=np.float32)
    tensor_list[9] = quantoffset2 = input_data.kwargs["quantOffset2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantOffset2Optional"] != None else np.array([], dtype=np.float32)
    tensor_list[10] = antiquantscale = input_data.kwargs["antiquantScaleOptional"].to(dtype=torch.float32).numpy() if input_data.kwargs["antiquantScaleOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[11] = antiquantoffset = input_data.kwargs["antiquantOffsetOptional"].to(dtype=torch.float32).numpy() if input_data.kwargs["antiquantOffsetOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[12] = blocktable = input_data.kwargs["blockTableOptional"] if input_data.kwargs["blockTableOptional"] != None else np.array([], dtype=np.int32)

    tensor_list[13] = q_padding_size = input_data.kwargs["queryPaddingSizeOptional"] if input_data.kwargs["queryPaddingSizeOptional"] != None else np.array([], dtype=np.uint64)
    tensor_list[14] = padding_size = input_data.kwargs["kvPaddingSizeOptional"] if input_data.kwargs["kvPaddingSizeOptional"] != None else np.array([], dtype=np.uint64)
    tensor_list[15] = k_antiquantscale = input_data.kwargs["keyAntiquantScaleOptional"] if input_data.kwargs["keyAntiquantScaleOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[16] = k_antiquantoffset = input_data.kwargs["keyAntiquantOffsetOptional"] if input_data.kwargs["keyAntiquantOffsetOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[17] = v_antiquantscale = input_data.kwargs["valueAntiquantScaleOptional"] if input_data.kwargs["valueAntiquantScaleOptional"] != None else np.array([], dtype=np.float32)

    tensor_list[18] = v_antiquantoffset = input_data.kwargs["valueAntiquantOffsetOptional"] if input_data.kwargs["valueAntiquantOffsetOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[19] = k_prefix = input_data.kwargs["keySharedPrefixOptional"] if input_data.kwargs["keySharedPrefixOptional"] != None else np.array([], dtype=np.int8)
    tensor_list[20] = v_prefix = input_data.kwargs["valueSharedPrefixOptional"] if input_data.kwargs["valueSharedPrefixOptional"] != None else np.array([], dtype=np.int8)

    
    params['shape_input'] = [[1]] * 23
    params['shape_input'][0] = list(input_data.kwargs["query"].shape)
    params['dtype_input'][0] = dtype_map[input_data.kwargs["query"].dtype]
    params['dtype_output'][0] = dtype_map[input_data.kwargs["query"].dtype]
    params['shape_input'][1] = list(input_data.kwargs["key"][0].shape)
    params['dtype_input'][1] = dtype_map[input_data.kwargs["key"][0].dtype]
    params['shape_input'][2] = list(input_data.kwargs["value"][0].shape)
    params['dtype_input'][2] = dtype_map[input_data.kwargs["value"][0].dtype]
    
    if input_data.kwargs["pseShiftOptional"] != None:
        params['flaglist'][3] = 1
        params['dtype_input'][3] = dtype_map[input_data.kwargs["pseShiftOptional"].dtype]
        params['shape_input'][3] = list(input_data.kwargs["pseShiftOptional"].shape)
    
    if input_data.kwargs["attenMaskOptional"] != None:
        params['flaglist'][4] = 1
        params['dtype_input'][4] = dtype_map[input_data.kwargs["attenMaskOptional"].dtype]
        params['shape_input'][4] = list(input_data.kwargs["attenMaskOptional"].shape)
    
    if input_data.kwargs["actualSeqLengthsOptional"] != None:
        params['flaglist'][5] = 1

    if input_data.kwargs["actualSeqLengthsKvOptional"] != None:
        params['flaglist'][6] = 1

    if input_data.kwargs["deqScale1Optional"] != None:
        params['flaglist'][7] = 1
        params['dtype_input'][5] = dtype_map[input_data.kwargs["deqScale1Optional"].dtype]
        params['shape_input'][5] = list(input_data.kwargs["deqScale1Optional"].shape)

    if input_data.kwargs["quantScale1Optional"] != None:
        params['flaglist'][8] = 1
        params['dtype_input'][6] = dtype_map[input_data.kwargs["quantScale1Optional"].dtype]
        params['shape_input'][6] = list(input_data.kwargs["quantScale1Optional"].shape)

    if input_data.kwargs["deqScale2Optional"] != None:
        params['flaglist'][9] = 1
        params['dtype_input'][7] = dtype_map[input_data.kwargs["deqScale2Optional"].dtype]
        params['shape_input'][7] = list(input_data.kwargs["deqScale2Optional"].shape)

    if input_data.kwargs["quantScale2Optional"] != None:
        params['flaglist'][10] = 1
        params['dtype_input'][8] = dtype_map[input_data.kwargs["quantScale2Optional"].dtype]
        params['shape_input'][8] = list(input_data.kwargs["quantScale2Optional"].shape)
    
    if input_data.kwargs["quantOffset2Optional"] != None:
        params['flaglist'][11] = 1
        params['dtype_input'][9] = dtype_map[input_data.kwargs["quantOffset2Optional"].dtype]
        params['shape_input'][9] = list(input_data.kwargs["quantOffset2Optional"].shape)

    if input_data.kwargs["antiquantScaleOptional"] != None:
        params['flaglist'][12] = 1
        params['dtype_input'][10] = dtype_map[input_data.kwargs["antiquantScaleOptional"].dtype]
        params['shape_input'][10] = list(input_data.kwargs["antiquantScaleOptional"].shape)
    
    if input_data.kwargs["antiquantOffsetOptional"] != None:
        params['flaglist'][13] = 1
        params['dtype_input'][11] = dtype_map[input_data.kwargs["antiquantOffsetOptional"].dtype]
        params['shape_input'][11] = list(input_data.kwargs["antiquantOffsetOptional"].shape)
    
    if input_data.kwargs["blockTableOptional"] != None:
        params['flaglist'][14] = 1
        params['dtype_input'][12] = dtype_map[input_data.kwargs["blockTableOptional"].dtype]
        params['shape_input'][12] = list(input_data.kwargs["blockTableOptional"].shape)
        actual_seq = key.shape[1] if input_data.kwargs["inputLayout"] in ["BSH", "BSND"] else key.shape[2]
        headdim = key.shape[3] if input_data.kwargs["inputLayout"] in ["BSND", "BNSD"] else key.shape[2] / input_data.kwargs["numKeyValueHeads"]
        block_num = math.ceil(actual_seq / input_data.kwargs["blockSize"])
        cache_shape = [block_num, input_data.kwargs["blockSize"], input_data.kwargs["numKeyValueHeads"] * headdim]
        params['shape_input'][21] = cache_shape
        params['shape_input'][22] = cache_shape

    params["actualseqlengths"] = list(input_data.kwargs["actualSeqLengthsOptional"])
    params["actualseqlengthskv"] = list(input_data.kwargs["actualSeqLengthsKvOptional"])
    params["prefix_act_lens"] = list(input_data.kwargs["actualSharedPrefixLenOptional"])
    params["numheads"] = input_data.kwargs["numHeads"]
    params["scalevalue"] = input_data.kwargs["scaleValue"]
    params["pretokens"] = input_data.kwargs["preTokens"]
    params["nexttokens"] = input_data.kwargs["nextTokens"]
    params["inputlayout"] = input_data.kwargs["inputLayout"]
    params["numkeyvalueheads"] = input_data.kwargs["numKeyValueHeads"]
    params["sparsemode"] = input_data.kwargs["sparseMode"]
    params["innerprecise"] = input_data.kwargs["innerPrecise"]
    params["blocksize"] = input_data.kwargs["blockSize"]
    params["antiquant_mode"] = input_data.kwargs["antiquantMode"]
    params["softmax_lse_flag"] = input_data.kwargs["softmaxLseFlag"]
    params["k_antiquant_mode"] = input_data.kwargs["keyAntiquantMode"]
    params["v_antiquant_mode"] = input_data.kwargs["valueAntiquantMode"]
    
    return tensor_list, params

def trans_input_to_pfa_params(input_data : InputDataset, case_id, is_benchmark_task=False):
  
    pfa_tensor_list, pfa_params = trans_input_to_params(input_data, case_id, is_benchmark_task)

    if input_data.kwargs["blockTableOptional"] != None: 
        pfa_tensor_list, pfa_params = create_kv_cache(pfa_tensor_list, pfa_params, case_id)

    if input_data.kwargs["quantScale2Optional"] != None:
        gloden_output = aclnnPromptFlashAttention_unification(pfa_tensor_list, pfa_params).to(dtype=torch.int8)
    else:
        gloden_output = aclnnPromptFlashAttention_unification(pfa_tensor_list, pfa_params).to(dtype=input_data.kwargs["query"].dtype)

    return gloden_output

def trans_input_to_ifa_params(input_data : InputDataset, case_id, is_benchmark_task=False):
    ifa_tensor_list, ifa_params = trans_input_to_params(input_data, case_id, is_benchmark_task)
    if input_data.kwargs["blockTableOptional"] != None: 
        ifa_tensor_list, ifa_params = create_kv_cache(ifa_tensor_list, ifa_params, case_id)
    if input_data.kwargs["quantScale2Optional"] != None:
        gloden_output = aclnn_op_func_ifa_cpu(ifa_tensor_list, ifa_params).to(dtype=torch.int8)
    else:
        gloden_output = aclnn_op_func_ifa_cpu(ifa_tensor_list, ifa_params).to(dtype=input_data.kwargs["query"].dtype)
    return gloden_output

def aclnn_op_func_fia_cpu(input_data : InputDataset, case_id, is_benchmark_task=False):
    query = input_data.kwargs["query"].to(dtype=torch.float32).numpy()
    key = input_data.kwargs["key"][0].to(dtype=torch.float32).numpy()
    inputLaout = input_data.kwargs["inputLayout"]
    pfaFlag = False
    if inputLaout == "BNSD":
        pfaFlag = query.shape[2] > 1
    if inputLaout in ['BSH', 'BSND']:
        pfaFlag = query.shape[1] > 1
    if pfaFlag:
        return trans_input_to_pfa_params(input_data, case_id, is_benchmark_task)
    else:
        return trans_input_to_ifa_params(input_data, case_id, is_benchmark_task)

@register("executor_fused_infer_attention_score_v2")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)
    
    def init_by_input_data(self, input_data: InputDataset):
        input_data = overwrite_structured_mask(input_data)
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 单标杆比对方法
        output = aclnn_op_func_fia_cpu(input_data, case_id = self.task_result.case_config.id, is_benchmark_task = True)
        # 双标杆比对方法
        # output = aclnn_op_func_fia_cpu(input_data, case_id = self.task_result.case_config.id, is_benchmark_task = self.task_result.is_benchmark_task)
        return output

@register("executor_aclnn_fused_infer_attention_score_v2")
class aclnnFusedInferAttentionScoreApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnFusedInferAttentionScoreApi, self).__init__(task_result, backend)
    
    def init_by_input_data(self, input_data: InputDataset):
        torch.npu.synchronize()
        input_args = []  # 算子的入参列表
        if input_data.kwargs["blockTableOptional"] != None: 

            load_kv_cache(input_data, self.task_result.case_config.id)

        input_args, output_packages = super().init_by_input_data(input_data)
        output_packages = []  # 算子的出参数据包列表
        input_args.pop()
        output_packages.append(input_args[-2])
        # 将所有type是tensor values是None的输入 改为AclTensor类型的空指针
        for i, (name, kwarg) in enumerate(input_data.kwargs.items()):
            if kwarg is None and self.task_result.case_config.inputs[i].type == "tensor":
                from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
                input_args[i] = TensorPtr()
            elif name == "actualSeqLengthsOptional" and self.task_result.case_config.inputs[i][0].dtype == "bool":
                input_args[i] = pointer(AclIntArray)()
        return input_args, output_packages

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output