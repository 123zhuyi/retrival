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
    from ml_dtypes import bfloat16
    new_tensor = np.zeros_like(tensor).astype(bfloat16)
    with np.nditer(tensor, flags=['multi_index'], op_flags=['readwrite']) as it:
        for x in it:
            new_value = ieee_754_conversion(0, int(x), 0)
            new_tensor[it.multi_index] = new_value
    return new_tensor

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

def antiquant(k, v, scale, offset, kv_dtype ="int8"):
    N = k.shape[1]
    S = k.shape[2]
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1","float4_e1m2","float4_e2m1"]:
        scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
        scale = scale_bf16.astype(np.float16)
        scale1 = scale[0]
        scale2 = scale[1]
        if kv_dtype in ["fp4_e1m2", "float4_e1m2"]:
            key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
            value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
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
        scale = scale.astype(np.float32)
        offset = offset.astype(np.float32)
        if kv_dtype == "int8":
            key = k.astype(np.float32)
            value = v.astype(np.float32)
        elif kv_dtype == "int4":
            scale = scale.astype(np.float16)
            offset = offset.astype(np.float16)
            key = k.astype(np.float16)
            value = v.astype(np.float16)
        elif kv_dtype == "hifloat8":
            # uint8->fp16，精度转换
            key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
            value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
        elif kv_dtype in ["float8_e5m2", "float8_e4m3fn"]:
            key = k.astype(np.float16)
            value = v.astype(np.float16)
        else:
            print(f"[ERROR]Invalid input kv dtype: {kv_dtype}")
            exit(1)

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

def softmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_sum

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

    op_type = "IncreFlashAttention"
    compile_info = {}

    inputs = [{'shape': q_shape, 'ori_shape': q_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': q_dtype,
               'range': [(0, None)], 'param_name': 'query'},
              {'shape': k_shape, 'ori_shape': k_shape, 'format': 'ND', 'ori_format': 'ND',
               'dtype': 'int8', 'range': [(0, None)], 'param_name': 'key'},
              {'shape': v_shape, 'ori_shape': v_shape, 'format': 'ND', 'ori_format': 'ND',
               'dtype': 'int8', 'range': [(0, None)], 'param_name': 'value'}]
    if ifa_param['p_flag']:
        inputs.extend([{'shape': p_shape, 'ori_shape': p_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': p_dtype,
                        'range': [(0, None)], 'param_name': 'padding_mask'}])
    else:
        inputs.extend([None])
    if ifa_param['m_flag']:
        inputs.extend([{'shape': m_shape, 'ori_shape': m_shape, 'format': 'ND', 'ori_format': 'ND',
                        'dtype': m_dtype, 'range': [(0, None)], 'param_name': 'atten_mask'}])
    else:
        inputs.extend([None])
    if ifa_param['as_flag']:
        act_seq = ifa_param['actualSeqLengths_raw']
        inputs.extend([{'shape': [len(act_seq)], 'ori_shape': [len(act_seq)], 'format': 'ND', 'ori_format': 'ND',
                        'range': [(0, None)], 'const_value': act_seq, 'dtype': 'int64',
                        'param_name': 'actual_seq_lengths'}])
    else:
        inputs.extend([None])
    if ifa_param['in_quant_flag']:
        inputs.extend([{'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'uint64', 'range': [(0, None)], 'param_name': 'dequant_scale_1'},
                       {'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'float32', 'range': [(0, None)], 'param_name': 'quant_scale_1'},
                       {'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'uint64', 'range': [(0, None)], 'param_name': 'dequant_scale_2'}])
    else:
        inputs.extend([None, None, None])

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

    if ifa_param['padding_flag']:
        inputs.extend([{'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'int64', 'range': [(0, None)], 'param_name': 'kv_padding_size'}])
    else:
        inputs.extend([None])

    outputs = [{'shape': out_shape, 'ori_shape': out_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': out_dtype,
                'range': [(0, None)], 'param_name': 'attention_out'}]

    attrs = [{'name': 'num_heads', 'dtype': 'int', 'value': ifa_param['numHeads']},
             {'name': 'scale_value', 'dtype': 'float', 'value': ifa_param['scaleValue']},
             {'name': 'input_layout', 'dtype': 'str', 'value': ifa_param['inputLayout']},
             {'name': 'num_key_value_heads', 'dtype': 'int', 'value': ifa_param['numKeyValueHeads']},
             {'name': 'block_size', 'dtype': 'int', 'value': ifa_param['blocksize']},
             {'name': 'inner_precise', 'dtype': 'int', 'value': ifa_param['innerprecise']}]
    ret = do_op_tiling(op_type, compile_info, inputs, outputs, attrs=attrs)
    tiling_key = ret.get("tiling_key")
    cache_tiling = ret.get("tiling_data")
    tiling_data_pr = np.frombuffer(cache_tiling, dtype=np.int32)
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

def cut_padding_size(tensor, list, padding_size):
    shape = tensor.shape
    ms = shape[-1]
    for i in range(len(list)):
        cut_len = ms - padding_size - list[i]
        tensor[i:(i + 1), ..., :int(cut_len)] = 1
    return tensor

def cut_padding_size_w(tensor, list, padding_size):
    new_tensor = torch.ones(tensor.shape, dtype=torch.bool).numpy()
    for i in range(len(list)):
        s_begin
        new_tensor[i:(i + 1), ..., :list[i]] = tensor[i:(i + 1), ..., padding_size:list[i] + padding_size]
    return new_tensor

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
    m_tensor = torch.tensor(m_tensor)
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

def _t_softmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum

def _t_ifaattention_act(ifa_param):
    q_tensor = ifa_param['q_tensor_cur']
    k_tensor = ifa_param['k_sub_tensor']
    v_tensor = ifa_param['v_sub_tensor']
    m_tensor = ifa_param['mask_cur']
    m_dtype = ifa_param['m_dtype']
    p_tensor = ifa_param['pse_cur']
    scalar = ifa_param['scaleValue']
    act_seq = ifa_param['act_seq']
    sinner = ifa_param['sinner']
    if ifa_param['padding_flag']:
        s_begin = k_tensor.shape[2] - act_seq - ifa_param['padding_size']
        s_end = k_tensor.shape[2] - ifa_param['padding_size']
    # matmul_dtype 赋值
    matmul_dtype = np.float32
    
    if ifa_param['in_quant_flag']:
        # 输入int8 场景，mm使用int32
        matmul_dtype = np.int32
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
            s_begin = int(s_begin)
            s_end = int(s_end)
            k_cur = k_tensor[:, :, s_begin:s_end, :]
            v_cur = v_tensor[:, :, s_begin:s_end, :]
        else:
            k_cur = k_tensor[:, :, :act_seq, :]
            v_cur = v_tensor[:, :, :act_seq, :]
    if ifa_param['kv_quant_ptog_flag']:
        if ifa_param['padding_flag']:
            ifa_param['antiquantScale'] = ifa_param['antiquantScale'][:, :, :, s_begin:s_end, :]
        else:
            ifa_param['antiquantScale'] = ifa_param['antiquantScale'][:, :, :, :act_seq, :]

    if ifa_param['kv_quant_flag']:
        k_cur, v_cur = antiquant(k_cur, v_cur, ifa_param['antiquantScale'], ifa_param['antiquantOffset'],
                                 ifa_param['k_dtype'])
    qkBmmRes = np.matmul(q_tensor, k_cur.transpose(0, 1, 3, 2), dtype=matmul_dtype)
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
        qkEleRes = qkEleRes + pse_cur
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
        softmax_res, softmax_sum = softmax(qkEleRes)
        #低精度模式会判断q_dtype为fp16还是bf16，增加一个cast过程
        if ifa_param['q_dtype'] == "float16":
            bmm2Res = np.matmul(softmax_res.astype(np.float16).astype(np.float32), v_cur, dtype=matmul_dtype) / softmax_sum
        elif ifa_param['q_dtype'] == "bfloat16":
            bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_cur, dtype=matmul_dtype) / softmax_sum
        # bm_output_gold命令会把q_dtype转成fp32，走到下面这个else分支，不加cast
        else:
            bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype) / softmax_sum
    if not ifa_param["is_benchmark_task"]:
        if ifa_param['q_dtype'] == "float16":
            bmm2Res = np.float16(bmm2Res).astype(np.float32)
        elif ifa_param['q_dtype'] == "bfloat16":
            bmm2Res = bmm2Res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32)
        
    return bmm2Res

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
    input_tensor = input_list[0]
    split_data = np.split(input_tensor, input_tensor.shape[0])
    return split_data

def _trans_antiparam_to_2n1d(shape, tensor, layout, numKeyValueHeads, d):
    # pertensor
    if shape == [2]:
        new_tensor = np.zeros((2, numKeyValueHeads, 1, d))
        new_tensor[0, :, :, :] = tensor[0]
        new_tensor[1, :, :, :] = tensor[1]
        return new_tensor

    elif layout == "BSH":
        d_num = int(shape[1] / numKeyValueHeads)
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
        return new_tensor
    elif layout == "BSND":
        if len(shape) == 4:
            new_tensor = tensor.transpose(0, 2, 1, 3)
        elif len(shape) == 3:
            new_tensor = tensor.reshape(2, 1, shape[1], shape[2]).transpose(0, 2, 1, 3)
        else:
            new_tensor = tensor
            exit(-1)
        return new_tensor
    else:
        return tensor

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

def broadcast_kv_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    if numKeyValueHeads == numheads:
        return tensor
    else:
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

def _t_increattention_bnsd(ifa_param):
    batch_value = ifa_param['q_shape_bnsd'][0]
    k_tensor_list = ifa_param['k_tensor_list_bnsd']
    v_tensor_list = ifa_param['v_tensor_list_bnsd']
    k_shape_list = ifa_param['k_shape_list_bnsd']
    v_shape_list = ifa_param['v_shape_list_bnsd']
    actseqlens = ifa_param['actualSeqLengths']
    if ifa_param['kv_quant_ptog_flag']:
        antiquantScale = ifa_param['antiquantScale']

    actseqlens_size = len(actseqlens)
    y = np.zeros(ifa_param['q_shape_bnsd'], dtype=np.float32)
    # 连续场景预处理
    if ifa_param['continue_flag']:
        # 处理连续场景：将单个tensor shape依据B值拆成列表
        k_shape_list = split_tensor_shape_by_b(k_shape_list)
        v_shape_list = split_tensor_shape_by_b(v_shape_list)
        # 处理连续场景：将单个tensor依据B值拆成列表
        k_tensor_list = split_tensor_by_b(k_tensor_list)
        v_tensor_list = split_tensor_by_b(v_tensor_list)

    for b_index in range(batch_value):
        # 处理实际参与计算的kv s
        if ifa_param['as_flag']:
            ifa_param['act_seq'] = act_seq = actseqlens[b_index]
        else:
            ifa_param['act_seq'] = act_seq = k_shape_list[b_index][2]

        ifa_param['k_sub_shape'] = k_sub_shape = k_shape_list[b_index]
        ifa_param['v_sub_shape'] = v_sub_shape = v_shape_list[b_index]

        if act_seq == 0 or 0 in k_sub_shape or 0 in v_sub_shape:
            continue
        ifa_param['k_sub_tensor'] = k_tensor_list[b_index]
        ifa_param['v_sub_tensor'] = v_tensor_list[b_index]
        ifa_param['q_tensor_cur'] = ifa_param['q_tensor_bnsd'][b_index:(b_index + 1), :, :, :]
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
        # [TEMP]ptog伪量化参数切分
        if ifa_param['kv_quant_ptog_flag']:
            ifa_param['antiquantScale'] = antiquantScale[:, b_index:(b_index + 1), :, :, :]

        y[b_index:(b_index + 1), :, :, :] = _t_ifaattention_act(ifa_param)

    if ifa_param['out_quant_flag']:
        if ifa_param['out_quant_pc_flag']:
            y = quant_pc(y, ifa_param['quantScale2'], ifa_param['quantOffset2'])
        else:
            y = quant(y, ifa_param['quantScale2'][0], ifa_param['quantOffset2'][0])
    return y

def get_kvs_list(shape_list, layout):
    kvs_list = []
    for shape in shape_list:
        if layout == "BNSD":
            kvs = shape[2]
        else:
            kvs = shape[1]
        kvs_list.append(kvs)
    return kvs_list

def get_d(shape, layout, n):
    if layout == "BSH":
        h = shape[2]
        d = h / n
    else:
        d = shape[3]
    return d

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
    ifa_param['out_quant_offset_flag'] = ifa_param['kv_quant_offset_flag'] = False
    ifa_param['kv_quant_ptog_flag'] = False
    ifa_param['continue_flag'] = False
    ifa_param['padding_flag'] = False
    ifa_param['out_quant_pc_flag'] = False
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
        ifa_param['kv_quant_flag'] = True
        if flag_list[12]:
            ifa_param['kv_quant_offset_flag'] = True
    if flag_list[13]:
        ifa_param['pa_flag'] = True
        ifa_param['bt_shape'] = params['shape_input'][12]
        ifa_param['bt_dtype'] = trans_input_dtype(params['dtype_input'][12])
        ifa_param['k_cache_shape'] = params['shape_input'][14]
        ifa_param['v_cache_shape'] = params['shape_input'][15]

    if flag_list[14] and ifa_param['as_flag'] and ifa_param['continue_flag'] and not ifa_param['pa_flag']:
        ifa_param['padding_flag'] = True
        ifa_param['padding_size'] = params['range_input'][v_end_index + 11][0]
        if ifa_param['padding_size'] < 0:
            ifa_param['padding_size'] = 0

    # 获取kv量化参数
    if ifa_param['kv_quant_flag']:
        ifa_param['antiquantscale_shape_raw'] = antiquantscale_shape_raw = params['shape_input'][v_end_index + 8]
        if 0 in ifa_param['antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
        ifa_param['antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 8])
        ifa_param['antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 8]
        if ifa_param['kv_quant_offset_flag']:
            ifa_param['antiquantoffset_shape_raw'] = antiquantoffset_shape_raw = params['shape_input'][v_end_index + 9]
            if 0 in antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
            ifa_param['antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 9]
        else:
            ifa_param['antiquantoffset_shape_raw'] = antiquantscale_shape_raw
            ifa_param['antiquantoffset_tensor_raw'] = torch.zeros(antiquantscale_shape_raw).numpy()

    return ifa_param

def ifa_aclnn_op_func(torch_tensor_list, params, case_id, is_benchmark_task, output_path):
    ifa_param = {}
    action_type = params["action_type"]
    device_type = "cpu"
    if device_type == "cpu":
        ifa_param = get_param(torch_tensor_list, params)
        if not ifa_param['normal_flag']:
            return torch.zeros(ifa_param['out_shape'])
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
        ifa_param['actualSeqLengths'] = actualSeqLengths

        # >> q 预处理：将q转换为bnsd
        q_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(torch_tensor_list[0], q_shape, inputLayout, numHeads)
        ifa_param['q_tensor_bnsd'] = q_tensor
        ifa_param['q_shape_bnsd'] = q_bnsd_shape
        ifa_param['q_d'] = q_bnsd_shape[3]

        # >> kv预处理：1、将kv list 转换为bnsd 2、GQA场景，将kvn扩展为qn
        k_tensor_list = []
        v_tensor_list = []
        k_shape_list = []
        v_shape_list = []
        # 给pa使用的未扩展列表
        k_tensor_list_bnsd = []
        v_tensor_list_bnsd = []
        k_shape_list_bnsd = []
        v_shape_list_bnsd = []
        x = 0
        ifa_param['ks_max'] = 0
        for i in range(k_start_index, k_end_index + 1):
            k_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(torch_tensor_list[i], k_ori_shape_list[x], inputLayout,
                                                            numKeyValueHeads)
            k_tensor_list_bnsd.append(k_tensor)
            k_shape_list_bnsd.append(k_bnsd_shape)
            x = x + 1
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
            v_tensor_list_bnsd.append(v_tensor)
            v_shape_list_bnsd.append(v_bnsd_shape)
            x = x + 1
            if numKeyValueHeads != numHeads:
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
                return torch.zeros(out_shape)

        if ifa_param['m_flag']:
            m_rewrite_flag = False
            if ifa_param['padding_flag']:
                m_tensor = cut_padding_size(m_tensor, actualSeqLengths, ifa_param['padding_size'])
                m_rewrite_flag = True
            if ifa_param['m_dtype'] == "float16":
                m_tensor = torch.where(m_tensor < 0.8, torch.tensor(0, dtype=torch.float16), torch.tensor(1, dtype=torch.float16))
                m_rewrite_flag = True
            if m_rewrite_flag:
                print(f"[INFO]rewrite attenmask")
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

        # >> kv 量化预处理
        if ifa_param['kv_quant_flag']:
            antiquantscale_shape_raw = ifa_param['antiquantscale_shape_raw']
            antiquantscale_tensor_raw = ifa_param['antiquantscale_tensor_raw']
            antiquantoffset_shape_raw = ifa_param['antiquantoffset_shape_raw']
            antiquantoffset_tensor_raw = ifa_param['antiquantoffset_tensor_raw']

            # >> [TEMP]pertoken_group：1、将kv量化参数统一转换成2,B,N,S,D/32 2、GQA场景下，扩展kv量化参数
            if ifa_param['kv_quant_ptog_flag']:
                antiquantscale_tensor = _trans_antiparam_to_2bnsg(antiquantscale_shape_raw, antiquantscale_tensor_raw,
                                                                  inputLayout, numKeyValueHeads)
                antiquantoffset_tensor = _trans_antiparam_to_2bnsg(antiquantoffset_shape_raw,
                                                                   antiquantoffset_tensor_raw,
                                                                   inputLayout, numKeyValueHeads)
                antiquantscale_tensor = broadcast_kv_dequant_tensor_fp8e8m0(antiquantscale_tensor, numKeyValueHeads,
                                                                            numHeads)
                antiquantoffset_tensor = broadcast_kv_dequant_tensor_fp8e8m0(antiquantoffset_tensor, numKeyValueHeads,
                                                                             numHeads)
            else:
                # pertensor/perchannel ->1、将kv量化参数统一转换成2n1d 2、GQA场景下，扩展kv量化参数
                antiquantscale_tensor = _trans_antiparam_to_2n1d(antiquantscale_shape_raw, antiquantscale_tensor_raw,
                                                                 inputLayout, numKeyValueHeads, ifa_param['q_d'])
                antiquantoffset_tensor = _trans_antiparam_to_2n1d(antiquantoffset_shape_raw, antiquantoffset_tensor_raw,
                                                                  inputLayout, numKeyValueHeads, ifa_param['q_d'])

                # GQA场景，扩展kv反量化参数
                q_dtype_np = np.float32
                antiquantscale_tensor = broadcast_kv_dequant_tensor(antiquantscale_tensor, numKeyValueHeads, numHeads,
                                                                    q_dtype_np)
                antiquantoffset_tensor = broadcast_kv_dequant_tensor(antiquantoffset_tensor, numKeyValueHeads, numHeads,
                                                                     q_dtype_np)

            ifa_param['antiquantScale'] = antiquantscale_tensor
            ifa_param['antiquantOffset'] = antiquantoffset_tensor

        if ifa_param['pa_flag']:
            blockSize = ifa_param['blocksize']
            blockTableShape = ifa_param['bt_shape']
            blockNum = ifa_param['k_cache_shape'][0]
            if 0 in ifa_param['bt_shape']:
                return torch.zeros(out_shape)
            blockNumPerBlock = []
            block_num_min = 0
            for actual_seq in actualSeqLengths:
                blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
                block_num_min += math.ceil(actual_seq / blockSize)
            if block_num_min > blockNum:
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
            ifa_param['block_table'] = block_table

            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            target_base_dir = os.path.dirname(current_file_dir)
            target_block_table_dir = os.path.join(target_base_dir, "block_table", output_path)
            if not os.path.exists(target_block_table_dir):
                os.makedirs(target_block_table_dir)
            if not is_benchmark_task:
                save_path = os.path.join(target_block_table_dir, f"block_table_{case_id}.npy")
                np.save(save_path, block_table)
            k_cache = np.zeros(ifa_param['k_cache_shape']).astype(np.float32)
            v_cache = np.zeros(ifa_param['v_cache_shape']).astype(np.float32)

            if len(ifa_param['k_cache_shape']) == 3:

                k_tensor_bsh_raw = trans_bnsd_to_bsh(k_tensor_list_bnsd[0], k_shape_list_bnsd[0])
                v_tensor_bsh_raw = trans_bnsd_to_bsh(v_tensor_list_bnsd[0], v_shape_list_bnsd[0])

                kvh = numKeyValueHeads * q_bnsd_shape[3]
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
                if not is_benchmark_task:
                    np.save(f"{target_block_table_dir}/k_cache_{case_id}.npy", k_cache)
                    np.save(f"{target_block_table_dir}/v_cache_{case_id}.npy", v_cache)
            elif len(ifa_param['k_cache_shape']) == 4:
                kvn = k_tensor_list_bnsd[0].shape[1]
                kvs = k_tensor_list_bnsd[0].shape[2]
                kvd = k_tensor_list_bnsd[0].shape[3]
                k_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(np.float32)
                v_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(np.float32)
                k_tensor_bnsd[:, :, :kvs, :] = k_tensor_list_bnsd[0][:, :, :, :]
                v_tensor_bnsd[:, :, :kvs, :] = v_tensor_list_bnsd[0][:, :, :, :]

                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            for n in range(q_bnsd_shape[1]):
                                k_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_tensor_bnsd[b, n:n + 1,
                                                                                    block_offset:(
                                                                                                block_offset + blockSize),
                                                                                    :]
                                v_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = v_tensor_bnsd[b, n:n + 1,
                                                                                    block_offset:(
                                                                                                block_offset + blockSize),
                                                                                    :]
            else:
                return torch.zeros(out_shape)
            # 将kv cache 生成新的bin文件
            if params['dtype_input'][1] == "fp16" and params['dtype_input'][2] == "fp16":
                k_cache = torch.tensor(k_cache, dtype=torch.float16)
                v_cache = torch.tensor(v_cache, dtype=torch.float16)
            elif params['dtype_input'][1] == "bf16" and params['dtype_input'][2] == "bf16":
                k_cache = torch.tensor(k_cache, dtype=torch.bfloat16)
                v_cache = torch.tensor(v_cache, dtype=torch.bfloat16)
            # [新增] 支持 fp32，这是 CPU 运算最常用的类型
            elif params['dtype_input'][1] == "fp32" and params['dtype_input'][2] == "fp32":
                k_cache = torch.tensor(k_cache, dtype=torch.float32)
                v_cache = torch.tensor(v_cache, dtype=torch.float32)
            elif params['dtype_input'][1] == "int8" and params['dtype_input'][2] == "int8":
                k_cache = torch.tensor(k_cache, dtype=torch.int8)
                v_cache = torch.tensor(v_cache, dtype=torch.int8)
            else:
                print(f"[ERROR][PageAtten]Wrong input dtype! Got Key: {params['dtype_input'][1]}, Value: {params['dtype_input'][2]}")
            k_cache_index = 14
            v_cache_index = 15

        # >> 输入量化预处理：获取切分使用的sinner
        ifa_param['sinner'] = 0
        if ifa_param['in_quant_flag']:
            sinner = get_sinner(ifa_param)
            ifa_param['sinner'] = sinner

        ifa_param["is_benchmark_task"] = is_benchmark_task
        # ===计算入口===
        out_shape = q_shape
        y_all = _t_increattention_bnsd(ifa_param)
        y_all = torch.from_numpy(y_all)
        y_all = trans_bnsd_to_layout(y_all, out_shape, inputLayout)
        return y_all

# pfa
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

def _random_fill_tensor(tensor, shape, random_number, value=0):
    for i in range(0, random_number):
        point = []
        for k in range(0, len(shape)):
            point.append(random.randint(1, shape[k]) - 1)
        tensor[point[0], point[1]] = value
    return tensor

def _np_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor):
    if numHeads == numKeyValueHeads:
        return kv_tensor, kv_tensor.shape
    factor = numHeads // numKeyValueHeads
    kv_res = np.repeat(kv_tensor, factor, axis=1)
    return kv_res, kv_res.shape

def _create_mask_band(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch, numheads, m_dtype, random_ones=0):

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
        if len(actualSeqLengthsKV) == 0:
            S2 = mask_s_kv
        else:
            S2 = actualSeqLengthsKV[i]
        pre_tokens_new = S1-S2+pre_tokens
        pre_tokens_list.append(pre_tokens_new)

        next_tokens_new = S2-S1+next_tokens
        next_tokens_list.append(next_tokens_new)
        atten_masks = _create_mask(m_shape, pre_tokens_new, next_tokens_new)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, pre_tokens_list, next_tokens_list

def _create_mask_right_down(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch, numheads, m_dtype, random_ones=0):

    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) == 0:
            S2 = mask_s_kv
        else:
            S2 = actualSeqLengthsKV[i]
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

def _create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch=1, numheads=1, sp_mode=0, random_ones=0):
    # mask shape [sq,skv]  #mshape  npu  fshape cpu
    if sp_mode == 0:
        cpu_mask ,npu_mask = _create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads,m_dtype, random_ones)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 1:
        pre_tokens = 214748647
        next_tokens = 214748647
        cpu_mask ,npu_mask = _create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads,m_dtype, random_ones)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens

    if sp_mode == 2:
        pre_tokens = 214748647
        next_tokens = 0
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, batch, numheads,m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype),pre_tokens,next_tokens
    if sp_mode == 3: #rightdown
        pre_tokens = 214748647
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask,next_tokens_new = _create_mask_right_down(cpu_m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch, numheads, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens_new
    if sp_mode ==4:
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, pre_tokens_new, next_tokens_new = _create_mask_band(cpu_m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch, numheads, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens_new, next_tokens_new

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
    elif inputLayout == "TND" or inputLayout == "NTD_TND":
        N = bsh_shape[1]
        D = bsh_shape[2]
        B = len(actSeqLength)
        S = max(actSeqLength)
        if inputLayout=="NTD_TND":
            N = bsh_shape[0]
            tensor = tensor.transpose(1, 0, 2)
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        return new_tensor, [B, N, S, D]
    else:
        return tensor, bsh_shape

def get_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m
def alibi_biases(pre_shift_shape):
    q_seq = pre_shift_shape[2]
    kv_seq = pre_shift_shape[3]
    alibi_biases = np.zeros([q_seq, kv_seq],dtype=np.float32)
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
    return pse_shift.astype(pse_dtype),m.astype(pse_dtype)

def _np_broadcast_pseShift_n(pse_shift_tensor, pse_shift_shape, q_batch):#pre_shift, pre_shift_shape, q_bnsd_shape[0]
    #1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        pse_res = np.zeros([B, pse_shift_tensor.shape[1],pse_shift_tensor.shape[2], pse_shift_tensor.shape[3]],dtype=np.float32)
        for i in range(B):
            pse_res[i:i+1] = pse_shift_tensor[0]
        return pse_res

def _np_broadcast_mask_n(m_tensor, m_shape, mask_cur_shape, numheads, q_batch):
    if len(m_shape) == 4:
        # b1ss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):

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
        exp_max = np.exp(max_front - x_max)
        out = x_exp * x_exp_new
        if is_fp16:
            out = out.astype(np.float16)
            exp_max = exp_max.astype(np.float16)

        return out, x_max, x_sum, exp_max

def _np_pfaattention_act_int8(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, innerprecise,
                              dequant_scale1=None,
                              dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                              out_dtype="<class 'numpy.int8'>"):
    S = None
    if act_seq ==0:
        pass
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]
        S = act_kv_seq
    else:
        S = k_tensor.shape[2]

    if mask_tensor is not None:
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq].astype(np.float16)
    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xffffe000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xffffe000)
    dequant_scale2.dtype = np.float32

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

        qk_i = qk_i* np.float32(scalar)
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
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

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
    if quant_scale2 is not None:
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
    return o_front

def truncate_to_bf16_sim(x_fp32):
    x_int = x_fp32.view(np.int32)
    x_int_truncated = x_int & 0xFFFF0000 
    return x_int_truncated.view(np.float32)

def _np_pfaattention_act(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, innerprecise, pfa_param, dequant_scale1=None,
                         dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None, out_dtype="<class 'numpy.int8'>"):
    input_dtype = q_tensor.dtype
    is_low_precision = (input_dtype == np.float16)
    def fast_cast_op(op_result):
        if is_low_precision:
            return op_result.astype(np.float16).astype(np.float32)
        return op_result

    q_f32 = q_tensor.astype(np.float32)
    k_f32 = k_tensor.astype(np.float32)
    v_f32 = v_tensor.astype(np.float32)

    if act_seq is not None:
        q_f32 = q_f32[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_f32 = k_f32[:, :, :act_kv_seq, :]
        v_f32 = v_f32[:, :, :act_kv_seq, :]
        
    # === Matmul 1 ===
    qkBmmRes = np.matmul(q_f32, k_f32.transpose([0, 1, 3, 2]))
    # [关键] 模拟 FP16 截断
    qkBmmRes = fast_cast_op(qkBmmRes)
    
    # Scale
    qkEleRes = qkBmmRes * scalar
    qkEleRes = fast_cast_op(qkEleRes)

    # === Add PSE ===
    if pse_tensor is not None:
        pse_f32 = pse_tensor.astype(np.float32)
        if act_seq is not None:
            pse_f32 = pse_f32[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_f32 = pse_f32[:, :, :, :act_kv_seq]
        qkEleRes += pse_f32
        # 加法后也模拟截断
        qkEleRes = fast_cast_op(qkEleRes)

    # === Add Mask ===
    if mask_tensor is not None:
        mask_bool = mask_tensor.astype(bool)
        if act_seq is not None:
            mask_bool = mask_bool[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_bool = mask_bool[:, :, :, :act_kv_seq]
            
        while mask_bool.ndim < qkEleRes.ndim:
            mask_bool = np.expand_dims(mask_bool, axis=0)
            
        qkEleRes[mask_bool] = -1.7e38 # Mask 值通常不需要截断，保持极小即可

    # === Softmax ===
    softmax_exp, _, x_sum = npSoftmax_new(qkEleRes)
    
    # Softmax 结果非常容易溢出，必须模拟截断
    softmax_exp = fast_cast_op(softmax_exp)
    x_sum = fast_cast_op(x_sum)

    # === Matmul 2 ===
    bmm2Res = np.matmul(softmax_exp, v_f32)
    bmm2Res = fast_cast_op(bmm2Res)

    # === Normalize ===
    x_sum = np.maximum(x_sum, 1e-9)
    bmm2Res = bmm2Res / x_sum
    bmm2Res = fast_cast_op(bmm2Res)

    # === Output Convert ===
    if "int8" in str(out_dtype):
        if quant_scale2 is not None:
            qs2 = quant_scale2.astype(np.float32)
            bmm2Res = bmm2Res * qs2
            
            if quant_offset2 is not None:
                qo2 = quant_offset2.astype(np.float32)
                bmm2Res += qo2
                
            bmm2Res = np.around(bmm2Res)
            bmm2Res = np.clip(bmm2Res, -128, 127)
            bmm2Res = bmm2Res.astype(np.int8)
    else:
        if out_dtype == np.float16 or "float16" in str(out_dtype):
            bmm2Res = bmm2Res.astype(np.float16)
        elif "bfloat16" in str(out_dtype):
            pass 
        
    return bmm2Res

def _np_pfaattention_act_backup(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, innerprecise,pfa_param,dequant_scale1=None,
                        dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None, out_dtype="<class 'numpy.int8'>"):

    qdtype = pfa_param['q_dtype']
    q_tensor_or = copy.deepcopy(q_tensor)
    k_tensor_or = copy.deepcopy(k_tensor)
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]
    qDtype = q_tensor.dtype 
    qkBmmRes = np.matmul(q_tensor, k_tensor.transpose([0, 1, 3, 2]))
    qkEleRes = qkBmmRes * scalar 

    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :, :act_kv_seq]
        qkEleRes += pse_tensor.astype(np.float32)

    if mask_tensor is not None:
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :, :act_kv_seq]
        qkEleRes[mask_tensor.astype(np.bool_)] = -1.7 * 10 ** 38
    softmax_res, x_max, x_sum = npSoftmax_new(qkEleRes)

    
    if qdtype==np.float16 or qdtype == np.float32:
        bmm2Res = np.matmul(softmax_res, v_tensor)
    else:
        bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_tensor)

    #/sum
    bmm2Res = bmm2Res/x_sum
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                bmm2Res[:, :, i, :] = 0

    if str(out_dtype) == "<class 'numpy.int8'>":
        bmm2Res = bmm2Res * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            bmm2Res+= quant_offset2.astype(np.float16)
        bmm2Res = np.around(bmm2Res)
        # 转成int8防止溢出
        bmm2Res = np.where(bmm2Res > 127, 127, bmm2Res)
        bmm2Res = np.where(bmm2Res < -128, -128, bmm2Res)
        bmm2Res = bmm2Res.astype(np.int8)

    return bmm2Res

def _np_promtattention_bnsd(q_tensor, q_shape, k_tensor, k_shape, v_tensor, v_shape, pse_bnsd_tensor, mask_tensor, scale, actseqlens,
                           actkvseqlens, preTokens_list, nextTokens_list, sparse, innerprecise, pfa_param, dequant_scale1=None, dequant_scale2=None, quant_scale1=None, quant_scale2=None,
                           quant_offset2=None, out_dtype="<class 'numpy.int8'>"):

    batch_value = q_shape[0]
    numhead_value = q_shape[1]
    actseqlens_size = len(actseqlens)
    outshape = q_shape
    outshape[-1] = v_shape[-1]
    y = np.zeros(outshape, dtype=np.float32)
    preTokens = preTokens_list
    nextTokens = nextTokens_list
    for b_index in range(batch_value):
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
        # bnsd
        if sparse ==4:
            preTokens = preTokens_list if not isinstance(preTokens_list, list) else preTokens_list[0]
        if sparse == 3 or sparse ==4:
            preTokens = preTokens_list if not isinstance(preTokens_list, list) else preTokens_list[0]
        mask_cur = np.zeros([1, 1, q_shape[2], k_shape[2]], dtype='uint8')
        if mask_tensor is None:
            mask_cur = None
        else:
            if mask_tensor.ndim == 2:
                src_mask = mask_tensor
            else:
                src_mask = mask_tensor[b_index]
            while src_mask.ndim > 2:
                src_mask = src_mask[0]
            h_dst, w_dst = mask_cur.shape[2], mask_cur.shape[3]
            # 来源 Mask 尺寸
            h_src, w_src = src_mask.shape
            h_valid = min(h_dst, h_src)
            w_valid = min(w_dst, w_src)
            mask_cur[0, 0, :h_valid, :w_valid] = src_mask[:h_valid, :w_valid]

        for n_index in range(numhead_value):
            if len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4:
                q_tensor_cur = q_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                k_tensor_cur = k_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                v_tensor_cur = v_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

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

                if act_seq is None:
                    if q_tensor.dtype == "int8" or q_tensor.dtype == "int16" :
                        y[b_index:(b_index + 1), n_index:(n_index + 1), :, :] = _np_pfaattention_act_int8(q_tensor_cur,
                                                                                                         k_tensor_cur,
                                                                                                         v_tensor_cur,
                                                                                                         pse_cur,
                                                                                                         mask_cur,
                                                                                                         scale, act_seq,
                                                                                                         act_kv_seq,
                                                                                                         preTokens,nextTokens,
                                                                                                         innerprecise,
                                                                                                         dequant_scale1,
                                                                                                         dequant_scale2,
                                                                                                         quant_scale1,
                                                                                                         quant_scale2_cur,
                                                                                                         quant_offset2_cur,
                                                                                                         out_dtype)
                    else:
                        if q_tensor.dtype == "fp16":
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :, :] = _np_pfaattention_act(q_tensor_cur,
                                                                                                        k_tensor_cur,
                                                                                                        v_tensor_cur,
                                                                                                        pse_cur,
                                                                                                        mask_cur, scale,
                                                                                                        act_seq, act_kv_seq,
                                                                                                        preTokens, nextTokens,
                                                                                                        innerprecise,pfa_param,
                                                                                                        dequant_scale1,
                                                                                                        dequant_scale2,
                                                                                                        quant_scale1,
                                                                                                        quant_scale2_cur,
                                                                                                        quant_offset2_cur,
                                                                                                            out_dtype)
                        else:
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :, :] = _np_pfaattention_act_backup(q_tensor_cur,
                                                                                                        k_tensor_cur,
                                                                                                        v_tensor_cur,
                                                                                                        pse_cur,
                                                                                                        mask_cur, scale,
                                                                                                        act_seq, act_kv_seq,
                                                                                                        preTokens, nextTokens,
                                                                                                        innerprecise,pfa_param,
                                                                                                        dequant_scale1,
                                                                                                        dequant_scale2,
                                                                                                        quant_scale1,
                                                                                                        quant_scale2_cur,
                                                                                                        quant_offset2_cur,
                                                                                                            out_dtype)
                else:
                    if q_tensor.dtype == "int8":
                        if act_seq==0 or act_kv_seq==0:
                            pass
                        else:
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :act_seq, :] = _np_pfaattention_act_int8(
                                q_tensor_cur, k_tensor_cur, v_tensor_cur, pse_cur,mask_cur, scale, act_seq, act_kv_seq,preTokens,nextTokens,innerprecise,
                                dequant_scale1, dequant_scale2, quant_scale1, quant_scale2_cur, quant_offset2_cur, out_dtype)
                    else:
                        if q_tensor.dtype == "fp16":
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :act_seq, :] = _np_pfaattention_act(q_tensor_cur,
                                                                                                           k_tensor_cur,
                                                                                                           v_tensor_cur,
                                                                                                           pse_cur,
                                                                                                           mask_cur,
                                                                                                           scale,
                                                                                                           act_seq,
                                                                                                           act_kv_seq,
                                                                                                           preTokens,nextTokens,
                                                                                                            innerprecise,pfa_param,
                                                                                                           dequant_scale1,
                                                                                                           dequant_scale2,
                                                                                                           quant_scale1,
                                                                                                           quant_scale2_cur,
                                                                                                           quant_offset2_cur,
                                                                                                         out_dtype)
                        else:
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :act_seq, :] = _np_pfaattention_act_backup(q_tensor_cur,
                                                                                                           k_tensor_cur,
                                                                                                           v_tensor_cur,
                                                                                                           pse_cur,
                                                                                                           mask_cur,
                                                                                                           scale,
                                                                                                           act_seq,
                                                                                                           act_kv_seq,
                                                                                                           preTokens,nextTokens,
                                                                                                            innerprecise,pfa_param,
                                                                                                           dequant_scale1,
                                                                                                           dequant_scale2,
                                                                                                           quant_scale1,
                                                                                                           quant_scale2_cur,
                                                                                                           quant_offset2_cur,
                                                                                                         out_dtype)
    return y

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


def gen_outshape(layout,qshape,vshape,numKeyValueHeads,numHeads):
    outshape = copy.deepcopy(qshape)
    if layout=="BSH":
        vd = int(vshape[-1]/numKeyValueHeads)
        # outshape[-1] = vd*numKeyValueHeads
        outshape[-1] = vd * numHeads
        return outshape
    else:
        outshape[-1] = vshape[-1]
        return outshape
def trans_tnd_actseq(list):
    list_len = len(list)
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i + 1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            print(f"[ERROR] trans_tnd_actseq: Wrong input actseq:{list}, in loop {i}, item {new_item} < 0")
    return list_new

def pfa_aclnn_op_func(torch_tensor_list, params, is_benchmark_task):
    action_type = params["action_type"]
    gold = ["bm_output", "bm_output_gold"]
    device_type = "cpu"
    innerprecise = params['innerprecise']
    if device_type == "cpu":
        pfa_param = {}
        pfa_param['action_type'] = params["action_type"]
        flagList = params['flaglist']
        actualSeqLengths = params['actualseqlengths']
        actualSeqLengthsKV = params['actualseqlengthskv']
        numHeads = int(params['numheads'])
        #设置参数默认值
        scaleValue = 1
        preTokens = 214748647
        nextTokens = 0
        inputLayout = "BSH"
        numKeyValueHeads = 0
        sp_mode = 0
        if flagList[13]:
            scaleValue = params['scalevalue']
        if flagList[14]:
            preTokens = params['pretokens']
        if flagList[15]:
            nextTokens = params['nexttokens']
        if flagList[16]:
            inputLayout = params['inputlayout']
        if flagList[16]:
            numKeyValueHeads = int(params['numkeyvalueheads'])
        if flagList[17]:
            sp_mode = params['sparsemode']
        if  numKeyValueHeads == 0:
            numKeyValueHeads = numHeads

        q_shape = params['shape_input'][0]
        q_dtype = tools.get_np_dtype(params['dtype_input'][0])  # numpy类型
        pfa_param['q_dtype'] = q_dtype
        out_dtype = tools.get_np_dtype(params['dtype_output'][0])  # numpy类型
        v_shape = params['shape_input'][2]
        out_shape = gen_outshape(inputLayout, q_shape,v_shape,numKeyValueHeads,numHeads)
        if inputLayout == "SH":
            actualSeqLengthsKV = actualSeqLengths
        if inputLayout == "TND" or inputLayout=="NTD_TND":
            actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
            actualSeqLengthsKV = trans_tnd_actseq(actualSeqLengthsKV)

        if 0 in q_shape or len(q_shape) == 0:
            return torch.from_numpy(torch_tensor_list[0])
        else:
            q_tensor, q_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[0], q_shape, numHeads, actualSeqLengths,inputLayout)
        if inputLayout !="TND" and inputLayout !="NTD_TND":
            # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
            if actualSeqLengths != None:
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

        kv_empty_flag = False
        k_shape = params['shape_input'][1]

        if 0 in k_shape or len(k_shape) == 0:
            kv_empty_flag = True
            return torch.zeros(out_shape)
        else:
            k_tensor, k_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[1], k_shape, numKeyValueHeads, actualSeqLengthsKV,
                                                    inputLayout)
            if numKeyValueHeads != numHeads:
                k_tensor, k_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor)


        if 0 in v_shape or len(v_shape) == 0 or kv_empty_flag:
            kv_empty_flag = True
            return torch.zeros(out_shape)
        else:
            v_tensor, v_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[2], v_shape, numKeyValueHeads, actualSeqLengthsKV,
                                                    inputLayout)

            if numKeyValueHeads != numHeads:
                v_tensor, v_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor)
        qs = q_bnsd_shape[2]

        kvs = k_bnsd_shape[2]
        pse_bnsd_tensor = None

        pse_shift_shape = params['shape_input'][3] ##1nss or bnss
        if flagList[3] == 0 or 0 in pse_shift_shape:
            pse_bnsd_tensor = None
        else:
            pse_dtype_torch = tools.get_pt_dtype(params['dtype_input'][3])
            pse_shift = None
            pse_shift_random_flag = True
            if action_type in gold:
                pse_shift = torch_tensor_list[3]
            else:
                if 'prandom' in params:
                    if params['prandom']!=0:
                        pse_shift = torch_tensor_list[3]
                    else:
                        pse_shift_random_flag = False
                else:
                    pse_shift_random_flag = False
                if not pse_shift_random_flag:
                    pse_shift ,m = get_all_alibi(numHeads, pse_shift_shape)
                    pse_shift[:, :, :, kvs:] = 1
                    pse_shift[:, :, qs:, :] = 1
                    npu_pse_shift = torch.from_numpy(pse_shift).to(pse_dtype_torch)
                    pse_shift = npu_pse_shift.to(torch.float32).numpy().astype(np.float32)
            cpu_pse_shift = pse_shift[:,:,:qs,:kvs]
            pse_bnsd_tensor = _np_broadcast_pseShift_n(cpu_pse_shift, pse_shift_shape, q_bnsd_shape[0]) #to bnsd
            pse_bnsd_tensor = torch_tensor_list[3]
        m_bnsd_tensor = None
        npu_m_shape = params['shape_input'][4]
        m_dtype = tools.get_np_dtype(params['dtype_input'][4])

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
            m_bnsd_tensor = torch_tensor_list[4]
        dequant_scale1, dequant_scale2, quant_scale1, quant_scale2, quant_offset2 = None, None, None, None, None

        if flagList[7]:
            dequant_scale1 = torch_tensor_list[5]
        if flagList[8]:
            quant_scale1 = torch_tensor_list[6]
        if flagList[9]:
            dequant_scale2 = torch_tensor_list[7]
        if flagList[10]:
            quant_scale2 = torch_tensor_list[8]

            quant_scale2_shape = params['shape_input'][8]
            per_channel = True
            if len(quant_scale2_shape)==1:
                if quant_scale2_shape[0]==1:
                    per_channel = False
            if per_channel:
                quant_scale2 = _trans_1h_to_1n1d(quant_scale2_shape, quant_scale2,
                                                               numHeads,inputLayout)
        
        if flagList[11]:
            quant_offset2 = torch_tensor_list[9]
            quant_offset2_shape = params['shape_input'][9]
            per_channel = True
            if len(quant_offset2_shape) == 1:
                if quant_offset2_shape[0] == 1:
                    per_channel = False
            if per_channel:
                quant_offset2 = _trans_1h_to_1n1d(quant_offset2_shape, quant_offset2,
                                                 numHeads, inputLayout)
        
        y_all = _np_promtattention_bnsd(q_tensor, q_bnsd_shape, k_tensor, k_bnsd_shape, v_tensor, v_bnsd_shape,pse_bnsd_tensor,
                                       m_bnsd_tensor, scaleValue, actualSeqLengths, actualSeqLengthsKV, preTokens, nextTokens, sp_mode, innerprecise, pfa_param,dequant_scale1,
                                       dequant_scale2, quant_scale1, quant_scale2, quant_offset2, out_dtype)

        if inputLayout == "BSH":
            y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape)
        elif inputLayout == "NSD":
            y_all = y_all.reshape(out_shape)
        elif inputLayout == "TND"  or inputLayout == "NTD_TND":
            T = sum(actualSeqLengths)
            B = y_all.shape[0]
            if inputLayout == "TND":
                N = out_shape[1]
            else:
                N = out_shape[0]
            D = y_all.shape[3]
            output = np.zeros((T, N, D), dtype=np.float32)
            t_start = 0
            for b_index in range(B):
                act_s = actualSeqLengths[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = y_all[b_index, n_index, :act_s, :]
                t_start += act_s
            y_all = output
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
        return y_all
    
    else:
        pass


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

def aclnn_op_func_fia_cpu(input_data : InputDataset, case_id, is_benchmark_task, output_path):
    query = input_data.kwargs["query"].to(dtype=torch.float32).numpy()
    key = input_data.kwargs["key"][0].to(dtype=torch.float32).numpy()
    inputLaout = input_data.kwargs["inputLayout"]
    pfaFlag = False
    if inputLaout == "BNSD":
        pfaFlag = query.shape[2] == key.shape[2]
    if inputLaout in ['BSH', 'BSND']:
        pfaFlag = query.shape[1] == key.shape[1]
    if pfaFlag:
        return aclnn_op_func_pfa_cpu(input_data, is_benchmark_task)
    else:
        return aclnn_op_func_ifa_cpu(input_data, case_id, is_benchmark_task, output_path)

def aclnn_op_func_pfa_cpu(input_data : InputDataset, is_benchmark_task):

    torch_tensor_list = [
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.uint64), 
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.uint64), 
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32)
    ]

    pfa_params = {
        'shape_input': [[1, 70, 512], [1, 70, 512], [1, 70, 512], [1], [1], [1], [1], [1], [1], [1]], 
        'range_input': [[-1, 1], [-1, 1], [-1, 1], ['null', 'null'], [0, 1], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null']], 
        'dtype_input': ['fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'uint64', 'fp32', 'uint64', 'fp16', 'fp16'], 
        'format_input': ['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND'], 
        'dtype_output': ['fp16'], 
        'type_input': ['tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor'], 
        'attr_1': 'actualseqlengths', 'actualseqlengths': [58], 'required_actualseqlengths': 1, 
        'attr_2': 'actualseqlengthskv', 'actualseqlengthskv': [58], 'required_actualseqlengthskv': 1, 
        'attr_3': 'numheads', 'numheads': 1, 'required_numheads': 1, 
        'attr_4': 'scalevalue', 'scalevalue': 0.044194173824159216, 'required_scalevalue': 1, 
        'attr_5': 'pretokens', 'pretokens': 2147483647, 'required_pretokens': 1, 
        'attr_6': 'nexttokens', 'nexttokens': 2147483647, 'required_nexttokens': 1, 
        'attr_7': 'inputlayout', 'inputlayout': 'BSH', 'required_inputlayout': 1, 
        'attr_8': 'numkeyvalueheads', 'numkeyvalueheads': 1, 'required_numkeyvalueheads': 1, 
        'attr_9': 'sparsemode', 'sparsemode': 0, 'required_sparsemode': 1, 
        'attr_10': 'mrandomtype', 'mrandomtype': 'Normal', 'required_mrandomtype': 1, 
        'attr_11': 'mrandom', 'mrandom': 0, 'required_mrandom': 1, 
        'attr_12': 'innerprecise', 'innerprecise': 1, 'required_innerprecise': 1, 
        'attr_13': 'enablegpu', 'enablegpu': 'True', 'required_enablegpu': 1, 
        'attr_14': 'prandom', 'prandom': 0, 'required_prandom': 1, 
        'attr_15': 'flaglist', 
        'flaglist': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 'required_flaglist': 1, 
        'shape_output': [[1, 70, 512]], 'test_type': 'training', 'action_type': 'bm', 'framework_type': 'aclnn', 'device_type': 'npu'
    }

    if input_data.kwargs["deqScale1Optional"] != None:
        query = input_data.kwargs["query"].to(dtype=torch.int8).numpy()
        key = input_data.kwargs["key"][0].to(dtype=torch.int8).numpy()
        value = input_data.kwargs["value"][0].to(dtype=torch.int8).numpy()
    else:
        query = input_data.kwargs["query"].to(dtype=torch.float32).numpy()
        key = input_data.kwargs["key"][0].to(dtype=torch.float32).numpy()
        value = input_data.kwargs["value"][0].to(dtype=torch.float32).numpy()

    pse = input_data.kwargs["pseShiftOptional"].to(dtype=torch.float32).numpy() if input_data.kwargs["pseShiftOptional"] != None else torch.Tensor(0).numpy()
    mask = input_data.kwargs["attenMaskOptional"].to(dtype=torch.bool).numpy() if input_data.kwargs["attenMaskOptional"] != None else torch.Tensor(0).numpy()
    dequantScale1 = input_data.kwargs["deqScale1Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["deqScale1Optional"] != None else torch.Tensor(0).numpy()
    quantScale1 = input_data.kwargs["quantScale1Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantScale1Optional"] != None else torch.Tensor(0).numpy()
    dequantScale2 = input_data.kwargs["deqScale2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["deqScale2Optional"] != None else torch.Tensor(0).numpy()
    quantScale2 = input_data.kwargs["quantScale2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantScale2Optional"] != None else torch.Tensor(0).numpy()
    quantOffset2 = input_data.kwargs["quantOffset2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantOffset2Optional"] != None else torch.Tensor(0).numpy()

    pfa_params["actualseqlengths"] = list(input_data.kwargs["actualSeqLengthsOptional"])
    pfa_params["actualseqlengthskv"] = list(input_data.kwargs["actualSeqLengthsKvOptional"])
    pfa_params["numheads"] = input_data.kwargs["numHeads"]
    pfa_params["scalevalue"] = input_data.kwargs["scaleValue"]
    pfa_params["pretokens"] = input_data.kwargs["preTokens"]
    pfa_params["nexttokens"] = input_data.kwargs["nextTokens"]
    pfa_params["inputlayout"] = input_data.kwargs["inputLayout"]
    pfa_params["numkeyvalueheads"] = input_data.kwargs["numKeyValueHeads"]
    pfa_params["sparsemode"] = input_data.kwargs["sparseMode"]
    pfa_params["innerprecise"] = input_data.kwargs["innerPrecise"]

    torch_tensor_list[0] = query
    pfa_params["shape_input"][0] = list(input_data.kwargs["query"].shape)
    pfa_params["shape_output"] = list(input_data.kwargs["query"].shape)
    pfa_params["dtype_input"][0] = dtype_map[input_data.kwargs["query"].dtype]

    torch_tensor_list[1] = key
    pfa_params["shape_input"][1] = list(input_data.kwargs["key"][0].shape)
    pfa_params["dtype_input"][1] = dtype_map[input_data.kwargs["key"][0].dtype]

    torch_tensor_list[2] = value
    pfa_params["shape_input"][2] = list(input_data.kwargs["value"][0].shape)
    pfa_params["dtype_input"][2] = dtype_map[input_data.kwargs["value"][0].dtype]

    if input_data.kwargs["pseShiftOptional"] != None:
        torch_tensor_list[3] = pse
        pfa_params["shape_input"][3] = list(input_data.kwargs["pseShiftOptional"].shape)
        pfa_params["dtype_input"][3] = dtype_map[input_data.kwargs["pseShiftOptional"].dtype]
        pfa_params["flaglist"][3] = 1

    if input_data.kwargs["attenMaskOptional"] != None:
        torch_tensor_list[4] = mask
        pfa_params["shape_input"][4] = list(input_data.kwargs["attenMaskOptional"].shape)
        pfa_params["dtype_input"][4] = dtype_map[input_data.kwargs["attenMaskOptional"].dtype]
        pfa_params["flaglist"][4] = 1
    
    if input_data.kwargs["deqScale1Optional"] != None:
        torch_tensor_list[5] = dequantScale1 
        pfa_params["shape_input"][5] = list(input_data.kwargs["deqScale1Optional"].shape)
        pfa_params["dtype_input"][5] = dtype_map[input_data.kwargs["deqScale1Optional"].dtype]
        pfa_params["flaglist"][7] = 1

    if input_data.kwargs["quantScale1Optional"] != None:
        torch_tensor_list[6] = quantScale1 
        pfa_params["shape_input"][6] = list(input_data.kwargs["quantScale1Optional"].shape)
        pfa_params["dtype_input"][6] = dtype_map[input_data.kwargs["quantScale1Optional"].dtype]
        pfa_params["flaglist"][8] = 1
    
    if input_data.kwargs["deqScale2Optional"] != None:
        torch_tensor_list[7] = dequantScale2 
        pfa_params["shape_input"][7] = list(input_data.kwargs["deqScale2Optional"].shape)
        pfa_params["dtype_input"][7] = dtype_map[input_data.kwargs["deqScale2Optional"].dtype]
        pfa_params["flaglist"][9] = 1

    if input_data.kwargs["quantScale2Optional"] != None:
        torch_tensor_list[8] = quantScale2 
        pfa_params["shape_input"][8] = list(input_data.kwargs["quantScale2Optional"].shape)
        pfa_params["dtype_input"][8] = dtype_map[input_data.kwargs["quantScale2Optional"].dtype]
        pfa_params["flaglist"][10] = 1
    
    if input_data.kwargs["quantOffset2Optional"] != None:
        torch_tensor_list[9] = quantOffset2 
        pfa_params["shape_input"][9] = list(input_data.kwargs["quantOffset2Optional"].shape)
        pfa_params["dtype_input"][9] = dtype_map[input_data.kwargs["quantOffset2Optional"].dtype]
        # pfa_params["flaglist"][11] = 1

    gloden_output = pfa_aclnn_op_func(torch_tensor_list, pfa_params, is_benchmark_task).to(dtype=input_data.kwargs["attentionOut"].dtype)

    return gloden_output

def aclnn_op_func_ifa_cpu(input_data : InputDataset, case_id, is_benchmark_task=False, output_path=""):

    torch_tensor_list =[
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32),
        np.array([], dtype=np.uint64), 
        np.array([], dtype=np.float32),
        np.array([], dtype=np.uint64), 
        np.array([], dtype=np.float32), 
        np.array([], dtype=np.float32), 
	    np.array([], dtype=np.float32), 
	    np.array([], dtype=np.float32),
        np.array([], dtype=np.float32), 
	    np.array([], dtype=np.float32)
    ]
    
    ifa_params = {
        'shape_input': [[1, 8, 1, 128], [1, 1, 256, 128], [1, 1, 256, 128], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]],
        'range_input': [[-1, 1], [-1, 1], [-1, 1], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], [-1, 1], [-1, 1]],
        'dtype_input': ['fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'uint64', 'fp32', 'uint64', 'fp16', 'fp16', 'fp16', 'fp16', 'int32', 'int64', 'fp16', 'fp16'],
        'format_input': ['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND'],
        'dtype_output': ['fp16'],
        'type_input': ['tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor'],
        'attr_1': 'actualseqlengths', 'actualseqlengths': [], 'required_actualseqlengths': 1,
        'attr_2': 'actualseqlengthskv', 'actualseqlengthskv': [], 'required_actualseqlengthskv': 1,
        'attr_3': 'numheads', 'numheads': 8, 'required_numheads': 1,
        'attr_4': 'scalevalue', 'scalevalue': 0.08838834764831843, 'required_scalevalue': 1,
        'attr_5': 'inputlayout', 'inputlayout': 'BNSD', 'required_inputlayout': 1,
        'attr_6': 'numkeyvalueheads', 'numkeyvalueheads': 1, 'required_numkeyvalueheads': 1,
        'attr_7': 'blocksize', 'blocksize': 0, 'required_blocksize': 1,
        'attr_8': 'innerprecise', 'innerprecise': 0, 'required_innerprecise': 1,
        'attr_9': 'flaglist', 'flaglist': [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'required_flaglist': 1, 'case_name': 'aclnnIncreFlashAttentionV4_IFA_default_CON_FP16_1_8_256_128_GQA',
        'shape_output': [[1, 8, 1, 128]], 'test_type': 'training', 'action_type': 'bm', 'framework_type': 'aclnn', 'device_type': 'cpu', 'bin_dir': None, 'precision_method': 0, 'op_name': 'aclnnIncreFlashAttentionV4', 'case_path': '/home/wangsong/aclnn_fuzz_1029/libs/../testcase/aclnn_case/aclnnIncreFlashAttentionV4'}
    
    query = input_data.kwargs["query"].to(dtype=torch.float32).numpy()
    if input_data.kwargs["antiquantScaleOptional"] != None:
        key = input_data.kwargs["key"][0].to(dtype=torch.int8).numpy()
        value = input_data.kwargs["value"][0].to(dtype=torch.int8).numpy()
    else:
        key = input_data.kwargs["key"][0].to(dtype=torch.float32).numpy()
        value = input_data.kwargs["value"][0].to(dtype=torch.float32).numpy()

    pse = input_data.kwargs["pseShiftOptional"].to(dtype=torch.float32).numpy() if input_data.kwargs["pseShiftOptional"] != None else torch.Tensor(0).numpy()
    mask = input_data.kwargs["attenMaskOptional"].to(dtype=torch.bool).numpy() if input_data.kwargs["attenMaskOptional"] != None else torch.Tensor(0).numpy()
    quantScale2 = input_data.kwargs["quantScale2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantScale2Optional"] != None else torch.Tensor(0).numpy()
    quantOffset2 = input_data.kwargs["quantOffset2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantOffset2Optional"] != None else torch.Tensor(0).numpy()
    antiquantScale = input_data.kwargs["antiquantScaleOptional"].to(dtype=torch.float32).numpy() if input_data.kwargs["antiquantScaleOptional"] != None else torch.Tensor(0).numpy()
    antiquantOffset = input_data.kwargs["antiquantOffsetOptional"].to(dtype=torch.float32).numpy() if input_data.kwargs["antiquantOffsetOptional"] != None else torch.Tensor(0).numpy()
    kvPaddingSize = input_data.kwargs["kvPaddingSizeOptional"].to(dtype=torch.int32).numpy() if input_data.kwargs["kvPaddingSizeOptional"] != None else torch.Tensor(0).numpy()
    blocktable = input_data.kwargs["blockTableOptional"].to(dtype=torch.int32).numpy() if input_data.kwargs["blockTableOptional"] != None else torch.Tensor(0).numpy()
    
    ifa_params["actualseqlengths"] = list(input_data.kwargs["actualSeqLengthsOptional"])
    ifa_params["actualseqlengthskv"] = list(input_data.kwargs["actualSeqLengthsOptional"])
    ifa_params["numheads"] = input_data.kwargs["numHeads"]
    ifa_params["scalevalue"] = input_data.kwargs["scaleValue"]
    ifa_params["inputlayout"] = input_data.kwargs["inputLayout"]
    ifa_params["numkeyvalueheads"] = input_data.kwargs["numKeyValueHeads"]
    ifa_params["blocksize"] = input_data.kwargs["blockSize"]
    ifa_params["innerprecise"] = input_data.kwargs["innerPrecise"]

    torch_tensor_list[0] = query
    ifa_params["shape_input"][0] = list(input_data.kwargs["query"].shape)
    ifa_params["shape_output"] = list(input_data.kwargs["query"].shape)
    ifa_params["dtype_input"][0] = dtype_map[input_data.kwargs["query"].dtype]

    torch_tensor_list[1] = key
    ifa_params["shape_input"][1] = list(input_data.kwargs["key"][0].shape)
    ifa_params["dtype_input"][1] = dtype_map[input_data.kwargs["key"][0].dtype]

    torch_tensor_list[2] = value
    ifa_params["shape_input"][2] = list(input_data.kwargs["value"][0].shape)
    ifa_params["dtype_input"][2] = dtype_map[input_data.kwargs["value"][0].dtype]

    if input_data.kwargs["pseShiftOptional"] != None:
        torch_tensor_list[3] = pse
        ifa_params["shape_input"][3] = list(input_data.kwargs["pseShiftOptional"].shape)
        ifa_params["dtype_input"][3] = dtype_map[input_data.kwargs["pseShiftOptional"].dtype]
        ifa_params["flaglist"][3] = 1
    
    if input_data.kwargs["attenMaskOptional"] != None:
        torch_tensor_list[4] = mask
        ifa_params["shape_input"][4] = list(input_data.kwargs["attenMaskOptional"].shape)
        ifa_params["dtype_input"][4] = dtype_map[input_data.kwargs["attenMaskOptional"].dtype]
        ifa_params["flaglist"][4] = 1
    
    if input_data.kwargs["quantScale2Optional"] != None:
        torch_tensor_list[8] = quantScale2 
        ifa_params["shape_input"][8] = list(input_data.kwargs["quantScale2Optional"].shape)
        ifa_params["dtype_input"][8] = dtype_map[input_data.kwargs["quantScale2Optional"].dtype]
        ifa_params["flaglist"][9] = 1
    
    if input_data.kwargs["quantOffset2Optional"] != None:
        torch_tensor_list[9] = quantOffset2 
        ifa_params["shape_input"][9] = list(input_data.kwargs["quantOffset2Optional"].shape)
        ifa_params["dtype_input"][9] = dtype_map[input_data.kwargs["quantOffset2Optional"].dtype]
        ifa_params["flaglist"][10] = 1
    
    if input_data.kwargs["antiquantScaleOptional"] != None:
        torch_tensor_list[10] = antiquantScale 
        ifa_params["shape_input"][10] = list(input_data.kwargs["antiquantScaleOptional"].shape)
        ifa_params["dtype_input"][10] = dtype_map[input_data.kwargs["antiquantScaleOptional"].dtype]
        ifa_params["flaglist"][11] = 1
    
    if input_data.kwargs["antiquantOffsetOptional"] != None:
        torch_tensor_list[11] = antiquantOffset 
        ifa_params["shape_input"][11] = list(input_data.kwargs["antiquantOffsetOptional"].shape)
        ifa_params["dtype_input"][11] = dtype_map[input_data.kwargs["antiquantOffsetOptional"].dtype]
        ifa_params["flaglist"][12] = 1
    
    if len(ifa_params["shape_input"][0]) == 3:
        H = ifa_params["shape_input"][0][2]
    else:
        H = ifa_params["numheads"] * ifa_params["shape_input"][0][3]

    if input_data.kwargs["blockTableOptional"] != None:
        torch_tensor_list[12] = blocktable 
        ifa_params["shape_input"][12] = list(input_data.kwargs["blockTableOptional"].shape)
        ifa_params["dtype_input"][12] = dtype_map[input_data.kwargs["blockTableOptional"].dtype]
        ifa_params["flaglist"][13] = 1
        ifa_params["shape_input"][14] = [ifa_params["shape_input"][12][1], ifa_params["blocksize"], H] 
        ifa_params["shape_input"][15] = [ifa_params["shape_input"][12][1], ifa_params["blocksize"], H] 

    if input_data.kwargs["kvPaddingSizeOptional"] != None:
        torch_tensor_list[13] = kvPaddingSize 
        ifa_params["shape_input"][13] = list(input_data.kwargs["kvPaddingSizeOptional"].shape)
        ifa_params["dtype_input"][13] = dtype_map[input_data.kwargs["kvPaddingSizeOptional"].dtype]
        ifa_params["range_input"][13] = list(input_data.kwargs["kvPaddingSizeOptional"])
        ifa_params["flaglist"][14] = 1

    gloden_output = ifa_aclnn_op_func(torch_tensor_list, ifa_params, case_id, is_benchmark_task, "").to(dtype=input_data.kwargs["attentionOut"].dtype)
    
    return gloden_output

def load_kv_cache(input_data: InputDataset, case_id):
    key_shape = list(input_data.kwargs["key"][0].shape)
    if len(key_shape) == 3:
        H = key_shape[2]
    else:
        H = input_data.kwargs["numHeads"] * key_shape[3]

    blocktable_shape = list(input_data.kwargs["blockTableOptional"].shape)
    cache_shape = [blocktable_shape[1], input_data.kwargs["blockSize"], H]
    k_cache = np.load(f"./block_table/k_cache_{case_id}.npy")
    v_cache = np.load(f"./block_table/v_cache_{case_id}.npy")
    k_cache_tensor = torch.tensor(k_cache, dtype=torch.float32).to(dtype=input_data.kwargs["key"][0].dtype).reshape(cache_shape).npu()
    v_cache_tensor = torch.tensor(v_cache, dtype=torch.float32).to(dtype=input_data.kwargs["value"][0].dtype).reshape(cache_shape).npu()
    input_data.kwargs["key"][0] = k_cache_tensor
    input_data.kwargs["value"][0] = v_cache_tensor

    block_table = np.load(f"./block_table/block_table_{case_id}.npy")
    block_table_tensor = torch.tensor(block_table, dtype=torch.int32)
    input_data.kwargs["blockTableOptional"] = torch.tensor(block_table_tensor, dtype=torch.int32).reshape(blocktable_shape).npu()
    return

@register("executor_fused_infer_attention_score_v3")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = aclnn_op_func_fia_cpu(input_data, self.task_result.case_config.id, self.task_result.is_benchmark_task, "")
        return output

@register("executor_aclnn_fused_infer_attention_score_v3")
class aclnnFusedInferAttentionScoreApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnFusedInferAttentionScoreApi, self).__init__(task_result, backend)
    
    def init_by_input_data(self, input_data: InputDataset):
        input_args = []  # 算子的入参列表
        if input_data.kwargs["blockTableOptional"] != None: 
            load_kv_cache(input_data, self.task_result.case_config.id)
        input_args, output_packages = super().init_by_input_data(input_data)
        output_packages = []  # 算子的出参数据包列表
        input_args.pop()
        output_packages.append(input_args[-2])
        return input_args, output_packages

   
    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack)
            output.append(temp_output_pack)
        return output