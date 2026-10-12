#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You file may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy
from functools import reduce
from copy import deepcopy


__golden__ = {
    "kernel": {
        "trans_data": "trans_data_golden"
    }
}


def ceil_div(a, b):
    return (a + b - 1) // b


def is_nchw_like(format_: str) -> bool:
    if len(set(format_)) != 4:
        return False
    return all([c in format_ for c in "NCHW"])


def is_ndhwc_like(format_: str) -> bool:
    if len(set(format_)) != 5:
        return False
    return all([c in format_ for c in "NDCHW"])


def is_5hd(format_: str) -> bool:
    if any([c in format_ for c in "PQ"]):
        return False
    format_ = format_.replace("C1", "P")
    format_ = format_.replace("C0", "Q")
    if len(set(format_)) != 5:
        return False
    return all([c in format_ for c in "NPHWQ"])


def is_6hd(format_: str) -> bool:
    if any([c in format_ for c in "PQ"]):
        return False
    format_ = format_.replace("C1", "P")
    format_ = format_.replace("C0", "Q")
    if len(set(format_)) != 6:
        return False
    return all([c in format_ for c in "NDPHWQ"])


class FormatTrans:
    def __init__(self, src_data, c0, n0, out_shape):
        self.src_data = src_data
        self.c0 = c0
        self.n0 = n0
        self.out_shape = out_shape

    @staticmethod
    def _pad_len(x, y):
        return (y - x % y) % y

    def trans_fractalz_to_nd(self):
        output_y_shape = self.out_shape
        input_x_shape = numpy.shape(self.src_data)
        if len(output_y_shape) == 1:
            axis_h, axis_c, axis_n = 1, 1, output_y_shape[0]
        elif len(output_y_shape) == 2:
            axis_h, axis_c, axis_n = 1, output_y_shape[0], output_y_shape[1]
        else:
            axis_h, axis_c, axis_n = reduce(lambda x, y: x * y, output_y_shape[:-2]), output_y_shape[-2], output_y_shape[-1]
        axis_c0 = input_x_shape[-1]
        axis_ni = input_x_shape[-2]
        axis_no = input_x_shape[-3]
        axis_c1 = input_x_shape[-4] // axis_h
        c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
        n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
        tmp_input_tensor = self.src_data.reshape(axis_h, axis_c1, axis_no, axis_ni, axis_c0)
        tmp_input_tensor = numpy.transpose(tmp_input_tensor, axes=(0, 1, 4, 2, 3))
        tmp_input_tensor = tmp_input_tensor.reshape([axis_h, axis_c1 * axis_c0, axis_no * axis_ni])
        data_y = tmp_input_tensor[:, :c_pad, :n_pad]
        return data_y

    def trans_fractalnz_to_nd(self):
        output_y_shape = self.out_shape
        input_x_data = self.src_data
        input_x_shape = numpy.shape(input_x_data)
        if len(output_y_shape) == 1:
            axis_h, axis_n, axis_c = 1, 1, output_y_shape[0]
        elif len(output_y_shape) == 2:
            axis_h, axis_n, axis_c = 1, output_y_shape[0], output_y_shape[1]
        else:
            axis_h, axis_n, axis_c = reduce(lambda x, y: x * y, output_y_shape[:-2]), output_y_shape[-2], output_y_shape[-1]
        axis_c0 = input_x_shape[-1]
        axis_ni = input_x_shape[-2]
        axis_no = input_x_shape[-3]
        axis_c1 = input_x_shape[-4]
        c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
        n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
        tmp_input_tensor = input_x_data.reshape(axis_h, axis_c1, axis_no, axis_ni, axis_c0)
        tmp_input_tensor = numpy.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
        tmp_input_tensor = tmp_input_tensor.reshape([axis_h, axis_no * axis_ni, axis_c1 * axis_c0])
        data_y = tmp_input_tensor[:, :n_pad, :c_pad]
        return data_y

    def trans_fractalnz_to_nc1hwc0(self):
        output_y_shape = self.out_shape
        input_x_data = self.src_data
        input_x_shape = numpy.shape(input_x_data)
        axis_n, axis_c1, axis_h, axis_w, axis_c0 = output_y_shape
        axis_ni = input_x_shape[-2]
        axis_no = input_x_shape[-3]
        n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
        tmp_input_tensor = input_x_data.reshape(axis_c1, axis_h, axis_w, axis_no, axis_ni, axis_c0)
        tmp_input_tensor = numpy.transpose(tmp_input_tensor, axes=(3, 4, 0, 1, 2, 5))
        tmp_input_tensor = tmp_input_tensor.reshape([axis_no * axis_ni, axis_c1, axis_h, axis_w, axis_c0])
        data_y = tmp_input_tensor[:n_pad, :, :, :, :]
        return data_y

    def trans_nd_to_fractalnz(self):
        in_shape = numpy.shape(self.src_data)
        if len(in_shape) == 1:
            axis_h, axis_n, axis_c = 1, 1, in_shape[0]
        elif len(in_shape) == 2:
            axis_h, axis_n, axis_c = 1, in_shape[0], in_shape[1]
        else:
            axis_h, axis_n, axis_c = reduce(lambda x, y: x * y, in_shape[:-2]), in_shape[-2], in_shape[-1]
        axis_ni = self.n0
        axis_c0 = self.c0
        axis_no = ceil_div(axis_n, axis_ni)
        axis_c1 = ceil_div(axis_c, axis_c0)
        c_pad = self._pad_len(axis_c, axis_c0)
        n_pad = self._pad_len(axis_n, axis_ni)

        tmp_input_tensor = numpy.pad(numpy.reshape(self.src_data, (axis_h, axis_n, axis_c)), ((0, 0), (0, n_pad), (0, c_pad)),
                                     mode="constant", constant_values=(0, 0))
        tmp_input_tensor = tmp_input_tensor.reshape([axis_h, axis_no, axis_ni, axis_c1, axis_c0])
        outputArr = numpy.transpose(tmp_input_tensor, axes=(0, 3, 1, 2, 4))
        return outputArr

    def trans_nd_to_fractalzn(self):
        in_shape = numpy.shape(self.src_data)
        input_data = self.src_data
        if len(in_shape) == 1:
            axis_h, axis_c, axis_n = 1, 1, in_shape[0]
        elif len(in_shape) == 2:
            axis_h, axis_c, axis_n = 1, in_shape[0], in_shape[1]
        else:
            axis_h, axis_c, axis_n = reduce(lambda x, y: x * y, in_shape[:-2]), in_shape[-2], in_shape[-1]
        axis_c0 = self.c0
        axis_ni = self.n0
        c_pad = self._pad_len(axis_c, axis_c0)
        n_pad = self._pad_len(axis_n, axis_ni)
        axis_no = ceil_div(axis_n, axis_ni)
        axis_c1 = ceil_div(axis_c, axis_c0)
        tmp_input_tensor = numpy.pad(input_data.reshape(axis_h, axis_c, axis_n), ((0, 0), (0, c_pad), (0, n_pad)),
                                     mode="constant", constant_values=(0, 0))
        tmp_input_tensor = tmp_input_tensor.reshape([axis_h, axis_c1, axis_c0, axis_no, axis_ni])
        outputArr = numpy.transpose(tmp_input_tensor, axes=(0, 1, 3, 4, 2))
        return outputArr

    def trans_nc1hwc0_to_fractalz(self):
        input_x_shape = numpy.shape(self.src_data)
        axis_n = input_x_shape[0]
        axis_c1 = input_x_shape[1]
        axis_h = input_x_shape[2]
        axis_w = input_x_shape[3]
        axis_c0 = input_x_shape[4]
        axis_ni = self.n0
        axis_no = ceil_div(axis_n, axis_ni)
        n_pad = self._pad_len(axis_n, axis_ni)

        tmp_input_tensor = numpy.pad(self.src_data, ((0, n_pad), (0, 0), (0, 0), (0, 0), (0, 0)),
                                     mode="constant", constant_values=(0, 0))
        tmp_input_tensor = tmp_input_tensor.reshape([axis_no, axis_ni, axis_c1, axis_h, axis_w, axis_c0])
        data_y = numpy.transpose(tmp_input_tensor, axes=(2, 3, 4, 0, 1, 5))
        return data_y

    def trans_fractalz_xd_to_xnchw(self, src_format: str, dst_format: str):
        out_shape = self.out_shape
        in_shape = numpy.shape(self.src_data)
        dst_dim_d = out_shape[dst_format.index('D')] if 'D' in dst_format else 1
        dst_dim_n = out_shape[dst_format.index('N')]
        dst_dim_c = out_shape[dst_format.index('C')]
        dst_dim_h = out_shape[dst_format.index('H')]
        dst_dim_w = out_shape[dst_format.index('W')]

        src_dim_dc1hw = in_shape[0]
        src_dim_n1 = in_shape[1]
        src_dim_n0 = in_shape[2]
        src_dim_c0 = in_shape[3]
        src_dim_c1 = src_dim_dc1hw // (dst_dim_d * dst_dim_h * dst_dim_w)

        c_pad = None if src_dim_c1 * src_dim_c0 == dst_dim_c else dst_dim_c - src_dim_c1 * src_dim_c0
        n_pad = None if src_dim_n1 * src_dim_n0 == dst_dim_n else dst_dim_n - src_dim_n1 * src_dim_n0
        split_shape = [src_dim_c1, dst_dim_h, dst_dim_w, src_dim_n1, src_dim_n0, src_dim_c0]
        if 'D' in dst_format:
            split_shape.insert(0, dst_dim_d)
        tmp_input_tensor = self.src_data.reshape(split_shape)

        transpose_axes = []
        dst_format_split = dst_format.replace('N', 'RS').replace('C', 'PQ')
        src_format_split = 'PHWRSQ' if 'D' not in dst_format else 'DPHWRSQ'
        for c in dst_format_split:
            transpose_axes.append(src_format_split.index(c))
        tmp_input_tensor = numpy.transpose(tmp_input_tensor, axes=transpose_axes)

        out_shape_pad = list(out_shape)
        out_shape_pad[dst_format.index('N')] = src_dim_n1 * src_dim_n0
        out_shape_pad[dst_format.index('C')] = src_dim_c1 * src_dim_c0
        tmp_input_tensor = tmp_input_tensor.reshape(out_shape_pad)

        eval_str = [":"] * len(dst_format)
        eval_str[dst_format.index('N')] = ':n_pad'
        eval_str[dst_format.index('C')] = ':c_pad'
        eval_str = 'tmp_input_tensor[' + ','.join(eval_str) + ']'
        return eval(eval_str)

    def trans_xhd_to_xnchw(self, src_format: str, dst_format: str):
        in_shape = numpy.shape(self.src_data)
        src_format = src_format.replace('C1', 'P').replace('C0', 'Q')
        dst_dim_c = self.out_shape[dst_format.index('C')]
        src_dim_c1 = in_shape[src_format.index('P')]
        src_dim_c0 = in_shape[src_format.index('Q')]
        c_pad = None if src_dim_c1 * src_dim_c0 == dst_dim_c else dst_dim_c - src_dim_c1 * src_dim_c0

        transpose_axes = []
        dst_format_tmp = dst_format.replace('C', 'PQ')
        for c in dst_format_tmp:
            transpose_axes.append(src_format.index(c))
        tmp_input_tensor = numpy.transpose(self.src_data, axes=transpose_axes)

        out_shape_pad = list(self.out_shape)
        out_shape_pad[dst_format.index('C')] = -1
        tmp_input_tensor = tmp_input_tensor.reshape(out_shape_pad)
        eval_str = [":"] * len(dst_format)
        eval_str[dst_format.index('C')] = ':c_pad'
        eval_str = 'tmp_input_tensor[' + ','.join(eval_str) + ']'
        return eval(eval_str)

    def trans_xnchw_to_xhd(self, src_format: str, dst_format: str):
        in_shape = numpy.shape(self.src_data)
        src_c_idx = src_format.index('C')
        src_dim_c = in_shape[src_c_idx]

        dst_dim_c1 = ceil_div(src_dim_c, self.c0)
        c_pad = self._pad_len(src_dim_c, self.c0)

        pad_shape = [[0, 0]] * len(src_format)
        pad_shape[src_c_idx] = [0, c_pad]
        tmp_input_tensor = numpy.pad(self.src_data, pad_shape, mode="constant", constant_values=(0, 0))

        split_c_shape = list(in_shape)
        split_c_shape[src_c_idx] = self.c0
        split_c_shape.insert(src_c_idx, dst_dim_c1)
        tmp_input_tensor = tmp_input_tensor.reshape(split_c_shape)

        transpose_axes, split_c_format = [], src_format.replace('C', 'PQ')
        dst_format = dst_format.replace('C1', 'P').replace('C0', 'Q')
        for c in dst_format:
            transpose_axes.append(split_c_format.index(c))
        return numpy.transpose(tmp_input_tensor, axes=transpose_axes)

    def trans_xnchw_to_fractalz_xd(self, src_format: str, dst_format: str):
        in_shape = numpy.shape(self.src_data)
        src_c_idx = src_format.index('C')
        src_dim_c = in_shape[src_c_idx]
        dst_dim_c1 = ceil_div(src_dim_c, self.c0)
        c_pad = self._pad_len(src_dim_c, self.c0)

        src_n_idx = src_format.index('N')
        src_dim_n = in_shape[src_n_idx]
        dst_dim_n1 = ceil_div(src_dim_n, self.n0)
        n_pad = self._pad_len(src_dim_n, self.n0)

        pad_shape = [[0, 0]] * len(src_format)
        pad_shape[src_c_idx] = [0, c_pad]
        pad_shape[src_n_idx] = [0, n_pad]
        tmp_tensor = numpy.pad(self.src_data, pad_shape, mode="constant", constant_values=(0, 0))

        split_shape = list(in_shape)
        split_shape[src_c_idx] = self.c0
        split_shape[src_n_idx] = self.n0
        split_shape.insert(src_c_idx, dst_dim_c1)
        split_shape.insert(src_n_idx if src_n_idx < src_c_idx else src_n_idx + 1, dst_dim_n1)
        tmp_tensor = tmp_tensor.reshape(split_shape)

        transpose_axes, split_format = [], src_format.replace('C', 'PQ').replace('N', 'RS')
        dst_format_tmp = 'PHWRSQ' if 'D' not in src_format else 'DPHWRSQ'
        for c in dst_format_tmp:
            transpose_axes.append(split_format.index(c))
        tmp_tensor = numpy.transpose(tmp_tensor, axes=transpose_axes)
        if dst_format not in ('FRACTAL_ZN',):
            tmp_tensor = tmp_tensor.reshape([-1, dst_dim_n1, self.n0, self.c0])
        return tmp_tensor

    def trans_within_nchw(self, src_format: str, dst_format: str):
        transpose_axes = []
        for c in dst_format:
            transpose_axes.append(src_format.index(c))
        return numpy.transpose(self.src_data, axes=transpose_axes)


def _get_format_trans_func(params, src_data):
    src_format, dst_format = deepcopy(params["src_format"]), deepcopy(params["dst_format"])
    out_shape = params["shape_output"]
    c0 = out_shape[-1]
    n0 = 16 if params["dtype_input"] in ["int8", "uint8", "bool"] else 16
    if src_format in ["NC1HWC0", "NDC1HWC0", "FRACTAL_Z", "FRACTAL_NZ", "FRACTAL_Z_3D"]:
        out_shape = params["shape_output"]
        trans_fun = FormatTrans(src_data, c0, n0, out_shape)
    else:
        trans_fun = FormatTrans(src_data, c0, n0, None)

    if is_nchw_like(src_format) and is_nchw_like(dst_format):
        return trans_fun.trans_within_nchw(src_format, dst_format)
    elif (is_nchw_like(src_format) and is_5hd(dst_format)) \
            or (is_ndhwc_like(src_format) and is_6hd(dst_format)):
        return trans_fun.trans_xnchw_to_xhd(src_format, dst_format)
    elif (is_nchw_like(src_format) and dst_format in ("FRACTAL_Z", "FRACTAL_ZN")) \
            or (is_ndhwc_like(src_format) and dst_format == "FRACTAL_Z_3D"):
        return trans_fun.trans_xnchw_to_fractalz_xd(src_format, dst_format)
    elif src_format == "NC1HWC0" and dst_format == "FRACTAL_Z":
        return trans_fun.trans_nc1hwc0_to_fractalz()
    elif src_format == "ND" and dst_format in ["FRACTAL_Z", "FRACTAL_ZN"]:
        return trans_fun.trans_nd_to_fractalzn()
    elif src_format == "ND" and dst_format == "FRACTAL_NZ":
        return trans_fun.trans_nd_to_fractalnz()
    elif (is_5hd(src_format) and is_nchw_like(dst_format)) or (is_6hd(src_format) and is_ndhwc_like(dst_format)):
        return trans_fun.trans_xhd_to_xnchw(src_format, dst_format)
    elif (src_format == "FRACTAL_Z_3D" and is_ndhwc_like(dst_format)) \
            or (src_format == "FRACTAL_Z" and is_nchw_like(dst_format)):
        return trans_fun.trans_fractalz_xd_to_xnchw(src_format, dst_format)
    elif src_format == "FRACTAL_NZ" and dst_format == "ND":
        return trans_fun.trans_fractalnz_to_nd()
    elif src_format == "FRACTAL_NZ" and dst_format == "NC1HWC0":
        return trans_fun.trans_fractalnz_to_nc1hwc0()
    elif src_format == "FRACTAL_Z" and dst_format == "ND":
        return trans_fun.trans_fractalz_to_nd()
    else:
        raise RuntimeError(f"TransData from {src_format} to {dst_format} is not supported yet !!!")


def trans_data_golden(src, src_format, dst_format, src_subformat=0, dst_subformat=0, groups=1, **kwargs):
    '''
    Kernel golden for trans_data.
    All the parameters follow @trans_data_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    dtype_input = src.dtype.name
    shape_input = src.shape
    shape_output = kwargs.get("output_ori_shapes", [[]])[0]
    
    params = {
        "src_format": src_format,
        "dst_format": dst_format,
        "dtype_input": dtype_input,
        "shape_input": shape_input,
        "shape_output": shape_output
    }
    return _get_format_trans_func(params, src)