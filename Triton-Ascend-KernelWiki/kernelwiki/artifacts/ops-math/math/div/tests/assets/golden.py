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

import numpy

__golden__ = {
    "kernel": {
        "div": "div_golden"
    }
}


def broadcast_to_maxshape(shapes: list):
    def _max(_shape):
        no_one_shape = [s for s in _shape if s != 1]
        if len(no_one_shape) == 0:
            max_value = 1
        else:
            max_value = no_one_shape[0]
        return max_value
    max_dim_length = max(len(list(shape)) for shape in shapes)
    input_shapes = []
    for shape in shapes:
        input_shapes.append([1 for _ in range(max_dim_length - len(shape))] + list(shape))
    input_shapes = list(map(list, zip(*input_shapes)))
    max_shape = [_max(shape) for shape in input_shapes]
    input_shapes = list(map(list, zip(*input_shapes)))
    return (*input_shapes, max_shape)


def div_golden(x1, x2, **kwargs):
    '''
    Golden function for div.
    All the parameters (names and order) follow @div_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    ori_dtype = x1.dtype
    input_dtypes = kwargs.get('input_dtypes', [ori_dtype.name, ori_dtype.name])
    
    if "float16" in str(ori_dtype): #include bfloat16 & float16
        x1, x2 = x1.astype("float32"), x2.astype("float32")

    if "complex32" in input_dtypes:
        import torch

        def complex32_div(reala, imaga, realb, imagb):
            abs_b1 = numpy.abs(realb)
            abs_b2 = numpy.abs(imagb)
            if abs_b1 >= abs_b2:
                if abs_b1 == 0 and abs_b2 == 0:
                    real_ = reala / abs_b1
                    imag_ = imaga / abs_b2
                else:
                    temp1 = imagb / realb
                    temp2 = realb + imagb * temp1
                    tensor_one = numpy.array(1.0, dtype=numpy.float16)
                    cm = tensor_one / temp2
                    real_ = (reala + imaga * temp1) * cm
                    imag_ = (imaga - reala * temp1) * cm
            else:
                temp1 = realb / imagb
                temp2 = imagb + realb * temp1
                tensor_one = numpy.array(1.0, dtype=numpy.float16)
                cm = tensor_one / temp2
                real_ = (imaga + reala * temp1) * cm
                imag_ = (imaga * temp1 - reala) * cm

            return real_, imag_

        x1, x2 = numpy.broadcast_arrays(x1, x2)
        xreal, ximag = numpy.split(x1, 2, axis=-1)
        yreal, yimag = numpy.split(x2, 2, axis=-1)
        ori_shape = xreal.shape
        xreal = xreal.reshape(-1)
        ximag = ximag.reshape(-1)
        yreal = yreal.reshape(-1)
        yimag = yimag.reshape(-1)
        input_xr = torch.from_numpy(xreal)
        input_xi = torch.from_numpy(ximag)
        input_yr = torch.from_numpy(yreal)
        input_yi = torch.from_numpy(yimag)
        zreal = torch.zeros_like(input_xr)
        zimag = torch.zeros_like(input_xr)
        for i in range(len(input_xr)):
            zreal[i], zimag[i] = complex32_div(input_xr[i],input_xi[i], input_yr[i], input_yi[i])

        zreal = zreal.numpy()
        zimag = zimag.numpy()
        zreal = zreal.reshape(ori_shape)
        zimag = zimag.reshape(ori_shape)
        res = numpy.concatenate((zreal, zimag), axis=-1)
        return res
    else:
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        _, _, shape_max = broadcast_to_maxshape([x1.shape, x2.shape])
        x1 = numpy.broadcast_to(x1, shape_max)
        x2 = numpy.broadcast_to(x2, shape_max)
        x = tf.compat.v1.placeholder(shape=x1.shape, dtype=x1.dtype)
        y = tf.compat.v1.placeholder(shape=x2.shape, dtype=x2.dtype)
        out = tf.compat.v1.div(x, y)
        feed_dict = {x: x1, y: x2}
        init_op = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            res = sess.run(out, feed_dict=feed_dict)
        return res.astype(ori_dtype, copy=False)
