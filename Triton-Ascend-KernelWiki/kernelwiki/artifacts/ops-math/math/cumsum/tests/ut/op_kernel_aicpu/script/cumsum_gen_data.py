# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes
import random
# prama1: file_name: the file which store the data
# param2: data: data which will be stored
# param3: fmt: format
def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')
# prama1: file_name: the file which store the data
# param2: dtype: data type
# param3: delim: delimiter which is used to split data
def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()
# prama1: file_name: the file which store the data
# param2: delim: delimiter which is used to split data
def read_file_txt_to_bool(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)
# prama1: data_file: the file which store the generation data
# param2: shape: data shape
# param3: dtype: data type
# param4: rand_type: the method of generate data, select from "randint, uniform"
# param5: data lower limit
# param6: data upper limit
def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    elif rand_type == "uniform":
        rand_data = np.random.uniform(low, high, size=shape)
    elif rand_type == "complex":
        r1 = np.random.uniform(low, high, size=shape)
        r2 = np.random.uniform(low, high, size=shape)
        rand_data = np.empty((shape[0], shape[1], shape[2]), dtype=dtype)
        for i in range(0, shape[0]):
            for p in range(0, shape[1]):
                for k in range(0, shape[2]):
                    rand_data[i, p, k] = complex(r1[i, p, k], r2[i, p, k])
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data

def gen_data_file2(data_file, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = random.randint(low, high)
    else:
        rand_data = random.uniform(low, high)
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_int8_1():
    data_files=["cumsum/data/cumsum_data_input_int8.txt",
                "cumsum/data/cumsum_data_axis_int32_int8.txt",
                "cumsum/data/cumsum_data_output_int8_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int8,"randint",-10,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int8,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int16_1():
    data_files=["cumsum/data/cumsum_data_input_int16.txt",
                "cumsum/data/cumsum_data_axis_int32_int16.txt",
                "cumsum/data/cumsum_data_output_int16_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int16,"randint",-10,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int16,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int32_1():
    data_files=["cumsum/data/cumsum_data_input_int32.txt",
                "cumsum/data/cumsum_data_axis_int32_int32.txt",
                "cumsum/data/cumsum_data_output_int32_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int32,"randint",-100,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int32,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int64_1():
    data_files=["cumsum/data/cumsum_data_input_int64.txt",
                "cumsum/data/cumsum_data_axis_int32_int64.txt",
                "cumsum/data/cumsum_data_output_int64_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int64,"randint",-100,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_uint8_1():
    data_files=["cumsum/data/cumsum_data_input_uint8.txt",
                "cumsum/data/cumsum_data_axis_int32_uint8.txt",
                "cumsum/data/cumsum_data_output_uint8_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.uint8,"randint",-10,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.uint8,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_uint16_1():
    data_files=["cumsum/data/cumsum_data_input_uint16.txt",
                "cumsum/data/cumsum_data_axis_int32_uint16.txt",
                "cumsum/data/cumsum_data_output_uint16_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.uint16,"randint",-10,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.uint16,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_float16_1():
    data_files=["cumsum/data/cumsum_data_input_float16.txt",
                "cumsum/data/cumsum_data_axis_int32_float16.txt",
                "cumsum/data/cumsum_data_output_float16_EF_RF.txt",
                "cumsum/data/cumsum_data_output_float16_ET_RF.txt",
                "cumsum/data/cumsum_data_output_float16_EF_RT.txt",
                "cumsum/data/cumsum_data_output_float16_ET_RT.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]  
    a = gen_data_file(data_files[0],shape_input_data,np.float16,"uniform",-100,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.float16,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)  
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")  
    re = tf.cumsum(input_data, axis_data, True, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[3],data,fmt="%s")
    re = tf.cumsum(input_data, axis_data, False, True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[4],data,fmt="%s")
    re = tf.cumsum(input_data, axis_data, True, True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[5], data, fmt="%s")

def gen_random_data_float32_1():
    data_files=["cumsum/data/cumsum_data_input_float32.txt",
                "cumsum/data/cumsum_data_axis_int32_float32.txt",
                "cumsum/data/cumsum_data_output_float32_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.float32,"uniform",-100,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.float32,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_float64_1():
    data_files=["cumsum/data/cumsum_data_input_float64.txt",
                "cumsum/data/cumsum_data_axis_int32_float64.txt",
                "cumsum/data/cumsum_data_output_float64_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.float64,"uniform",-100,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.float64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_complex64_1():
    data_files=["cumsum/data/cumsum_data_input_complex64.txt",
                "cumsum/data/cumsum_data_axis_int32_complex64.txt",
                "cumsum/data/cumsum_data_output_complex64_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.complex64,"complex",0,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.complex64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_complex128_1():
    data_files=["cumsum/data/cumsum_data_input_complex128.txt",
                "cumsum/data/cumsum_data_axis_int32_complex128.txt",
                "cumsum/data/cumsum_data_output_complex128_EF_RF.txt",
                "cumsum/data/cumsum_data_output_complex128_ET_RT.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.complex128,"complex",0,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.complex128,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")
    re = tf.cumsum(input_data, axis_data, True, True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[3],data,fmt="%s")

def gen_random_data_double_big_1():
    data_files=["cumsum/data/cumsum_data_input_double.txt",
                "cumsum/data/cumsum_data_axis_int32_double.txt",
                "cumsum/data/cumsum_data_output_double_EF_RF.txt"]
    np.random.seed(3457)
    shape_input_data = [17,4,1024]
    a = gen_data_file(data_files[0],shape_input_data,np.float64,"uniform",-100,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.float64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.cumsum(input_data, axis_data, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def run():
    gen_random_data_int8_1()
    gen_random_data_int16_1()
    gen_random_data_int32_1()
    gen_random_data_int64_1()
    gen_random_data_uint8_1()
    gen_random_data_uint16_1()
    gen_random_data_float16_1()
    gen_random_data_float32_1()
    gen_random_data_float64_1()
    gen_random_data_complex64_1()
    gen_random_data_complex128_1()
    gen_random_data_double_big_1()

if __name__ == '__main__':
    os.system("mkdir -p cumsum/data")
    run()