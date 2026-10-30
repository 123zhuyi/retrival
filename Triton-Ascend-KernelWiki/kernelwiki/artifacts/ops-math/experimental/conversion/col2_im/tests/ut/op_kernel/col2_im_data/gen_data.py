#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
import re
import tensorflow as tf


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def col2im_forward(cols, height, width, kernel_h, kernel_w):
    """
    Perform col2im operation to reconstruct image from column matrix
    
    Args:
        cols: input column matrix of shape [N, C*kH*kW, L]
        height: output image height
        width: output image width
        kernel_h: kernel height
        kernel_w: kernel width
    Returns:
        image of shape [N, C, H, W]
    """
    N, C_times_kernel_size, L = cols.shape
    kernel_size = kernel_h * kernel_w
    C = C_times_kernel_size // kernel_size
    
    # Reshape input to [N, C, kernel_h, kernel_w, L]
    cols_reshaped = cols.reshape(N, C, kernel_h, kernel_w, L)
    
    # Initialize output image
    output = np.zeros((N, C, height, width), dtype=cols.dtype)
    
    # Reconstruct image by adding patches at their original positions
    for h in range(height - kernel_h + 1):
        for w in range(width - kernel_w + 1):
            # Calculate corresponding column index
            col_idx = h * (width - kernel_w + 1) + w
            
            if col_idx < L:  # Ensure we don't exceed column bounds
                # Extract patch from columns
                patch = cols_reshaped[:, :, :, :, col_idx]
                # Add patch to output image at position (h, w)
                output[:, :, h:h+kernel_h, w:w+kernel_w] += patch
    
    return output


def gen_data_and_golden(shape_str, d_type="float32", 
                       height=2, width=2, kernel_h=2, kernel_w=2):
    """
    Generate input data and golden output for col2im operation
    
    Args:
        shape_str: input shape string in format "(N, C*kH*kW, L)"
        d_type: data type (float32, float16, bfloat16)
        height: output image height
        width: output image width
        kernel_h: kernel height
        kernel_w: kernel width
    """
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    
    # Parse input shape
    shape = parse_str_to_shape_list(shape_str)
    if len(shape) != 3:
        print(f"Error: Input shape must have 3 dimensions, got {len(shape)}")
        exit(1)
    
    N, C_times_kernel_size, L = shape
    kernel_size = kernel_h * kernel_w
    
    # Validate shape parameters
    if C_times_kernel_size % kernel_size != 0:
        print(f"Error: C*kH*kW ({C_times_kernel_size}) must be divisible by kernel_size ({kernel_size})")
        exit(1)
    
    C = C_times_kernel_size // kernel_size
    
    if L != (height - kernel_h + 1) * (width - kernel_w + 1):
        print(f"Error: L ({L}) must equal (H-kH+1)*(W-kW+1) = ({height}-{kernel_h}+1)*({width}-{kernel_w}+1) = {(height - kernel_h + 1) * (width - kernel_w + 1)}")
        exit(1)
    
    # Generate random input data
    size = np.prod(shape)
    # Use random values between -1 and 1 for better numerical stability
    tmp_input = np.random.uniform(-1, 1, size=size)
    tmp_input = tmp_input.reshape(shape).astype(np_type)
    
    # Generate golden output using col2im operation
    tmp_golden = col2im_forward(tmp_input, height, width, kernel_h, kernel_w)
    
    # Save to files
    tmp_input.astype(np_type).tofile(f"{d_type}_input_col2im.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_col2im.bin")
    
    # Print shape information for verification
    print(f"Input shape: {shape} -> [N={N}, C*kH*kW={C_times_kernel_size}, L={L}]")
    print(f"Kernel: {kernel_h}x{kernel_w}")
    print(f"Channels: C = {C}")
    print(f"Output shape: [N={N}, C={C}, H={height}, W={width}]")


if __name__ == "__main__":
    # Parse command line arguments
    # Expected format: python script.py "(N, C*kH*kW, L)" dtype [H] [W] [kH] [kW]
    if len(sys.argv) < 3:
        print("Usage: python script.py \"(N, C*kH*kW, L)\" dtype [H] [W] [kH] [kW]")
        print("Example: python script.py \"(2, 8, 9)\" float32 4 4 2 2")
        exit(1)
    
    # Clean bin files
    os.system("rm -rf *.bin")
    
    # Get required parameters
    shape_str = sys.argv[1]
    dtype = sys.argv[2]
    
    # Set default values for optional parameters
    height = 4
    width = 4
    kernel_h = 2
    kernel_w = 2
    
    # Parse optional parameters if provided
    if len(sys.argv) > 3:
        height = int(sys.argv[3])
    if len(sys.argv) > 4:
        width = int(sys.argv[4])
    if len(sys.argv) > 5:
        kernel_h = int(sys.argv[5])
    if len(sys.argv) > 6:
        kernel_w = int(sys.argv[6])
    
    gen_data_and_golden(shape_str, dtype, height, width, kernel_h, kernel_w)