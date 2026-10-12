#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import struct

def float32_to_bfloat16_bytes(f):
    """Convert float32 to bfloat16 bytes representation."""
    f_bytes = struct.pack('>f', f)
    # bfloat16: take upper 16 bits of float32 (sign + exponent + upper 7 bits of mantissa)
    bf16_bytes = f_bytes[:2]
    return bf16_bytes

def bfloat16_bytes_to_float32(bf16_bytes):
    """Convert bfloat16 bytes to float32."""
    # Pad with zeros for lower 16 bits of mantissa
    f_bytes = bf16_bytes + b'\x00\x00'
    return struct.unpack('>f', f_bytes)[0]

def gen_data(shape_str, dtype_str, scalar_value):
    """Generate input and golden data for Adds operator."""
    shape = eval(shape_str)
    np.random.seed(42)
    
    if dtype_str == 'float32':
        dtype = np.float32
        input_data = np.random.uniform(-2, 2, size=shape).astype(dtype)
        output_data = (input_data + scalar_value).astype(dtype)
        input_data.tofile('./input.bin')
        output_data.tofile('./golden.bin')
    elif dtype_str == 'float16':
        dtype = np.float16
        input_data = np.random.uniform(-2, 2, size=shape).astype(dtype)
        output_data = (input_data + scalar_value).astype(dtype)
        input_data.tofile('./input.bin')
        output_data.tofile('./golden.bin')
    elif dtype_str == 'bfloat16':
        # Generate float32 data and convert to bfloat16 bytes
        input_fp32 = np.random.uniform(-2, 2, size=shape).astype(np.float32)
        output_fp32 = (input_fp32 + scalar_value).astype(np.float32)
        
        with open('./input.bin', 'wb') as f:
            for val in input_fp32.flatten():
                f.write(float32_to_bfloat16_bytes(val))
        
        with open('./golden.bin', 'wb') as f:
            for val in output_fp32.flatten():
                f.write(float32_to_bfloat16_bytes(val))
    elif dtype_str == 'int16':
        dtype = np.int16
        input_data = np.random.randint(-100, 100, size=shape, dtype=dtype)
        output_data = np.clip(input_data + int(scalar_value), -32768, 32767).astype(dtype)
        input_data.tofile('./input.bin')
        output_data.tofile('./golden.bin')
    elif dtype_str == 'int32':
        dtype = np.int32
        input_data = np.random.randint(-1000, 1000, size=shape, dtype=dtype)
        output_data = (input_data + int(scalar_value)).astype(dtype)
        input_data.tofile('./input.bin')
        output_data.tofile('./golden.bin')
    elif dtype_str == 'int64':
        dtype = np.int64
        input_data = np.random.randint(-10000, 10000, size=shape, dtype=dtype)
        output_data = (input_data + int(scalar_value)).astype(dtype)
        input_data.tofile('./input.bin')
        output_data.tofile('./golden.bin')
    
    print(f"Generated data: shape={shape}, dtype={dtype_str}, scalar={scalar_value}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 gen_data.py <shape> <dtype> <scalar>")
        sys.exit(1)
    
    shape_str = sys.argv[1]
    dtype_str = sys.argv[2]
    scalar_value = float(sys.argv[3])
    
    gen_data(shape_str, dtype_str, scalar_value)