#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
# A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the
# License.
# -----------------------------------------------------------------------------
# NOTE: Batch tests must use --proc-no-reuse; reused workers may retain invalid
# ACL/torch_npu contexts between test cases.

import ctypes
import os

import numpy as np
import torch


__golden__ = {
    "aclnn": {
        "aclnnMultinomial": "multinomial_golden",
        "aclnnMultinomialTensor": "multinomial_tensor_golden",
    }
}

PHILOX_W32_A = 0x9E3779B9
PHILOX_W32_B = 0xBB67AE85
PHILOX_M4X32_A = 0xD2511F53
PHILOX_M4X32_B = 0xCD9E8D57
UINT32_MASK = 0xFFFFFFFF
UINT64_MASK = 0xFFFFFFFFFFFFFFFF

# Uniform conversion factor for the Philox u32 stream. The device kernels map a
# counter output u32 to a uniform float as u * RAND_2POW32_INV + RAND_2POW32_INV_HALF;
# golden must use the same rounding to stay bit-aligned with the operator.
RAND_2POW32_INV = 2.3283064e-10
RAND_2POW32_INV_HALF = RAND_2POW32_INV / 2.0

SAMPLE_ALIGNMENT = 128

_torch_npu = None
_acl_lib = None


def _worker_device_id():
    try:
        from ttk.core_modules.tbe_multiprocessing import get_process_context

        process_context = get_process_context()
        if process_context is None:
            return None
        device = process_context.storage.get("device")
        device_id = getattr(device, "device_id", None)
        return None if device_id is None else int(device_id)
    except Exception:
        return None


def _acl_device_id():
    global _acl_lib
    if _acl_lib is None:
        try:
            _acl_lib = ctypes.CDLL("libascendcl.so")
            _acl_lib.aclrtGetDevice.restype = ctypes.c_int32
            _acl_lib.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        except OSError:
            _acl_lib = False
    if not _acl_lib:
        return None

    device_id = ctypes.c_int32(-1)
    if _acl_lib.aclrtGetDevice(ctypes.byref(device_id)) != 0:
        return None
    return device_id.value


def _load_torch_npu():
    global _torch_npu
    if _torch_npu is not None:
        return _torch_npu

    import torch_npu

    device_id = _worker_device_id()
    if device_id is None:
        device_id = _acl_device_id()
    override = os.getenv("MULTINOMIAL_GOLDEN_DEV")
    if override is not None:
        device_id = int(override)
    if device_id is not None and device_id >= 0:
        torch_npu.npu.set_device(device_id)
    _torch_npu = torch_npu
    return _torch_npu


def _philox_batch4_u32(seed, counter_lo, counter_hi):
    counter_lo = np.asarray(counter_lo, dtype=np.uint64)
    counter_hi = np.asarray(counter_hi, dtype=np.uint64)

    c0 = (counter_lo & np.uint64(UINT32_MASK)).astype(np.uint32)
    c1 = (counter_lo >> np.uint64(32)).astype(np.uint32)
    c2 = (counter_hi & np.uint64(UINT32_MASK)).astype(np.uint32)
    c3 = (counter_hi >> np.uint64(32)).astype(np.uint32)
    key0 = np.uint32(seed & UINT32_MASK)
    key1 = np.uint32((seed >> 32) & UINT32_MASK)
    multiplier0 = np.uint64(PHILOX_M4X32_A)
    multiplier1 = np.uint64(PHILOX_M4X32_B)

    for _ in range(10):
        product0 = multiplier0 * c0.astype(np.uint64)
        product1 = multiplier1 * c2.astype(np.uint64)
        lo0 = product0.astype(np.uint32)
        hi0 = (product0 >> np.uint64(32)).astype(np.uint32)
        lo1 = product1.astype(np.uint32)
        hi1 = (product1 >> np.uint64(32)).astype(np.uint32)
        c0, c1, c2, c3 = (
            hi1 ^ c1 ^ key0,
            lo1,
            hi0 ^ c3 ^ key1,
            lo0,
        )
        key0 = np.uint32((int(key0) + PHILOX_W32_A) & UINT32_MASK)
        key1 = np.uint32((int(key1) + PHILOX_W32_B) & UINT32_MASK)
    return c0, c1, c2, c3


def _uniform_u32(seed, offset, subsequence):
    subsequence = np.asarray(subsequence, dtype=np.uint64)
    counter_lo = np.full(subsequence.shape, offset, dtype=np.uint64)
    return _philox_batch4_u32(seed, counter_lo, subsequence)[0]


def _u32_to_float(value):
    inverse = np.float32(RAND_2POW32_INV)
    return value.astype(np.float32) * inverse + inverse / np.float32(2.0)


def _without_replacement(weights, seed, offset, numsamples):
    torch_npu = _load_torch_npu()
    weights_npu = weights.npu()
    generator = torch.Generator(device=weights_npu.device)
    generator.manual_seed(int(seed))
    # torch_npu Generator.set_offset requires offset to be a multiple of 4.
    # The device kernel aligns the offset via ceil(offset / 4), so round up to
    # the next multiple of 4 to start the exponential stream at the same Philox
    # counter the kernel will consume.
    aligned_offset = ((int(offset) + 3) // 4) * 4
    generator.set_offset(aligned_offset)
    exponential = torch.empty_like(weights_npu)
    torch_npu.npu_sim_exponential_(exponential, lambd=1.0, generator=generator)
    scores = torch.div(weights_npu, exponential)
    if numsamples == 1:
        result = torch.argmax(scores, dim=-1, keepdim=True)
    else:
        result = torch.topk(
            scores, numsamples, dim=-1, largest=True, sorted=True
        ).indices
    return result.to(torch.int64).cpu()


def _normalized_cdf(weights):
    _load_torch_npu()
    weights_npu = weights.npu()
    total = torch.sum(weights_npu, dim=-1, keepdim=True, dtype=weights_npu.dtype)
    probabilities = torch.div(weights_npu, total)
    cdf = torch.cumsum(probabilities, dim=-1, dtype=weights_npu.dtype)
    return cdf.cpu(), probabilities.cpu()


def _sample_from_cdf(cdf, probabilities, seed, offset, numsamples):
    squeeze_output = cdf.ndim == 1
    if squeeze_output:
        cdf = cdf.unsqueeze(0)
        probabilities = probabilities.unsqueeze(0)

    dtype = cdf.dtype
    distribution_count, category_count = cdf.shape
    invalid_rows = torch.nonzero(~(cdf[:, -1] > 0), as_tuple=False).reshape(-1)
    if invalid_rows.numel():
        values = cdf[invalid_rows, -1].tolist()
        raise RuntimeError(
            "StatelessSampleMultinomial assertion failed: "
            f"cdf[:, -1] must be positive; rows={invalid_rows.tolist()}, "
            f"values={values}"
        )

    aligned_samples = (
        (numsamples + SAMPLE_ALIGNMENT - 1) // SAMPLE_ALIGNMENT * SAMPLE_ALIGNMENT
    )
    offset_u64 = int(offset) & UINT64_MASK
    base_offset = ((offset_u64 + 3) & UINT64_MASK) // 4
    flat_index = np.arange(distribution_count * numsamples, dtype=np.uint64)
    distribution_index = flat_index // np.uint64(numsamples)
    sample_index = flat_index % np.uint64(numsamples)
    subsequence = distribution_index * np.uint64(aligned_samples) + sample_index
    random_u32 = _uniform_u32(int(seed) & UINT64_MASK, base_offset, subsequence)
    random_value = torch.from_numpy(_u32_to_float(random_u32)).to(dtype)
    random_value = random_value.reshape(distribution_count, numsamples)

    start = torch.zeros((distribution_count, numsamples), dtype=torch.int64)
    end = torch.full_like(start, category_count)
    active = end > start
    while bool(active.any()):
        midpoint = start + ((end - start) >> 1)
        values = torch.gather(cdf, 1, midpoint.clamp_max(category_count - 1))
        move_right = (values < random_value) & active
        start = torch.where(move_right, midpoint + 1, start)
        end = torch.where(active & ~move_right, midpoint, end)
        active = end > start
    start.clamp_max_(category_count - 1)

    category_index = torch.arange(category_count, dtype=torch.int64)
    category_index = category_index.unsqueeze(0).expand(
        distribution_count, category_count
    )
    nonzero_index = torch.where(
        probabilities != torch.zeros((), dtype=dtype),
        category_index,
        torch.full_like(category_index, -1),
    )
    previous_nonzero = torch.cummax(nonzero_index, dim=1).values
    result = torch.gather(previous_nonzero, 1, start)
    result = torch.where(result >= 0, result, torch.zeros_like(result))
    return result.reshape(numsamples) if squeeze_output else result


def _with_replacement(weights, seed, offset, numsamples):
    squeeze_output = weights.ndim == 1
    if squeeze_output:
        weights = weights.unsqueeze(0)
    cdf, probabilities = _normalized_cdf(weights)
    if squeeze_output:
        cdf = cdf.squeeze(0)
        probabilities = probabilities.squeeze(0)
    return _sample_from_cdf(cdf, probabilities, seed, offset, numsamples)


def _as_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    if value.dtype.name == "bfloat16":
        return torch.frombuffer(
            bytearray(value.tobytes()), dtype=torch.bfloat16
        ).reshape(value.shape)
    return torch.from_numpy(np.ascontiguousarray(value))


def _compute(weights, numsamples, replacement, seed, offset):
    weights = _as_torch(weights).detach().cpu()
    numsamples = int(numsamples)
    if not bool(replacement) or numsamples == 1:
        return _without_replacement(weights, int(seed), int(offset), numsamples)
    return _with_replacement(weights, int(seed), int(offset), numsamples)


def multinomial_golden(self, numsamples, replacement, seed, offset, out=None, **kwargs):
    """
    Aclnn golden for aclnnMultinomial.
    Parameters follow @aclnnMultinomialGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.

    replacement=False (or numsamples == 1): sample via the exponential path
    (weights / Exp(1.0)), then argmax (numsamples == 1) or top-k (numsamples > 1).
    replacement=True otherwise: normalized-CDF binary search against a Philox
    stream. The sample index deterministically depends on (seed, offset).
    """
    del out, kwargs
    return (_compute(self, numsamples, replacement, seed, offset).numpy(),)


def multinomial_tensor_golden(
    self,
    numsamples,
    replacement,
    seedTensor,
    offsetTensor,
    offset,
    out=None,
    **kwargs,
):
    """
    Aclnn golden for aclnnMultinomialTensor.
    Parameters follow @aclnnMultinomialTensorGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.

    seedTensor / offsetTensor carry the generator seed and offset; `offset` is the
    stream-local intragraph offset added on top of offsetTensor. The combined offset
    wraps at 64 bits and is interpreted as signed for the exponential path.
    """
    del out, kwargs
    seed = int(_as_torch(seedTensor).reshape(-1)[0].item())
    tensor_offset = int(_as_torch(offsetTensor).reshape(-1)[0].item())
    combined_offset = (tensor_offset + int(offset)) & UINT64_MASK
    if combined_offset >= 1 << 63:
        combined_offset -= 1 << 64
    return (_compute(self, numsamples, replacement, seed, combined_offset).numpy(),)
