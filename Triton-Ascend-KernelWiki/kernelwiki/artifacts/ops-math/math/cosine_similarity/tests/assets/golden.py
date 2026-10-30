#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch

__golden__ = {
    "kernel": {"cosine_similarity": "cosine_similarity_golden"},
    "aclnn": {"aclnnCosineSimilarity": "aclnn_cosine_similarity_golden"}
}


def cosine_similarity_golden(input_x1, input_x2, *, dim=1, eps=1e-8, **kwargs):
    '''
    Golden function for cosine_similarity.
    All the parameters (names and order) follow @cosine_similarity_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    # Save original output dtype (only float32 is supported)
    out_dtype = input_x1.dtype

    x1_torch = torch.from_numpy(np.ascontiguousarray(input_x1)).float()
    x2_torch = torch.from_numpy(np.ascontiguousarray(input_x2)).float()

    result = torch.nn.functional.cosine_similarity(x1_torch, x2_torch, dim=dim, eps=eps)
    result_np = result.numpy().astype(out_dtype, copy=False)

    return result_np


def aclnn_cosine_similarity_golden(inputX1, inputX2, dim, eps, outputY, **kwargs):
    '''
    Aclnn golden for aclnnCosineSimilarity.
    All the parameters (name & order) follow \
        function `aclnnCosineSimilarityGetWorkspaceSize` in @aclnn_cosine_similarity.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    if isinstance(inputX1, np.ndarray):
        x1_torch = torch.from_numpy(np.ascontiguousarray(inputX1)).float()
        x2_torch = torch.from_numpy(np.ascontiguousarray(inputX2)).float()
    else:
        x1_torch = inputX1.float()
        x2_torch = inputX2.float()

    dim_val = dim.item() if hasattr(dim, 'item') else int(dim)
    eps_val = eps.item() if hasattr(eps, 'item') else float(eps)

    result = torch.nn.functional.cosine_similarity(x1_torch, x2_torch, dim=dim_val, eps=eps_val)

    # Convert back to output dtype
    if isinstance(outputY, np.ndarray):
        out_dtype = outputY.dtype
        return result.numpy().astype(out_dtype, copy=False)
    else:
        return result.to(outputY.dtype)
