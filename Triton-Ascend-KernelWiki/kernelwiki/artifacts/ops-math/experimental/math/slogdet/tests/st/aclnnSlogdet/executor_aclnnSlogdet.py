#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


_PATTERNS = (
    "positive_det",
    "negative_det",
    "singular_zero",
    "singular_zero_row",
    "diag_dominant",
    "ill_conditioned",
)


def _case_id(api) -> int:
    try:
        case_config = api.task_result.case_config
        value = getattr(case_config, "id", None)
        if value is None and isinstance(case_config, dict):
            value = case_config.get("id")
        return int(value) if value is not None else 0
    except Exception:
        return 0


def _single_matrix(pattern: str, n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
    eye = torch.eye(n, dtype=torch.float32)

    if pattern == "positive_det":
        diag = torch.arange(1, n + 1, dtype=torch.float32)
        return torch.diag(diag)
    if pattern == "negative_det":
        diag = torch.arange(1, n + 1, dtype=torch.float32)
        mat = torch.diag(diag)
        if n >= 2:
            mat[[0, 1]] = mat[[1, 0]].clone()
            return mat
        return -mat
    if pattern == "singular_zero":
        return torch.zeros(n, n, dtype=torch.float32)

    base = (torch.rand(n, n, generator=generator, dtype=torch.float32) - 0.5) * 0.2 + (n + 1.0) * eye
    if pattern == "singular_zero_row":
        if n >= 2:
            base[0, :] = 0.0
            return base
        return torch.zeros(n, n, dtype=torch.float32)
    if pattern == "diag_dominant":
        return base
    if pattern == "ill_conditioned" and n >= 2:
        base[0, :] = base[1, :] + 0.2 * base[0, :]
        return base
    return base


def _build_like(tensor: torch.Tensor, case_id: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        return tensor
    shape = list(tensor.shape)
    if len(shape) < 2 or tensor.numel() == 0 or shape[-1] != shape[-2]:
        return tensor

    n = int(shape[-1])
    batch_shape = shape[:-2]
    batch_count = 1
    for dim in batch_shape:
        batch_count *= int(dim)

    pattern = _PATTERNS[case_id % len(_PATTERNS)]
    mats = [_single_matrix(pattern, n, (case_id + 1) * 1009 + i) for i in range(max(1, batch_count))]
    if batch_shape:
        built = torch.stack(mats).reshape(*batch_shape, n, n)
    else:
        built = mats[0]
    return built.to(dtype=tensor.dtype, device=tensor.device).contiguous()


def _inject_content(api, input_data: InputDataset):
    case_id = _case_id(api)
    for index, value in enumerate(list(input_data.args)):
        if isinstance(value, torch.Tensor):
            input_data.args[index] = _build_like(value, case_id)
    for key, value in list(input_data.kwargs.items()):
        if isinstance(value, torch.Tensor):
            input_data.kwargs[key] = _build_like(value, case_id)


def _get_input_tensor(input_data: InputDataset):
    if input_data.args:
        return input_data.args[0]
    if "A" in input_data.kwargs:
        return input_data.kwargs["A"]
    if "self" in input_data.kwargs:
        return input_data.kwargs["self"]
    if input_data.kwargs:
        return next(iter(input_data.kwargs.values()))
    raise ValueError("aclnnSlogdet executor requires one tensor input")


@register("function_aclnnSlogdet")
class AclnnSlogdetExecutor(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _inject_content(self, input_data)
        result = torch.linalg.slogdet(_get_input_tensor(input_data).float())
        return (result.sign.float(), result.logabsdet.float())


@register("aclnn_slogdet_custom")
class AclnnSlogdetCustomApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        _inject_content(self, input_data)
        return super().init_by_input_data(input_data)

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND

    @classmethod
    def get_cpp_func_signature_type(cls):
        return ("aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor* self, "
                "aclTensor* signOut, aclTensor* logOut, uint64_t* workspaceSize, "
                "aclOpExecutor** executor)")
