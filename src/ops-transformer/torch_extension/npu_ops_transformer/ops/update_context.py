# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class UpdateContextOpBuilder(OpBuilder):
    def __init__(self):
        super(UpdateContextOpBuilder, self).__init__("update_context")

    def sources(self):
        """Path to C++ source conde."""
        return ['ops/csrc/update_context.cpp']
    
    def schema(self) -> str:
        """PyTorch operator signature."""
        return "update_context(str group_ep, int ep_world_size, int ccl_buffer_size, Tensor context_tensor) -> bool"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def update_context_meta(group_ep, ep_world_size, ccl_buffer_size, context_tensor):
            return False

# Instantiate the builder
update_context_op_builder = UpdateContextOpBuilder()
op_module = update_context_op_builder.load() # Compiles/loads the .so file


@impl(AS_LIBRARY, update_context_op_builder.name, "PrivateUse1")
def update_context(group_ep, ep_world_size, ccl_buffer_size, context_tensor):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.update_context(group_ep, ep_world_size, ccl_buffer_size, context_tensor)