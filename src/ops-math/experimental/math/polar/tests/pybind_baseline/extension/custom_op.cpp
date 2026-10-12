/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

// aclnnPolar(input, angle, out)：
//   out = input * (cos(angle) + i*sin(angle))
//   input/angle: float32（可广播）；out: complex64，shape = broadcast(input, angle)
at::Tensor my_op_impl_npu(const at::Tensor& input, const at::Tensor& angle)
{
    // numpy 广播规则推导输出 shape（与参考实现 OP_CHECK_BROADCAST_AND_INFER_SHAPE 对齐）
    auto out_shape = at::infer_size(input.sizes(), angle.sizes());

    auto round = 50; // msprof 取平均：重复 50 次（与 S8 一致）
    at::Tensor result;
    for (size_t i = 0; i < round; i++) {
        result = at::empty(out_shape, input.options().dtype(c10::kComplexFloat));
        EXEC_NPU_CMD(aclnnPolar, input, angle, result);
    }
    return result;
}

// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) { m.def("my_op(Tensor input, Tensor angle) -> Tensor"); }

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) { m.impl("my_op", &my_op_impl_npu); }

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("custom_op", &my_op_impl_npu, "custom_polar"); }
