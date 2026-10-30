/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PY_EXT_CATLASS_KERNEL_WRAPPER_H
#define PY_EXT_CATLASS_KERNEL_WRAPPER_H

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "catlass_kernel.h"

namespace CatlassKernelWrapper {

at::Tensor RunMLA(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const int32_t,
    const std::vector<int32_t> &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const std::string &,
    const float
);

using ConstTensorRef = const at::Tensor&;
std::vector<uint64_t> KernelGetKVSplit(
    ConstTensorRef,
    ConstTensorRef,
    ConstTensorRef,
    ConstTensorRef,
    const int32_t,
    const std::vector<int32_t> &,
    ConstTensorRef,
    ConstTensorRef,
    ConstTensorRef,
    ConstTensorRef,
    ConstTensorRef,
    ConstTensorRef,
    ConstTensorRef,
    const std::string &
);

} // namespace CatlassKernelWrapper

#endif // PY_EXT_CATLASS_KERNEL_WRAPPER_H
