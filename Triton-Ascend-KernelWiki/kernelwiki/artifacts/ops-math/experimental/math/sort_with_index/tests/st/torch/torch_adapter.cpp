/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file torch_adapter.cpp
 * @brief PyTorch 适配层 —— SortWithIndex（ascend910b）ACLNN 两段式封装 + PyTorch 算子注册
 *
 * 两段式调用架构（对齐 add_example 样例）：
 *   - 主线程：分配输出 tensor + GetWorkspace
 *   - lambda：OpExecute 入 OpCommand queue
 *   - workspace 内存由 PyTorch tensor 管理（torch::empty(kByte)），不用 aclrtMalloc
 *
 * SortWithIndex 双输入双输出 + 3 属性：
 *   forward(Tensor x, Tensor index, int axis, bool descending, bool stable)
 *       -> (Tensor y, Tensor sorted_index)
 *   y.dtype=x.dtype、sorted_index.dtype=index.dtype；y.shape=x.shape、sorted_index.shape=index.shape。
 *
 * 使用：
 *   import torch
 *   torch.ops.load_library("libtorch_adapter.so")
 *   x = torch.randn(8, device="npu"); idx = torch.arange(8, dtype=torch.int32, device="npu")
 *   y, si = torch.ops.sort_with_index.forward(x, idx, -1, False, True)
 */

#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "aclnn_sort_with_index.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#ifndef ACLNN_STATUS_DEFINED
typedef int aclnnStatus;
#define ACLNN_STATUS_DEFINED
#endif

namespace {

// ============================================================================
// PyTorch c10::ScalarType -> CANN aclDataType 映射
// ============================================================================

aclDataType ScalarTypeToAclDType(c10::ScalarType st)
{
    switch (st) {
        case c10::ScalarType::Byte:     return ACL_UINT8;
        case c10::ScalarType::Char:     return ACL_INT8;
        case c10::ScalarType::Short:    return ACL_INT16;
        case c10::ScalarType::Int:      return ACL_INT32;
        case c10::ScalarType::Long:     return ACL_INT64;
        case c10::ScalarType::Half:     return ACL_FLOAT16;
        case c10::ScalarType::Float:    return ACL_FLOAT;
        case c10::ScalarType::Double:   return ACL_DOUBLE;
        case c10::ScalarType::Bool:     return ACL_BOOL;
        case c10::ScalarType::BFloat16: return ACL_BF16;
        default:                        return ACL_DT_UNDEFINED;
    }
}

// ============================================================================
// 从裸指针创建 aclTensor（ND）
// ============================================================================

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

aclTensor* CreateAclTensor(const void* data_ptr, const std::vector<int64_t>& shape, aclDataType dtype)
{
    auto strides = ComputeStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), const_cast<void*>(data_ptr));
}

// ============================================================================
// Workspace 封装（内存由 PyTorch tensor 管理；析构销毁 aclTensor）
// ============================================================================

struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* x_acl = nullptr;
    aclTensor* index_acl = nullptr;
    aclTensor* y_acl = nullptr;
    aclTensor* si_acl = nullptr;

    ~OpWorkspace()
    {
        if (x_acl) { aclDestroyTensor(x_acl); x_acl = nullptr; }
        if (index_acl) { aclDestroyTensor(index_acl); index_acl = nullptr; }
        if (y_acl) { aclDestroyTensor(y_acl); y_acl = nullptr; }
        if (si_acl) { aclDestroyTensor(si_acl); si_acl = nullptr; }
        executor = nullptr;
    }
};

OpWorkspace* OpGetWorkspace(const void* x_ptr, const void* index_ptr, void* y_ptr, void* si_ptr,
                            const std::vector<int64_t>& shape, aclDataType x_dtype, aclDataType index_dtype,
                            int64_t axis, bool descending, bool stable)
{
    auto ws = new OpWorkspace();
    ws->x_acl = CreateAclTensor(x_ptr, shape, x_dtype);
    ws->index_acl = CreateAclTensor(index_ptr, shape, index_dtype);
    ws->y_acl = CreateAclTensor(y_ptr, shape, x_dtype);
    ws->si_acl = CreateAclTensor(si_ptr, shape, index_dtype);

    if (!ws->x_acl || !ws->index_acl || !ws->y_acl || !ws->si_acl) {
        TORCH_CHECK(false, "OpGetWorkspace: CreateAclTensor failed");
        delete ws;
        return nullptr;
    }

    aclnnStatus status = aclnnSortWithIndexGetWorkspaceSize(ws->x_acl, ws->index_acl, axis, descending, stable,
                                                            ws->y_acl, ws->si_acl, &ws->workspace_size,
                                                            &ws->executor);
    if (status != ACL_SUCCESS) {
        TORCH_CHECK(false, "OpGetWorkspace: aclnnSortWithIndexGetWorkspaceSize failed, aclnnStatus=", status);
        delete ws;
        return nullptr;
    }
    return ws;
}

aclnnStatus OpExecute(OpWorkspace* ws, void* workspace_ptr, aclrtStream stream)
{
    if (!ws || !ws->executor) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclnnSortWithIndex(workspace_ptr, ws->workspace_size, ws->executor, stream);
}

}  // anonymous namespace

// ============================================================================
// Meta 函数：形状/类型推导（y 跟随 x，sorted_index 跟随 index）
// ============================================================================

static std::tuple<torch::Tensor, torch::Tensor> forward_meta(const torch::Tensor& x, const torch::Tensor& index,
                                                             int64_t axis, bool descending, bool stable)
{
    (void)axis;
    (void)descending;
    (void)stable;
    TORCH_CHECK(x.sizes() == index.sizes(), "SortWithIndex: x/index shapes must match, got ", x.sizes(), " vs ",
                index.sizes());
    auto y = torch::empty_like(x);
    auto sorted_index = torch::empty_like(index);
    return std::make_tuple(y, sorted_index);
}

// ============================================================================
// NPU 实现：contiguous + OpCommand 异步入 queue
// ============================================================================

static std::tuple<torch::Tensor, torch::Tensor> forward_npu(const torch::Tensor& x, const torch::Tensor& index,
                                                            int64_t axis, bool descending, bool stable)
{
    TORCH_CHECK(x.sizes() == index.sizes(), "SortWithIndex: x/index shapes must match, got ", x.sizes(), " vs ",
                index.sizes());

    auto x_contig = x.contiguous();
    auto index_contig = index.contiguous();
    auto y = torch::empty_like(x_contig).contiguous();
    auto sorted_index = torch::empty_like(index_contig).contiguous();

    auto x_dtype = ScalarTypeToAclDType(x.scalar_type());
    auto index_dtype = ScalarTypeToAclDType(index.scalar_type());
    auto shape = x.sizes().vec();

    OpWorkspace* ws = OpGetWorkspace(x_contig.data_ptr(), index_contig.data_ptr(), y.data_ptr(),
                                     sorted_index.data_ptr(), shape, x_dtype, index_dtype, axis, descending, stable);
    TORCH_CHECK(ws != nullptr, "OpGetWorkspace returned null (see above for details)");

    torch::Tensor workspace_tensor;
    void* workspace_ptr = nullptr;
    if (ws->workspace_size > 0) {
        workspace_tensor =
            torch::empty({static_cast<int64_t>(ws->workspace_size)}, torch::dtype(torch::kByte).device(x.device()));
        workspace_ptr = workspace_tensor.data_ptr();
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [ws, workspace_ptr, acl_stream]() -> int {
        aclnnStatus status = OpExecute(ws, workspace_ptr, acl_stream);
        delete ws;
        return status == ACL_SUCCESS ? 0 : 1;
    };

    at_npu::native::OpCommand::RunOpApiV2("ascendc_sort_with_index", acl_call);

    return std::make_tuple(y, sorted_index);
}

// ============================================================================
// PyTorch 算子注册
// ============================================================================

TORCH_LIBRARY_FRAGMENT(sort_with_index, m)
{
    m.def("forward(Tensor x, Tensor index, int axis, bool descending, bool stable) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(sort_with_index, Meta, m)
{
    m.impl("forward", forward_meta);
}

TORCH_LIBRARY_IMPL(sort_with_index, PrivateUse1, m)
{
    m.impl("forward", forward_npu);
}
