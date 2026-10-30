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
 * @brief PyTorch 适配层 - FusedMulAddN ACLNN 两段式调用 + PyTorch 算子注册
 *
 * 算子公式: y_i = x1_i * x3[0] + x2_i （x3 单元素标量张量，仅取 x3[0]）
 *
 * ACLNN 两段式签名（aclnn_fused_mul_add_n.h）：
 *   aclnnFusedMulAddNGetWorkspaceSize(x1, x2, x3, y, &workspaceSize, &executor)
 *   aclnnFusedMulAddN(workspace, workspaceSize, executor, stream)
 *
 * 两段式调用架构：
 *   - 主线程：分配输出 tensor + workspace tensor，GetWorkspaceSize
 *   - lambda：通过 OpCommand 异步入 queue 执行算子，执行后释放 OpWorkspace
 *
 * 内存管理：workspace 内存由 PyTorch tensor 管理（torch::empty），不使用 aclrtMalloc
 *
 * 使用：
 *   import torch, torch_npu
 *   torch.ops.load_library("libtorch_adapter.so")
 *   y = torch.ops.fused_mul_add_n.forward(x1, x2, x3)
 */

#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "aclnn_fused_mul_add_n.h"
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

aclDataType ScalarTypeToAclDType(c10::ScalarType st) {
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
// 辅助函数：从裸指针创建 aclTensor（行优先连续 strides）
// rank=0（标量）时 shape 为空向量，stride 为空向量，合法。
// ============================================================================

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

aclTensor* CreateAclTensor(const void* data_ptr, const std::vector<int64_t>& shape, aclDataType dtype) {
    auto strides = ComputeStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), const_cast<void*>(data_ptr));
}

// ============================================================================
// Workspace 封装（内存由 PyTorch tensor 管理，析构释放 aclTensor）
// ============================================================================

struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* x1_acl = nullptr;
    aclTensor* x2_acl = nullptr;
    aclTensor* x3_acl = nullptr;
    aclTensor* y_acl = nullptr;

    ~OpWorkspace() {
        if (x1_acl) { aclDestroyTensor(x1_acl); x1_acl = nullptr; }
        if (x2_acl) { aclDestroyTensor(x2_acl); x2_acl = nullptr; }
        if (x3_acl) { aclDestroyTensor(x3_acl); x3_acl = nullptr; }
        if (y_acl) { aclDestroyTensor(y_acl); y_acl = nullptr; }
        executor = nullptr;
    }
};

OpWorkspace* OpGetWorkspace(const void* x1_ptr, const void* x2_ptr, const void* x3_ptr, void* y_ptr,
                            const std::vector<int64_t>& shape, const std::vector<int64_t>& x3_shape,
                            aclDataType dtype) {
    auto ws = new OpWorkspace();
    ws->x1_acl = CreateAclTensor(x1_ptr, shape, dtype);
    ws->x2_acl = CreateAclTensor(x2_ptr, shape, dtype);
    ws->x3_acl = CreateAclTensor(x3_ptr, x3_shape, dtype);
    ws->y_acl = CreateAclTensor(y_ptr, shape, dtype);

    if (!ws->x1_acl || !ws->x2_acl || !ws->x3_acl || !ws->y_acl) {
        TORCH_CHECK(false, "OpGetWorkspace: CreateAclTensor failed, x1=", (ws->x1_acl ? "ok" : "null"),
                    ", x2=", (ws->x2_acl ? "ok" : "null"), ", x3=", (ws->x3_acl ? "ok" : "null"),
                    ", y=", (ws->y_acl ? "ok" : "null"));
        delete ws;
        return nullptr;
    }

    aclnnStatus status = aclnnFusedMulAddNGetWorkspaceSize(ws->x1_acl, ws->x2_acl, ws->x3_acl, ws->y_acl,
                                                           &ws->workspace_size, &ws->executor);
    if (status != ACL_SUCCESS) {
        TORCH_CHECK(false, "OpGetWorkspace: aclnnFusedMulAddNGetWorkspaceSize failed, aclnnStatus=", status);
        delete ws;
        return nullptr;
    }
    return ws;
}

aclnnStatus OpExecute(OpWorkspace* ws, void* workspace_ptr, aclrtStream stream) {
    if (!ws || !ws->executor) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclnnFusedMulAddN(workspace_ptr, ws->workspace_size, ws->executor, stream);
}

}  // anonymous namespace

// ============================================================================
// Meta 函数：形状推导（y 与 x1 同 shape 同 dtype）
// ============================================================================

static torch::Tensor forward_meta(const torch::Tensor& x1, const torch::Tensor& x2, const torch::Tensor& x3) {
    TORCH_CHECK(x1.sizes() == x2.sizes(), "forward: x1/x2 shapes must match, got ", x1.sizes(), " vs ",
                x2.sizes());
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type() && x1.scalar_type() == x3.scalar_type(),
                "forward: x1/x2/x3 dtypes must match");
    TORCH_CHECK(x3.numel() == 1, "forward: x3 must be a single-element scalar tensor, got numel=", x3.numel());
    return torch::empty_like(x1);
}

// ============================================================================
// NPU 实现：contiguous + OpCommand 异步入 queue
// ============================================================================

static torch::Tensor forward_npu(const torch::Tensor& x1, const torch::Tensor& x2, const torch::Tensor& x3) {
    TORCH_CHECK(x3.numel() == 1, "forward: x3 must be a single-element scalar tensor, got numel=", x3.numel());

    auto y = torch::empty_like(x1).contiguous();
    auto x1_contig = x1.contiguous();
    auto x2_contig = x2.contiguous();
    auto x3_contig = x3.contiguous();

    auto dtype = ScalarTypeToAclDType(x1.scalar_type());
    auto shape = x1.sizes().vec();
    auto x3_shape = x3.sizes().vec();

    OpWorkspace* ws = OpGetWorkspace(x1_contig.data_ptr(), x2_contig.data_ptr(), x3_contig.data_ptr(),
                                     y.data_ptr(), shape, x3_shape, dtype);
    TORCH_CHECK(ws != nullptr, "OpGetWorkspace returned null (see above for details)");

    torch::Tensor workspace_tensor;
    void* workspace_ptr = nullptr;
    if (ws->workspace_size > 0) {
        workspace_tensor = torch::empty({static_cast<int64_t>(ws->workspace_size)},
                                        torch::dtype(torch::kByte).device(x1.device()));
        workspace_ptr = workspace_tensor.data_ptr();
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    auto acl_call = [ws, workspace_ptr, acl_stream]() -> int {
        aclnnStatus status = OpExecute(ws, workspace_ptr, acl_stream);
        delete ws;
        return status == ACL_SUCCESS ? 0 : 1;
    };

    at_npu::native::OpCommand::RunOpApiV2("ascendc_fused_mul_add_n", acl_call);

    return y;
}

// ============================================================================
// PyTorch 算子注册
// ============================================================================

TORCH_LIBRARY_FRAGMENT(fused_mul_add_n, m) {
    m.def("forward(Tensor x1, Tensor x2, Tensor x3) -> Tensor");
}

TORCH_LIBRARY_IMPL(fused_mul_add_n, Meta, m) {
    m.impl("forward", forward_meta);
}

TORCH_LIBRARY_IMPL(fused_mul_add_n, PrivateUse1, m) {
    m.impl("forward", forward_npu);
}
