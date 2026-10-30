/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"
#include "op_host/rope_matrix_tiling.h"
#include "op_kernel/rope_matrix.h"
#include "kernel_operator.h"

namespace npu_ops_transformer_ext {
namespace RopeMatrix {

template <typename T>
__global__ __aicore__ void rope_matrix_kernel(
    GM_ADDR x, GM_ADDR y, 
    GM_ADDR sin, GM_ADDR cos, 
    GM_ADDR out, GM_ADDR workspace, 
    RopeMatrixTiling ropeTilingGm, TCubeTiling mmTilingDataGm) 
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tpipe;
    TCubeTiling cubeTiling;
    RopeMatrixTiling ropeTiling;
    CopyTiling(&cubeTiling, &ropeTiling, &mmTilingDataGm, &ropeTilingGm);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if ASCEND_IS_AIC {
        // if not want compiler optmize this branch: add __asm__ volatile("NOP_BAR.MTE2")
        MatmulBatchKernel<T, T, T> mmkernel;
        mmkernel.Init(x, y, user, workspace, cubeTiling, &tpipe);
        mmkernel.Process(&tpipe, &ropeTiling);

        // set flag after matmul, need wait in vector: AscendC::CrossCoreSetFlag<modeID. pipe>(flagId)
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
    }

    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(0x8);
        RopeVec<T> vec;

        vec.Init(x, user, sin, cos, out, &ropeTiling, &tpipe);
        vec.Process();
    }
}

void PackInputInfo(at::Tensor &x, uint32_t blockDim, RopeMatrixTiling *ropeTiling)
{
    ropeTiling->blockDim = blockDim;
    uint32_t *ropePtr = reinterpret_cast<uint32_t *>(ropeTiling);
    ropePtr++; // skip blockDim
    for (uint32_t size : x.sizes()) {
        // set bnsd=size[0,4] value into ropeTiling
        *ropePtr = size;
        ropePtr++;
    }
    return;
}

at::Tensor rope_matrix_npu(at::Tensor &x, at::Tensor &y, at::Tensor &sin, at::Tensor &cos){
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDims = 20;
    //only support bnsd
    RopeMatrixTiling ropeTiling;
    PackInputInfo(x, blockDims, &ropeTiling);
    uint64_t len = x.numel();
    // set device
    int devidx = x.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream(false);

    // output if use "at::Tensor output = torch::zeros(x.sizes(), x.options());", will have extra cost
    at::Tensor output = torch::empty(x.sizes(), x.options());

    // workspace
    auto userWorkspaceSize = len * 2; // need when need copy out and copy in tmp; x.numel() * sizeof(bf16)
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    int64_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    auto workspaceTensor = 
        at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));
    
    // mmtiling: notice TCubeTiling and optiling::TCubeTiling may be different, one maybe more
    TCubeTiling mmTilingData; 
    uint32_t mmSize = sizeof(TCubeTiling) / sizeof(uint32_t);
    auto ret = GenerateTiling(&ropeTiling, reinterpret_cast<uint32_t *>(&mmTilingData), mmSize);
    if (ret == nullptr) {
        std::cout << "Error: generateTiling fail" << std::endl;
        return output;
    }

    auto acl_call = [=]() -> int {
        rope_matrix_kernel<bfloat16_t><<<blockDims, nullptr, aclstream>>>(
            (GM_ADDR)(x.storage().data()),
            (GM_ADDR)(y.storage().data()),
            (GM_ADDR)(sin.storage().data()),
            (GM_ADDR)(cos.storage().data()),
            (GM_ADDR)(output.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            ropeTiling,
            mmTilingData);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("RopeMatrix", acl_call);
    return output;
}

torch::Tensor rope_matrix_meta(torch::Tensor &x, torch::Tensor &y, torch::Tensor &sin,
                               torch::Tensor &cos)
{
    TORCH_CHECK(x.defined(), "Input tensor must be defined");
    return x;
}

// Register Ascend implementations for RopeMatrix
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("rope_matrix", rope_matrix_npu);
}

// Register Meta Function for RopeMatrix
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("rope_matrix", TORCH_FN(rope_matrix_meta));
}

} // namespace RopeMatrix
} // namespace npu_ops_transformer_ext
