/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

namespace py = pybind11;
using namespace CatlassKernel;

namespace CatlassKernelWrapper {

static const uint32_t NUM_MLA_KERNEL_INPUTS = 7;
static const uint32_t NUM_MLA_KERNEL_OUTPUTS = 5;
static const uint32_t Q_NUMHEAD_IND = 1;
static const uint32_t Q_EMBEDDING_IND = 2;

torch::Dtype getTorchDtype(const std::string &dtype_str){
    torch::Dtype torchDtype;
    if (dtype_str == "float16") {
        torchDtype = torch::kFloat16;
    } else if(dtype_str == "bf16") {
        torchDtype = torch::kBFloat16;
    } else {
        throw std::runtime_error("unsupported dtype");;
    }
    return torchDtype;
}

MLAKernelInfo ConstructMlaKernelInfo(
    const std::array<const at::Tensor, NUM_MLA_KERNEL_INPUTS> &inputs,
    const std::array<const at::Tensor, NUM_MLA_KERNEL_OUTPUTS> &outputs,
    const int32_t kv_seqlen,
    const std::string &dtype_str
)
{
    auto [q, q_rope, k, k_rope, block_table, s, p] = inputs;
    auto [result, result_temp, global_o, l, o_core_tmp] = outputs;

    MLAKernelInfo mlaKernelInfo;

    mlaKernelInfo.dTypeKey = (dtype_str == "float16") ? 0 : 1;

    mlaKernelInfo.inputAddr = {
        static_cast<uint8_t *>(const_cast<void *>(q.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(q_rope.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(k.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(k_rope.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(block_table.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(s.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(p.storage().data()))
    };

    mlaKernelInfo.outputAddr = {
        static_cast<uint8_t *>(const_cast<void *>(result.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(result_temp.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(global_o.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(l.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(o_core_tmp.storage().data()))
    };

    int32_t batch = q.sizes().at(0); // Q shape: [batch, kv_heads, head_size]
    int32_t qSeqLen = 1;  // Assume no spec-dec & MTP
    int32_t numBlocks = k.sizes().at(0);  // K shape: [num_block, block_size, kv_heads, head_size]
    int32_t blockSize = k.sizes().at(1);
    int32_t kvSeqLen = kv_seqlen;

    mlaKernelInfo.batch = batch;
    mlaKernelInfo.numHeads = q.sizes().at(Q_NUMHEAD_IND);
    mlaKernelInfo.embeddingSize = q.sizes().at(Q_EMBEDDING_IND);
    mlaKernelInfo.embeddingSizeRope = q_rope.sizes().at(Q_EMBEDDING_IND);
    mlaKernelInfo.numTokens = batch * qSeqLen;
    mlaKernelInfo.kvHeads = 1;
    mlaKernelInfo.numBlocks = numBlocks;
    mlaKernelInfo.blockSize = blockSize;
    mlaKernelInfo.maxKvSeqlen = kvSeqLen; // ignore padding for now

    return mlaKernelInfo;
}


at::Tensor RunMLA(
    const at::Tensor &q, const at::Tensor &q_rope,
    const at::Tensor &k, const at::Tensor &k_rope,
    const int32_t kv_seqlen, const std::vector<int32_t> &kv_seqlens,
    const at::Tensor &block_table, const at::Tensor &s, const at::Tensor &p,
    const at::Tensor &result_temp, const at::Tensor &global_o, const at::Tensor &l,
    const at::Tensor &o_core_tmp, const std::string &dtype_str, const float softmax_scale
)
{
    torch::Dtype outputDataType = getTorchDtype(dtype_str);

    at::TensorOptions options = at::TensorOptions();
    options = options.dtype(outputDataType).layout(at::kStrided).requires_grad(false).device(
        torch_npu::utils::get_npu_device_type());;

    // NOTE: MLA output has same shape as q_nope
    torch::Tensor result =  at_npu::native::empty_with_format(q.sizes(), options, ACL_FORMAT_ND);

    const std::array<const at::Tensor, NUM_MLA_KERNEL_INPUTS> inputs = {q, q_rope, k, k_rope, block_table, s, p};
    const std::array<const at::Tensor, NUM_MLA_KERNEL_OUTPUTS> outputs = {result, result_temp, global_o, l, o_core_tmp};

    MLAKernelInfo mlaKernelInfo = ConstructMlaKernelInfo(inputs, outputs, kv_seqlen, dtype_str);

    int32_t batch = q.sizes().at(0); // Q shape: [batch, kv_heads, head_size]
    int32_t qSeqLen = 1;  // NOTE: for now assume no spec-dec & MTP

    // NOTE: assume equal length for qSeqLen and kvSeqLen
    void *qSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&qSeq, batch * sizeof(int32_t)));
    int32_t *qSeq_int = static_cast<int32_t *>(qSeq);
    for (int i=0; i<batch; i++){
        qSeq_int[i] = qSeqLen;
    }

    void *kvSeq = nullptr;;
    ACL_CHECK(aclrtMallocHost(&kvSeq, batch * sizeof(int32_t)));
    int32_t *kvSeq_int = static_cast<int32_t *>(kvSeq);
    for (int i = 0; i < batch; i++) {
        kvSeq_int[i] = kv_seqlens[i];;
    }

    mlaKernelInfo.qSeqLen = qSeq_int;
    mlaKernelInfo.kvSeqLen = kvSeq_int;
    mlaKernelInfo.softmaxScale = softmax_scale;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    LaunchMLA(aicCoreNum, stream, mlaKernelInfo);

    // clean memory
    aclrtFreeHost(qSeq);
    aclrtFreeHost(kvSeq);

    return result;
}

using ConstTensorRef = const at::Tensor&;
std::vector<uint64_t> KernelGetKVSplit(
    ConstTensorRef q,
    ConstTensorRef q_rope,
    ConstTensorRef k,
    ConstTensorRef k_rope,
    const int32_t kv_seqlen,
    const std::vector<int32_t> &kv_seqlens,
    ConstTensorRef block_table,
    ConstTensorRef s,
    ConstTensorRef p,
    ConstTensorRef result_temp,
    ConstTensorRef global_o,
    ConstTensorRef l,
    ConstTensorRef o_core_tmp,
    const std::string &dtype_str
)
{
    torch::Dtype outputDataType = getTorchDtype(dtype_str);

    at::TensorOptions options = at::TensorOptions();
    options = options.dtype(outputDataType).layout(at::kStrided).requires_grad(false).device(
        torch_npu::utils::get_npu_device_type());

    // MLA output has same shape as q_nope
    torch::Tensor result =  at_npu::native::empty_with_format(
        q.sizes(), options, ACL_FORMAT_ND);

    const std::array<const at::Tensor, NUM_MLA_KERNEL_INPUTS> inputs = {
        q, q_rope, k, k_rope, block_table, s, p
    };
    const std::array<const at::Tensor, NUM_MLA_KERNEL_OUTPUTS> outputs = {
        result, result_temp, global_o, l, o_core_tmp
    };

    MLAKernelInfo mlaKernelInfo = ConstructMlaKernelInfo(inputs, outputs, kv_seqlen, dtype_str);

    int32_t batch = q.sizes().at(0); // Q shape: [batch, kv_heads, head_size]
    int32_t qSeqLen = 1;  // Assume no spec-dec & MTP

    // Assume equal length for qSeqLen and kvSeqLen
    void *qSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&qSeq, batch * sizeof(int32_t)));
    int32_t *qSeq_int = static_cast<int32_t *>(qSeq);
    for (int i=0; i<batch; i++){
        qSeq_int[i] = qSeqLen;
    }

    void *kvSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvSeq, batch * sizeof(int32_t)));
    int32_t *kvSeq_int = static_cast<int32_t *>(kvSeq);
    for (int i = 0; i < batch; i++) {
        kvSeq_int[i] = kv_seqlens[i];
    }

    mlaKernelInfo.qSeqLen = qSeq_int;
    mlaKernelInfo.kvSeqLen = kvSeq_int;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    std::vector<uint64_t> kernel_prep_params = GetKVSplitKernel(aicCoreNum, stream, mlaKernelInfo);

    // clean memory
    aclrtFreeHost(qSeq);
    aclrtFreeHost(kvSeq);

    return kernel_prep_params;
}

} // namespace Catlass
