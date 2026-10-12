/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bernoulli.h"

#include <cstdint>

#include "random/dsa_gen_bit_mask/op_host/op_api/dsa_gen_bit_mask.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Bernoulli);

namespace {

constexpr int64_t kMaskModeRandomAliased = 0;
constexpr int64_t kMaskModeZero = 1;
constexpr int64_t kMaskModeOne = 2;
constexpr uint64_t kMaskAlignmentBits = 128U;
constexpr uint64_t kBitsPerByte = 8U;

uint64_t GetDataTypeBytes(op::DataType dtype)
{
    switch (dtype) {
        case op::DataType::DT_UINT8:
        case op::DataType::DT_INT8:
        case op::DataType::DT_BOOL:
            return 1U;
        case op::DataType::DT_FLOAT16:
        case op::DataType::DT_BF16:
        case op::DataType::DT_INT16:
            return 2U;
        case op::DataType::DT_FLOAT:
        case op::DataType::DT_INT32:
            return 4U;
        case op::DataType::DT_DOUBLE:
        case op::DataType::DT_INT64:
            return 8U;
        default:
            return 0U;
    }
}

aclTensor* AllocateOutput(const aclTensor* input, uint64_t minStorageElements, aclOpExecutor* executor)
{
    const uint64_t logicalElements = static_cast<uint64_t>(input->Numel());
    if (minStorageElements <= logicalElements) {
        return executor->AllocTensor(input->GetViewShape(), input->GetDataType(), input->GetViewFormat());
    }

    // CreateView checks its shape before the UINT8 reinterpretation. DSA's
    // minimum 128-bit block therefore needs 16 elements of output storage,
    // while the public view shape remains unchanged.
    return executor->AllocTensor(op::Shape{static_cast<int64_t>(minStorageElements)}, input->GetViewShape(),
                                 input->GetDataType(), op::Format::FORMAT_ND, input->GetViewFormat());
}

aclTensor* CreateMaskAlias(aclTensor* output, uint64_t maskBytes, aclOpExecutor* executor)
{
    const auto& storageShape = output->GetStorageShape();
    uint64_t storageElements = 1;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        const int64_t dim = storageShape.GetDim(i);
        if (dim < 0 || static_cast<uint64_t>(dim) > UINT64_MAX / storageElements) {
            return nullptr;
        }
        storageElements *= static_cast<uint64_t>(dim);
    }
    const uint64_t typeBytes = GetDataTypeBytes(output->GetDataType());
    if (typeBytes == 0U || storageElements > UINT64_MAX / typeBytes || maskBytes > storageElements * typeBytes) {
        return nullptr;
    }
    auto mask = executor->CreateView(output, op::Shape{static_cast<int64_t>(maskBytes)}, output->GetViewOffset());
    if (mask != nullptr) {
        mask->SetDataType(op::DataType::DT_UINT8);
    }
    return mask;
}

const aclTensor* LaunchBernoulli(const aclTensor* input, const aclTensor* mask, int64_t mode, aclTensor* output,
                                 aclOpExecutor* executor)
{
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Bernoulli, OP_INPUT(input, mask), OP_ATTR(mode), OP_OUTPUT(output));
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Bernoulli launch failed"), return nullptr);
    return output;
}

} // namespace

const aclTensor* BernoulliRandom(const aclTensor* input, double probability, int64_t seed, int64_t offset,
                                 aclTensor* directOut, aclOpExecutor* executor)
{
    L0_DFX(BernoulliRandom, input, probability, seed, offset);
    const uint64_t elementCount = static_cast<uint64_t>(input->Numel());
    const uint64_t alignedBits = ((elementCount + kMaskAlignmentBits - 1U) / kMaskAlignmentBits) * kMaskAlignmentBits;
    const uint64_t maskBytes = alignedBits / kBitsPerByte;
    auto output = directOut != nullptr ? directOut : AllocateOutput(input, maskBytes, executor);
    if (output == nullptr) {
        return nullptr;
    }

    auto mask = CreateMaskAlias(output, maskBytes, executor);
    if (mask == nullptr) {
        return nullptr;
    }

    auto dropout = executor->AllocScalar(static_cast<float>(1.0 - probability));
    if (dropout == nullptr) {
        return nullptr;
    }
    l0op::DSAGenBitMask(alignedBits, seed, offset, dropout, mask, executor);
    return LaunchBernoulli(input, mask, kMaskModeRandomAliased, output, executor);
}

const aclTensor* BernoulliConstant(const aclTensor* input, bool value, aclOpExecutor* executor)
{
    L0_DFX(BernoulliConstant, input, value);
    auto output = AllocateOutput(input, 1U, executor);
    if (output == nullptr) {
        return nullptr;
    }
    auto mask = CreateMaskAlias(output, 1U, executor);
    if (mask == nullptr) {
        return nullptr;
    }
    return LaunchBernoulli(input, mask, value ? kMaskModeOne : kMaskModeZero, output, executor);
}

} // namespace l0op
