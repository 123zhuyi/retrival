/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

namespace l0op {
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;
constexpr int32_t DIM3 = 3;

OP_TYPE_REGISTER(Transpose);
OP_TYPE_REGISTER(TransposeV2);

static const std::initializer_list<op::DataType> TRANSPOSEV2_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,    op::DataType::DT_FLOAT16,     op::DataType::DT_INT8,         op::DataType::DT_INT16,
    op::DataType::DT_INT32,    op::DataType::DT_INT64,       op::DataType::DT_UINT8,        op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,   op::DataType::DT_UINT64,      op::DataType::DT_BOOL,         op::DataType::DT_BF16,
    op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN};

static bool IsTransposeV2021Support(const aclTensor* self, const aclIntArray* perm)
{
    // 2 is perm value
    if ((*perm)[DIM0] == 0 && (*perm)[DIM1] == 2 && (*perm)[DIM2] == 1) {
        // 128 is limit
        if (self->GetViewShape().GetDim(DIM1) <= 128 && self->GetViewShape().GetDim(DIM2) <= 128) {
            return true;
        }
    }
    return false;
}

static bool IsTransposeV2102Support(const aclTensor* self, const aclIntArray* perm)
{
    // 64 is limitSize of fp16 and bf16, 32 is limitSize of fp32
    int32_t limitSize = self->GetDataType() == op::DataType::DT_FLOAT ? 32 : 64;
    if ((*perm)[DIM0] == 1 && (*perm)[DIM1] == 0 && (*perm)[DIM2] == 2) { // 2 is perm value
        if ((self->GetViewShape().GetDim(DIM2) < limitSize ||
             // 256 is large limit
             (self->GetViewShape().GetDim(DIM2) >= limitSize && self->GetViewShape().GetDim(DIM2) <= 256 &&
              // 64 is copy limit
              (self->GetViewShape().GetDim(DIM0) <= 64 || self->GetViewShape().GetDim(DIM1) <= 64)))) {
            return true;
        }
    }
    return false;
}

static bool IsTransposeV20213Support(const aclTensor* self, const aclIntArray* perm)
{
    // 64 is limitSize of fp16 and bf16, 32 is limitSize of fp32
    int32_t typeBlock = self->GetDataType() == op::DataType::DT_FLOAT ? 8 : 16;
    // 2/3 is perm value
    if ((*perm)[DIM0] == 0 && (*perm)[DIM1] == 2 && (*perm)[DIM2] == 1 && (*perm)[DIM3] == 3) {
        // 64 is limit
        if (self->GetViewShape().GetDim(DIM3) <= 64 && self->GetViewShape().GetDim(DIM3) % typeBlock == 0 &&
            // 1024/2048 is limit of dim0
            self->GetViewShape().GetDim(DIM0) >= 1024 && self->GetViewShape().GetDim(DIM0) <= 2048 &&
            // 64 is copy limit
            (self->GetViewShape().GetDim(DIM1) <= 64 || self->GetViewShape().GetDim(DIM2) <= 64)) {
            return true;
        }
    }
    return false;
}

static bool IsTransposeV2AiCoreSupport(const aclTensor* self, const aclIntArray* perm)
{
    uint64_t permSize = perm->Size();
    bool IsSupport = false;
    // 3 is permSize
    if (permSize == 3U && self->GetViewShape().GetDim(DIM0) != 1 && self->GetViewShape().GetDim(DIM1) != 1 &&
        self->GetViewShape().GetDim(DIM2) != 1) {
        IsSupport = (IsTransposeV2021Support(self, perm) || IsTransposeV2102Support(self, perm));
        // 4 is permSize
    } else if (permSize == 4U && self->GetViewShape().GetDim(DIM0) != 1 && self->GetViewShape().GetDim(DIM1) != 1 &&
               self->GetViewShape().GetDim(DIM2) != 1 && self->GetViewShape().GetDim(DIM3) != 1) {
        IsSupport = IsTransposeV20213Support(self, perm);
    }
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B ||
        op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_93) {
        return (IsSupport && op::CheckType(self->GetDataType(), TRANSPOSEV2_AICORE_DTYPE_SUPPORT_LIST));
    }
    return false;
}

const aclTensor* TransposeV2AiCore(const aclTensor* x, const aclTensor* y, const aclTensor* perm,
                                   aclOpExecutor* executor)
{
    L0_DFX(TransposeV2AiCore, x, y, perm);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(TransposeV2, OP_INPUT(x, perm), OP_OUTPUT(y));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "TransposeV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return y;
}

static bool IsAiCoreSupport(const aclTensor* self)
{
    return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* TransposeAiCore(const aclTensor* x, const aclTensor* y, const aclTensor* perm, aclOpExecutor* executor)
{
    L0_DFX(TransposeAiCore, x, y, perm);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Transpose, OP_INPUT(x, perm), OP_OUTPUT(y));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "Transpose ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return y;
}

const aclTensor* TransposeAiCpu(const aclTensor* x, const aclTensor* y, const aclTensor* perm, aclOpExecutor* executor)
{
    L0_DFX(TransposeAiCpu, x, y, perm);
    static op::internal::AicpuTaskSpace space("Transpose", ge::DEPEND_CONST_VALUE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Transpose, OP_ATTR_NAMES({"T", "Tperm"}), OP_INPUT(x, perm), OP_OUTPUT(y),
                                          OP_ATTR(x->GetDataType(), perm->GetDataType()));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return y;
}

const aclTensor* Transpose(const aclTensor* x, const aclTensor* y, const aclTensor* perm, aclOpExecutor* executor)
{
    if (IsAiCoreSupport(x)) {
        return TransposeAiCore(x, y, perm, executor);
    }
    return TransposeAiCpu(x, y, perm, executor);
}

const aclTensor* Transpose(const aclTensor* x, const aclIntArray* perm, aclOpExecutor* executor)
{
    if (!op::IsContiguous(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor should be contiguous.");
        return nullptr;
    }

    auto permTensor = executor->ConvertToTensor(perm, op::ToOpDataType(ACL_INT64));
    auto out = executor->AllocTensor(x->GetDataType(), static_cast<op::Format>(x->GetStorageFormat()),
                                     x->GetOriginalFormat());
    INFER_SHAPE(Transpose, OP_INPUT(x, permTensor), OP_OUTPUT(out), OP_EMPTY_ARG);
    if (IsTransposeV2AiCoreSupport(x, perm)) {
        return TransposeV2AiCore(x, out, permTensor, executor);
    }

    return Transpose(x, out, permTensor, executor);
}

} // namespace l0op
