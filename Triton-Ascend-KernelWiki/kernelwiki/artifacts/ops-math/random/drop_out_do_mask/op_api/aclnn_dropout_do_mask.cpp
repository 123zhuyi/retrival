/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_dropout_do_mask.h"
#include "aclnn_kernels/cast.h"
#include "random/dsa_gen_bit_mask/op_host/op_api/dsa_gen_bit_mask.h"
#include "aclnn_kernels/contiguous.h"
#include "dropout_do_mask.h"
#include "random/stateless_drop_out_gen_mask/op_api/stateless_dropout_gen_mask.h"
#include "math/zero_op/op_api/zero_op.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
static const int64_t BIT_NUMBER = 128;
static const int64_t UINT8_BIT_NUMBER = 8;
static const int64_t FLOAT_BYTE = 4;
static const int64_t FLOAT_BIT = 32;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {op::DataType::DT_UINT8};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* mask, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510: {
            return ASCEND910B_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* mask, const aclTensor* out)
{
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);

    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    // 检查input的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查mask的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(mask, MASK_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool IsDoubleEqual(double f1, double f2)
{
    return std::abs(f1 - f2) <= std::numeric_limits<double>::epsilon();
}

static inline bool CheckProbability(double prob)
{
    if (prob > 1 || prob < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of prob has to be between 0 and 1, but current is %f.", prob);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* mask, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    int64_t inputSize = self->GetViewShape().GetShapeSize();
    int64_t maskSize = mask->GetViewShape().GetShapeSize();
    int64_t needMaskSizde = (inputSize + BIT_NUMBER - 1) / BIT_NUMBER * BIT_NUMBER / UINT8_BIT_NUMBER;
    if (maskSize < needMaskSizde) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of mask has to be %ld, but current is %ld.", needMaskSizde, maskSize);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor* self, const aclTensor* mask, double prob, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mask, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, mask, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查p是否符合规则
    CHECK_RET(CheckProbability(prob), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出输出shape
    CHECK_RET(CheckShape(self, mask, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static inline const aclIntArray* GenerateShapeIntArray(const op::Shape& shape, aclOpExecutor* executor)
{
    FVector<int64_t> shapeVector;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        shapeVector.push_back(shape.GetDim(i));
    }
    return executor->AllocIntArray(shapeVector.data(), shapeVector.size());
}

static inline const aclTensor* DoMask(
    const aclTensor* inputContiguous, const aclTensor* mask, double prob, aclOpExecutor* executor)
{
    if (IsDoubleEqual(prob, 0.0)) {
        return inputContiguous;
    } else if (IsDoubleEqual(prob, 1.0)) {
        return l0op::ZerosLike(inputContiguous, executor);
    } else {
        FVector<float> probVector = {static_cast<float>(1 - prob)};
        auto probTensor =
            executor->ConvertToTensor(probVector.data(), probVector.size(), inputContiguous->GetDataType());
        return l0op::DropoutDoMask(inputContiguous, mask, probTensor, executor);
    }
}

aclnnStatus aclnnDropoutDoMaskGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mask, double prob, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDropoutDoMask, DFX_IN(self, mask, prob), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    
    if (mask->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    
    // 固定写法，参数检查
    auto ret = CheckParams(self, mask, prob, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，将输入input转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto maskContiguous = l0op::Contiguous(mask, uniqueExecutor.get());
    CHECK_RET(maskContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 进行do mask
    auto doMaskOut = DoMask(inputContiguous, maskContiguous, prob, uniqueExecutor.get());
    CHECK_RET(doMaskOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(doMaskOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDropoutDoMask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDropoutDoMask);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
