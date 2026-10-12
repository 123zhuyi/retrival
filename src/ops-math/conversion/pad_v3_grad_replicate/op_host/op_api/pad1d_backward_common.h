/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PAD1D_BACKWARD_COMMON_H
#define PAD1D_BACKWARD_COMMON_H

#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "../../../squeeze/op_host/op_api/squeeze.h"
#include "../../../unsqueeze/op_host/op_api/unsqueeze.h"
#include "../../../pad_v3_grad/op_api/padv3grad.h"
#include "op_api/aclnn_check.h"

namespace op {

// Pad1d backward mode: REFLECT for reflection_pad, REPLICATE for replication_pad
enum class Pad1dBackwardMode { REFLECT, REPLICATE };

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> kPad1dBwdDtypeSupportList = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16,   op::DataType::DT_BF16,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const int64_t kPad1dReflectAicpuShape = 3000;
static const int64_t kPad1dReplicatePaddingFp32Max = 7200;

inline static bool Pad1dBwdCheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding,
                                        const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

inline static bool Pad1dBwdCheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self,
                                           const aclTensor* gradInput)
{
    // 检查gradOutput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, kPad1dBwdDtypeSupportList, return false);

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, kPad1dBwdDtypeSupportList, return false);

    // 检查gradInput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, kPad1dBwdDtypeSupportList, return false);

    // gradOutput, self和gradInput数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradInput, self->GetDataType(), return false);
    return true;
}

inline static bool Pad1dBwdCheckFormat(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(gradOutput->GetStorageFormat()) || op::IsPrivateFormat(self->GetStorageFormat()) ||
        op::IsPrivateFormat(gradInput->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND.");
        return false;
    }

    OP_CHECK(
        gradOutput->GetViewFormat() == self->GetViewFormat() &&
            gradOutput->GetViewFormat() == gradInput->GetViewFormat(),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format of input and output should be equal, gradOutput [%s], self [%s], gradInput [%s].",
                op::ToString(gradOutput->GetViewFormat()).GetString(), op::ToString(self->GetViewFormat()).GetString(),
                op::ToString(gradInput->GetViewFormat()).GetString()),
        return false);
    return true;
}

// 公共shape校验（不含mode相关的padding范围校验）
static bool Pad1dBwdCheckShapeCommon(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding,
                                     const aclTensor* gradInput)
{
    auto selfDimnum = self->GetViewShape().GetDimNum();
    // self和gradInput的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, gradInput, return false);

    // 2 and 3 are dims, self只支持2维和3维
    OP_CHECK_MIN_DIM(self, 2, return false);
    OP_CHECK_MAX_DIM(self, 3, return false);

    // gradOutput, self, gradInput维度需要一致
    OP_CHECK(
        gradOutput->GetViewShape().GetDimNum() == selfDimnum && gradInput->GetViewShape().GetDimNum() == selfDimnum,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput, self, gradInput dim should be same."), return false);

    // padding长度为2
    OP_CHECK(padding->Size() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 2, but got %lu.", padding->Size()),
             return false);

    // check the last dim value of gradOutput. 0, 1 are indexes
    OP_CHECK(gradOutput->GetViewShape().GetDim(selfDimnum - 1) ==
                 self->GetViewShape().GetDim(selfDimnum - 1) + (*padding)[0] + (*padding)[1],
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "gradOutput shape does not match the expected shape derived from input and padding."),
             return false);
    return true;
}

static bool Pad1dBwdCheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding,
                               const aclTensor* gradInput, Pad1dBackwardMode mode)
{
    if (!Pad1dBwdCheckShapeCommon(gradOutput, self, padding, gradInput)) {
        return false;
    }

    // reflect模式额外校验: padding值需小于self最后一维度
    if (mode == Pad1dBackwardMode::REFLECT) {
        auto selfDimnum = self->GetViewShape().GetDimNum();
        OP_CHECK((*padding)[0] < self->GetViewShape().GetDim(selfDimnum - 1) &&
                     (*padding)[1] < self->GetViewShape().GetDim(selfDimnum - 1),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding size should be less than the corresponding self dimension."),
                 return false);
    }
    return true;
}

inline static aclnnStatus Pad1dBwdCheckParams(const aclTensor* gradOutput, const aclTensor* self,
                                              const aclIntArray* padding, const aclTensor* gradInput,
                                              Pad1dBackwardMode mode)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(Pad1dBwdCheckNotNull(gradOutput, self, padding, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(Pad1dBwdCheckDtypeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(Pad1dBwdCheckFormat(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(Pad1dBwdCheckShape(gradOutput, self, padding, gradInput, mode), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* Pad1dBwdGetPaddingTensor(int64_t dim, const aclIntArray* padding, aclOpExecutor* executor)
{
    FVector<int64_t, op::MAX_DIM_NUM> paddingsVector;
    // 2 is the magnification
    for (size_t i = 2 * dim; i > 0; i -= 2) {
        if (i <= (size_t)padding->Size()) {
            // 2 and 1 indicate the element of padding is put into paddingsVector from the back to the front
            paddingsVector.emplace_back((*padding)[i - 2]);
            paddingsVector.emplace_back((*padding)[i - 1]);
        } else {
            paddingsVector.emplace_back(0);
            paddingsVector.emplace_back(0);
        }
    }
    // 2 is the magnification
    auto newpadding = executor->AllocIntArray(paddingsVector.data(), 2 * dim);
    auto paddingsTensor = executor->ConvertToTensor(newpadding, static_cast<op::DataType>(ACL_INT64));
    return paddingsTensor;
}

// 校验padding值，根据mode采用不同的上限策略:
// REFLECT: padding值不超过kPad1dReflectAicpuShape (3000)
// REPLICATE: FP32类型且非RegBase时，padding值不超过kPad1dReplicatePaddingFp32Max (7200)
static bool Pad1dBwdCheckPaddingValue(const aclIntArray* padding, const aclTensor* gradOutput, Pad1dBackwardMode mode)
{
    // padding的每一维度的数值要大于等于0
    if ((*padding)[0] < 0 || (*padding)[1] < 0) {
        OP_LOGW("in AICore situation, padding values should be greater than or equal to 0.");
        return false;
    }

    if (mode == Pad1dBackwardMode::REFLECT) {
        if ((*padding)[0] > kPad1dReflectAicpuShape || (*padding)[1] > kPad1dReflectAicpuShape) {
            OP_LOGW(
                "in AICore situation, padding values should be greater than or equal to 0 and less than or equal to "
                "the shape limit value %ld.",
                kPad1dReflectAicpuShape);
            return false;
        }
    } else {
        // fp32类型下，AtlasA2 padding最多不超过7200
        if (!IsRegBase() && gradOutput->GetDataType() == op::DataType::DT_FLOAT &&
            ((*padding)[0] >= kPad1dReplicatePaddingFp32Max || (*padding)[1] >= kPad1dReplicatePaddingFp32Max)) {
            OP_LOGW("in AICore situation, padding values should be less than %ld.", kPad1dReplicatePaddingFp32Max);
            return false;
        }
    }
    return true;
}

static aclnnStatus Pad1dBwdInputPreprocess(const aclTensor*& gradOutput, const aclTensor*& self,
                                           const aclIntArray* dimArray, int64_t dimCp, aclOpExecutor* executor)
{
    // 如果非连续，需要转连续
    gradOutput = l0op::Contiguous(gradOutput, executor);
    CHECK_RET(gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    self = l0op::Contiguous(self, executor);
    CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    self = l0op::UnsqueezeNd(self, dimArray, executor);
    gradOutput = l0op::UnsqueezeNd(gradOutput, dimArray, executor);
    // 2 is dim
    if (dimCp == 2) {
        self = l0op::UnsqueezeNd(self, dimArray, executor);
        gradOutput = l0op::UnsqueezeNd(gradOutput, dimArray, executor);
    }
    CHECK_RET(gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// 空tensor处理，根据mode在dim==2时采用不同的校验策略:
// REFLECT: 检查GetDim(1)是否为0
// REPLICATE: 直接报错
static aclnnStatus Pad1dBwdHandleEmptyTensor(const aclTensor* self, Pad1dBackwardMode mode)
{
    // 2 is dim number
    if (self->GetViewShape().GetDimNum() == 2) {
        if (mode == Pad1dBackwardMode::REFLECT) {
            // 1 is index
            if (self->GetViewShape().GetDim(1) == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input should not be empty.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input should not be empty.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 3 is dim number
    if (self->GetViewShape().GetDimNum() == 3) {
        // 1, 2 are indexes
        if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input should not be empty.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

// Pad1d backward公共计算逻辑
// mode: REFLECT对应reflection_pad, REPLICATE对应replication_pad
// modeStr: l0op::PadV3Grad的mode字符串 ("reflect" 或 "edge")
// logTag: 日志标签 (如 "[PadV4Grad]" 或 "[PadV3Grad]")
static aclnnStatus Pad1dBackwardCompute(const aclTensor*& gradOutput, const aclTensor*& self,
                                        const aclIntArray* padding, aclTensor* gradInput, uint64_t* workspaceSize,
                                        aclOpExecutor** executor, Pad1dBackwardMode mode, const std::string& modeStr,
                                        const char* logTag)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = Pad1dBwdCheckParams(gradOutput, self, padding, gradInput, mode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (gradOutput->IsEmpty() || self->IsEmpty() || gradInput->IsEmpty()) {
        *workspaceSize = 0;
        auto emptyRet = Pad1dBwdHandleEmptyTensor(self, mode);
        CHECK_RET(emptyRet == ACLNN_SUCCESS, emptyRet);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用l0算子来进行计算
    auto dim = self->GetViewShape().GetDimNum();
    auto dimCp = dim;
    // 0 is index
    const int64_t appendDim[] = {0};
    // 1 is the dim num to be unsqueezed
    aclIntArray* dimArray = (uniqueExecutor.get())->AllocIntArray(appendDim, 1);
    ret = Pad1dBwdInputPreprocess(gradOutput, self, dimArray, dimCp, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    dim = self->GetViewShape().GetDimNum();
    auto paddingsTensor = Pad1dBwdGetPaddingTensor(dim, padding, uniqueExecutor.get());
    auto padFlag = Pad1dBwdCheckPaddingValue(padding, gradOutput, mode);
    const aclTensor* pad1dbackwardResult = nullptr;
    auto originOutDataType = gradOutput->GetDataType();
    // cast to fp32 from fp16 or bf16
    // REFLECT: always cast (reconstruct pre-dedup behavior, aicpu needs fp32)
    // REPLICATE: only cast when padFlag (preserve original behavior)
    if ((mode == Pad1dBackwardMode::REFLECT || padFlag) &&
        (originOutDataType == op::DataType::DT_FLOAT16 || originOutDataType == op::DataType::DT_BF16)) {
        gradOutput = l0op::Cast(gradOutput, op::DataType::DT_FLOAT, uniqueExecutor.get());
        OP_LOGD("%s FP16 or BF16 Cast to FP32: true", logTag);
    }

    pad1dbackwardResult = l0op::PadV3Grad(gradOutput, paddingsTensor, modeStr, true, padFlag, uniqueExecutor.get());
    CHECK_RET(pad1dbackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    pad1dbackwardResult = l0op::SqueezeNd(pad1dbackwardResult, dimArray, uniqueExecutor.get());
    // 2 is dim
    if (dimCp == 2) {
        pad1dbackwardResult = l0op::SqueezeNd(pad1dbackwardResult, dimArray, uniqueExecutor.get());
    }
    CHECK_RET(pad1dbackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // cast to fp16 or bf16
    if ((mode == Pad1dBackwardMode::REFLECT || padFlag) &&
        (originOutDataType == op::DataType::DT_FLOAT16 || originOutDataType == op::DataType::DT_BF16)) {
        pad1dbackwardResult = l0op::Cast(pad1dbackwardResult, originOutDataType, uniqueExecutor.get());
        OP_LOGD("%s FP16 or BF16 Cast to FP32: true", logTag);
    }

    // 如果出参gradInput是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(pad1dbackwardResult, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace op

#endif // PAD1D_BACKWARD_COMMON_H
