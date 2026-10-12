/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_grouped_matmul_inplace_add_950_checker.h"

using namespace QGmmInPlaceAdd;

namespace QGmmInPlaceAdd {
template class AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<aclTensor>;
}

namespace {
const aclTensor *GetInputTensor(const aclTensorList *input, size_t index = 0)
{
    if (input == nullptr || index >= input->Size()) {
        return nullptr;
    }
    return (*input)[index];
}

const aclTensor *GetInputTensor(const aclTensor *input, size_t index = 0)
{
    (void)index;
    return input;
}

size_t GetInputTensorSize(const aclTensorList *input)
{
    if (input == nullptr) {
        return 0;
    }
    return input->Size();
}

size_t GetInputTensorSize(const aclTensor *input)
{
    (void)input;
    return 1;
}
} // namespace

template <typename T>
void AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::SetInputName(const std::string &xName,
                                                                      const std::string &weightName,
                                                                      const std::string &perTokenScaleName,
                                                                      const std::string &scaleName,
                                                                      const std::string &groupTensorName)
{
    this->xName_ = xName;
    this->weightName_ = weightName;
    this->perTokenScaleName_ = perTokenScaleName;
    this->scaleName_ = scaleName;
    this->groupTensorName_ = groupTensorName;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckTensorListSizeForEachInput() const
{
    CHECK_COND(GetInputTensorSize(gmmParams_.scaleOptional) == GetInputTensorSize(gmmParams_.x),
               ACLNN_ERR_PARAM_INVALID,
               "In quant case, the scale size must equal x size but scale size is [%zu] x size is [%zu].",
               GetInputTensorSize(gmmParams_.scaleOptional), GetInputTensorSize(gmmParams_.x));
    if (gmmParams_.perTokenScaleOptional != nullptr) {
        CHECK_COND(GetInputTensorSize(gmmParams_.perTokenScaleOptional) == GetInputTensorSize(gmmParams_.x),
                   ACLNN_ERR_PARAM_INVALID,
                   "In quant case, the per-token scale size must equal x size but scale per-token size is [%zu] x size "
                   "is [%zu].",
                   GetInputTensorSize(gmmParams_.perTokenScaleOptional), GetInputTensorSize(gmmParams_.x));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckGeneralQuantShape() const
{
    CHECK_RET(CheckTensorListSizeForEachInput() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        auto yNIndex = GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum() - 1;
        CHECK_COND(GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(0) > 0,
                    ACLNN_ERR_PARAM_INVALID, "In quant case, the K value[%ld] in %s should be positive.",
                    GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(0), weightName_.c_str());
         CHECK_COND(GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(1) > 0, ACLNN_ERR_PARAM_INVALID,
                    "In quant case, the K value[%ld] in %s should be positive.",
                    GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(1), xName_.c_str());
        CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(0) == groupNum, ACLNN_ERR_PARAM_INVALID,
                   "When groupType is 2 (split K), the first dim of %s[%ld] should be equal to that of \
%s[%ld].",
                   yName_.c_str(), GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(0), groupTensorName_.c_str(),
                   groupNum);
        // y shape dim num must 3
        CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID,
                   "The %s dim num should be equal 3, but actual dim num is [%zu]", yName_.c_str(),
                   GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum());
        CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(1) ==
                       GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(0),
                   ACLNN_ERR_PARAM_INVALID,
                   "The m dim of %s should be equal %s m dim, but actual %s m dim is [%ld], %s m dim is [%ld]",
                   yName_.c_str(), xName_.c_str(), yName_.c_str(),
                   GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(1), xName_.c_str(),
                   GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(0));
        CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(yNIndex) ==
                       GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex),
                   ACLNN_ERR_PARAM_INVALID,
                   "The n dim of %s should be equal %s n dim, but actual %s n dim is [%ld], %s n dim is [%ld]",
                   yName_.c_str(), weightName_.c_str(), yName_.c_str(),
                   GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(yNIndex), weightName_.c_str(),
                   GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckQuantCasesFormat() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.x, i)->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of %s[%zu] %s is invalid. It should only be ND.", xName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.x, i)->GetStorageFormat()).GetString());
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.weight, i)->GetStorageFormat()),
                   ACLNN_ERR_PARAM_INVALID, "The format of %s[%zu] %s is invalid. It should only be ND.",
                   weightName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.weight, i)->GetStorageFormat()).GetString());
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.y, i)->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of %s[%zu] %s is invalid. It should only be ND.", yName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.y, i)->GetStorageFormat()).GetString());
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::IsGmmInplaceAddTCQuantMode() const
{
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.weight); ++i) {
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto weightDimNumber = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum();
        // 2 is the dimension for scale in GmmInplaceAdd T-C case
        CHECK_COND(scaleDimNumber == 2 && weightDimNumber == scaleDimNumber, ACLNN_ERR_PARAM_INVALID,
                   "The %s dim should be 2 and dim equal to %s in hifloat8 case, but actual %s dim is [%zu], "
                   "%s dim is [%zu]",
                   scaleName_.c_str(), weightName_.c_str(), scaleName_.c_str(), scaleDimNumber, weightName_.c_str(),
                   weightDimNumber);
        auto xDimNumber = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDimNum();
        auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();
        // 2 is the maxmum dimension for perTokenScaleOptional in GmmInplaceAdd T-C case
        CHECK_COND(xDimNumber == 2 && perTokenDimNumber <= xDimNumber, ACLNN_ERR_PARAM_INVALID,
                   "The %s dim should be 2 and %s dim should <= %s in hifloat8 case, but actual %s dim is [%zu], "
                   "%s dim is [%zu]",
                   xName_.c_str(), perTokenScaleName_.c_str(), xName_.c_str(), xName_.c_str(), xDimNumber,
                   perTokenScaleName_.c_str(), perTokenDimNumber);
        auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(scaleDimNumber - 1);
        auto scaleGDim = GetInputTensor(gmmParams_.scaleOptional, i)
                             ->GetViewShape()
                             .GetDim(scaleDimNumber - gmm::LAST_TWO_DIM_INDEX);
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - 1);
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(0);
        CHECK_COND(scaleGDim == groupNum && scaleNDim == weightNDim, ACLNN_ERR_PARAM_INVALID,
                   "The %s G dim must equal with groupnum dim and %s N dim must equal with %s N dim in "
                   "hifloat8 case, but actual %s G dim is [%zu], %s N dim is [%zu], %s N dim is [%zu]",
                   scaleName_.c_str(), scaleName_.c_str(), weightName_.c_str(), scaleName_.c_str(), scaleGDim,
                   scaleName_.c_str(), scaleNDim, weightName_.c_str(), weightNDim);
        if (perTokenDimNumber != 1) {
            auto perTokenKDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(1);
            CHECK_COND(perTokenKDim == 1 && perTokenMDim == groupNum, ACLNN_ERR_PARAM_INVALID,
                       "The %s K dim should be 1 and %s M dim must equal groupnum in hifloat8 case, but "
                       "actual %s K dim is [%zu], %s M dim is [%zu], groupnum is [%zu]",
                       perTokenScaleName_.c_str(), perTokenScaleName_.c_str(), perTokenScaleName_.c_str(), perTokenKDim,
                       perTokenScaleName_.c_str(), perTokenMDim, groupNum);
        } else {
            CHECK_COND(perTokenMDim == groupNum, ACLNN_ERR_PARAM_INVALID,
                       "The %s M dim must equal groupnum in hifloat8 case, but actual %s M dim is [%zu], "
                       "groupnum dim is [%zu]",
                       perTokenScaleName_.c_str(), perTokenScaleName_.c_str(), perTokenMDim, groupNum);
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckHif8QuantParams() const
{
    CHECK_COND(gmmParams_.biasOptional == nullptr, ACLNN_ERR_PARAM_INVALID, "Hifloat8 case does not support bias.");
    CHECK_COND(gmmParams_.perTokenScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "Hifloat8 case perTokenScaleOptional not be null.");
    DataType perTokenScaleDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
    CHECK_COND(perTokenScaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "The %s dtype should be float32 in hifloat8 case, but actual dtype is %s", perTokenScaleName_.c_str(),
               op::ToString(perTokenScaleDtype).GetString());
    CHECK_COND(gmmParams_.y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Hifloat8 case y not be null.");
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    CHECK_COND(yDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "Expect yDtype to be float32 in hifloat8 quant case, but actual dtype is %s",
               op::ToString(yDtype).GetString());
    CHECK_RET(IsGmmInplaceAddTCQuantMode() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckQuantGroupedMatmulInplaceAddDAV3510() const
{
    DataType xDtype = gmmParams_.xDtype;
    CHECK_COND(gmmParams_.weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In quant case, weight should not be nullptr.");
    DataType weightDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    CHECK_COND(gmmParams_.scaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In quant case, scaleOptional should not be nullptr.");
    CHECK_COND(gmmParams_.groupTensorOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In quant case, groupListOptional should not be nullptr.");
    CHECK_COND(gmmParams_.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID, "Quant case does not support offset.");
    CHECK_COND(GetInputTensorSize(gmmParams_.x) == 1 && GetInputTensorSize(gmmParams_.weight) == 1 &&
                   GetInputTensorSize(gmmParams_.y) == 1,
               ACLNN_ERR_PARAM_INVALID,
               "In quant case, the size of x, weight and y should all be 1, but actual sizes are %zu, %zu and %zu.",
               GetInputTensorSize(gmmParams_.x), GetInputTensorSize(gmmParams_.weight),
               GetInputTensorSize(gmmParams_.y));
    int64_t groupListLen = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    CHECK_COND(groupListLen <= 1024, ACLNN_ERR_PARAM_INVALID, // The group number should not be greater than 1024
                "The length of groupList should not be greater than 1024, but actual is %ld.", groupListLen);
    CHECK_RET(CheckQuantCasesFormat() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGeneralQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    if (xDtype == DataType::DT_HIFLOAT8 && weightDtype == DataType::DT_HIFLOAT8) {
        CHECK_COND(scaleDtype == DataType::DT_FLOAT || scaleDtype == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
                   "With hifloat8 inputs, scale dtype should be float8_e8m0 or float32, but actual dtype is %s",
                   op::ToString(scaleDtype).GetString());
        return CheckHif8QuantParams();
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GmmInplaceAdd T-C Quant case with x dtype %s and weight dtype %s is not supported.",
                op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
}