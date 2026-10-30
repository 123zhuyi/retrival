/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_d.cpp
 * \brief
 */
#include "concat_d.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

#define NUM_32 32
#define NUM_2048 2048

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(ConcatD);
OP_TYPE_REGISTER(ConcatDV2);

std::map<op::DataType, int64_t> type_size = {
    {DataType::DT_FLOAT, 4},      {DataType::DT_INT32, 4},  {DataType::DT_INT64, 8},  {DataType::DT_FLOAT16, 2},
    {DataType::DT_INT16, 2},      {DataType::DT_INT8, 1},   {DataType::DT_UINT8, 1},  {DataType::DT_DOUBLE, 8},
    {DataType::DT_COMPLEX64, 8},  {DataType::DT_BF16 ,2},   {DataType::DT_BOOL, 1}};

bool IsSupportConcatDV2(const aclTensorList* inputs, int64_t dim)
{
    CHECK_RET(inputs != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != op::SocVersion::ASCEND910_93 && socVersion != op::SocVersion::ASCEND910B) {
        return false;
    }
    if (dim != 0){
        return false;
    }
    if (inputs->Size() > NUM_2048 || inputs->Size() <= NUM_32) {
        return false;
    }

    auto promoteType = (*inputs)[0]->GetDataType();
    for (uint64_t i = 0; i < inputs->Size(); i++) {
        op::Shape shape = (*inputs)[i]->GetViewShape();
        int64_t dim_num = shape.GetDimNum();
        int64_t tail_dim = shape.GetDim(dim_num - 1); // 获取尾轴维度
        if (tail_dim * type_size[promoteType] % NUM_32 != 0) {
            return false;
        }
    }

    return true;
}

static op::Shape ConcatDInferShape(const aclTensorList* inputs, int64_t dim)
{
    op::Shape concatShape = (*inputs)[0]->GetViewShape();
    int64_t concatDimSize = 0;
    for (uint64_t i = 0; i < inputs->Size(); i++) {
        concatDimSize += (*inputs)[i]->GetViewShape().GetDim(dim);
    }
    concatShape.SetDim(dim, concatDimSize);
    return concatShape;
}

aclTensor* ConcatD(const aclTensorList* inputs, int64_t dim, op::DataType outDtype, aclOpExecutor* executor)
{
    L0_DFX(ConcatD, inputs, dim, outDtype);
    op::Shape concatShape = ConcatDInferShape(inputs, dim);
    auto out = executor->AllocTensor(concatShape, outDtype, (*inputs)[0]->GetViewFormat());

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ConcatD, OP_INPUT(inputs), OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ConcatD ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

aclTensor* ConcatD(const aclTensorList* inputs, int64_t dim, aclOpExecutor* executor)
{
    L0_DFX(ConcatD, inputs, dim);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    size_t catMaxInputSize = (IsRegBase(npuArch)) ? NUM_2048 : NUM_32;

    if (IsSupportConcatDV2(inputs, dim)) {
        catMaxInputSize = NUM_2048;
    }

    if (inputs->Size() == 0 || inputs->Size() > catMaxInputSize) {
        OP_LOGE(
            ACLNN_ERR_INNER, "Inputs tensor list's size should be (0, %zu]) but current size is %zu.", catMaxInputSize,
            inputs->Size());
        return nullptr;
    }
    op::Shape concatShape = ConcatDInferShape(inputs, dim);
    auto out = executor->AllocTensor(concatShape, (*inputs)[0]->GetDataType(), (*inputs)[0]->GetViewFormat());
    if (IsSupportConcatDV2(inputs, dim)) {
        auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ConcatDV2, OP_INPUT(inputs), OP_OUTPUT(out), OP_ATTR(dim));
        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
            retAicore != ACLNN_SUCCESS, return nullptr, "ConcatDV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    } else {
        auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ConcatD, OP_INPUT(inputs), OP_OUTPUT(out), OP_ATTR(dim));
        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
            retAicore != ACLNN_SUCCESS, return nullptr, "ConcatD ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
    return out;
}

} // namespace l0op
