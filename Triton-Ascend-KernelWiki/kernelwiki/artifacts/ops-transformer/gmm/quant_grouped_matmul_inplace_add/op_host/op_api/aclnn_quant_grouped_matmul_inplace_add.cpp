/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_quant_grouped_matmul_inplace_add.h"

#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#include "../../../grouped_matmul/op_host/op_api/grouped_matmul_util.h"
#include "../../../grouped_matmul/op_host/op_api/grouped_matmul_950_checker.h"
#include "quant_grouped_matmul_inplace_add_util.h"
#include "quant_grouped_matmul_inplace_add.h"
#include "quant_grouped_matmul_inplace_add_950_checker.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static aclnnStatus CheckNotNull(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    CHECK_COND(params.x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x1 must not be nullptr.");
    CHECK_COND(params.x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x2 must not be nullptr.");
    CHECK_COND(params.scale2 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale2 must not be nullptr.");
    CHECK_COND(params.groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "groupList must not be nullptr.");
    CHECK_COND(params.yRef != nullptr, ACLNN_ERR_PARAM_NULLPTR, "yRef must not be nullptr.");
    CHECK_COND(params.scale1Optional != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale1Optional must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    CHECK_COND(params.x1->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "Format of x1 should be ND, current format is invalid.");
    CHECK_COND(params.x2->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "Format of x2 should be ND, current format is invalid.");
    CHECK_COND(params.scale2->GetStorageFormat() == Format::FORMAT_ND ||
                   params.scale2->GetStorageFormat() == Format::FORMAT_NCL,
               ACLNN_ERR_PARAM_INVALID, "Format of scale2 should be ND or NCL, current format is invalid.");
    CHECK_COND(params.groupList->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "Format of groupList should be ND, current format is invalid.");
    CHECK_COND(params.yRef->GetStorageFormat() == Format::FORMAT_ND ||
                   params.yRef->GetStorageFormat() == Format::FORMAT_NCL,
               ACLNN_ERR_PARAM_INVALID, "Format of yRef should be ND or NCL, current format is invalid.");
    CHECK_COND(params.scale1Optional->GetStorageFormat() == Format::FORMAT_ND ||
                   params.scale1Optional->GetStorageFormat() == Format::FORMAT_NCL,
               ACLNN_ERR_PARAM_INVALID, "Format of scale1 should be ND or NCL, current format is invalid.");
    return ACLNN_SUCCESS;
}

static aclnnStatus IsTcQuant(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    auto x1ScaleDimNum = params.scale1Optional->GetViewShape().GetDimNum();
    CHECK_COND(x1ScaleDimNum == 1 || x1ScaleDimNum == 2, ACLNN_ERR_PARAM_INVALID, // 2 max dim num in T-C quant
               "The dimension of scale1 should be 1 or 2 in T-C quant mode, but actual is %zu.", x1ScaleDimNum);
    auto x2ScaleDimNum = params.scale2->GetViewShape().GetDimNum();
    CHECK_COND(x2ScaleDimNum == 2, ACLNN_ERR_PARAM_INVALID, // 2 max dim num in T-C quant
               "The dimension of scale2 should be 2 in T-C quant mode, but actual is %zu.", x2ScaleDimNum);
    auto nDim = params.x2->GetViewShape().GetDim(1);
    auto g = params.groupList->GetViewShape().GetDim(0);
    auto x1ScaleLastDim = params.scale1Optional->GetViewShape().GetDim(x1ScaleDimNum - 1);
    auto x1ScaleFirstDim = params.scale1Optional->GetViewShape().GetDim(0);
    if (x1ScaleDimNum == 1) {
        CHECK_COND(x1ScaleFirstDim == g, ACLNN_ERR_PARAM_INVALID,
                   "In T-C quant mode, the expected shape of scale1 is (%ld, ) or (%ld, 1), \
but the actual is (%ld, ).",
                   g, g, x1ScaleFirstDim);
    } else {
        CHECK_COND(x1ScaleFirstDim == g && x1ScaleLastDim == 1, ACLNN_ERR_PARAM_INVALID,
                   "In T-C quant mode, the expected shape of scale1 is (%ld, ) or (%ld, 1), \
but the actual is (%ld, %ld).",
                   g, g, x1ScaleFirstDim, x1ScaleLastDim);
    }

    auto x2ScaleLastDim = params.scale2->GetViewShape().GetDim(x2ScaleDimNum - 1);
    auto x2ScaleFirstDim = params.scale2->GetViewShape().GetDim(0);
    CHECK_COND(x2ScaleFirstDim == g && x2ScaleLastDim == nDim, ACLNN_ERR_PARAM_INVALID,
               "In T-C quant mode, the expected shape of scale2 is (%ld, %ld), but the actual is (%ld, %ld).", g, nDim,
               x2ScaleFirstDim, x2ScaleLastDim);
    return ACLNN_SUCCESS;
}


static aclnnStatus IsMxQuantDim(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    auto x1ScaleDimNum = params.scale1Optional->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.scale2->GetViewShape().GetDimNum();
    CHECK_COND(x2ScaleDimNum == gmm::MX_SPLIT_K_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
               "In Mx Quant, the scale2 dim num should be 3, but actual is [%zu].", x2ScaleDimNum);
    CHECK_COND(x1ScaleDimNum == gmm::MX_SPLIT_K_PER_TOKEN_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
               "In Mx Quant, the scale1 dim num should be 3, but actual is [%zu].", x1ScaleDimNum);
    auto scale1LastDimValue = params.scale1Optional->GetViewShape().GetDim(gmm::MX_SPLIT_K_PER_TOKEN_SCALE_DIM - 1);
    auto scale2LastDimValue = params.scale2->GetViewShape().GetDim(gmm::MX_SPLIT_K_SCALE_DIM - 1);
    CHECK_COND(scale1LastDimValue == 2, ACLNN_ERR_PARAM_INVALID, // last dim should be 2 in mx quant mode
               "The last dim of scale1 should be 2 in mx quant mode, but actual is %ld.", scale1LastDimValue);
    CHECK_COND(scale2LastDimValue == 2, ACLNN_ERR_PARAM_INVALID, // last dim should be 2 in mx typek quant mode
               "The last dim of scale2 should be 2 in mx quant mode, but actual is %ld.", scale2LastDimValue);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto groupListDimNum = params.groupList->GetViewShape().GetDimNum();
    auto yDimNum = params.yRef->GetViewShape().GetDimNum();
    CHECK_COND(x1DimNum == 2, ACLNN_ERR_PARAM_INVALID, // 2 max dim num
               "The dimension of x1 should be 2, but actual is %zu.", x1DimNum);
    CHECK_COND(x2DimNum == 2, ACLNN_ERR_PARAM_INVALID, // 2 max dim num
               "The dimension of x2 should be 2, but actual is %zu.", x2DimNum);
    CHECK_COND(groupListDimNum == 1, ACLNN_ERR_PARAM_INVALID,
               "The dimension of groupList should be 1, but actual is %ld.", groupListDimNum);
    CHECK_COND(yDimNum == 3, ACLNN_ERR_PARAM_INVALID, // 3 max dim num
               "The dimension of yRef should be 3, but actual is %zu.", yDimNum);
    auto aKDim = params.x1->GetViewShape().GetDim(1);
    auto bKDim = params.x2->GetViewShape().GetDim(0);
    auto nDim = params.x2->GetViewShape().GetDim(1);
    auto mDim = params.x1->GetViewShape().GetDim(0);
    auto gDim = params.groupList->GetViewShape().GetDim(0);

    auto yGDim = params.yRef->GetViewShape().GetDim(0);
    auto yMDim = params.yRef->GetViewShape().GetDim(1);
    auto yNDim = params.yRef->GetViewShape().GetDim(2);

    CHECK_COND(mDim >= 0, ACLNN_ERR_PARAM_INVALID, "The M value[%ld] in x1 should be positive.", mDim);

    CHECK_COND(aKDim == bKDim, ACLNN_ERR_PARAM_INVALID,
               "The kDimNum of x1/x2 should be equal, but the actual is %ld/%ld.", aKDim, bKDim);
    CHECK_COND(gDim == yGDim && mDim == yMDim && nDim == yNDim, ACLNN_ERR_PARAM_INVALID,
               "The expected shape of yRef is (%ld, %ld, %ld), but the actual is (%ld, %ld, %ld).", gDim, mDim, nDim,
               yGDim, yMDim, yNDim);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    auto x1Dtype = params.x1->GetDataType();
    auto x2Dtype = params.x2->GetDataType();
    CHECK_COND(params.yRef->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "Input yRef dtype should be FLOAT32, actual dtype is %s.",
               op::ToString(params.yRef->GetDataType()).GetString());
    CHECK_COND(params.groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "Input groupList dtype should be INT64, actual dtype is %s.",
               op::ToString(params.groupList->GetDataType()).GetString());
    if (x1Dtype == DataType::DT_HIFLOAT8 && x2Dtype == DataType::DT_HIFLOAT8) {
        CHECK_COND(IsTcQuant(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "With HIFLOAT8 inputs, only support T-C quant.");
        CHECK_COND(params.scale2->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "With HIFLOAT8 inputs, scale2 dtype should be FLOAT32, actual dtype is %s.",
                   op::ToString(params.scale2->GetDataType()).GetString());
        CHECK_COND(params.scale1Optional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "With HIFLOAT8 inputs, scale1 dtype should be FLOAT32, actual dtype is %s.",
                   op::ToString(params.scale1Optional->GetDataType()).GetString());
    } else if ((x1Dtype == DataType::DT_FLOAT8_E4M3FN || x1Dtype == DataType::DT_FLOAT8_E5M2) &&
               (x2Dtype == DataType::DT_FLOAT8_E4M3FN || x2Dtype == DataType::DT_FLOAT8_E5M2)) {
        CHECK_COND(IsMxQuantDim(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check IsMxQuantDim failed.");
        CHECK_COND(params.scale2->GetDataType() == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
                   "With FLOAT8_E4M3FN/FLOAT8_E5M2 inputs, scale2 dtype should be FLOAT8_E8M0, actual dtype is %s.",
                   op::ToString(params.scale2->GetDataType()).GetString());
        CHECK_COND(params.scale1Optional->GetDataType() == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
                   "With FLOAT8_E4M3FN/FLOAT8_E5M2 inputs, scale1 dtype should be FLOAT8_E8M0, actual dtype is %s.",
                   op::ToString(params.scale1Optional->GetDataType()).GetString());
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Quant case with x1 dtype %s and x2 dtype %s is not supported.",
                op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    gmm::GroupedMatmulParamsBase<aclTensor> gmmParams;
    gmmParams.x = params.x1;
    gmmParams.weight = params.x2;
    gmmParams.scaleOptional = params.scale2;
    gmmParams.perTokenScaleOptional = params.scale1Optional;
    gmmParams.y = params.yRef;
    gmmParams.groupTensorOptional = params.groupList;
    gmmParams.groupListType = params.groupListType;
    gmmParams.groupType = gmm::SPLIT_K;
    gmmParams.xDtype = params.x1->GetDataType();
    gmmParams.transposeX = true;
    gmmParams.transposeWeight = false;
    if (params.x1->GetDataType() == DataType::DT_HIFLOAT8 && params.x2->GetDataType() == DataType::DT_HIFLOAT8) {
        auto checkerTC = QGmmInPlaceAdd::AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<aclTensor>(gmmParams);
        checkerTC.SetInputName("x1", "x2", "scale1Optional", "scale2", "groupList");
        CHECK_RET(checkerTC.CheckQuantGroupedMatmulInplaceAddDAV3510() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    } else {
        auto checker = gmm::AclnnGroupedMatmulDAV3510Checker<aclTensor>(gmmParams);
        checker.SetInputName("x1", "x2", "scale1Optional", "scale2", "groupList");
        CHECK_RET(checker.CheckGroupedMatmulDAV3510() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static void SetTransViewShape(const aclTensor *&inputTensor, aclOpExecutor *executor)
{
    op::Shape viewShape = inputTensor->GetViewShape();
    uint32_t viewShapeDimsNum = viewShape.GetDimNum();
    op::Shape shape;
    shape.SetScalar();
    // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain unchanged.
    for (uint32_t i = 0; i < viewShapeDimsNum - 2; ++i) {
        shape.AppendDim(viewShape.GetDim(i));
    }
    // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second last dim.
    shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
    shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2)); // 2:the second last dim.
    inputTensor = executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset());
}

static aclnnStatus SetTransViewShapeForPertoken(const aclTensor *&inputTensor, aclOpExecutor *executor)
{
    op::Shape viewShape = inputTensor->GetViewShape();
    op::Shape shape;
    shape.SetScalar();
    if (viewShape.GetDimNum() < 3) { // only pertoken in mx typek quant mode have to trans, which dim num is 3
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In Mx Quant, pertoken dim num should be 3, actual is %zu",
                viewShape.GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // swap first two dim
    shape.AppendDim(viewShape.GetDim(1));
    shape.AppendDim(viewShape.GetDim(0));
    shape.AppendDim(viewShape.GetDim(2)); // 2 is last dim contiguous in k axis in mx typek quant mode
    inputTensor =
        executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.x1, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x1 failed.");
    CHECK_COND(DataContiguous(params.x2, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x2 failed.");
    CHECK_COND(DataContiguous(params.scale1Optional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous scale1Optional failed.");
    CHECK_COND(DataContiguous(params.scale2, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous scale2 failed.");
    CHECK_COND(DataContiguous(params.groupList, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous groupList failed.");
    return ACLNN_SUCCESS;
}

static bool IsSpecialTranspose(const aclTensor* const inputTensor)
{
    const auto inputShape = inputTensor->GetViewShape();
    int64_t dim1 = inputShape.GetDimNum() - gmm::LAST_FIRST_DIM_INDEX;
    int64_t dim2 = inputShape.GetDimNum() - gmm::LAST_SECOND_DIM_INDEX;
    return inputShape.GetDim(dim1) == 1 && inputShape.GetDim(dim2) == 1;
}

static aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSizeCommon(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params,
                                                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto x1MDim = params.x1->GetViewShape().GetDim(0);
    auto x2NDim = params.x2->GetViewShape().GetDim(1);
    if (x1MDim == 0 || x2NDim == 0) {
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    bool transposeX = gmm::IsTransposeLastTwoDims(params.x1);      // check is transpose x
    bool transposeWeight = gmm::IsTransposeLastTwoDims(params.x2); // check is transpose weight
    // when the last two dims of weight are (1, 1), consider tranB as false
    transposeWeight = transposeWeight && !IsSpecialTranspose(params.x2);
    CHECK_COND(transposeX == true && transposeWeight == false, ACLNN_ERR_PARAM_INVALID,
               "Only support when transpose of x1 is true and transpose of x2 is false.");

    if (transposeX) {
        SetTransViewShape(params.x1, executorPtr);
        if (params.scale1Optional->GetDataType() == DataType::DT_FLOAT8_E8M0) {
            CHECK_RET(SetTransViewShapeForPertoken(params.scale1Optional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        }
    }
    if (transposeWeight) {
        SetTransViewShape((params.x2), executorPtr);
        if (params.scale2->GetDataType() == DataType::DT_FLOAT8_E8M0) {
            CHECK_RET(SetTransViewShapeForPertoken(params.scale2, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        }
    }
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    // Invoke l0 operator QuantGroupedMatmulInplaceAdd for calculation.
    auto result =
        l0op::QuantGroupedMatmulInplaceAdd(params.x1, params.x2, params.scale1Optional, params.scale2, params.groupList,
                                           params.yRef, params.groupListType, params.groupSize, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result, params.yRef, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                              const aclTensor *scale1Optional, const aclTensor *scale2,
                                                              const aclTensor *groupList, aclTensor *yRef,
                                                              int64_t groupListType, int64_t groupSize,
                                                              uint64_t *workspaceSize, aclOpExecutor **executor)
{
    QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params{x1,        x2,   scale1Optional, scale2,
                                                       groupList, yRef, groupListType,  groupSize};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnQuantGroupedMatmulInplaceAdd,
                   DFX_IN(x1, x2, scale1Optional, scale2, groupList, yRef, groupListType, groupSize), DFX_OUT(yRef));
    return aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSizeCommon(params, workspaceSize, executor);
}

aclnnStatus aclnnQuantGroupedMatmulInplaceAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantGroupedMatmulInplaceAdd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in QuantGMMInplaceAdd launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
