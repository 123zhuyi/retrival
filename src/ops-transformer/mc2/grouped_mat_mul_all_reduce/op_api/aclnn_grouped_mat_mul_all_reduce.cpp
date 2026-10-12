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
 * \file aclnn_grouped_mat_mul_all_reduce.cpp
 * \brief
 */
#include "aclnn_grouped_mat_mul_all_reduce.h"
#include <new>
#include "securec.h"

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
#include "hccl_util.h"

static constexpr int64_t MAX_GROUP_LIST_SIZE = 64; // tiling data size only support 8192 bytes.

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t X_Y_SEPARATED = 0; // x,y不切?
static const int64_t Y_SEPARATED = 1;   // x切分
static const int64_t X_SEPARATED = 2;   // y切分
static const int64_t NO_SEPARATED = 3;  // x,y切分

static const size_t MAX_FM_DIM = 6;
static const size_t MIN_FM_DIM = 2;
static const size_t SPLIT_DIM = 2;

const std::map<DataType, aclDataType> BIAS_DTYPE{
    {DataType::DT_FLOAT16, aclDataType::ACL_FLOAT16}, {DataType::DT_BF16, aclDataType::ACL_FLOAT}};

struct NnopbaseDfxId {
    uint32_t id;
    const char* funcName;
    bool hasReg;
};

extern aclnnStatus aclnnInnerGroupedMatMulAllReduceGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* bias,
    const aclIntArray* groupListOptional, int64_t splitItem, const char* group, const char* reduceOp, int64_t commTurn,
    const aclTensorList* out, uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerGroupedMatMulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

extern uint64_t NnopbaseMsprofSysTime();
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);

static void SplitTensorList(
    const std::vector<std::vector<int64_t>>& shapes, const aclTensorList* tensorList, std::vector<aclTensor*>& tensors)
{
    uint8_t* addr = static_cast<uint8_t*>((*tensorList)[0]->GetStorageAddr());
    aclDataType dataType;
    aclGetDataType((*tensorList)[0], &dataType);
    aclFormat format;
    aclGetFormat((*tensorList)[0], &format);
    uint64_t groupListSum = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
        if (shapes[i].empty()) {
            aclTensor* tensor = nullptr;
            tensors.emplace_back(tensor);
            continue;
        }
        aclTensor* tensor = aclCreateTensor(
            shapes[i].data(), shapes[i].size(), dataType, nullptr, 0, format, shapes[i].data(), shapes[i].size(),
            addr + groupListSum * shapes[i][1] * TypeSize((*tensorList)[0]->GetDataType()));
        groupListSum += shapes[i][0];
        tensors.emplace_back(tensor);
    }
}

struct GroupedMatMulAllReduceParams {
    const aclTensorList* x = nullptr;
    const aclTensorList* weight = nullptr;
    const aclTensorList* bias = nullptr;
    const aclIntArray* groupListOptional = nullptr;
    int64_t splitItemOptional;
    const char* group = nullptr;
    const char* reduceOp = nullptr; // reduce type, now 'sum'
    int64_t commTurn;               // communication trun num
    const aclTensorList* y = nullptr;
    DataType xDtype;
};

static bool CheckNotNull(const GroupedMatMulAllReduceParams& gmmParams)
{
    CHECK_COND(gmmParams.x != nullptr, false, "x must not be nullptr.");
    CHECK_COND(gmmParams.weight != nullptr, false, "weight must not be nullptr.");
    CHECK_COND(gmmParams.y != nullptr, false, "y must not be nullptr.");
    return true;
}

static bool CheckGroupListOptional(const GroupedMatMulAllReduceParams& gmmParams)
{
    CHECK_COND(
        gmmParams.groupListOptional != nullptr, false, "When splitItem is 1/3, groupList size must not be nullptr.");
    uint64_t groupListSize = gmmParams.groupListOptional->Size();
    CHECK_COND(
        groupListSize == gmmParams.weight->Size(), false,
        "When splitItem is 1/3, groupList size[%lu] must be equal with weight group size[%lu].", groupListSize,
        gmmParams.weight->Size());
    int64_t preGoupList = 0;
    for (size_t i = 0; i < groupListSize; i++) {
        CHECK_COND(
            (*gmmParams.groupListOptional)[i] >= preGoupList, false,
            "groupListOptional should be an incremental sequence.");
        preGoupList = (*gmmParams.groupListOptional)[i];
    }
    CHECK_COND(
        (*gmmParams.x)[0]->GetViewShape().GetDim(0) == preGoupList, false,
        "The last value of group list(%lu) must equal with x shape[0] (%lu).", preGoupList,
        (*gmmParams.x)[0]->GetViewShape().GetDim(0));
    return true;
}

static aclnnStatus CheckParamDimAndLengthGmmAr(const GroupedMatMulAllReduceParams& gmmParams)
{
    uint64_t xGroupedSize = gmmParams.x->Size();
    uint64_t weightGroupedSize = gmmParams.weight->Size();
    uint64_t yGroupedSize = gmmParams.y->Size();
    CHECK_COND(
        weightGroupedSize <= MAX_GROUP_LIST_SIZE, ACLNN_ERR_PARAM_INVALID,
        "The group size of weight should not "
        "exceed %ld, but now is %lu.",
        MAX_GROUP_LIST_SIZE, weightGroupedSize);
    if (gmmParams.splitItemOptional == Y_SEPARATED || gmmParams.splitItemOptional == NO_SEPARATED) {
        CHECK_COND(
            xGroupedSize == 1, ACLNN_ERR_PARAM_INVALID,
            "When splitItemOptional is 1/3, x group size must be 1, but now is %lu.", xGroupedSize);
        CHECK_RET(CheckGroupListOptional(gmmParams), ACLNN_ERR_PARAM_INVALID);
    } else if (gmmParams.splitItemOptional == X_SEPARATED || gmmParams.splitItemOptional == X_Y_SEPARATED) {
        CHECK_COND(
            xGroupedSize == weightGroupedSize, ACLNN_ERR_PARAM_INVALID,
            "When splitItem is 0/2, "
            "x group size[%lu] must be equal with weight group size[%lu].",
            xGroupedSize, weightGroupedSize);
        CHECK_COND(
            gmmParams.groupListOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
            "When splitItem is 0/2, groupList must be nullptr.");
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "splitItem only support 0/1/2/3.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gmmParams.splitItemOptional == X_SEPARATED || gmmParams.splitItemOptional == NO_SEPARATED) {
        CHECK_COND(
            yGroupedSize == 1, ACLNN_ERR_PARAM_INVALID,
            "When splitItemOptional is 2/3, y group size must be 1, but now is %lu.", yGroupedSize);
    } else {
        CHECK_COND(
            yGroupedSize == weightGroupedSize, ACLNN_ERR_PARAM_INVALID,
            "When splitItemOptional is 0/1, "
            "y group size[%lu] must be equal with weight group size[%lu].",
            yGroupedSize, weightGroupedSize);
    }
    for (size_t i = 0; i < xGroupedSize; ++i) {
        OP_CHECK_NULL((*gmmParams.x)[i], continue);
        size_t xDims = (*gmmParams.x)[i]->GetViewShape().GetDimNum();
        CHECK_COND(
            xDims <= MAX_FM_DIM && xDims >= MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID,
            "x[%lu] Dim only support 2-6, but now is %lu.", i, xDims);
    }
    for (size_t i = 0; i < weightGroupedSize; ++i) {
        OP_CHECK_NULL((*gmmParams.weight)[i], continue);
        size_t weightDims = (*gmmParams.weight)[i]->GetViewShape().GetDimNum();
        CHECK_COND(
            weightDims == MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID, "weight[%lu] Dim only support 2, but now is %lu.", i,
            weightDims);
    }
    if (gmmParams.bias != nullptr) {
        uint64_t biasGroupedSize = gmmParams.bias->Size();
        CHECK_COND(
            weightGroupedSize == biasGroupedSize, ACLNN_ERR_PARAM_INVALID,
            "bias group size [%lu] should be equal to weight group size [%lu].", biasGroupedSize, weightGroupedSize);
        for (size_t i = 0; i < biasGroupedSize; ++i) {
            OP_CHECK_NULL((*gmmParams.bias)[i], continue);
            size_t biasDims = (*gmmParams.bias)[i]->GetViewShape().GetDimNum();
            CHECK_COND(biasDims == 1, ACLNN_ERR_PARAM_INVALID, "bias[%lu] Dim must be 1, but now is %lu.", i, biasDims);
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDimK(const GroupedMatMulAllReduceParams& gmmParams)
{
    bool isXSeparated = gmmParams.splitItemOptional == X_Y_SEPARATED || gmmParams.splitItemOptional == X_SEPARATED;
    auto xTensor = (*gmmParams.x)[0]; // 0: first x tensor
    CHECK_COND(xTensor != nullptr, ACLNN_ERR_PARAM_INVALID, "x[0] should not be empty.");
    for (size_t i = 0; i < gmmParams.weight->Size(); ++i) {
        if (isXSeparated) {
            xTensor = (*gmmParams.x)[i];
            CHECK_COND(xTensor != nullptr, ACLNN_ERR_PARAM_INVALID, "When splitItem = 0/2, x[%lu] cannot be empty.", i);
        }
        auto wTensor = (*gmmParams.weight)[i];
        size_t xDims = xTensor->GetViewShape().GetDimNum();
        int64_t xDimLast = xTensor->GetViewShape().GetDim(xDims - 1);
        CHECK_COND(wTensor != nullptr, ACLNN_ERR_PARAM_INVALID, "w[%lu] should not be empty", i);
        int64_t wDimFirst = wTensor->GetViewShape().GetDim(0);
        CHECK_COND(
            xDimLast == wDimFirst, ACLNN_ERR_PARAM_INVALID,
            "The last dim of x[%lu] = [%lu], which is not equal"
            " with the first dim of weight[%lu] = [%lu]",
            xDims, xDimLast, i, wDimFirst);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDims(const GroupedMatMulAllReduceParams& gmmParams)
{
    if (gmmParams.splitItemOptional == X_Y_SEPARATED) {
        for (size_t i = 0; i < gmmParams.x->Size(); ++i) {
            auto xTensor = (*gmmParams.x)[i];
            auto yTensor = (*gmmParams.y)[i];
            if (xTensor == nullptr) {
                CHECK_COND(
                    yTensor == nullptr, ACLNN_ERR_PARAM_INVALID, "x[%lu] is empty, but y[%lu] is not empty", i, i);
                continue;
            } else {
                CHECK_COND(
                    yTensor != nullptr, ACLNN_ERR_PARAM_INVALID, "x[%lu] is not empty, but y[%lu] is empty", i, i);
            }
            size_t xDims = xTensor->GetViewShape().GetDimNum();
            size_t yDims = yTensor->GetViewShape().GetDimNum();
            CHECK_COND(
                xDims == yDims && xDims < MAX_FM_DIM, ACLNN_ERR_PARAM_INVALID,
                "When splitItem is 0, x dims[%lu] must be equal with y dims[%lu] and not greater than %lu", xDims,
                yDims, MAX_FM_DIM);
        }
    } else {
        for (size_t i = 0; i < gmmParams.x->Size(); i++) {
            auto xTensor = (*gmmParams.x)[i];
            if (xTensor != nullptr) {
                auto xDims = xTensor->GetViewShape().GetDimNum();
                CHECK_COND(
                    xDims == MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID,
                    "When splitItem is not 0, x dims must be %lu, but now is %lu.", MIN_FM_DIM, xDims);
            }
        }
        for (size_t i = 0; i < gmmParams.y->Size(); i++) {
            auto yTensor = (*gmmParams.y)[i];
            if (yTensor != nullptr) {
                auto yDims = yTensor->GetViewShape().GetDimNum();
                CHECK_COND(
                    yDims == MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID,
                    "When splitItem is not 0, y dims must be %lu, but now is %lu.", MIN_FM_DIM, yDims);
            }
        }
    }

    CHECK_COND(
        CheckDimK(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
        "K dim of x tensors and weight tensors does match");

    return ACLNN_SUCCESS;
}

static bool CheckMatmulShape(const GroupedMatMulAllReduceParams& gmmParams)
{
    for (size_t i = 0; i < gmmParams.x->Size(); ++i) {
        CHECK_COND((*gmmParams.weight)[i] != nullptr, ACLNN_ERR_INNER_NULLPTR, "weight[%zu] cannot be nullptr.", i);
        if ((*gmmParams.x)[i] == nullptr && (*gmmParams.y)[i] == nullptr) {
            continue;
        }
        CHECK_COND(
            (*gmmParams.x)[i] != nullptr && (*gmmParams.y)[i] != nullptr, ACLNN_ERR_INNER_NULLPTR,
            "x or y is nullptr.");
        op::Shape xShape = (*gmmParams.x)[i]->GetViewShape();
        op::Shape yShape = (*gmmParams.y)[i]->GetViewShape();
        size_t checkDim = xShape.GetDimNum() - 1;
        for (size_t dim = 0; dim < checkDim; dim++) {
            int64_t xMDimValue = xShape.GetDim(dim);
            int64_t yMDimValue = yShape.GetDim(dim);
            CHECK_COND(
                xMDimValue == yMDimValue, false,
                "Dim[%zu] of x[%zu] and y[%zu] are %ld and %ld, respectively, "
                "which should be equal to each other.",
                dim, i, i, xMDimValue, yMDimValue);
        }
        int64_t xKDimValue = (*gmmParams.x)[i]->GetViewShape().GetDim(checkDim);
        int64_t weightKDimValue = (*gmmParams.weight)[i]->GetViewShape().GetDim(0);
        int64_t weightNDimValue = (*gmmParams.weight)[i]->GetViewShape().GetDim(1);
        int64_t yNDimValue = (*gmmParams.y)[i]->GetViewShape().GetDim(checkDim);
        CHECK_COND(
            xKDimValue == weightKDimValue, false,
            "KDim[%ld] of weight[%zu] should be equal with "
            "KDim[%ld] of x[%zu].",
            weightKDimValue, i, xKDimValue, i);
        CHECK_COND(
            weightNDimValue == yNDimValue, false,
            "NDim[%ld] of weight[%zu] should be equal with "
            "NDim[%ld] of y[%zu].",
            weightNDimValue, i, yNDimValue, i);
        if (gmmParams.bias == nullptr) {
            continue;
        }
        CHECK_COND((*gmmParams.bias)[i] != nullptr, false, "bias[%ld] should not be nullptr.", i);
        int64_t biasDimValue = (*gmmParams.bias)[i]->GetViewShape().GetDim(0);
        CHECK_COND(
            biasDimValue == weightNDimValue, false,
            "NDim[%ld] of weight[%zu] should be equal with "
            "NDim[%ld] of bias[%zu].",
            weightNDimValue, i, biasDimValue, i);
    }
    return true;
}

static bool CheckTensorListDataType(const aclTensorList* tensorList, const DataType dtype)
{
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        const aclTensor* tensor = (*tensorList)[i];
        OP_CHECK_NULL(tensor, continue);
        OP_CHECK_DTYPE_NOT_MATCH(tensor, dtype, return false);
    }
    return true;
}

static bool CheckMatmulDataType(
    const GroupedMatMulAllReduceParams& gmmParams, const DataType comDtype, const DataType weightDtype,
    const DataType biasDtype)
{
    CHECK_COND(
        CheckTensorListDataType(gmmParams.x, comDtype), ACLNN_ERR_PARAM_INVALID,
        "GMMAr:x dtype does not match with required dtype[%s].", op::ToString(comDtype).GetString());
    CHECK_COND(
        CheckTensorListDataType(gmmParams.weight, weightDtype), ACLNN_ERR_PARAM_INVALID,
        "GMMAr:weight dtype does not match with required dtype[%s].", op::ToString(weightDtype).GetString());
    CHECK_COND(
        CheckTensorListDataType(gmmParams.y, comDtype), ACLNN_ERR_PARAM_INVALID,
        "GMMAr:y dtype does not match with required dtype[%s].", op::ToString(comDtype).GetString());
    if (gmmParams.bias != nullptr) {
        CHECK_COND(
            CheckTensorListDataType(gmmParams.bias, biasDtype), ACLNN_ERR_PARAM_INVALID,
            "GMMAr:bias dtype does not match with required dtype[%s].", op::ToString(biasDtype).GetString());
    }
    return true;
}

static aclnnStatus CheckParamOptionGmmAr(const GroupedMatMulAllReduceParams& gmmParams)
{
    DataType weightDtype = (*gmmParams.weight)[0]->GetDataType();
    if ((gmmParams.xDtype == DataType::DT_BF16 || gmmParams.xDtype == DataType::DT_FLOAT16) &&
        gmmParams.xDtype == weightDtype) {
        DataType biasDtype = gmmParams.xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT : DataType::DT_FLOAT16;
        CHECK_RET(CheckMatmulDataType(gmmParams, gmmParams.xDtype, weightDtype, biasDtype), ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only float16/bfloat16 are supported for x and weight.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParam(const GroupedMatMulAllReduceParams& gmmParams)
{
    CHECK_RET(CheckNotNull(gmmParams), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckParamDimAndLengthGmmAr(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParamOptionGmmAr(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDims(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static void InitShape(
    const GroupedMatMulAllReduceParams& gmmParams, const bool isXTensor, std::vector<std::vector<int64_t>>& shapes)
{
    shapes.reserve(gmmParams.weight->Size());
    int64_t preOffset = 0;
    for (uint64_t i = 0; i < gmmParams.weight->Size(); i++) {
        std::vector<int64_t> shape;
        if ((*gmmParams.weight)[i] == nullptr) {
            shapes.emplace_back(shape);
            continue;
        }
        shape.reserve(SPLIT_DIM);
        if (isXTensor) {
            shape.emplace_back((*gmmParams.groupListOptional)[i] - preOffset);
            preOffset = (*gmmParams.groupListOptional)[i];
            shape.emplace_back((*gmmParams.weight)[i]->GetViewShape().GetDim(0));
        } else {
            shape.emplace_back((*gmmParams.x)[i]->GetViewShape().GetDim(0));
            shape.emplace_back((*gmmParams.weight)[i]->GetViewShape().GetDim(1));
        }
        shapes.emplace_back(shape);
    }
}

static void CheckOptionalTensorListEmpty(const aclTensorList*& tensorList)
{
    if (tensorList != nullptr) {
        if (tensorList->Size() == 0) {
            tensorList = nullptr;
        } else if ((*tensorList)[0] == nullptr) {
            tensorList = nullptr;
        } else if (tensorList->Size() == 1 && (*tensorList)[0]->GetViewShape().GetDim(0) == 0) {
            tensorList = nullptr;
        }
    }
    return;
}

static void ResetEmptyTensor(GroupedMatMulAllReduceParams& gmmParams)
{
    if (gmmParams.groupListOptional != nullptr && gmmParams.groupListOptional->Size() == 0) {
        gmmParams.groupListOptional = nullptr;
    }
    CheckOptionalTensorListEmpty(gmmParams.bias);
    return;
}

static aclnnStatus CreateXYTensorList(
    const aclTensorList* xAfterSplit, const aclTensorList* yAfterSplit, GroupedMatMulAllReduceParams& gmmParams)
{
    if (gmmParams.splitItemOptional == Y_SEPARATED || gmmParams.splitItemOptional == NO_SEPARATED) {
        std::vector<std::vector<int64_t>> shapes;
        InitShape(gmmParams, true, shapes);
        std::vector<aclTensor*> xTensorList;
        SplitTensorList(shapes, gmmParams.x, xTensorList);
        xAfterSplit = aclCreateTensorList(xTensorList.data(), xTensorList.size());
        gmmParams.x = xAfterSplit;
    }
    if (gmmParams.splitItemOptional == X_SEPARATED || gmmParams.splitItemOptional == NO_SEPARATED) {
        std::vector<std::vector<int64_t>> shapes;
        InitShape(gmmParams, false, shapes);
        std::vector<aclTensor*> yTensorList;
        SplitTensorList(shapes, gmmParams.y, yTensorList);
        yAfterSplit = aclCreateTensorList(yTensorList.data(), yTensorList.size());
        gmmParams.y = yAfterSplit;
    }
    return ACLNN_SUCCESS;
}

static void CreateEmptyTensor(
    const aclDataType dataType, const aclTensorList*& gmmTensorList, aclTensorList*& tensorList)
{
    if (gmmTensorList == nullptr) {
        std::vector<aclTensor*> emptyTensors;
        aclTensor* emptyTensor = aclCreateTensor({}, 0, dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, {}, 0, nullptr);
        emptyTensors.emplace_back(emptyTensor);
        tensorList = aclCreateTensorList(emptyTensors.data(), emptyTensors.size());
        gmmTensorList = tensorList;
    }
    return;
}

static aclnnStatus PreAndPostProcessForInner(
    GroupedMatMulAllReduceParams& gmmParams, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    aclTensorList* emptyBiasList = nullptr; // init emptyBiasList
    if (BIAS_DTYPE.find(gmmParams.xDtype) == BIAS_DTYPE.cend()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "GroupedMatMulAllReduce: Cannot find bias dtype match with xDtype[%s].",
            op::ToString(gmmParams.xDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    CreateEmptyTensor(BIAS_DTYPE.at(gmmParams.xDtype), gmmParams.bias, emptyBiasList);
    aclnnStatus ret = aclnnInnerGroupedMatMulAllReduceGetWorkspaceSize(
        gmmParams.x, gmmParams.weight, gmmParams.bias, gmmParams.groupListOptional, gmmParams.splitItemOptional,
        gmmParams.group, gmmParams.reduceOp, gmmParams.commTurn, gmmParams.y, workspaceSize, executor);

    if (emptyBiasList != nullptr) { // destroy tensorList
        aclDestroyTensorList(emptyBiasList);
    }
    return ret;
}

aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* bias,
    const aclIntArray* groupListOptional, int64_t splitItem, const char* group, const char* reduceOp, int64_t commTurn,
    int64_t streamMode, const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    OP_API_CHECK(streamMode != 1, {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "streamMode is %ld, only support 1!", streamMode);
        return ACLNN_ERR_PARAM_INVALID;
    });
    OP_API_CHECK(commTurn != 0, {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "commTurn is %ld, only support 0!", commTurn);
        return ACLNN_ERR_PARAM_INVALID;
    });
    OP_API_CHECK(strcmp(reduceOp, "sum") != 0, {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "reduceOp is %s, only support sum!", reduceOp);
        return ACLNN_ERR_PARAM_INVALID;
    });
    DataType xDtype = DataType::DT_UNDEFINED;
    for (size_t i = 0; i < x->Size(); ++i) {
        if ((*x)[i] != nullptr) {
            xDtype = (*x)[i]->GetDataType();
            break;
        }
    }
    GroupedMatMulAllReduceParams gmmParams{x,        weight, bias,  groupListOptional, splitItem, group, reduceOp,
                                           commTurn, y,      xDtype};
    ResetEmptyTensor(gmmParams);
    CHECK_RET(CheckParam(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclTensorList* xAfterSplit = nullptr;
    aclTensorList* yAfterSplit = nullptr;
    CreateXYTensorList(xAfterSplit, yAfterSplit, gmmParams);
    CHECK_RET(CheckMatmulShape(gmmParams), ACLNN_ERR_PARAM_INVALID);

    // GetWorkspaceSize
    aclnnStatus ret = PreAndPostProcessForInner(gmmParams, workspaceSize, executor);
    OP_LOGD("GroupedMatMulAllReduce, end ret %d.", ret);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (xAfterSplit != nullptr) {
        aclDestroyTensorList(xAfterSplit);
    }
    if (yAfterSplit != nullptr) {
        aclDestroyTensorList(yAfterSplit);
    }

    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnGroupedMatMulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // Kernel Func
    CHECK_COND(
        aclnnInnerGroupedMatMulAllReduce(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
        "GroupedMatMulAllReduce, This is an error in launch aicore");
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
