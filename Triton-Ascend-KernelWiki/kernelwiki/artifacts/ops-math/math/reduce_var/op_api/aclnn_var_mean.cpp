/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_var_mean.h"
#include "math/reduce_std_v2_update/op_host/op_api/reduce_std_v2_update.h"
#include "math/reduce_mean/op_api/reduce_mean.h"
#include "reduce_var.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "math/expand/op_api/expand.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"
#include "op_api/level2_base_caculation.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 算子支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ARCH3510_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
    if (IsRegBase()) {
        return ARCH3510_DTYPE_SUPPORT_LIST;
    } else {
        return DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self, aclTensor* meanOut, aclTensor* varOut) {
    auto dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(meanOut, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(varOut, dtypeSupportList, return false);
    return true;
}

static bool CheckDimValid(const aclTensor* self, const aclIntArray* dim) {
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    // self为标量时，dim range [-1, 0]
    selfDimNum = selfDimNum <= 0 ? 1 : selfDimNum;
    if (dim == nullptr) {
        return true;
    }
    // 获取dim元素,dim值不能重复
    uint64_t dimMask[64] = {0};
    for (size_t i = 0; i < dim->Size(); i++) {
        if (dim->operator[](i) >= selfDimNum || dim->operator[](i) < (-selfDimNum)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Provided dim %ld must be in the range of [%ld, %ld].",
                dim->operator[](i), -selfDimNum, selfDimNum - 1);
            return false;
        }
        if (dim->operator[](i) < 0) {
            if (dimMask[selfDimNum + dim->operator[](i)] == 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims.",
                    selfDimNum + dim->operator[](i));
                return false;
            }
            dimMask[selfDimNum + dim->operator[](i)] = 1;
            continue;
        }
        if (dimMask[dim->operator[](i)] == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims.", dim->operator[](i));
            return false;
        }
        dimMask[dim->operator[](i)] = 1;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* dim, bool keepdim,
                       aclTensor* meanOut, aclTensor* varOut) {
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    op::Shape reduceShape = ReduceShapeGetWithVar(self, dim, keepdim);
    // 检查输出shape
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(meanOut, reduceShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(varOut, reduceShape, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim, bool keepdim,
                               aclTensor* meanOut, aclTensor* varOut) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull3Tensor(self, meanOut, varOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, meanOut, varOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查reduce的轴是否合理范围
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self的shape是否在最大维度范围之内
    CHECK_RET(CheckShape(self, dim, keepdim, meanOut, varOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnVarMeanImplUnify(const aclTensor *self, const aclIntArray *dim, int64_t correction,
    bool keepdim, aclTensor *varOut, aclTensor *meanOut, uint64_t* workspaceSize,
    UniqueExecutor &uniqueExecutor, aclOpExecutor **executor)
{
    bool isMeanOut = true;
    auto reduceVarOut = l0op::ReduceVar(self, dim, correction, keepdim, isMeanOut, uniqueExecutor.get());

    auto varOpOut = std::get<0>(reduceVarOut);
    CHECK_RET(varOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut = l0op::Cast(varOpOut, varOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(castOut, varOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanOpOut = std::get<1>(reduceVarOut);
    CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut1 = l0op::Cast(meanOpOut, meanOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(castOut1, meanOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnVarMeanGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim,
                                         int64_t correction, bool keepdim,
                                         aclTensor* varOut, aclTensor* meanOut,
                                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnVarMean, DFX_IN(self, dim, correction, keepdim), DFX_OUT(varOut, meanOut));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 参数检查
    auto ret = CheckParams(self, dim, keepdim, meanOut, varOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 算子的空tensor处理
    if (self->IsEmpty()) {
        // 空tensor填充NAN
        ret = CheckFillScalarShapeStdAndVar(varOut, NAN, uniqueExecutor.get());
        ret = CheckFillScalarShapeStdAndVar(meanOut, NAN, uniqueExecutor.get());
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }
    // dim为空指针
    const aclIntArray* dimArray = dim;
    if (dim == nullptr || dim->Size() == 0) {
        dimArray = CalcDimWithVar(self, uniqueExecutor.get());
    }
    auto selfViewShape = self->GetViewShape();
    size_t dimSize = dimArray->Size();
    std::vector<int64_t> tmpDim(dimSize, 0);
    for (size_t i = 0; i < dimSize; i++) {
        auto dimValue = dimArray->operator[](i);
        if (dimValue < 0) {
            dimValue += selfViewShape.GetDimNum();
        }
        tmpDim[i] = dimValue;
    }
    dimArray = uniqueExecutor->AllocIntArray(tmpDim.data(), dimSize);
    CHECK_RET(dimArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfReformat = l0op::ReFormat(selfContiguous, Format::FORMAT_ND, uniqueExecutor.get());
    CHECK_RET(selfReformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (IsRegBase()) {
        return aclnnVarMeanImplUnify(selfReformat, dimArray, correction, keepdim, varOut, meanOut,
                                     workspaceSize, uniqueExecutor, executor);
    }

    // 调用mean算子kernel
    auto meanOpOut = l0op::ReduceMean(selfReformat, dimArray, keepdim, uniqueExecutor.get());
    CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出meanOut的数据类型
    auto castMeanOut = l0op::Cast(meanOpOut, meanOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castMeanOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
    auto viewCopyMeanResult = l0op::ViewCopy(castMeanOut, meanOut, uniqueExecutor.get());
    CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // shapeProd小于等于correction场景
    int64_t shapeProd = 1;
    shapeProd = CalcShapeProdStdAndVarMean(self, dimArray);
    if ((shapeProd == 1) && (shapeProd <= correction)) {
        // 返回NAN
        ret = CheckFillScalarShapeStdAndVar(varOut, NAN, uniqueExecutor.get());
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }
    if ((correction > 1) && (shapeProd <= correction)) {
        // 返回INF
        ret = CheckFillScalarShapeStdAndVar(varOut, INFINITY, uniqueExecutor.get());
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }

    // 调用mean算子kernel获取用于计算var的mean
    if (keepdim == false) {
        meanOpOut = l0op::ReduceMean(selfReformat, dimArray, true, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用Expand算子kernel
    FVector<int64_t> shapeVector;
    auto selfShape = self->GetViewShape();
    size_t selfDimNum = selfShape.GetDimNum();
    if (selfDimNum > 0) {
        for (size_t i = 0; i < selfDimNum; i++) {
            shapeVector.emplace_back(selfShape[i]);
        }
        auto shapeArray = uniqueExecutor.get()->AllocIntArray(shapeVector.data(), selfDimNum);
        CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (!keepdim) {
            FVector<int64_t> reShapeVector(shapeVector);
            for (size_t i = 0; i < dimArray->Size(); i++) {
                reShapeVector[dimArray->operator[](i)] = 1;
            }
            auto reShapeArray = uniqueExecutor.get()->AllocIntArray(reShapeVector.data(), selfDimNum);
            CHECK_RET(reShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
            meanOpOut = l0op::Reshape(meanOpOut, reShapeArray, uniqueExecutor.get());
            CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        meanOpOut = l0op::Expand(meanOpOut, shapeArray, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用算子完成方差计算
    auto varOpOut = l0op::ReduceStdV2UpdateCorrection(selfReformat, meanOpOut, dimArray,
                                                      correction, keepdim, uniqueExecutor.get());
    CHECK_RET(varOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castVarOut = l0op::Cast(varOpOut, varOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castVarOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 将计算结果拷贝到输出varOut上，varOut可能是非连续的tensor
    auto viewCopyVarResult = l0op::ViewCopy(castVarOut, varOut, uniqueExecutor.get());
    CHECK_RET(viewCopyVarResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnVarMean(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnVarMean);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif