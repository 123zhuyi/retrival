/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_var.h"
#include "math/reduce_std_v2_update/op_host/op_api/reduce_std_v2_update.h"
#include "math/reduce_mean/op_api/reduce_mean.h"
#include "reduce_var.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "math/expand/op_api/expand.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
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

using namespace op;


#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

// 算子支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* self, aclTensor* out) {
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);

    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
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
    // 获取dim元素
    for (size_t i = 0; i < dim->Size(); i++) {
        if (dim->operator[](i) >= selfDimNum || dim->operator[](i) < (-selfDimNum)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Provided dim %ld must be in the range of [%ld, %ld].",
                dim->operator[](i), -selfDimNum, selfDimNum - 1);
            return false;
        }
    }
    // dim值不能重复
    uint64_t dimMask[64] = {0};
    for (size_t i = 0; i < dim->Size(); i++) {
        if (dim->operator[](i) < 0) {
            if (dimMask[selfDimNum + dim->operator[](i)] == 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims.",
                    selfDimNum + dim->operator[](i));
                return false;
            } else {
                dimMask[selfDimNum + dim->operator[](i)] = 1;
            }
            continue;
        }
        if (dimMask[dim->operator[](i)] == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims.", dim->operator[](i));
            return false;
        } else {
            dimMask[dim->operator[](i)] = 1;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* dim,
                       bool keepdim, aclTensor* out) {
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    op::Shape reduceShape = ReduceShapeGetWithVar(self, dim, keepdim);
    // 检查输出shape
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, reduceShape, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim,
                               bool keepdim, aclTensor* out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查reduce的轴是否合理范围
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入输出shape支持能力
    CHECK_RET(CheckShape(self, dim, keepdim, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static int64_t CalcShapeProd(const aclTensor* self, const aclIntArray* dim) {
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    int64_t shapeProd = 1;
    if (selfDimNum == 0) {
        return shapeProd;
    }
    for (size_t i = 0; i < dim->Size(); i++) {
        auto dimValue = dim->operator[](i);
        dimValue = dimValue >= 0 ? dimValue : dimValue + selfDimNum;
        shapeProd *= selfViewShape.GetDim(dimValue);
    }
    return shapeProd;
}

static aclnnStatus aclnnVarImplUnify(const aclTensor *self, const aclIntArray *dim, bool unbiased, bool keepdim,
    aclTensor *out, uint64_t* workspaceSize, UniqueExecutor &uniqueExecutor, aclOpExecutor **executor)
{
    int64_t correction = unbiased ? 1 : 0;
    bool isMeanOut = false;
    auto reduceVarOut = l0op::ReduceVar(self, dim, correction, keepdim, isMeanOut, uniqueExecutor.get());

    auto varOut = std::get<0>(reduceVarOut);
    CHECK_RET(varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut = l0op::Cast(varOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnVarGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim,  bool unbiased, bool keepdim,
                                     aclTensor* out, uint64_t* workspaceSize, aclOpExecutor* *executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnVar, DFX_IN(self, dim, unbiased, keepdim), DFX_OUT(out));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, dim, keepdim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 算子的空tensor处理
    if (self->IsEmpty()) {
        // 空tensor填充NAN
        ret = CheckFillScalarShapeStdAndVar(out, NAN, uniqueExecutor.get());
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }
    // dim为空指针
    const aclIntArray* dimArray = dim;
    if (dim == nullptr || dim->Size() == 0) {
        dimArray = CalcDimWithVar(self, uniqueExecutor.get());
    }
    CHECK_RET(dimArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfReformat = l0op::ReFormat(selfContiguous, Format::FORMAT_ND);
    CHECK_RET(selfReformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (IsRegBase()) {
        return aclnnVarImplUnify(selfReformat, dimArray, unbiased, keepdim, out, workspaceSize,
                                 uniqueExecutor, executor);
    }

    // 调用mean算子kernel
    auto meanOpOut = l0op::ReduceMean(selfReformat, dimArray, true, uniqueExecutor.get());
    CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

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
        meanOpOut = l0op::Expand(meanOpOut, shapeArray, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用算子完成方差计算
    auto varOut = l0op::ReduceStdV2Update(selfReformat, meanOpOut, dimArray, unbiased, keepdim, uniqueExecutor.get());
    CHECK_RET(varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(varOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnVar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnVar);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static aclnnStatus aclnnVarCorrectionImplUnify(const aclTensor *self, const aclIntArray *dim, int64_t correction,
    bool keepdim, aclTensor *out, uint64_t* workspaceSize, UniqueExecutor &uniqueExecutor, aclOpExecutor **executor)
{
    bool isMeanOut = false;
    auto reduceVarOut = l0op::ReduceVar(self, dim, correction, keepdim, isMeanOut, uniqueExecutor.get());

    auto varOut = std::get<0>(reduceVarOut);
    CHECK_RET(varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut = l0op::Cast(varOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnVarCorrectionGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, int64_t correction, bool keepdim,
                                               aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnVarCorrection, DFX_IN(self, dim, correction, keepdim), DFX_OUT(out));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, dim, keepdim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 算子的空tensor处理
    if (self->IsEmpty()) {
        // 空tensor填充NAN
        ret = CheckFillScalarShapeStdAndVar(out, NAN, uniqueExecutor.get());
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }
    // dim为空指针
    const aclIntArray* dimArray = dim;
    if (dim == nullptr || dim->Size() == 0) {
        dimArray = CalcDimWithVar(self, uniqueExecutor.get());
    }
    CHECK_RET(dimArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // shapeProd小于等于correction场景
    int64_t shapeProd = 1;
    shapeProd = CalcShapeProd(self, dimArray);
    if ((shapeProd == 1) && (shapeProd <= correction)) {
        // 返回NAN
        ret = CheckFillScalarShapeStdAndVar(out, NAN, uniqueExecutor.get());
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }
    if ((correction > 1) && (shapeProd <= correction)) {
        // 返回INF
        ret = CheckFillScalarShapeStdAndVar(out, INFINITY, uniqueExecutor.get());
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }

    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfReformat = l0op::ReFormat(selfContiguous, Format::FORMAT_ND);
    CHECK_RET(selfReformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (IsRegBase()) {
        return aclnnVarCorrectionImplUnify(selfReformat, dimArray, correction, keepdim, out, workspaceSize,
                                           uniqueExecutor, executor);
    }

    // 调用mean算子kernel
    auto meanOpOut = l0op::ReduceMean(selfReformat, dimArray, true, uniqueExecutor.get());
    CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

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
        meanOpOut = l0op::Expand(meanOpOut, shapeArray, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用算子完成方差计算
    auto varOut = l0op::ReduceStdV2UpdateCorrection(selfReformat, meanOpOut, dimArray,
                                                    correction, keepdim, uniqueExecutor.get());
    CHECK_RET(varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(varOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnVarCorrection(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnVarCorrection);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif