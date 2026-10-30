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
 * \file aclnn_triangular_solve.cpp
 * \brief
 */

#include "aclnn_triangular_solve.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "../../../add/op_api/add.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "../../../eye/op_api/eye.h"
#include "../../../mul/op_api/mul.h"
#include "../../../ones_like/op_api/ones_like.h"
#include "../../../sub/op_api/sub.h"
#include "triangular_solve.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

inline static bool CheckNotNull(const aclTensor *self, const aclTensor *A,
                                const aclTensor *xOut)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(A, return false);
    OP_CHECK_NULL(xOut, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor *self, const aclTensor *A,
                                   const aclTensor *xOut, const aclTensor *mOut)
{
    // 检查self的数据类型是否在triangularSolve算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    // 检查A的数据类型是否在triangularSolve算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(A, DTYPE_SUPPORT_LIST, return false);
    // 检查xOut的数据类型是否在triangularSolve算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(xOut, DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SAME(self, A, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, xOut, return false);
    if (mOut != nullptr) {
        // 检查mOut的数据类型是否在triangularSolve算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(mOut, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SAME(self, mOut, return false);
    }

    return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *A,
                       const aclTensor *xOut, const aclTensor *mOut)
{
    // self,A 维度至少为2且最多为8
    OP_CHECK_MIN_DIM(self, 2, return false);
    OP_CHECK_MIN_DIM(A, 2, return false);
    OP_CHECK_MIN_DIM(xOut, 2, return false);
    OP_CHECK_MAX_DIM(self, 8, return false);
    OP_CHECK_MAX_DIM(A, 8, return false);
    OP_CHECK_MAX_DIM(xOut, 8, return false);
    // A最后两维相等
    auto aShape = A->GetViewShape();
    size_t aDimNum = aShape.GetDimNum();
    auto aHDim = aShape.GetDim(aDimNum - 2);
    auto aWDim = aShape.GetDim(aDimNum - 1);
    OP_CHECK(aHDim == aWDim,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "A must be batchs of square matrices, but they are %ld by %ld matrices.",
            aHDim, aWDim),
        return false);
    // self, A 最后两维度需要满足A(*,m,m), self(*,m,n)
    auto bShape = self->GetViewShape();
    size_t bDimNum = bShape.GetDimNum();
    auto bHDim = bShape.GetDim(bDimNum - 2);
    auto bWDim = bShape.GetDim(bDimNum - 1);
    OP_CHECK(bHDim == aHDim,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Incompatible matrix sizes for triangular_solve: each A matrix is %ld by %ld \
            but each b matrix is %ld by %ld.",
            aHDim, aWDim, bHDim, bWDim),
        return false);
    // self, A 除最后两维外需要满足broadcast条件
    auto tempShape = bShape;
    tempShape.SetDim(bDimNum - 1, bHDim);
    op::Shape broadcastShape;
    OP_CHECK(BroadcastInferShape(aShape, tempShape, broadcastShape),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Broadcast fail, the size of tensor a must match the size of tensor b at non-singleton diemnsion."),
        return false);
    // xOut shape满足broadcast后的A，self （*, m, n）
    auto xShape = broadcastShape;
    xShape.SetDim(xShape.GetDimNum() - 1, bWDim);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(xOut, xShape, return false);
    if (mOut != nullptr) {
        // mOut shape与broadcast后的A相同(*, m, m)
        OP_CHECK_MIN_DIM(mOut, 2, return false);
        OP_CHECK_MAX_DIM(mOut, 8, return false);
        auto mShape = broadcastShape;
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(mOut, mShape, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *A,
                               const aclTensor *xOut, const aclTensor *mOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, A, xOut), ACLNN_ERR_INNER_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, A, xOut, mOut), ACLNN_ERR_PARAM_INVALID);
    // 3. ND 算子不检查格式
    // 4. 检查输入输出shape
    CHECK_RET(CheckShape(self, A, xOut, mOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclTensor* BroadcastTensor(const op::Shape dstShape, const aclTensor *src, aclOpExecutor *executor)
{
    auto dstTensor = executor->AllocTensor(dstShape, src->GetDataType());
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(dstShape);
    auto shape = executor->ConvertToTensor(broadcastDims.data(), broadcastDims.size(),
        static_cast<op::DataType>(ACL_INT64));
    auto result = l0op::BroadcastTo(src, dstTensor, shape, executor);
    return const_cast<aclTensor*>(result);
}

static aclnnStatus CalcuUnitriangular(const aclTensor *&aBroadcast, aclOpExecutor *executor)
{
    // eye仅支持float32数据类型
    OP_CHECK(aBroadcast->GetDataType() == op::DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "unitriangular mode only support float."),
             return ACLNN_ERR_PARAM_INVALID);
    auto aBroadcastShape = aBroadcast->GetViewShape();
    size_t aBroadcastDimNum = aBroadcastShape.GetDimNum();
    auto WDim = aBroadcastShape.GetDim(aBroadcastDimNum - 1);
    op::Shape eyeShape;
    eyeShape.AppendDim(WDim);
    eyeShape.AppendDim(WDim);
    auto out = executor->AllocTensor(eyeShape, aBroadcast->GetDataType());
    // 生成一个eye矩阵
    auto eyeResult = l0op::Eye(out, WDim, WDim, executor);
    CHECK_RET(eyeResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 生成一个全1矩阵
    auto ones = l0op::OnesLike(eyeResult, executor);
    CHECK_RET(ones != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // aUnitriangular = A * (1 - eye) + eye
    auto tempSub = l0op::Sub(ones, eyeResult, executor);
    CHECK_RET(tempSub != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto tempMul = l0op::Mul(aBroadcast, tempSub, executor);
    CHECK_RET(tempMul != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aBroadcast = l0op::Add(tempMul, eyeResult, executor);
    CHECK_RET(aBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTriangularSolveGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *A,
    bool upper,
    bool transpose,
    bool unitriangular,
    aclTensor *xOut,
    aclTensor *mOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnTriangularSolve, DFX_IN(self, A, upper, transpose, unitriangular), DFX_OUT(xOut, mOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, A, xOut, mOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // triangular算子的空tensor在kernel中不支持，对标竞品根据算子实际情况补充
    if (A->IsEmpty() || self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto aContiguous = l0op::Contiguous(A, uniqueExecutor.get());
    CHECK_RET(aContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 若输出tensor形状不同，则进行broadcast
    const aclTensor *selfBroadcast = selfContiguous;
    const aclTensor *aBroadcast = aContiguous;
    if (selfContiguous->GetViewShape() != xOut->GetViewShape()) {
        selfBroadcast = BroadcastTensor(xOut->GetViewShape(), selfContiguous, uniqueExecutor.get());
        CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (mOut != nullptr) {
        if (aContiguous->GetViewShape() != mOut->GetViewShape()) {
            aBroadcast = BroadcastTensor(mOut->GetViewShape(), aContiguous, uniqueExecutor.get());
            CHECK_RET(aBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // a broadcast之后的结果作为输出mOut
        auto resultM = aBroadcast;
        auto viewCopyResultM = l0op::ViewCopy(resultM, mOut, uniqueExecutor.get());
        CHECK_RET(viewCopyResultM != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 若 unitriangular为true，需要归一A主对角线
    if (unitriangular) {
        ret = CalcuUnitriangular(aBroadcast, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 调用MatrixTriangularSolve
    auto resultX = l0op::TriangularSolve(selfBroadcast, aBroadcast, upper, transpose, xOut, uniqueExecutor.get());
    CHECK_RET(resultX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出xOut, mOut上，可能是非连续的tensor
    auto viewCopyResultX = l0op::ViewCopy(resultX, xOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResultX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTriangularSolve(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnTriangularSolve);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif