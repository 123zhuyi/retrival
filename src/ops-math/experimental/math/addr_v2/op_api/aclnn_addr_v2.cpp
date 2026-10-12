/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_addr_v2.h"
#include "../../../../math/add/op_api/add.h"
#include "addr_v2.h"
#include "../../../../math/mul/op_api/mul.h"
#include "../../../../math/logical_or/op_api/logical_or.h"
#include "../../../../math/logical_and/op_api/logical_and.h"
#include "aclnn_kernels/cast.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ADDR_V2_INPUT_DIMS_NUMS 2
#define DEFAULT_OUTER_VEC_DIMS_NUMS 1

enum OuterExpandMode { All, IgnoreInput, IgnoreInputScaling };

// A2 (Ascend910B) 算子对外声明的支持 dtype 列表（含走 stub 分解的类型）
static const std::initializer_list<DataType> ADDR_V2_DTYPE_SUPPORT_LIST = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT64, DataType::DT_INT32,
    DataType::DT_INT16,  DataType::DT_INT8,  DataType::DT_UINT8,   DataType::DT_BOOL,  DataType::DT_BF16};

// A2 (Ascend910B) 原生 kernel 支持的 dtype 列表（DESIGN.md §10.1）
static const std::initializer_list<DataType> ADDR_V2_NATIVE_KERNEL_DTYPE_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16,
    DataType::DT_INT8,  DataType::DT_UINT8,   DataType::DT_BOOL};

static bool IsSupportAddrV2(op::DataType hightDtype)
{
    // 判断是否走原生 AICORE kernel 路径（l0op::AddrV2）。
    // 不使用 IsRegBase() 判断：A2 上 IsRegBase() 可能返回 true，导致框架以 RegBase
    // (A5/950) 方式加载 kernel，但安装的 kernel binary 是 A2 (910B) 格式，引发
    // ADD_TO_LAUNCHER_LIST_AICORE 失败 (561103 ACLNN_ERR_INNER_NULLPTR)。
    // 本算子仅支持 910B，直接按 dtype 列表判定即可。
    return CheckType(hightDtype, ADDR_V2_NATIVE_KERNEL_DTYPE_LIST);
}

static bool CheckBetaAndAlphaDtyeValid(const aclScalar* beta, const aclScalar* alpha, DataType highDtype,
                                       bool allIntInputs)
{
    // beta和alpha为bool类型值时，其他入参只能是bool类型
    if (beta && highDtype != DataType::DT_BOOL && beta->GetDataType() == DataType::DT_BOOL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Boolean beta only supported for Boolean results.");
        return false;
    }

    if (alpha && highDtype != DataType::DT_BOOL && alpha->GetDataType() == DataType::DT_BOOL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Boolean alpha only supported for Boolean results.");
        return false;
    }

    // 如果输入都是整型tensor，那beta和alpha不能是浮点型数
    if (allIntInputs && beta && !IsIntegralType(beta->GetDataType(), true)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "For integral input tensors, argument beta must not be a floating point number");
        return false;
    }

    if (allIntInputs && alpha && !IsIntegralType(alpha->GetDataType(), true)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "For integral input tensors, argument alpha must not be a floating point number");
        return false;
    }

    return true;
}

static inline const aclTensor* vecUnsqueezeWithDim(const aclTensor* vec, int64_t dim, aclOpExecutor* executor)
{
    const int64_t appendDim[] = {dim};
    auto dims = executor->AllocIntArray(appendDim, 1);
    auto vecMul = l0op::UnsqueezeNd(vec, dims, executor);
    CHECK_RET(vecMul != nullptr, nullptr);

    return vecMul;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(vec1, return false);
    OP_CHECK_NULL(vec2, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2, const aclScalar* beta,
                            const aclScalar* alpha, aclTensor* out)
{
    const auto& supportList = ADDR_V2_DTYPE_SUPPORT_LIST;
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查vec1的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(vec1, supportList, return false);

    // 检查vec2的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(vec2, supportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    // 检查beta的数据类型是否在支持列表内，beta为空例外
    if (beta != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(beta, supportList, return false);
    }

    // 检查alpha的数据类型是否在支持列表内，alpha为空例外
    if (alpha != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(alpha, supportList, return false);
    }

    // 检查beta和alpha dtype是否符合要求
    auto hightDtype = op::PromoteType(self->GetDataType(), op::PromoteType(vec2->GetDataType(), vec1->GetDataType()));
    bool allIntInputs = (IsIntegralType(self->GetDataType(), true) && IsIntegralType(vec1->GetDataType(), true) &&
                         IsIntegralType(vec2->GetDataType(), true));
    auto optValid = CheckBetaAndAlphaDtyeValid(beta, alpha, hightDtype, allIntInputs);
    CHECK_RET(optValid, false);

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2, aclTensor* out)
{
    // vec1和vec2只能是1维
    OP_CHECK_WRONG_DIMENSION(vec1, 1, return false);
    OP_CHECK_WRONG_DIMENSION(vec2, 1, return false);

    // self的维度不能超过2
    OP_CHECK_MAX_DIM(self, MAX_ADDR_V2_INPUT_DIMS_NUMS, return false);

    if (self == out) { // 如果out就是self，则self只能是2维, for addr_v2_
        OP_CHECK_WRONG_DIMENSION(self, MAX_ADDR_V2_INPUT_DIMS_NUMS, return false);
    }

    // 检查self与vec1和vec2外积结果之间是否能broadcast；add_校验更严格，不需要进行broadcast
    op::Shape outerShape = {(vec1->GetViewShape())[0], (vec2->GetViewShape())[0]};
    op::Shape broadcastShape = self->GetViewShape();
    if (self != out && !BroadcastInferShape(self->GetViewShape(), outerShape, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the size of tensor self %s must match the size of tensor outer %s.",
                op::ToString(self->GetViewShape()).GetString(), op::ToString(outerShape).GetString());
        return false;
    }

    // broadcast之后的tensor size必须要与外积相同
    if (broadcastShape != outerShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "size mismatch, input: %s, v1: %s, v2: %s",
                op::ToString(broadcastShape).GetString(), op::ToString(vec1->GetViewShape()).GetString(),
                op::ToString(vec2->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2,
                               const aclScalar* beta, const aclScalar* alpha, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, vec1, vec2, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, vec1, vec2, beta, alpha, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, vec1, vec2, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* addr_v2StubMul(const aclTensor* left, const aclTensor* right, DataType highDtype,
                                       aclOpExecutor* executor)
{
    const aclTensor* mulOut = nullptr;
    if (highDtype == DataType::DT_BOOL) {
        mulOut = l0op::LogicalAnd(left, right, executor);
    } else {
        mulOut = l0op::Mul(left, right, executor);
    }

    return mulOut;
}

static const aclTensor* addr_v2StubAdd(const aclTensor* left, const aclTensor* right, DataType highDtype,
                                       aclOpExecutor* executor)
{
    const aclTensor* addOut = nullptr;
    if (highDtype == DataType::DT_BOOL) {
        addOut = l0op::LogicalOr(left, right, executor);
    } else {
        addOut = l0op::Add(left, right, executor);
    }

    return addOut;
}

static aclnnStatus addr_v2OutHandle(const aclTensor* outCalcuResult, aclTensor* out, aclOpExecutor* executor,
                                    DataType hightDtype)
{
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outCalcuResult->GetViewShape(), return ACLNN_ERR_PARAM_INVALID);
    // 输出类型转换成out dtype类型
    // 注意：原 A5 实现对 bool 做 bool→int8→bool Cast roundtrip 以归一化 (非零→1)。
    // 但 A2 custom opp 中 bool Cast kernel 不可用，会返回 nullptr (561103)。
    // 修复方案：原生 kernel 路径已在 kernel 内用 Mins 归一化 bool 输出为 0/1；
    //           stub 路径使用 LogicalOr/LogicalAnd 天然产生 0/1。故移除 Cast roundtrip。
    (void)hightDtype; // 保留参数以兼容调用方签名
    auto temp = outCalcuResult;
    auto addr_v2OutCast = l0op::Cast(temp, out->GetDataType(), executor);
    CHECK_RET(addr_v2OutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 计算结果拷贝，支持非连续
    auto viewCopyResult = l0op::ViewCopy(addr_v2OutCast, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus addr_v2Stub(const aclTensor* selfCast, const aclTensor* vec1Cast, const aclTensor* vec2Cast,
                               const aclScalar* beta, const aclScalar* alpha, aclTensor* out, aclOpExecutor* executor,
                               const op::DataType& hightDtype)
{
    // vec1要转置下，从1行m列扩展成m行1列，才能进行外积运算；vec2相应的也扩展成1行n列的二维tensor
    auto vecMul1 = vecUnsqueezeWithDim(vec1Cast, 1, executor);
    CHECK_RET(vecMul1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vecMul2 = vecUnsqueezeWithDim(vec2Cast, 0, executor);
    CHECK_RET(vecMul2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 计算vec1和vec2外积
    auto outerVec = addr_v2StubMul(vecMul1, vecMul2, hightDtype, executor);
    CHECK_RET(outerVec != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 处理外积比例因子, alpha * outer(vec1, vec2)
    // alpha为nullptr，以默认值做处理
    auto mulOuterVec = (const aclTensor*)outerVec;
    if (alpha != nullptr && alpha->ToFloat() != 1) {
        auto alphaTensor = executor->ConvertToTensor(alpha, hightDtype);
        CHECK_RET(alphaTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        mulOuterVec = addr_v2StubMul(outerVec, alphaTensor, hightDtype, executor);
        CHECK_RET(mulOuterVec != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 处理扩展矩阵, beta * self + alpha * outer(vec1, vec2)
    OuterExpandMode expandMode = All;
    if (beta != nullptr) {
        if (beta->ToFloat() == 0) { // beta为false或者0，则忽略self值
            expandMode = IgnoreInput;
        } else if (beta->ToFloat() == 1) { // beta为true或者1，直接进行矩阵扩展，无需考虑beta比例因子
            expandMode = IgnoreInputScaling;
        }
    } else { // beta为nullptr，以默认值做处理，不考虑beta比例因子
        expandMode = IgnoreInputScaling;
    }

    if (expandMode == IgnoreInput) {
        return addr_v2OutHandle(mulOuterVec, out, executor, DataType::DT_MAX);
    }

    auto selfScaling = selfCast;
    if (expandMode == All) {
        auto betaTensor = executor->ConvertToTensor(beta, hightDtype);
        CHECK_RET(betaTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        selfScaling = addr_v2StubMul(selfCast, betaTensor, hightDtype, executor);
        CHECK_RET(selfScaling != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto addr_v2Out = addr_v2StubAdd(selfScaling, mulOuterVec, hightDtype, executor);
    CHECK_RET(addr_v2Out != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return addr_v2OutHandle(addr_v2Out, out, executor, DataType::DT_MAX);
}

static aclnnStatus addr_v2Proc(const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2,
                               const aclScalar* beta, const aclScalar* alpha, aclTensor* out, aclOpExecutor* executor)
{
    // 计算输入最高类型
    auto hightDtype = op::PromoteType(self->GetDataType(), op::PromoteType(vec2->GetDataType(), vec1->GetDataType()));

    // 连续性转换
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vec1Contiguous = l0op::Contiguous(vec1, executor);
    CHECK_RET(vec1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vec2Contiguous = l0op::Contiguous(vec2, executor);
    CHECK_RET(vec2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 类型转换，都转换成最高类型进行计算
    auto selfCast = l0op::Cast(selfContiguous, hightDtype, executor);
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vec1Cast = l0op::Cast(vec1Contiguous, hightDtype, executor);
    CHECK_RET(vec1Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vec2Cast = l0op::Cast(vec2Contiguous, hightDtype, executor);
    CHECK_RET(vec2Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* alphaTensor;
    const aclTensor* betaTensor;
    if (alpha != nullptr) {
        alphaTensor = executor->ConvertToTensor(alpha, hightDtype);
        CHECK_RET(alphaTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        const aclScalar* valueScalar = executor->AllocScalar(1);
        CHECK_RET(valueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        alphaTensor = executor->ConvertToTensor(valueScalar, hightDtype);
        CHECK_RET(alphaTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (beta != nullptr) {
        betaTensor = executor->ConvertToTensor(beta, hightDtype);
        CHECK_RET(betaTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        const aclScalar* valueScalar = executor->AllocScalar(1);
        CHECK_RET(valueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        betaTensor = executor->ConvertToTensor(valueScalar, hightDtype);
        CHECK_RET(betaTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (IsSupportAddrV2(hightDtype)) {
        auto addr_v2Out = l0op::AddrV2(selfCast, vec1Cast, vec2Cast, betaTensor, alphaTensor, hightDtype, executor);
        CHECK_RET(addr_v2Out != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return addr_v2OutHandle(addr_v2Out, out, executor, hightDtype);
    } else {
        return addr_v2Stub(selfCast, vec1Cast, vec2Cast, beta, alpha, out, executor, hightDtype);
    }
}

aclnnStatus aclnnAddrV2GetWorkspaceSize(const aclTensor* self, const aclTensor* vec1, const aclTensor* vec2,
                                        const aclScalar* betaOptional, const aclScalar* alphaOptional, aclTensor* out,
                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddrV2, DFX_IN(self, vec1, vec2, betaOptional, alphaOptional), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, vec1, vec2, betaOptional, alphaOptional, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (vec1->IsEmpty() || vec2->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 开始addr_v2 计算
    auto addr_v2Ret = addr_v2Proc(self, vec1, vec2, betaOptional, alphaOptional, out, uniqueExecutor.get());
    CHECK_RET(addr_v2Ret == ACLNN_SUCCESS, addr_v2Ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddrV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddrV2);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAddrV2GetWorkspaceSize(aclTensor* selfRef, const aclTensor* vec1, const aclTensor* vec2,
                                               const aclScalar* betaOptional, const aclScalar* alphaOptional,
                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnAddrV2GetWorkspaceSize(selfRef, vec1, vec2, betaOptional, alphaOptional, selfRef, workspaceSize,
                                       executor);
}

aclnnStatus aclnnInplaceAddrV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                               const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceAddrV2);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
