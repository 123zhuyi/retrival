/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_cdist.h"
#include "cdist.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "conversion/fill/op_api/fill.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;
constexpr size_t MIN_DIM_LEN = 2;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(SocVersion socVersion)
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 || IsRegBase()) {
        return DTYPE_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    // 检查输入输出是否为空指针
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* out, SocVersion socVersion)
{
    static const std::initializer_list<DataType> dtypeSupportList = GetSupportDtypeList(socVersion);
    // 检查输入输出的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);
    // 检查out和输入的数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(x1, out, return false);
    OP_CHECK_DTYPE_NOT_SAME(x2, out, return false);
    return true;
}

static bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    // 输入输出的维度最多支持8维
    OP_CHECK_MAX_DIM(x1, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(x2, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    // 输入输出的维度要求最低为2维
    OP_CHECK_MIN_DIM(x1, MIN_DIM_LEN, return false);
    OP_CHECK_MIN_DIM(x2, MIN_DIM_LEN, return false);
    OP_CHECK_MIN_DIM(out, MIN_DIM_LEN, return false);
    // 检查x2的点特征维度与x1是否一致
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    int64_t x1LastDim = static_cast<int64_t>(x1->GetViewShape().GetDim(x1DimNum - 1));
    int64_t x2LastDim = static_cast<int64_t>(x2->GetViewShape().GetDim(x2DimNum - 1));
    if(x1LastDim != x2LastDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of X1 and X2 must be the same. X1 got: %ld, X2 got: %ld.", x1LastDim, x2LastDim);
        return false;
    }
    return true;
}

static bool CheckParamsLogic(float& p, const int64_t compute_mode, SocVersion socVersion)
{
    // p范数为非负数
    if(p < 0 || std::isnan(p)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cdist only supports non-negative p values.");
        return false;
    }
    // 强校验compute_mode
    if (std::fabs(p - 2.0) < 1e-6 && compute_mode != 2){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "compute_mode %lld is not supported, only support 2 when pValue equals 2.0",
                compute_mode);
        return false;
    }
    // A2A3特殊处理p值
    if ((socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) && std::isinf(p)){
        p = -1.0;
    }
    return true;
}

// 检查数据格式
static void CheckFormat(const aclTensor* x1, const aclTensor* x2)
{
    ge::Format x1StorageFormat = x1->GetStorageFormat();
    ge::Format x2StorageFormat = x2->GetStorageFormat();
    if (x1StorageFormat != ge::Format::FORMAT_ND || x2StorageFormat != ge::Format::FORMAT_ND) {
        OP_LOGW("aclnnCdist only support ND format.");
    }
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, float& p, const int64_t compute_mode, aclTensor* out,
    SocVersion socVersion)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(x1, x2, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(x1, x2, out, socVersion), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查输入输出的shape，双输入shape能否做broadcast
    CHECK_COND(CheckShape(x1, x2, out), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    // 4. 检查输入数据的值是否合理
    CHECK_COND(CheckParamsLogic(p, compute_mode, socVersion), ACLNN_ERR_PARAM_INVALID, "CheckParamsLogic failed!");

    // 5. 检查数据格式
    CheckFormat(x1, x2);
    return ACLNN_SUCCESS;
}

static bool CheckBatchDimNum(const aclTensor* x) {
    // 检查batch轴是否为0
    auto xDimNum = x->GetViewShape().GetDimNum();
    for(int64_t i = 0; i < static_cast<int64_t>(xDimNum - 1); i++) {
        if(x->GetViewShape().GetDim(i) == 0) {
            return true;
        }
    }
    return false;
}

// 定义aclnnCdist的第一段接口
aclnnStatus aclnnCdistGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, float p, int64_t compute_mode, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnCdist, DFX_IN(x1, x2, p, compute_mode), DFX_OUT(out));

    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x1, x2, p, compute_mode, out, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* CdistOutRet = nullptr;

    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();

    // batch为0的时候，输出空tensor
    if(CheckBatchDimNum(x1) || CheckBatchDimNum(x2)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    bool needCast = x1->GetDataType() == op::DataType::DT_BF16 && (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93);
    // 将输入x1和x2转换成连续的tensor
    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 计算广播的形状
    op::Shape x1BroadcastShape;
    op::Shape x2BroadcastShape;
    int64_t x1Rank = x1Contiguous->GetViewShape().GetDimNum();
    int64_t x2Rank = x2Contiguous->GetViewShape().GetDimNum();
    int64_t minRank = std::min(x1Rank, x2Rank);
    int64_t maxRank = std::max(x1Rank, x2Rank);

    // 前面的维度进行广播
    for (int64_t i = 0; i < static_cast<int64_t>(maxRank - MIN_DIM_LEN); i++) {
        // 较小维度的tensor前面补1
        int64_t dim1 = (x1Rank < maxRank) ?
                           ((i < maxRank - minRank) ? 1 : x1Contiguous->GetViewShape().GetDim(i - maxRank + minRank)) :
                           x1Contiguous->GetViewShape().GetDim(i);
        int64_t dim2 = (x2Rank < maxRank) ?
                           ((i < maxRank - minRank) ? 1 : x2Contiguous->GetViewShape().GetDim(i - maxRank + minRank)) :
                           x2Contiguous->GetViewShape().GetDim(i);
        // 判断是否符合广播规则
        if(dim1 != dim2) {
            if(dim1 != 1 && dim2 != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X1 shape: %s and X2 shape: %s cannot broadcast!", 
                        op::ToString(x1->GetViewShape()).GetString(), op::ToString(x2->GetViewShape()).GetString());
                return ACLNN_ERR_PARAM_INVALID; // 无法广播
            }
        }
        int64_t broadcastDim = std::max(dim1, dim2);
        x1BroadcastShape.AppendDim(broadcastDim);
        x2BroadcastShape.AppendDim(broadcastDim);
    }
    for (int64_t i = 1; i >= 0; i--) {
        x1BroadcastShape.AppendDim(x1Contiguous->GetViewShape().GetDim(x1DimNum - i - 1));
        x2BroadcastShape.AppendDim(x2Contiguous->GetViewShape().GetDim(x2DimNum - i - 1));
    }

    if(x1Contiguous->GetViewShape().GetDim(x1DimNum - 1) == 0) {
        // 执行L0 Fill算子
        aclScalar* scalar = uniqueExecutor.get()->AllocScalar(0);
        auto valueTensor = uniqueExecutor.get()->ConvertToTensor(scalar, out->GetDataType());
        op::Shape outShape;
        for(int64_t i = 0; i < static_cast<int64_t>(maxRank - 1); i++) {
            int64_t dim = x1BroadcastShape.GetDim(i);
            outShape.AppendDim(dim);
        }
        outShape.AppendDim(x2BroadcastShape.GetDim(maxRank - MIN_DIM_LEN));
        auto outputDims = op::ToShapeVector(outShape);
        aclIntArray* dimArray = uniqueExecutor.get()->AllocIntArray(outputDims.data(), outputDims.size());
        CHECK_RET(dimArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto dimTensor = uniqueExecutor.get()->ConvertToTensor(dimArray, op::DataType::DT_INT64);
        CdistOutRet = l0op::Fill(dimTensor, valueTensor, dimArray, uniqueExecutor.get());
    } else {
        // 将广播后的形状转换为数组
        op::FVector<int64_t, op::MAX_DIM_NUM> x1BroadcastDims = op::ToShapeVector(x1BroadcastShape);
        auto x1BroadcastShapeArray = uniqueExecutor.get()->AllocIntArray(x1BroadcastDims.data(), x1BroadcastDims.size());
        op::FVector<int64_t, op::MAX_DIM_NUM> x2BroadcastDims = op::ToShapeVector(x2BroadcastShape);
        auto x2BroadcastShapeArray = uniqueExecutor.get()->AllocIntArray(x2BroadcastDims.data(), x2BroadcastDims.size());

        // 执行广播操作
        auto x1Broadcast = l0op::BroadcastTo(x1Contiguous, x1BroadcastShapeArray, uniqueExecutor.get());
        CHECK_RET(x1Broadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto x2Broadcast = l0op::BroadcastTo(x2Contiguous, x2BroadcastShapeArray, uniqueExecutor.get());
        CHECK_RET(x2Broadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
            if (needCast) {
                x1Broadcast = l0op::Cast(x1Broadcast, op::DataType::DT_FLOAT, uniqueExecutor.get());
                CHECK_RET(x1Broadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
                x2Broadcast = l0op::Cast(x2Broadcast, op::DataType::DT_FLOAT, uniqueExecutor.get());
                CHECK_RET(x2Broadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }

            // 获取算子输入所需的shape，将其输入进行broadcast
            gert::Shape x1Shape = x1Broadcast->GetViewShape();
            gert::Shape x2Shape = x2Broadcast->GetViewShape();
            size_t dimNum = x1Shape.GetDimNum();
            int64_t tensorR = x2Shape.GetDim(dimNum - 2);
            int64_t tensorM = x1Shape.GetDim(dimNum - 1);

            gert::Shape x1UnsqueezedShape = x1Shape;
            gert::Shape x2UnsqueezedShape = x2Shape;
            x1UnsqueezedShape.AppendDim(tensorM);
            x2UnsqueezedShape.AppendDim(tensorM);
            x1UnsqueezedShape.SetDim(dimNum - 1, 1);
            x2UnsqueezedShape.SetDim(dimNum - 2, 1);
            x2UnsqueezedShape.SetDim(dimNum - 1, tensorR);
            auto x1Reshape = l0op::Reshape(x1Broadcast, x1UnsqueezedShape, uniqueExecutor.get());
            auto x2Reshape = l0op::Reshape(x2Broadcast, x2UnsqueezedShape, uniqueExecutor.get());

            gert::Shape finalInputShape = x1UnsqueezedShape;
            finalInputShape.SetDim(dimNum - 1, x2UnsqueezedShape[dimNum - 1]);

            FVector<int64_t, MAX_DIM_NUM> broadcastDims = ToShapeVector(finalInputShape);
            aclIntArray* inputBroadcastShapeArray =
                uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
            CHECK_RET(inputBroadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
            x1Broadcast = l0op::BroadcastTo(x1Reshape, inputBroadcastShapeArray, uniqueExecutor.get());
            x2Broadcast = l0op::BroadcastTo(x2Reshape, inputBroadcastShapeArray, uniqueExecutor.get());
        }

        // 进行计算
        CdistOutRet = l0op::Cdist(x1Broadcast, x2Broadcast, p, compute_mode, uniqueExecutor.get());

        if (needCast) {
            CdistOutRet = l0op::Cast(CdistOutRet, op::DataType::DT_BF16, uniqueExecutor.get());
            CHECK_RET(CdistOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    CHECK_RET(CdistOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out支持非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(CdistOutRet, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCdist(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCdist);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif