/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_normal_out.h"
#include "math/add/op_api/add.h"
#include "math/mul/op_api/mul.h"
#include "random/stateless_random_normal_v2/op_api/stateless_random_normal_v2.h"
#include "random/stateless_random_normal_v3/op_api/stateless_random_normal_v3.h"
#include "random/stateless_normal/op_api/stateless_normal.h"
#include "aclnn_kernels/cast.h"
#include "conversion/view_copy/op_api/view_copy.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "aclnn/aclnn_base.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "../../random_common/op_api/aclnn_set_pytorch_random.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE};

// 表示mean,std的数据类型是Tensor还是Scalar
enum class ScalarMode { TensorTensor, TensorScalar, ScalarTensor, ScalarScalar };

/* 查看TensorFloat的Dtype和Shape */
static bool CheckTensorAndFloatDtype(const aclTensor* mean, const aclTensor* out)
{
    // 检查mean的数据类型是否在normal算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(mean, DTYPE_SUPPORT_LIST, return false);
    // 检查out的数据类型是否在normal算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckTensorAndFloatShapeOfMean(const aclTensor* mean, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(mean, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(mean, out, broadcastShape, return false);

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
                op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckTensorAndFloatNotNull(const aclTensor* mean, aclTensor* out)
{
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

/* 查看FloatTensor的Dtype和Shape */
static bool CheckFloatAndTensorDtype(const aclTensor* std, const aclTensor* out)
{
    // 检查std的数据类型是否在normal算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(std, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckFloatAndTensorShapeOfStd(const aclTensor* std, const aclTensor* out)
{
    if (std->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The std can not be an empty tensor.");
        return false;
    }
    if (!(out->IsEmpty() && std->Size() == 1)) {
        if (out->GetViewShape() != std->GetViewShape()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of std should be match with out.");
            return false;
        }
    }

    OP_CHECK_MAX_DIM(std, MAX_DIM_LEN, return false);
    return true;
}

static inline bool CheckFloatAndTensorNotNull(const aclTensor* std, aclTensor* out)
{
    OP_CHECK_NULL(std, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

/* 查看TensorTensor的Dtype和Shape */
static bool CheckTensorAndTensorDtype(const aclTensor* mean, const aclTensor* std, const aclTensor* out)
{
    // 检查std/mean的数据类型是否在normal算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(mean, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(std, DTYPE_SUPPORT_LIST, return false);
    // 检查out的数据类型是否在normal算子的可支持列表
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckTensorAndTensorShape(const aclTensor* mean, const aclTensor* std, const aclTensor* out)
{
    // 检查std和mean的维度是否大于8
    OP_CHECK_MAX_DIM(mean, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(std, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(mean, std, broadcastShape, return false);

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
                op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckTensorAndTensorNotNull(const aclTensor* mean, const aclTensor* std, aclTensor* out)
{
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(std, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckPromoteType(const aclTensor* mean, const aclTensor* std, const aclTensor* out,
                             op::DataType promoteType)
{
    // 检查self和other能否做数据类型推导
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Mean dtype [%s] and std dtype [%s] can not promote dtype.",
                op::ToString(mean->GetDataType()).GetString(), op::ToString(std->GetDataType()).GetString());
        return false;
    }
    // 检查推导后的数据类型是否能转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static aclnnStatus CheckTensorAndFloatParams(const aclTensor* mean, aclTensor* out)
{
    // 检查空指针的场景
    CHECK_RET(CheckTensorAndFloatNotNull(mean, out), ACLNN_ERR_PARAM_NULLPTR);
    // 检查mean的shape场景
    CHECK_RET(CheckTensorAndFloatShapeOfMean(mean, out), ACLNN_ERR_PARAM_INVALID);
    // 检查dtype类型判断场景
    CHECK_RET(CheckTensorAndFloatDtype(mean, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFloatAndTensorParams(const aclTensor* std, aclTensor* out)
{
    // 检查空指针的场景
    CHECK_RET(CheckFloatAndTensorNotNull(std, out), ACLNN_ERR_PARAM_NULLPTR);
    // 检查std的shape场景
    CHECK_RET(CheckFloatAndTensorShapeOfStd(std, out), ACLNN_ERR_PARAM_INVALID);
    // 检查dtype类型判断场景
    CHECK_RET(CheckFloatAndTensorDtype(std, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFloatAndFloatParams(aclTensor* out)
{
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorAndTensorParams(const aclTensor* mean, const aclTensor* std, aclTensor* out)
{
    // 检查空指针的场景
    CHECK_RET(CheckTensorAndTensorNotNull(mean, std, out), ACLNN_ERR_PARAM_NULLPTR);
    // 检查mean和std的shape场景
    CHECK_RET(CheckTensorAndTensorShape(mean, std, out), ACLNN_ERR_PARAM_INVALID);
    // 检查dtype类型判断场景
    CHECK_RET(CheckTensorAndTensorDtype(mean, std, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

/**
 * common部分的调用流程
 */
aclnnStatus CommonLogicGeneralNormal(const aclTensor* mean, const aclTensor* std, int64_t seed, int64_t offset,
                                     aclTensor* self, aclTensor* out, UniqueExecutor& uniqueExecutor,
                                     uint64_t* workspaceSize, aclOpExecutor** executor,
                                     ScalarMode scalarMode = ScalarMode::TensorTensor)
{
    const aclTensor* addOut = nullptr;

    if (!aclnnGetPytorchRandom() && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
        self->GetDataType() != DataType::DT_DOUBLE) {
        OP_LOGD("compat mode, use V3 Normal");
        // seed转化为key
        FVector<int64_t, op::MAX_DIM_NUM> key_vec = {seed};
        auto keyArr = (uniqueExecutor.get())->AllocIntArray(key_vec.data(), key_vec.size());

        // offset转化为counter
        FVector<int64_t, op::MAX_DIM_NUM> counter_vec = {0, offset};
        auto counterArr = (uniqueExecutor.get())->AllocIntArray(counter_vec.data(), counter_vec.size());
        // V3 kernel 要求 mean/std 参数为 DT_FLOAT
        auto meanCasted = l0op::Cast(mean, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(meanCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto stdCasted = l0op::Cast(std, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(stdCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // V3 kernel 不支持广播，需要将 shape 与 out 不一致的多元素 tensor 广播到 out shape
        // Size()==1 的 tensor（scalar 转换而来）由 V3 kernel 的 scalar 路径原生处理，无需广播
        // self 需要单独广播以保持原始 dtype（决定 V3 kernel 的计算精度）
        bool selfNeedBcast = self->GetViewShape() != out->GetViewShape();
        bool meanNeedBcast = meanCasted->GetViewShape() != out->GetViewShape() && mean->Size() > 1;
        bool stdNeedBcast = stdCasted->GetViewShape() != out->GetViewShape() && std->Size() > 1;
        if (selfNeedBcast || meanNeedBcast || stdNeedBcast) {
            op::FVector<int64_t, op::MAX_DIM_NUM> outDims = op::ToShapeVector(out->GetViewShape());
            auto outShapeArray = uniqueExecutor.get()->AllocIntArray(outDims.data(), outDims.size());
            CHECK_RET(outShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

            if (selfNeedBcast) {
                auto selfBroadcast = l0op::BroadcastTo(self, outShapeArray, uniqueExecutor.get());
                CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
                self = const_cast<aclTensor*>(selfBroadcast);
            }
            if (meanNeedBcast) {
                meanCasted = l0op::BroadcastTo(meanCasted, outShapeArray, uniqueExecutor.get());
                CHECK_RET(meanCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
            if (stdNeedBcast) {
                stdCasted = l0op::BroadcastTo(stdCasted, outShapeArray, uniqueExecutor.get());
                CHECK_RET(stdCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
        addOut = l0op::StatelessRandomNormalV3(self, keyArr, counterArr, meanCasted, stdCasted, uniqueExecutor.get());
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
               self->GetDataType() != DataType::DT_DOUBLE) {
        if (scalarMode == ScalarMode::TensorTensor || scalarMode == ScalarMode::ScalarTensor) {
            FVector<float> zeroMeanVector = {0.0f};
            auto zeroMeanTensor = uniqueExecutor.get()->ConvertToTensor(zeroMeanVector.data(), zeroMeanVector.size(),
                                                                        op::DataType::DT_FLOAT);
            FVector<float> oneStdVector = {1.0f};
            auto oneStdTensor = uniqueExecutor.get()->ConvertToTensor(oneStdVector.data(), oneStdVector.size(),
                                                                      op::DataType::DT_FLOAT);

            auto normalOut = l0op::StatelessNormal(self, seed, offset, zeroMeanTensor, oneStdTensor,
                                                   uniqueExecutor.get());
            CHECK_RET(normalOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto mulOut = l0op::Mul(std, normalOut, uniqueExecutor.get());
            CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            // ScalarTensor 模式下，mean的类型一定为f32，mulout数据类型可为f16\f32\bf16，但mean是scalar,
            // 故不能用AddInplace
            if (scalarMode == ScalarMode::ScalarTensor) {
                addOut = l0op::Add(mulOut, mean, uniqueExecutor.get());
            } else {
                addOut = l0op::AddInplace(mean, mulOut, uniqueExecutor.get());
            }
        } else if (scalarMode == ScalarMode::TensorScalar) {
            FVector<float> zeroMeanVector = {0.0f};
            auto zeroMeanTensor = uniqueExecutor.get()->ConvertToTensor(zeroMeanVector.data(), zeroMeanVector.size(),
                                                                        op::DataType::DT_FLOAT);

            auto normalOut = l0op::StatelessNormal(self, seed, offset, zeroMeanTensor, std, uniqueExecutor.get());
            CHECK_RET(normalOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            addOut = l0op::AddInplace(mean, normalOut, uniqueExecutor.get());
        } else {
            addOut = l0op::StatelessNormal(self, seed, offset, mean, std, uniqueExecutor.get());
        }
    } else {
        // V2 路径：seed 转化为 key，offset 转化为 counter
        // 设置算法序号 int64_t -> scalar -> tensor
        int64_t alg = 1;
        auto algScalar = (uniqueExecutor.get())->AllocScalar((void*)&alg, DataType::DT_INT32);
        const aclTensor* algTensor = (uniqueExecutor.get())->ConvertToTensor(algScalar, op::ToOpDataType(ACL_INT32));

        FVector<int64_t, op::MAX_DIM_NUM> key_vec = {seed};
        auto keyArr = (uniqueExecutor.get())->AllocIntArray(key_vec.data(), key_vec.size());

        FVector<int64_t, op::MAX_DIM_NUM> counter_vec = {0, offset};
        auto counterArr = (uniqueExecutor.get())->AllocIntArray(counter_vec.data(), counter_vec.size());

        // 调用normal_算子kernel function(AI Cpu算子)
        auto stateLessOut = l0op::StatelessRandomNormalV2(self, keyArr, counterArr, algTensor, uniqueExecutor.get());
        CHECK_RET(stateLessOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用mul_算子kernel function(AI Core算子)，语义为 mulOut = std * stateLessOut。
        // stateLessOut是刚分配、算完即弃的中间tensor，两个分支都想把乘法输出绑定到它以省内存，
        // 但绑定机制不同，导致入参顺序相反：
        //   - MulInplace 复用【第二个】入参的内存做输出，故把 stateLessOut 放第二位，直接就地写、不再分配；
        //     前提是它已是完整shape且dtype能装下结果，即下面 canMulInplace 的两个条件。
        //   - 普通 Mul 会新分配内存，其输出dtype取【第一个】入参，为让新内存对齐 stateLessOut 的 dtype，
        //     故把 stateLessOut 放第一位。
        bool canMulInplace = stateLessOut->GetViewShape() == out->GetViewShape() &&
                             std->GetDataType() == stateLessOut->GetDataType();
        auto mulOut = canMulInplace ? l0op::MulInplace(std, stateLessOut, uniqueExecutor.get()) :
                                      l0op::Mul(stateLessOut, std, uniqueExecutor.get());
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 如果类型不一致，先做类型提升，再进行ADD算子运算
        auto meanCast = mean;
        auto mulOutCast = mulOut;
        auto meanType = mean->GetDataType();
        auto mulOutType = mulOut->GetDataType();
        if (meanType != mulOutType) {
            auto promoteType = op::PromoteType(meanType, mulOutType);
            meanCast = l0op::Cast(mean, promoteType, uniqueExecutor.get());
            CHECK_RET(meanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            mulOutCast = l0op::Cast(mulOut, promoteType, uniqueExecutor.get());
            CHECK_RET(mulOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 调用add_算子kernel function(AI Core算子)
        // mulOutCast是中间tensor，复用其内存做输出，省一次内存分配。
        // 上面的类型提升已保证meanCast与mulOutCast的dtype一致，且mean恒能广播进mulOutCast
        addOut = l0op::AddInplace(meanCast, mulOutCast, uniqueExecutor.get());
    }
    CHECK_RET(addOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果转换成输出self的数据类型
    auto castOut = l0op::Cast(addOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出self上，self可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// normal.Tensor_Tensor_out
aclnnStatus aclnnNormalTensorTensorGetWorkspaceSize(const aclTensor* mean, const aclTensor* std, int64_t seed,
                                                    int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNormalTensorTensor, DFX_IN(mean, std, seed, offset), DFX_OUT(out));

    // 固定写法， 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数dtype检查
    auto ret = CheckTensorAndTensorParams(mean, std, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if ((mean->IsEmpty() || std->IsEmpty()) && out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 需要对mean和std两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType = op::PromoteType(mean->GetDataType(), std->GetDataType());
    CHECK_RET(CheckPromoteType(mean, std, out, promoteType), ACLNN_ERR_PARAM_INVALID);

    // 将输入mean转换为特定类型的连续tensor
    auto meanContiguous = l0op::Contiguous(mean, uniqueExecutor.get());
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanCasted = l0op::Cast(meanContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(meanCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入std转换为特定类型的连续tensor
    auto stdContiguous = l0op::Contiguous(std, uniqueExecutor.get());
    CHECK_RET(stdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto stdCasted = l0op::Cast(stdContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(stdCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 拷贝mean副本
    auto self = const_cast<aclTensor*>(meanCasted);
    return CommonLogicGeneralNormal(meanCasted, stdCasted, seed, offset, self, out, uniqueExecutor, workspaceSize,
                                    executor, ScalarMode::TensorTensor);
}

// normal.Tensor_float_out
aclnnStatus aclnnNormalTensorFloatGetWorkspaceSize(const aclTensor* mean, float std, int64_t seed, int64_t offset,
                                                   aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNormalTensorFloat, DFX_IN(mean, std, seed, offset), DFX_OUT(out));

    // 固定写法， 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，检查参数mean的shape和dtype
    auto ret = CheckTensorAndFloatParams(mean, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto meanDim = mean->GetViewShape().GetDimNum();
    auto outDim = out->GetViewShape().GetDimNum();
    if ((meanDim == outDim) && (mean->IsEmpty() || mean->Size() == 1) && out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入mean转换为连续的tensor
    auto meanContiguous = l0op::Contiguous(mean, uniqueExecutor.get());
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入std转换为连续的tensor（保持DT_FLOAT精度，避免截断到bf16/fp16）
    auto stdScalar = (uniqueExecutor.get())->AllocScalar(std);
    auto stdTensor = (uniqueExecutor.get())->ConvertToTensor(stdScalar, op::DataType::DT_FLOAT);
    auto stdContiguous = l0op::Contiguous(stdTensor, uniqueExecutor.get());
    CHECK_RET(stdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 拷贝mean副本
    auto self = const_cast<aclTensor*>(out);
    return CommonLogicGeneralNormal(meanContiguous, stdContiguous, seed, offset, self, out, uniqueExecutor,
                                    workspaceSize, executor, ScalarMode::TensorScalar);
}

// normal.float_Tensor_out
aclnnStatus aclnnNormalFloatTensorGetWorkspaceSize(float mean, const aclTensor* std, int64_t seed, int64_t offset,
                                                   aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNormalFloatTensor, DFX_IN(mean, std, seed, offset), DFX_OUT(out));

    // 固定写法， 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，检查参数std的shape和dtype
    auto ret = CheckFloatAndTensorParams(std, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto stdDim = std->GetViewShape().GetDimNum();
    auto outDim = out->GetViewShape().GetDimNum();
    if (stdDim == outDim && std->Size() == 1 && out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入mean转换为连续的tensor（保持DT_FLOAT精度，避免截断到bf16/fp16）
    auto meanScalar = (uniqueExecutor.get())->AllocScalar(mean);
    auto meanTensor = (uniqueExecutor.get())->ConvertToTensor(meanScalar, op::DataType::DT_FLOAT);
    auto meanContiguous = l0op::Contiguous(meanTensor, uniqueExecutor.get());
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入std转换为连续的tensor
    auto stdContiguous = l0op::Contiguous(std, uniqueExecutor.get());
    CHECK_RET(stdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 拷贝std副本
    auto self = const_cast<aclTensor*>(stdContiguous);
    return CommonLogicGeneralNormal(meanContiguous, stdContiguous, seed, offset, self, out, uniqueExecutor,
                                    workspaceSize, executor, ScalarMode::ScalarTensor);
}

// normal.float_float_out
aclnnStatus aclnnNormalFloatFloatGetWorkspaceSize(float mean, float std, int64_t seed, int64_t offset, aclTensor* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNormalFloatFloat, DFX_IN(mean, std, seed, offset), DFX_OUT(out));

    // 固定写法， 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，检查参数out
    auto ret = CheckFloatAndFloatParams(out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入mean转换为连续的tensor（保持DT_FLOAT精度，避免截断到bf16/fp16）
    auto meanScalar = (uniqueExecutor.get())->AllocScalar(mean);
    auto meanTensor = (uniqueExecutor.get())->ConvertToTensor(meanScalar, op::DataType::DT_FLOAT);
    auto meanContiguous = l0op::Contiguous(meanTensor, uniqueExecutor.get());
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入std转换为连续的tensor（保持DT_FLOAT精度，避免截断到bf16/fp16）
    auto stdScalar = (uniqueExecutor.get())->AllocScalar(std);
    auto stdTensor = (uniqueExecutor.get())->ConvertToTensor(stdScalar, op::DataType::DT_FLOAT);
    auto stdContiguous = l0op::Contiguous(stdTensor, uniqueExecutor.get());
    CHECK_RET(stdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 拷贝out副本
    auto self = const_cast<aclTensor*>(out);
    return CommonLogicGeneralNormal(meanContiguous, stdContiguous, seed, offset, self, out, uniqueExecutor,
                                    workspaceSize, executor, ScalarMode::ScalarScalar);
}

aclnnStatus aclnnNormalTensorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNormalTensorTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnNormalTensorFloat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNormalTensorFloat);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnNormalFloatTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNormalFloatTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnNormalFloatFloat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNormalFloatFloat);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
