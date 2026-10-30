/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include "aclnn_range.h"
#include "arange.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

template <typename T>
inline static aclnnStatus CheckStep(T start, T end, T step)
{
    if (!(step > static_cast<T>(0) || step < static_cast<T>(0))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "step must be nonzero.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 在step大于0时结束值必须大于等于起始值；在step小于0时结束值必须小于等于起始值
    if ((step > static_cast<T>(0) && start > end) || (step < static_cast<T>(0) && start < end)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Range 算子的完整计算流程如下:
 *     start        step      end
 *       |           |         |
 *        \          |        /
 *         \         |       /
 *          Arange(workspace4)
 *                   |
 *            Cast(workspace5)
 *                   |
 *                ViewCopy
 *                   |
 *                 result
 */

// 根据API定义，需要列出Ascend910所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64,  DataType::DT_FLOAT16, DataType::DT_INT16,
    DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_BOOL};

static const std::initializer_list<DataType> ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_INT64, DataType::DT_INT32};

// 根据API定义，需要列出Ascend910B所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910B_INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64,  DataType::DT_FLOAT16, DataType::DT_INT16,
    DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_BOOL,    DataType::DT_BF16};

static const std::initializer_list<DataType> ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_DOUBLE,
    DataType::DT_INT64,   DataType::DT_INT32, DataType::DT_BF16};

// 检查输入是否是空指针
inline static bool CheckNotNull(
    const aclScalar* start, const aclScalar* end, const aclScalar* step, const aclTensor* out)
{
    OP_CHECK_NULL(start, return false);
    OP_CHECK_NULL(end, return false);
    OP_CHECK_NULL(step, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(
    const aclScalar* start, const aclScalar* end, const aclScalar* step, const aclTensor* out)
{
    // 获取芯片类型，判断芯片是否为Ascend910B
    bool isAscend910BSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
         IsRegBase());
    const std::initializer_list<op::DataType> CURRENT_INPUT_DTYPE_SUPPORT_LIST =
        isAscend910BSocVersion ? ASCEND910B_INPUT_DTYPE_SUPPORT_LIST : ASCEND910_INPUT_DTYPE_SUPPORT_LIST;
    const std::initializer_list<op::DataType> CURRENT_OUTPUT_DTYPE_SUPPORT_LIST =
        isAscend910BSocVersion ? ASCEND910B_OUTPUT_DTYPE_SUPPORT_LIST : ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST;

    // 检查start的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(start, CURRENT_INPUT_DTYPE_SUPPORT_LIST, return false);

    // 检查end的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(end, CURRENT_INPUT_DTYPE_SUPPORT_LIST, return false);

    // 检查step的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(step, CURRENT_INPUT_DTYPE_SUPPORT_LIST, return false);

    // 检查输出的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, CURRENT_OUTPUT_DTYPE_SUPPORT_LIST, return false);

    return true;
}

inline static aclnnStatus CheckStepCorrect(
    const aclScalar* start, const aclScalar* end, const aclScalar* step, const aclTensor* out)
{
    DataType outType = out->GetDataType();
    switch (outType) {
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
        case DataType::DT_FLOAT: {
            float startValueFloat = start->ToFloat();
            float endValueFloat = end->ToFloat();
            float stepValueFloat = step->ToFloat();
            if (CheckStep<float>(startValueFloat, endValueFloat, stepValueFloat) != ACLNN_SUCCESS) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "upper bound and lower bound inconsistent with step sign. start:%f, end:%f, step:%f.",
                    startValueFloat, endValueFloat, stepValueFloat);
                return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        case DataType::DT_DOUBLE: {
            double startValueDouble = start->ToDouble();
            double endValueDouble = end->ToDouble();
            double stepValueDouble = step->ToDouble();
            if (CheckStep<double>(startValueDouble, endValueDouble, stepValueDouble) != ACLNN_SUCCESS) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "upper bound and lower bound inconsistent with step sign. start:%lf, end:%lf, step:%lf.",
                    startValueDouble, endValueDouble, stepValueDouble);
                return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        case DataType::DT_INT32:
        case DataType::DT_INT64: {
            int64_t startValueInt64 = start->ToInt64();
            int64_t endValueInt64 = end->ToInt64();
            int64_t stepValueInt64 = step->ToInt64();
            if (CheckStep<int64_t>(startValueInt64, endValueInt64, stepValueInt64) != ACLNN_SUCCESS) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "upper bound and lower bound inconsistent with step sign. start:%ld, end:%ld, step:%ld.",
                    startValueInt64, endValueInt64, stepValueInt64);
                return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        default: {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputdtype invalid.");
            return ACLNN_ERR_PARAM_INVALID;
            break;
        }
    }
    return ACLNN_SUCCESS;
}

// 检查参数是否符合算子的逻辑
inline static aclnnStatus CheckParamsLogic(
    const aclScalar* start, const aclScalar* end, const aclScalar* step, const aclTensor* out)
{
    size_t dim_num = out->GetViewShape().GetDimNum();
    OP_CHECK(
        dim_num != 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected out a 1d tensor, but got %zu.", dim_num),
        return ACLNN_ERR_PARAM_INVALID);
    if (CheckStepCorrect(start, end, step, out) == ACLNN_ERR_PARAM_INVALID) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckParams(
    const aclScalar* start, const aclScalar* end, const aclScalar* step, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(start, end, step, out), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(start, end, step, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据的值是否合理
    CHECK_RET(CheckParamsLogic(start, end, step, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRangeGetWorkspaceSize(
    const aclScalar* start, const aclScalar* end, const aclScalar* step, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRange, DFX_IN(start, end, step), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(start, end, step, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool isClosed = true;
    const aclTensor* rangeOutRet = l0op::Arange(start, end, step, out, isClosed, uniqueExecutor.get());
    CHECK_RET(rangeOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(rangeOutRet, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRange(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRange);
    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
