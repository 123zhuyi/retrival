/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_linspace.h"
#include "linspace.h"
#include "conversion/fill/op_api/fill.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* Linspace 算子的完整计算流程如下:
 *     start        steps      end
 *       |           |         |
 *        \          |        /
 *         \         |       /
 *          Linspace(workspace4)
 *                   |
 *            Cast(workspace5)
 *                   |
 *                ViewCopy
 *                   |
 *                 result
 */

// 算子支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
      op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
      op::DataType::DT_INT32, op::DataType::DT_INT8,
      op::DataType::DT_UINT8,
      // AiCpu支持数据类型
      op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
      op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
      op::DataType::DT_INT32, op::DataType::DT_INT16, op::DataType::DT_INT8,
      op::DataType::DT_UINT8, op::DataType::DT_INT64,
      // AiCpu支持数据类型
      op::DataType::DT_DOUBLE};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(NpuArch npuArch) {
  static const std::initializer_list<DataType> emptyDtypes = {};
  if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } 
  if (npuArch == NpuArch::DAV_1001 || npuArch == NpuArch::DAV_2002) {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
  return emptyDtypes;
}

// 检查输入是否是空指针
inline static bool CheckNotNull(const aclScalar *start, const aclScalar *end, const aclTensor *out) {
  OP_CHECK_NULL(start, return false);
  OP_CHECK_NULL(end, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

inline static bool CheckDtypeValid(const aclTensor *out)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    const auto& DTYPE_SUPPORT_LIST_CURRENT = GetSupportDtypeList(npuArch);
    if (DTYPE_SUPPORT_LIST_CURRENT.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return false;
    }
    // 检查out的数据类型是否在linspace算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_CURRENT, return false);
    return true;
}

inline static bool CheckScalarDtypeValid(const aclScalar *scalar)
{   
    if (IsComplexType(scalar->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "complex for input is not implemented.");
        return false;
    }
    return true; 
}

// 判断out数据类型，非COMPLEX则返回原数据类型，否则按情况返回
inline static DataType OutPromoteType(DataType outDataType) {
    if (outDataType == op::DataType::DT_COMPLEX128) {
        return op::DataType::DT_DOUBLE;
    }
    else if (outDataType == op::DataType::DT_COMPLEX64) {
        return op::DataType::DT_FLOAT;
    }
    else if (outDataType == op::DataType::DT_INT64) {
        return op::DataType::DT_FLOAT; 
    }
    else {
        return outDataType;
    }
}

// 检查参数是否符合算子的逻辑
inline static aclnnStatus CheckParamsLogic(const aclTensor *out, int64_t steps) {
    // steps不能小于0
    if (steps < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "number of steps must be non-negative.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 检查out size与steps是否相同
    int64_t outSize = 1;
    op::Shape outShape = out->GetViewShape();
    size_t outDimNum = outShape.GetDimNum();
    for (size_t idx = 0; idx < outDimNum; idx++) {
        outSize *= outShape.GetDim(idx);
    }
    if (outSize != steps) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "size of out must equal to steps, but got size of out %ld, steps: %ld.",
                outSize, steps);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(const aclTensor *out, const aclScalar *start, aclOpExecutor *executor) {
    FVector<int64_t> tmp = {1};
    auto dims = executor->ConvertToTensor(tmp.data(), tmp.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());

    auto valTensor = executor->ConvertToTensor(start, out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclScalar *start, const aclScalar *end, int64_t steps,
                               const aclTensor *out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(start, end, out), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckScalarDtypeValid(start), ACLNN_ERR_PARAM_INVALID); 
    CHECK_RET(CheckScalarDtypeValid(end), ACLNN_ERR_PARAM_INVALID); 
    CHECK_RET(CheckDtypeValid(out), ACLNN_ERR_PARAM_INVALID); 

    // 3. 检查输入的数据的值是否合理
    CHECK_RET(CheckParamsLogic(out, steps) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinspaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps,
                                          aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnLinspace, DFX_IN(start, end, steps), DFX_OUT(out));
    // 创建OpExcutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(start, end, steps, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 当输入为bool时，需转化类型为float
    float startFloatValue = start->ToFloat();
    float endFloatValue = end->ToFloat();
    auto startWithBool = (start->GetDataType() == op::DataType::DT_BOOL) ?
                        (uniqueExecutor.get()->AllocScalar(startFloatValue)) : start;
    auto endWithBool = (end->GetDataType() == op::DataType::DT_BOOL) ?
                      (uniqueExecutor.get()->AllocScalar(endFloatValue)) : end;

    // steps等于0时返回空tensor, steps等于1时返回start。
    if (steps == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    } else if (steps == 1) {
        ret = FillScalar(out, startWithBool, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    DataType promoteType = OutPromoteType(out->GetDataType());
    // start、end按照out的数据类型生成tensor
    auto startTensor = (uniqueExecutor.get())->ConvertToTensor(startWithBool, promoteType);
    auto endTensor = (uniqueExecutor.get())->ConvertToTensor(endWithBool, promoteType);
    // 执行L0算子，进行Linspace计算
    auto linspaceOutRet = l0op::Linspace(startTensor, endTensor, steps, uniqueExecutor.get());
    CHECK_RET(linspaceOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    // 将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(linspaceOutRet, out->GetDataType(), uniqueExecutor.get());
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

aclnnStatus aclnnLinspace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnLinspace);
    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
