/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ge_scalar.h"
#include "greater_equal.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                   DataType::DT_INT32,
                                                                                   DataType::DT_FLOAT16,
                                                                                   DataType::DT_INT8,
                                                                                   DataType::DT_DOUBLE,
                                                                                   DataType::DT_INT16,
                                                                                   DataType::DT_INT64,
                                                                                   DataType::DT_UINT64,
                                                                                   DataType::DT_UINT32,
                                                                                   DataType::DT_UINT16,
                                                                                   DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                    DataType::DT_INT32,
                                                                                    DataType::DT_FLOAT16,
                                                                                    DataType::DT_INT8,
                                                                                    DataType::DT_DOUBLE,
                                                                                    DataType::DT_INT16,
                                                                                    DataType::DT_INT64,
                                                                                    DataType::DT_UINT64,
                                                                                    DataType::DT_UINT32,
                                                                                    DataType::DT_UINT16,
                                                                                    DataType::DT_UINT8,
                                                                                    DataType::DT_BF16};

static const std::initializer_list<DataType> REGBASE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                      DataType::DT_INT32,
                                                                                      DataType::DT_FLOAT16,
                                                                                      DataType::DT_INT8,
                                                                                      DataType::DT_DOUBLE,
                                                                                      DataType::DT_INT16,
                                                                                      DataType::DT_INT64,
                                                                                      DataType::DT_UINT64,
                                                                                      DataType::DT_UINT32,
                                                                                      DataType::DT_UINT16,
                                                                                      DataType::DT_UINT8,
                                                                                      DataType::DT_BOOL,
                                                                                      DataType::DT_BF16};

static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_INT32,
                                                                       DataType::DT_FLOAT16, DataType::DT_INT8,
                                                                       DataType::DT_DOUBLE, DataType::DT_INT16,
                                                                       DataType::DT_INT64, DataType::DT_UINT64,
                                                                       DataType::DT_UINT32, DataType::DT_UINT16,
                                                                       DataType::DT_UINT8, DataType::DT_COMPLEX128,
                                                                       DataType::DT_COMPLEX64, DataType::DT_BOOL,
                                                                       DataType::DT_BF16};

static const size_t DIM_BOUND = 8;

static inline double GetCastedDouble(const aclTensor *self, const aclScalar *other) {
  double castedRes = 0;
  switch (self->GetDataType()) {
    case DataType::DT_FLOAT:
      castedRes = static_cast<double>(other->ToFloat());
      break;
    case DataType::DT_FLOAT16:
      castedRes = static_cast<double>(other->ToFp16());
      break;
    case DataType::DT_BF16:
      castedRes = static_cast<double>(other->ToBf16());
      break;
    default:
      castedRes = other->ToDouble();
      break;
  }
  return castedRes;
}

static inline bool IsDoubleEqual(double a, double b) {
  if (std::abs(a - b) <= std::numeric_limits<float>::epsilon()) {
    return true;
  }
  return false;
}

static bool CheckNotNull(const aclTensor *self, const aclScalar *other, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckSocExtraType(const DataType dtype) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (dtype == op::DataType::DT_BF16 && 
    (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch))) {
    return true;
  }
  return false;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (IsRegBase(npuArch)) {
    return REGBASE_DTYPE_SUPPORT_LIST;
  }
  if (npuArch == NpuArch::DAV_2201) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (IsRegBase(npuArch)) {
    const auto& supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  }
  // 检查out的数据类型是否在支持列表内，当前CanCast能力与cast算子不一致，需要在这里判断
  if (!CheckType(out->GetDataType(), OUT_DTYPE_SUPPORT_LIST) && !CheckSocExtraType(out->GetDataType())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGeScalar not implemented for out dtype %s," \
            "should be in dtype support list %s.", op::ToString(out->GetDataType()).GetString(),
            op::ToString(OUT_DTYPE_SUPPORT_LIST).GetString());
    return false;
  }

  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, DIM_BOUND, return false);
  OP_CHECK_MAX_DIM(out, DIM_BOUND, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
  return true;
}
static op::DataType InnerTypeToComplexType(const op::DataType input)
{
    switch (input) {
        case op::DataType::DT_BF16:
            // BFloat16 has range equivalent to Float,
            // so we map it to ComplexFloat.
            return op::DataType::DT_COMPLEX64;
        case op::DataType::DT_FLOAT16:
            return op::DataType::DT_COMPLEX32;
        case op::DataType::DT_FLOAT:
            return op::DataType::DT_COMPLEX64;
        case op::DataType::DT_DOUBLE:
            return op::DataType::DT_COMPLEX128;
        case op::DataType::DT_COMPLEX32:
            return op::DataType::DT_COMPLEX32;
        case op::DataType::DT_COMPLEX64:
            return op::DataType::DT_COMPLEX64;
        case op::DataType::DT_COMPLEX128:
            return op::DataType::DT_COMPLEX128;
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unknown Complex ScalarType for [%s]", ToString(input).GetString());
            return op::DataType::DT_UNDEFINED;
    }
}

static op::DataType CombineCategoriesWithComplex(const op::DataType higher, const op::DataType lower)
{
    if (IsComplexType(higher)) {
        return higher;
    } else if (IsComplexType(lower)) {
        // preserve value type of higher if it is floating type.
        if (IsFloatingType(higher)) {
            return InnerTypeToComplexType(higher);
        }
        // in case of integral input
        // lower complex takes precedence.
        return lower;
    } else if (IsFloatingType(higher)) {
        return higher;
    }
    if (higher == op::DataType::DT_BOOL || IsFloatingType(lower)) {
        return op::PromoteType(higher, lower);
    }
    if (higher != op::DataType::DT_UNDEFINED) {
        return higher;
    }
    return lower;
}

static op::DataType GetScalarDefaultDtype(const op::DataType input)
{
    if (IsComplexType(input)) {
        return op::DataType::DT_COMPLEX64;
    } else if (IsFloatingType(input)) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

static bool CheckPromoteType(const aclTensor *self, const aclScalar *other,
                             const aclTensor *out, DataType &promoteType) {
  const auto& supportList = GetDtypeSupportList();
  // 类型提升判断，将提升后的数据类型返回
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (IsRegBase(npuArch)) {
    auto scalarDefaultDtype = GetScalarDefaultDtype(other->GetDataType());
    promoteType = CombineCategoriesWithComplex(self->GetDataType(), scalarDefaultDtype);
    if (promoteType == DataType::DT_BOOL) {
      promoteType = DataType::DT_INT8;
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), promoteType, return false);
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), promoteType, return false);
  } else {
    promoteType = PromoteType(self->GetDataType(), other->GetDataType());
    // 如果有经过评审可以进行的转换，在这里添加并修改promoteType的值
    if (promoteType == DataType::DT_BOOL) {
      promoteType = DataType::DT_INT8;
    }
    if (other->GetDataType() == DataType::DT_DOUBLE && IsFloatingType(self->GetDataType())) {
      double afterCast = GetCastedDouble(self, other);
      promoteType = IsDoubleEqual(afterCast, other->ToDouble()) ? self->GetDataType() : promoteType;
    }
  }

  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            ToString(self->GetDataType()).GetString(), ToString(other->GetDataType()).GetString());
    return false;
  }

  if (!CheckType(promoteType, supportList) && !CheckSocExtraType(promoteType)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGeScalar not implemented for input dtype %s," \
            "should be in dtype support list %s.", ToString(promoteType).GetString(),
            op::ToString(supportList).GetString());
    return false;
  }
  // 检查计算结果的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(DataType::DT_BOOL, out->GetDataType(), return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclScalar *other, const aclTensor *out,
                               DataType &promoteType) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和other的数据类型能否提升
  CHECK_RET(CheckPromoteType(self, other, out, promoteType), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeScalarCommon(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  DataType promoteType;
  auto ret = CheckParams(self, other, out, promoteType);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将other转换为tensor，并且数据类型转换为推导后的数据类型
  auto otherTensor = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
  CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成推导后的数据类型
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子GreaterEqual进行计算
  auto greaterEqualResult = l0op::GreaterEqual(selfCasted, otherTensor, uniqueExecutor.get());
  CHECK_RET(greaterEqualResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成推导后的数据类型
  auto greaterEqualResultCasted = l0op::Cast(greaterEqualResult, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(greaterEqualResultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(greaterEqualResultCasted, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnGeScalar, DFX_IN(self, other), DFX_OUT(out));
  return aclnnGeScalarCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnGeScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGeScalar);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceGeScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceGeScalar, DFX_IN(selfRef, other), DFX_OUT(selfRef));
  return aclnnGeScalarCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceGeScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceGeScalar);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
