/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ne_scalar.h"
#include "not_equal.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

static op::DataType InnerTypeToComplexType(const op::DataType input) {
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

static op::DataType CombineCategoriesWithComplex(const op::DataType higher, const op::DataType lower) {
  if(IsComplexType(higher)) {
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

static op::DataType GetScalarDefaultDtype(const op::DataType input) {
  if (IsComplexType(input)) {
    return op::DataType::DT_COMPLEX64;
  } else if (IsFloatingType(input)) {
    return op::DataType::DT_FLOAT;
  }
  return input;
}

// 列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT,  op::DataType::DT_INT32,     op::DataType::DT_INT64,      op::DataType::DT_FLOAT16,
  op::DataType::DT_INT16,  op::DataType::DT_INT8,      op::DataType::DT_UINT8,      op::DataType::DT_BOOL,
  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

// 910B和910_93支持数据类型，增加了op::DataType::DT_BF16
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
  op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_BOOL,
  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

// 950相对于之前，增加了DT_UINT64
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
  op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_BOOL,
  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16,
  op::DataType::DT_UINT64};

// 列出output所支持的dtype
// 除910B-950以外版本
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
  op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
  op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

//910B-910_93
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910B_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
  op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
  op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

// 950版本支持
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_REGBASE_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
  op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
  op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_BOOL, op::DataType::DT_UINT16,
  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetInputDtypeSupportList() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (IsRegBase()) {
    return REGBASE_DTYPE_SUPPORT_LIST;
  }
  if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  }
  return ASCEND910_DTYPE_SUPPORT_LIST;
}

static inline const std::initializer_list<op::DataType>& GetOutputDtypeSupportList() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (IsRegBase()){
    return OUT_DTYPE_SUPPORT_REGBASE_LIST;
  }
  if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E) {
    return OUT_DTYPE_SUPPORT_910B_LIST;
  }
  return OUT_DTYPE_SUPPORT_910_LIST;
}

static bool CheckNotNull(const aclTensor* self, const aclScalar* other, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool HasEmptyTensor(const aclTensor* self) {
  if (self->IsEmpty()) {
    return true;
  }

  return false;
}

static bool CheckDtypeValid(const aclTensor* self, const aclScalar* other, const aclTensor* out) {
  auto inputSupportList = GetInputDtypeSupportList();
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, inputSupportList, return false);

  // 检查other的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(other, inputSupportList, return false);
  auto outSuportList = IsRegBase()? 
                       GetOutputDtypeSupportList() : inputSupportList;

  // 检查out的数据类型
  OP_CHECK_DTYPE_NOT_SUPPORT(out, outSuportList, return false);

  return true;
}

static inline op::DataType PromoteTypeScalar(op::DataType selfDtype, op::DataType otherDtype) {
  if (IsRegBase()) {
    op::DataType promoteType;
    auto scalarDefaultDtype = GetScalarDefaultDtype(otherDtype);
    promoteType = CombineCategoriesWithComplex(selfDtype, scalarDefaultDtype);
    if (promoteType == op::DataType::DT_COMPLEX32){
      promoteType = op::DataType::DT_COMPLEX64;
    }
    return promoteType;
  }

  if (IsComplexType(selfDtype)) {
    return selfDtype;
  }

  if (!IsComplexType(otherDtype) && IsFloatingType(selfDtype)) {
    return selfDtype;
  }

  if (selfDtype == op::DataType::DT_BOOL || IsFloatingType(otherDtype) || IsComplexType(otherDtype)) {
    return op::PromoteType(selfDtype, otherDtype);
  }

  return selfDtype;
}

static bool CheckPromoteType(const aclTensor* self, const aclScalar* other, const aclTensor* out) {
  // 检查self和other能否做数据类型推导
  op::DataType promoteType = PromoteTypeScalar(self->GetDataType(), other->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
    return false;
  }
  if (IsRegBase()) {
    auto inputSupportList = GetInputDtypeSupportList();
    // 检查self的数据类型是否在NotEqual算子的支持列表内
    if (!CheckType(promoteType, inputSupportList)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "promote dtype %s should be in dtype support list [%s].",
        op::ToString(promoteType).GetString(), op::ToString(inputSupportList).GetString());
      return false;
    }
  }
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), promoteType, return false);
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), promoteType, return false);
  // 检查BOOL类型能否转换为输出的数据类型(算子返回的都是BOOL类型)
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(DataType::DT_BOOL, out->GetDataType(), return false);

  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out) {
  const size_t MAX_DIM = 8;
  OP_CHECK_MAX_DIM(self, MAX_DIM, return false);
  OP_CHECK_MAX_DIM(out, MAX_DIM, return false);

  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* other, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self的数据类型能否转换为输出数据类型
  CHECK_RET(CheckPromoteType(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnNeScalarCommon(const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize,
                                aclOpExecutor** executor) {
  // 固定写法，参数检查
  auto ret = CheckParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 空Tensor处理
  if (HasEmptyTensor(self)) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto promoteType = PromoteTypeScalar(self->GetDataType(), other->GetDataType());
  // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* otherTensor = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
  CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子Equal进行计算
  auto neResult = l0op::NotEqual(selfCasted, otherTensor, uniqueExecutor.get());
  CHECK_RET(neResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果(BOOL)转换成输出out的数据类型
  auto castOut = l0op::Cast(neResult, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnNeScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* other, aclTensor* out,
                                          uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnNeScalar, DFX_IN(self, other), DFX_OUT(out));
  return aclnnNeScalarCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnNeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnNeScalar);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceNeScalarGetWorkspaceSize(aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceNeScalar, DFX_IN(selfRef, other), DFX_OUT(selfRef));
  return aclnnNeScalarCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceNeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceNeScalar);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
