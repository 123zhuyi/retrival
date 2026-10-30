/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_pow.h"
#include "pow.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/fill/op_api/fill.h"
#include "math/pows/op_host/op_api/pows.h"
#include "aclnn_kernels/cast.h"
#include "math/square/op_api/square.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

const float SQRT_EXP = 0.5;
const float NOOP_EXP = 1.0;
const float SQUARE_EXP = 2.0;
const float CUBE_EXP = 3.0;
const float NEGTIVE_SQRT_EXP = -0.5;
const float NEGTIVE_ONE_EXP = -1.0;
const float NEGTIVE_SQUARE_EXP = -2.0;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,
    op::DataType::DT_INT16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> SQUARE_NEED_CAST_DTYPE_LIST = {
  op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL, op::DataType::DT_INT16
};

static const std::initializer_list<op::DataType> POWS_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16,  op::DataType::DT_FLOAT, op::DataType::DT_BF16
};

static op::DataType GetScalarDefaultDtype(const op::DataType input) {
  if (IsComplexType(input)) {
    return op::DataType::DT_COMPLEX64;
  } else if (IsFloatingType(input)) {
    return op::DataType::DT_FLOAT;
  }
  return input;
}

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

static bool CheckPowTensorScalarNotNull(const aclTensor *self, const aclScalar *exponent, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(exponent, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckPowScalarTensorNotNull(const aclScalar *self, const aclTensor *exponent, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(exponent, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckSocVersionIsSupportBf16(void) {
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

// 判断910B芯片上，pow是否走AICPU路径
static inline bool IsPowAiCpuOn910B(const op::DataType dtype, const aclScalar *self, const aclTensor *exponent) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion < SocVersion::ASCEND910B || socVersion > SocVersion::ASCEND910E) {
    return false;
  }
  if (IsRegBase()) {
    return false;
  }
  // self或exponent为bool或bf16类型时，不走AICPU路径
  const auto selfDtype = self->GetDataType();
  const auto expDtype = exponent->GetDataType();
  if (selfDtype == op::DataType::DT_BOOL || expDtype == op::DataType::DT_BOOL ||
      selfDtype == op::DataType::DT_BF16 || expDtype == op::DataType::DT_BF16) {
    return false;
  }
  // 910B AICORE支持的dtype
  static const std::initializer_list<op::DataType> AICORE_DTYPE_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_BF16};
  return !CheckType(dtype, AICORE_DTYPE_LIST);
}

static bool CheckDtypeValid(const op::DataType selfDtype, const op::DataType expDtype, const op::DataType outDtype) {
  if (!CheckSocVersionIsSupportBf16() &&
     (selfDtype == op::DataType::DT_BF16 || expDtype == op::DataType::DT_BF16)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of pow is not support bfloat16 in current socversion.");
    return false;
  }
  if (!CheckType(selfDtype, DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s should be in dtype support list %s.",
            op::ToString(selfDtype).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
    return false;
  }
  if (!CheckType(expDtype, DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "exp dtype %s should be in dtype support list %s.",
            op::ToString(expDtype).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
    return false;
  }
  // 检查self和exponent能否做数据类型推导
  op::DataType promoteType = op::PromoteType(selfDtype, expDtype);
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and exponent dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(expDtype).GetString());
    return false;
  }
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
  return true;
}

static inline bool isFloatType(const DataType type) {
  return type == op::DataType::DT_DOUBLE || type == op::DataType::DT_FLOAT ||
         type == op::DataType::DT_FLOAT16 || type == op::DataType::DT_BF16;
}

static inline op::DataType InferTensorScalarDtype(const aclTensor *self, const aclScalar* exponent,
                                                  const aclTensor *out) {
  if (IsRegBase()) {
    auto scalarDefaultDtype = GetScalarDefaultDtype(exponent->GetDataType());
    auto promoteType = CombineCategoriesWithComplex(self->GetDataType(), scalarDefaultDtype);
    if (promoteType == DataType::DT_COMPLEX32) {
      promoteType = DataType::DT_COMPLEX64;
    }
    return promoteType;
  }
  if (exponent->GetDataType() == op::DataType::DT_DOUBLE && out->GetDataType() == op::DataType::DT_FLOAT) {
    return op::DataType::DT_FLOAT;
  }

  if (IsComplexType(exponent->GetDataType())) {
    return PromoteType(self->GetDataType(), exponent->GetDataType());
  }
  return isFloatType(self->GetDataType()) ? self->GetDataType() :
         ((isFloatType(exponent->GetDataType()) || self->GetDataType() == op::DataType::DT_BOOL) ?
          PromoteType(self->GetDataType(), exponent->GetDataType()) : self->GetDataType());
}

static inline op::DataType InferScalarTensorDtype(const aclScalar *self, const aclTensor* exponent,
                                                  const aclTensor *out) {
  if (IsRegBase()) {
    auto scalarDefaultDtype = GetScalarDefaultDtype(self->GetDataType());
    auto promoteType = CombineCategoriesWithComplex(exponent->GetDataType(), scalarDefaultDtype);
    if (promoteType == DataType::DT_COMPLEX32) {
      promoteType = DataType::DT_COMPLEX64;
    }
    return promoteType;
  }
  if (exponent->GetDataType() == op::DataType::DT_DOUBLE && out->GetDataType() == op::DataType::DT_FLOAT) {
    return op::DataType::DT_FLOAT;
  }

  if (IsComplexType(self->GetDataType())) {
    return PromoteType(self->GetDataType(), exponent->GetDataType());
  }
  return isFloatType(exponent->GetDataType()) ? exponent->GetDataType() :
         ((isFloatType(self->GetDataType()) || exponent->GetDataType() == op::DataType::DT_BOOL) ?
          PromoteType(exponent->GetDataType(), self->GetDataType()) : exponent->GetDataType());
}

static bool CheckPromoteType(const op::DataType selfDtype, const op::DataType exponentDtype,
                             const op::DataType outDtype, op::DataType promoteType) {
  if (promoteType == op::DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and exponent dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(exponentDtype).GetString());
    return false;
  }
  if ((selfDtype == op::DataType::DT_BOOL) && (exponentDtype == op::DataType::DT_BOOL)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self and exponent dtype are bool is not supported.");
    return false;
  }
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static aclnnStatus CheckPowTensorScalarParams(const aclTensor *self, const aclScalar* exponent,
                                              const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckPowTensorScalarNotNull(self, exponent, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self->GetDataType(), exponent->GetDataType(), out->GetDataType()),
            ACLNN_ERR_PARAM_INVALID);

  op::DataType promoteType = InferTensorScalarDtype(self, exponent, out);
  CHECK_RET(CheckPromoteType(self->GetDataType(), exponent->GetDataType(), out->GetDataType(), promoteType),
            ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入shape
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool CheckPowTensorScalarExponet(const DataType inputDtype, const aclScalar* exponent) {
  // promoteType为整形的情况，exponent需要大于0
  if (IsIntegralType(inputDtype) && (exponent->ToInt64() < 0)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Base dtype is intergal and exponent negative integers is not allowed.");
    return false;
  }
  return true;;
}

static bool CheckNotOverflow(const DataType pmtType, const aclScalar *exponent) {
  int8_t overFlowFlag = 1;
  int8_t floatFlag = 1;
  int8_t intFlag = 2;
  int8_t complexFlag = 3;

  switch (pmtType) {
    case op::DataType::DT_FLOAT: {
      overFlowFlag = exponent->CheckOverflows<float>() ? overFlowFlag << floatFlag : overFlowFlag;
      break;
    }
    case op::DataType::DT_FLOAT16: {
      overFlowFlag = exponent->CheckOverflows<op::fp16_t>() ? overFlowFlag << floatFlag : overFlowFlag;
      break;
    }
    case op::DataType::DT_BF16: {
      overFlowFlag = exponent->CheckOverflows<op::bfloat16>() ? overFlowFlag << floatFlag : overFlowFlag;
      break;
    }
    case op::DataType::DT_INT8: {
      overFlowFlag = exponent->CheckOverflows<int8_t>() ? overFlowFlag << intFlag : overFlowFlag;
      break;
    }
    case op::DataType::DT_INT16: {
      overFlowFlag = exponent->CheckOverflows<int16_t>() ? overFlowFlag << intFlag : overFlowFlag;
      break;
    }
    case op::DataType::DT_INT32: {
      overFlowFlag = exponent->CheckOverflows<int32_t>() ? overFlowFlag << intFlag : overFlowFlag;
      break;
    }
    case op::DataType::DT_INT64: {
      overFlowFlag = exponent->CheckOverflows<int64_t>() ? overFlowFlag << intFlag : overFlowFlag;
      break;
    }
    case op::DataType::DT_UINT8: {
      overFlowFlag = exponent->CheckOverflows<uint8_t>() ? overFlowFlag << intFlag : overFlowFlag;
      break;
    }
    case op::DataType::DT_COMPLEX32:
    case op::DataType::DT_COMPLEX64: {
      overFlowFlag = exponent->CheckOverflows<std::complex<float>>() ? overFlowFlag << complexFlag : overFlowFlag;
      break;
    }
    default: {
      return true;
    }
  }

  if ((overFlowFlag >> floatFlag) == 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "exponent value cannot be converted to promote type %s without overflow: %lf.",
            op::ToString(pmtType).GetString(), exponent->ToDouble());
  } else if ((overFlowFlag >> intFlag) == 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "exponent value cannot be converted to promote type %s without overflow: %ld.",
            op::ToString(pmtType).GetString(), exponent->ToInt64());
  } else if ((overFlowFlag >> complexFlag) == 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "exponent value cannot be converted to promote type %s without overflow: real is %f, imag is %f.",
            op::ToString(pmtType).GetString(), exponent->ToComplex64().real(), exponent->ToComplex64().imag());
  }
  return overFlowFlag == 1;
}

static aclnnStatus CheckPowScalarTensorParams(const aclScalar *self, const aclTensor* exponent,
                                              const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckPowScalarTensorNotNull(self, exponent, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self->GetDataType(), exponent->GetDataType(), out->GetDataType()),
            ACLNN_ERR_PARAM_INVALID);

  op::DataType promoteType = InferScalarTensorDtype(self, exponent, out);
  CHECK_RET(CheckPromoteType(self->GetDataType(), exponent->GetDataType(), out->GetDataType(), promoteType),
            ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入shape
  CHECK_RET(CheckShape(exponent, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool CheckSupportPows(const aclTensor *selfCast, const aclScalar *exponent) {
  if (exponent->ToFloat() != SQRT_EXP && exponent->ToFloat() != SQUARE_EXP &&
      exponent->ToFloat() != CUBE_EXP && exponent->ToFloat() != NEGTIVE_SQRT_EXP &&
      exponent->ToFloat() != NEGTIVE_ONE_EXP && exponent->ToFloat() != NEGTIVE_SQUARE_EXP) {
    return false; 
  }

  if (!CheckType(selfCast->GetDataType(), POWS_DTYPE_SUPPORT_LIST)) {
    return false; 
  }

  if(GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P && 
     GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B && 
     GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
    return false;
  }

  return true;
}

static void CheckFormat(const aclTensor* self){
  ge::Format selfStorageFormat = self->GetStorageFormat();
  if (selfStorageFormat != ge::Format::FORMAT_ND){
    OP_LOGW("aclnnPowTensorScalar/aclnnInplacePowTensorScalar only support format ND.");
  }
}

aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(const aclTensor *self,
                                                 const aclScalar *exponent,
                                                 const aclTensor *out,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnPowTensorScalar, DFX_IN(self, exponent), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckPowTensorScalarParams(self, exponent, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  CheckFormat(self);

  // pow算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  auto promoteType = InferTensorScalarDtype(self, exponent, out);
  if (IsRegBase()) {
    // promoteType为整形的情况，exponent需要大于0
    CHECK_RET(CheckPowTensorScalarExponet(promoteType, exponent), ACLNN_ERR_PARAM_INVALID);

    // 检查exponent是否溢出
    CHECK_RET(CheckNotOverflow(promoteType, exponent), ACLNN_ERR_PARAM_INVALID);
  }
  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfCast = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  aclTensor* powOut = nullptr;
  bool canUseSquare = 
      static_cast<float>(exponent->ToFloat()) == SQUARE_EXP &&
      (!IsRegBase() ||
       (IsRegBase() && (selfCast->GetDataType() == op::DataType::DT_FLOAT ||
                                                   selfCast->GetDataType() == op::DataType::DT_BF16 ||
                                                   selfCast->GetDataType() == op::DataType::DT_FLOAT16 ||
                                                   selfCast->GetDataType() == op::DataType::DT_INT64 ||
                                                   selfCast->GetDataType() == op::DataType::DT_COMPLEX64 ||
                                                   selfCast->GetDataType() == op::DataType::DT_COMPLEX128)));
  bool canNoUseOp = static_cast<float>(exponent->ToFloat()) == NOOP_EXP && !IsRegBase();
  if (CheckSupportPows(selfCast, exponent)) {
    auto expTensor = uniqueExecutor.get()->ConvertToTensor(exponent, promoteType);
    CHECK_RET(expTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 调用pows进行计算
    powOut = const_cast<aclTensor *>(l0op::Pows(selfCast, expTensor, uniqueExecutor.get()));
  } else if (canUseSquare) {
    const aclTensor *squareInput = selfCast;
    if (CheckType(selfCast->GetDataType(), SQUARE_NEED_CAST_DTYPE_LIST)) {
      squareInput = l0op::Cast(selfCast, op::DataType::DT_INT32, uniqueExecutor.get());
      CHECK_RET(squareInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 当exponent为2.0时，使用square算子计算
    powOut = const_cast<aclTensor *>(l0op::Square(squareInput, uniqueExecutor.get()));
  } else if (canNoUseOp) {
    powOut = const_cast<aclTensor *>(selfCast);
  } else {
    auto expTensor = uniqueExecutor.get()->ConvertToTensor(exponent, promoteType);
    CHECK_RET(expTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用pow进行计算
    powOut = const_cast<aclTensor *>(l0op::Pow(selfCast, expTensor, uniqueExecutor.get()));
  }
  CHECK_RET(powOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(powOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPowTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnPowTensorScalar);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(const aclTensor *self,
                                                        const aclScalar *exponent,
                                                        uint64_t *workspaceSize,
                                                        aclOpExecutor **executor) {
  auto out = const_cast<aclTensor*>(self);
  return aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, workspaceSize, executor);
}

aclnnStatus aclnnInplacePowTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplacePowTensorScalar);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// self为1时的fill(1)分支
static aclnnStatus BuildPowScalarTensorFillOne(const aclTensor *out, aclOpExecutor *executor) {
  FVector<int64_t> shape;
  for (size_t idx = 0; idx < out->GetViewShape().GetDimNum(); idx++) {
    int64_t tmpVal = out->GetViewShape().GetDim(idx);
    shape.push_back(tmpVal);
  }
  auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
  CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());
  CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  FVector<float> valVector = {1.0};
  auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
  CHECK_RET(valTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto powOut = l0op::Fill(dims, valTensor, shapeArray, executor);
  CHECK_RET(powOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(powOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

// pow计算分支：contiguous -> cast -> pow -> cast -> viewcopy
static aclnnStatus BuildPowScalarTensorCompute(const aclScalar *self, const aclTensor *exponent,
                                               const aclTensor *out, const op::DataType promoteType,
                                               aclOpExecutor *executor) {
  auto expContiguous = l0op::Contiguous(exponent, executor);
  CHECK_RET(expContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 910B且走AICPU时，exponent不cast到promoteType
  bool skipCastForAicpu = IsPowAiCpuOn910B(promoteType, self, exponent);
  const aclTensor* powExp = expContiguous;
  if (!skipCastForAicpu) {
    powExp = l0op::Cast(expContiguous, promoteType, executor);
    CHECK_RET(powExp != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  auto selfTensor = executor->ConvertToTensor(self, promoteType);
  CHECK_RET(selfTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用pow进行计算
  auto powOut = l0op::Pow(selfTensor, powExp, executor);
  CHECK_RET(powOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(powOut, out->GetDataType(), executor);
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(const aclScalar *self,
                                                 const aclTensor *exponent,
                                                 const aclTensor *out,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnPowScalarTensor, DFX_IN(self, exponent), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

   // 固定写法，参数检查
  auto ret = CheckPowScalarTensorParams(self, exponent, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // pow算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (exponent->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // fill(1) 分支
  if (IsRegBase() &&
      static_cast<float>(self->ToFloat()) == 1.0 &&
      !IsComplexType(exponent->GetDataType()) && !IsComplexType(out->GetDataType())) {
    CHECK_RET(BuildPowScalarTensorFillOne(out, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto promoteType = InferScalarTensorDtype(self, exponent, out);

  CHECK_RET(BuildPowScalarTensorCompute(self, exponent, out, promoteType,
             uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPowScalarTensor(void *workspace, uint64_t workspaceSize,
                                 aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnPowScalarTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
