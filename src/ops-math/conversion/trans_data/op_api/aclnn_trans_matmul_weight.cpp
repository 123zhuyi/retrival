/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_trans_matmul_weight.h"

#include "util/math_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "conversion/tensor_move/op_api/tensor_move.h"
using namespace op;

static const int MIN_DIM_NUM_ND = 2;
static const int MAX_DIM_NUM_ND = 6;
static const int64_t MIN_SIZE_PER_BLOCK_FLOAT16 = 16;
static const int64_t MIN_SIZE_PER_BLOCK_INT8 = 32;

namespace {
static const int MIN_INT8_DIM_NUM_ND = 2;
static const int MAX_INT8_DIM_NUM_ND = 6;
static const int MIN_FLOAT16_DIM_NUM_ND = 2;
static const int MAX_FLOAT16_DIM_NUM_ND = 6;  
}

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                                    DataType::DT_BF16,
                                                                                    DataType::DT_INT8,
                                                                                    DataType::DT_FLOAT8_E4M3FN,
                                                                                    DataType::DT_HIFLOAT8,
                                                                                    DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND310P_DTYPE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                                    DataType::DT_INT8};

static inline const std::initializer_list<DataType> &GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
    return ASCEND310P_DTYPE_DTYPE_SUPPORT_LIST;
  } else if (IsRegBase()) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static inline bool NeedToCovertToPravateFormat() {
  // For now. All cases will be transfered into Private format
  return true;
}

static inline bool CheckShapeDim(const aclIntArray *tensorShape) {
  uint64_t dimSize = tensorShape->Size();
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    // Only support matmul operations with dimSize is between 2 and 6
    if (dimSize < MIN_FLOAT16_DIM_NUM_ND || dimSize > MAX_FLOAT16_DIM_NUM_ND) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected that tensorShape has 2-6 dimensions, but got dimensions %lu.", dimSize);
      return false;
    }
  } else {
    // Only support matmul operation with dimSize is 2
    if (dimSize != MIN_FLOAT16_DIM_NUM_ND) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected that tensorShape has 2 dimensions, but got dimensions %lu.", dimSize);
      return false;
    }
  }
  return true;
}

static inline bool CheckShapeDimV2(const aclIntArray *tensorShape) {
  uint64_t dimSize = tensorShape->Size();
  // Only support matmul operations with dimSize is between 2 and 6
  if (dimSize < MIN_INT8_DIM_NUM_ND || dimSize > MAX_INT8_DIM_NUM_ND) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected that tensorShape has 2-6 dimensions, but got dimensions %lu.", dimSize);
    return false;
  }
  return true;
}

static inline bool CheckNonZeroShape(const aclIntArray *tensorShape) {
  uint64_t dimSize = tensorShape->Size();
  for (size_t idx = 0; idx < dimSize; idx++) {
    if ((*tensorShape)[idx] == 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "No zeros are allow in shape");
      return false;
    }
  }
  return true;
}

static aclnnStatus CheckSocValid()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND310P || socVersion == SocVersion::ASCEND910B ||
        socVersion == SocVersion::ASCEND910_93 || IsRegBase()) {
        return ACLNN_SUCCESS;
    } else {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return ACLNN_ERR_RUNTIME_ERROR;
    }
}

static uint64_t CalculateMatmulWeightSize(const aclIntArray *tensorShape, aclDataType dataType) {
  uint64_t dimSize = tensorShape->Size();
  int64_t shapeSize = 1;
  bool privateFlag = NeedToCovertToPravateFormat();
  for (size_t idx = 0; idx < dimSize; idx++) {
    if (!privateFlag) {
      shapeSize *= (*tensorShape)[idx];
      continue;
    }
    // 多维场景只有最后两根轴需要做对齐处理
    if (idx < dimSize - 2) {
      shapeSize *= (*tensorShape)[idx];
    } else {
      if (dataType == aclDataType::ACL_FLOAT16 || dataType == aclDataType::ACL_BF16) {
        // Minimum block size in FP16/BF16 is 16.
        shapeSize *= Ops::Base::CeilAlign(static_cast<int64_t>((*tensorShape)[idx]), MIN_SIZE_PER_BLOCK_FLOAT16);
      } else {
        // Minimum block size in INT8 is 32.
        shapeSize *= (idx == dimSize - 2) ? Ops::Base::CeilAlign(static_cast<int64_t>((*tensorShape)[idx]), MIN_SIZE_PER_BLOCK_FLOAT16) : 
                      Ops::Base::CeilAlign(static_cast<int64_t>((*tensorShape)[idx]), MIN_SIZE_PER_BLOCK_INT8);
      }
    }
  }
  return static_cast<uint64_t>(shapeSize);
}

aclnnStatus aclnnCalculateMatmulWeightSize(const aclIntArray *tensorShape, uint64_t *weightTensorSize) {
  // 检查参数是否为空指针
  OP_CHECK_NULL(tensorShape, return ACLNN_ERR_PARAM_NULLPTR);
  if (weightTensorSize == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
            "expected a value of type number for argument weightTensorSize but instead found type null.");
    return ACLNN_ERR_PARAM_NULLPTR;
  }
  CHECK_RET(CheckShapeDim(tensorShape), ACLNN_ERR_PARAM_INVALID);
  *weightTensorSize = CalculateMatmulWeightSize(tensorShape, aclDataType::ACL_FLOAT16);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnCalculateMatmulWeightSizeV2(const aclIntArray* tensorShape, aclDataType dataType,
                                             uint64_t* weightTensorSize) {
  // 检查参数是否为空指针
  aclnnStatus socCheckRes = CheckSocValid();
  CHECK_RET(socCheckRes == ACLNN_SUCCESS, socCheckRes);
  OP_CHECK_NULL(tensorShape, return ACLNN_ERR_PARAM_NULLPTR);
  if (weightTensorSize == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
            "expected a value of type number for argument weightTensorSize but instead found type null.");
    return ACLNN_ERR_PARAM_NULLPTR;
  }
  CHECK_RET(CheckShapeDimV2(tensorShape), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckNonZeroShape(tensorShape), ACLNN_ERR_PARAM_INVALID);
  *weightTensorSize = CalculateMatmulWeightSize(tensorShape, dataType);
  return ACLNN_SUCCESS;
}

static bool CheckNotNull(const aclTensor *input) {
  OP_CHECK_NULL(input, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *input) {
  // 检查input的数据类型是否在支持列表内，out类型随动检查
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(input, supportList, return false);
  return true;
}

static bool CheckFormatValid(const aclTensor *input) {
  if (input->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the format of input should be ND. Actual is [%s].",
            op::ToString(input->GetStorageFormat()).GetString());
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *input) {
  // 所有算子的维度都不能超过8
  OP_CHECK_MAX_DIM(input, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *input) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(input), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(input), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查format是否满足约束
  CHECK_RET(CheckFormatValid(input), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足约束
  CHECK_RET(CheckShape(input), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTransMatmulWeightGetWorkspaceSize(aclTensor *mmWeightRef, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnTransMatmulWeight, DFX_IN(mmWeightRef), DFX_OUT(mmWeightRef));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  uniqueExecutor.get()->AbandonCache();

  // 固定写法，参数检查
  auto ret = CheckParams(mmWeightRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (mmWeightRef->IsEmpty() || !NeedToCovertToPravateFormat()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 生成一个tensorShape
  int64_t tensorSize = (int64_t)(mmWeightRef->GetViewShape().GetDimNum());
  std::vector<int64_t> tensorShape(tensorSize);
  for (int64_t i = 0; i < tensorSize; i++) {
      tensorShape[i] = (mmWeightRef->GetViewShape())[i];
  }
  auto tensorShapeArray = uniqueExecutor.get()->AllocIntArray(tensorShape.data(), tensorSize);
  aclDataType weightDtype = aclDataType::ACL_DT_UNDEFINED;
  aclGetDataType(mmWeightRef, &weightDtype);
  if (weightDtype == aclDataType::ACL_INT8) {
    CHECK_RET(CheckShapeDimV2(tensorShapeArray), ACLNN_ERR_PARAM_INVALID);
  } else {
    CHECK_RET(CheckShapeDim(tensorShapeArray), ACLNN_ERR_PARAM_INVALID);
  }
  uint64_t weightSize = CalculateMatmulWeightSize(tensorShapeArray, weightDtype);
  OP_LOGI("Returning calculated weight tensor sise is %lu", weightSize);
  // 如果input张量是非连续的，那么需要转换，否则需要复制
  auto weightContiguous = l0op::Contiguous(mmWeightRef, uniqueExecutor.get());
  if (weightContiguous == mmWeightRef){
      weightContiguous = uniqueExecutor.get()->CreateView(mmWeightRef, mmWeightRef->GetViewShape(), mmWeightRef->GetViewOffset());
  }
  CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 调用l0算子TransData进行计算，将内部计算格式转换为Nz
  auto PrivateFormatResult = l0op::TransData(weightContiguous, Format::FORMAT_FRACTAL_NZ, 0, uniqueExecutor.get());
  CHECK_RET(PrivateFormatResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 把计算结果写回tensor中
  mmWeightRef->SetOriginalShape(PrivateFormatResult->GetOriginalShape());
  mmWeightRef->SetStorageShape(PrivateFormatResult->GetStorageShape());
  mmWeightRef->SetStorageFormat(Format::FORMAT_FRACTAL_NZ);
  auto viewCopyResult = l0op::TensorMove(PrivateFormatResult, mmWeightRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTransMatmulWeight(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnTransMatmulWeight);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif