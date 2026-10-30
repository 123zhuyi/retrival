/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_acos.h"
#include "acos.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8, DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,   DataType::DT_BOOL,   DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8, DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,   DataType::DT_BOOL,   DataType::DT_UINT8, DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_OUT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                   DataType::DT_DOUBLE, DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      IsRegBase()) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

// 检查输入和输出的数据类型是否在算子的支持列表内
static bool CheckDtypeValid(const aclTensor* input, const aclTensor* out) {
  // 检查输入的数据类型是否在算子的支持列表内
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(input, supportList, return false);

  // 检查输出的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_OUT_LIST, return false);

  return true;
}

// 检查输入和输出是否是空指针
static inline bool CheckNotNull(const aclTensor* input, const aclTensor* out) {
  // 检查输入是否是空指针
  OP_CHECK_NULL(input, return false);

  // 检查输入是否是空指针
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckShape(const aclTensor* input, const aclTensor* out) {
  // input和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(input, out, return false);

  // input的维度必须小于等于8
  OP_CHECK_MAX_DIM(input, MAX_SUPPORT_DIMS_NUMS, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(input, out), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(input, out), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入的数据的值是否合理
  CHECK_RET(CheckShape(input, out), ACLNN_ERR_PARAM_INVALID);
  if (input->GetStorageFormat() != Format::FORMAT_ND) {
      OP_LOGW("Only support ND format for acos/inplaceAcos operator.");
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus ExecAcosGetWorkspaceSize(const aclTensor* input, aclTensor* out, uint64_t* workspaceSize,
                                            aclOpExecutor** executor) {
  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(input, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (input->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
  CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用cast算子将不支持的类型转化为float
  auto castDtype = inputContiguous->GetDataType();
  if (!CheckType(castDtype, DTYPE_OUT_LIST)) {
    castDtype = DataType::DT_FLOAT;
  }

  auto inputCast = l0op::Cast(inputContiguous, castDtype, uniqueExecutor.get());
  CHECK_RET(inputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 执行L0算子
  auto acosOutRet = l0op::Acos(inputCast, uniqueExecutor.get());
  CHECK_RET(acosOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto castOut = l0op::Cast(acosOutRet, out->GetDataType(), uniqueExecutor.get());
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

aclnnStatus aclnnAcosGetWorkspaceSize(const aclTensor* input, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnAcos, DFX_IN(input), DFX_OUT(out));
  return ExecAcosGetWorkspaceSize(input, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceAcosGetWorkspaceSize(aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceAcos, DFX_IN(inputRef), DFX_OUT(inputRef));
  auto out = const_cast<aclTensor*>(inputRef);
  return ExecAcosGetWorkspaceSize(inputRef, out, workspaceSize, executor);
}

aclnnStatus aclnnAcos(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnAcos);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAcos(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                             const aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnInplaceAcos);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
 
#ifdef __cplusplus
}
#endif
