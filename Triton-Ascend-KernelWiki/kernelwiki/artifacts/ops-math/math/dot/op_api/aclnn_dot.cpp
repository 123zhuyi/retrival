/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_dot.cpp
 * \brief
 */

#include "aclnn_dot.h"
#include "dot.h"
#include "conversion/fill/op_api/fill.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16,
  DataType::DT_INT8, DataType::DT_INT32, DataType::DT_UINT8
};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, 
  DataType::DT_INT8, DataType::DT_INT32, DataType::DT_UINT8
};

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* tensor, const aclTensor* out) {
  const auto& supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
  // 检查数据类型是否一致
  OP_CHECK_DTYPE_NOT_SAME(self, tensor, return false);
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  // 检查数据类型是否在dot算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* tensor, const aclTensor* out) {
  auto &selfViewShape = self->GetViewShape();
  auto &tensorViewShape = tensor->GetViewShape();
  auto &outViewShape = out->GetViewShape();

  if (selfViewShape.GetDimNum() != 1 || tensorViewShape.GetDimNum() != 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected 1D input tensors, but got %s and %s tensors.",
            ToString(selfViewShape).GetString(), ToString(tensorViewShape).GetString());
    return false;
  }

  if (selfViewShape.GetDim(0) != tensorViewShape.GetDim(0)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected consistent tensor size, but got %s and %s tensors.",
            ToString(selfViewShape).GetString(), ToString(tensorViewShape).GetString());
    return false;
  }

  if (outViewShape.GetDimNum() != 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected 0D output tensor, but got %s tensor.",
            ToString(outViewShape).GetString());
    return false;
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* tensor, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull3Tensor(self, tensor, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在支持范围内，且数据类型是否一致
  CHECK_RET(CheckDtypeValid(self, tensor, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足要求
  CHECK_RET(CheckShape(self, tensor, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnDotGetWorkspaceSize(const aclTensor* self, const aclTensor* tensor, aclTensor* out,
                                     uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  L2_DFX_PHASE_1(aclnnDot, DFX_IN(self, tensor), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, tensor, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 对标竞品，支持输入都是空tensor的情况，输出是0
  if (self->IsEmpty() && tensor->IsEmpty()) {
    int64_t dim = 0;
    const aclScalar *dimScalar = (uniqueExecutor.get())->AllocScalar(dim);
    CHECK_RET(dimScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *dimTensor = (uniqueExecutor.get())->ConvertToTensor(dimScalar, op::DataType::DT_INT64);
    CHECK_RET(dimTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclIntArray *outShape = (uniqueExecutor.get())->AllocIntArray(&dim, 0);
    CHECK_RET(outShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclScalar *valueScalar = (uniqueExecutor.get())->AllocScalar(0);
    CHECK_RET(valueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *valueTensor = (uniqueExecutor.get())->ConvertToTensor(valueScalar, out->GetDataType());
    CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Fill算子kernel，对一维一元张量赋予0值
    auto fillOut = l0op::Fill(dimTensor, valueTensor, outShape, uniqueExecutor.get());
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入tensor转换成连续的Tensor
  auto contiguousTensor = l0op::Contiguous(tensor, uniqueExecutor.get());
  CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Dot算子kernel
  auto dotOut = l0op::Dot(contiguousSelf, contiguousTensor, uniqueExecutor.get());
  CHECK_RET(dotOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(dotOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnDot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnDot);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
