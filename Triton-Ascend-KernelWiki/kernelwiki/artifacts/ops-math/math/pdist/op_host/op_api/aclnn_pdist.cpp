/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_pdist.h"
#include "pdist.h"
#include "conversion/fill/op_api/fill.h"
#include "conversion/view_copy/op_api/view_copy.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"


using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
   op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT
};

inline static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

inline static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // 检查self的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查out的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

  // 检查self和out的数据类型是否相等
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
  return true;
}

// 检查参数是否符合算子的逻辑
inline static aclnnStatus CheckParamsLogic(const aclTensor* self, float p, const aclTensor* out) {
  // self的shape要求为2维
  OP_CHECK_WRONG_DIMENSION(self, 2, return ACLNN_ERR_PARAM_INVALID);

  // out的shape要求为1维
  OP_CHECK_WRONG_DIMENSION(out, 1, return ACLNN_ERR_PARAM_INVALID);

  int64_t N = self->GetViewShape().GetDim(0);
  op::Shape expectOutShape = {N * (N - 1) / 2};
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectOutShape, return ACLNN_ERR_PARAM_INVALID);

  // 范数p要求为非负数
  if (p < 0 || std::isnan(p)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pdist only supports non-negative p values.");
    return ACLNN_ERR_PARAM_INVALID;
  }
  return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckParams(const aclTensor* self, float p, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");
  // 3. 检查输入的数据的值是否合理
  CHECK_COND(CheckParamsLogic(self, p, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckParamsLogic failed!");
  return ACLNN_SUCCESS;
}

// 定义aclnnPdist的第一段接口
aclnnStatus aclnnPdistGetWorkspaceSize(const aclTensor* self, float p, aclTensor* out, uint64_t* workspaceSize,
                                       aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnPdist, DFX_IN(self, p), DFX_OUT(out));

  // 固定写法， 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, p, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  const aclTensor* PdistOutRet = nullptr;
  // 2维self的第一维小于等于1，输出0维空tensor;
  if (self->GetViewShape().GetDim(0) <= 1) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  } else if (self->GetViewShape().GetDim(1) == 0) {
    // 执行L0 Fill算子
    aclScalar* scalar = uniqueExecutor.get()->AllocScalar(0);
    auto valueTensor = uniqueExecutor.get()->ConvertToTensor(scalar, out->GetDataType());
    auto outputDims = op::ToShapeVector(out->GetViewShape());
    aclIntArray* dimArray = uniqueExecutor.get()->AllocIntArray(outputDims.data(), outputDims.size());
    CHECK_RET(dimArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dimTensor = uniqueExecutor.get()->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    PdistOutRet = l0op::Fill(dimTensor, valueTensor, dimArray, uniqueExecutor.get());
  } else {
    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 执行L0 Pdist算子
    PdistOutRet = l0op::Pdist(selfContiguous, p, uniqueExecutor.get());
  }
  CHECK_RET(PdistOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出out上
  auto viewCopyResult = l0op::ViewCopy(PdistOutRet, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPdist(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnPdist);
  // 调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
