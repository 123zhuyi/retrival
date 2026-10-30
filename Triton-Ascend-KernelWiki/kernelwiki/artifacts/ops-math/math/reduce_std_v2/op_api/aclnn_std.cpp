/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_std.h"
#include <bitset>
#include "math/reduce_std_with_mean/op_host/op_api/reduce_std_with_mean.h"
#include "math/reduce_mean/op_api/reduce_mean.h"
#include "reduce_std_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/contiguous.h"
#include "math/expand/op_api/expand.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "opdev/platform.h"
#include "op_api/level2_base_caculation.h"

using namespace op;
using std::bitset;
#ifdef __cplusplus
extern "C" {
#endif

/* Std 算子的完整计算流程如下:
 *                   self
 *                    |
 *          Contiguous(workspace_0)
 *                    |
 *           ReduceMean(workspace_1)
 *                    |
 *       ReduceStdWithMean(workspace_2)
 *                    |
 *           Expand(workspace_3)
 *                    |
 *             Cast(workspace_4)
 *                    |
 *                ViewCopy
 *                    |
 *                 result
 */

constexpr size_t MAX_MASK_LEN = 64;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // 检查self的数据类型是否在std算子的支持列表内
  auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查out的数据类型是否在std算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

  // 检查self的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);

  return true;
}

static bool CheckDimValid(const aclTensor *self, const aclIntArray *dim) {
  auto selfViewShape = self->GetViewShape();
  auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
  // 0维tensor
  if (selfDimNum == 0) {
    selfDimNum = 1;
  }
  uint64_t dimMask[64] = {0};
  // dim可以为空指针
  if (dim == nullptr) {
    return true;
  }
  // 获取dim元素
  for (size_t i = 0; i < dim->Size(); i++) {
    // dim值不能超出范围
    if (dim->operator[](i) >= selfDimNum || dim->operator[](i) < (-selfDimNum)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Provided dim %ld must be in the range of [%ld, %ld].",
                dim->operator[](i), -selfDimNum, selfDimNum - 1);
      return false;
    }
    // dim值可以为负
    if (dim->operator[](i) < 0) {
      if (dimMask[selfDimNum + dim->operator[](i)] == 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims.",
          selfDimNum + dim->operator[](i));
        return false;
      } else {
        dimMask[selfDimNum + dim->operator[](i)] = 1;
      }
      continue;
    }
    // dim值不能重复
    if (dimMask[dim->operator[](i)] == 1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims.", dim->operator[](i));
      return false;
    } else {
      dimMask[dim->operator[](i)] = 1;
    }
  }
  return true;
}

static void StdInferShape(const op::Shape& selfShape, const aclIntArray* dim, bool keepDim, op::Shape& reduceShape) {
  bitset<MAX_MASK_LEN> dimMask = bitset<MAX_MASK_LEN>();
  for (size_t i = 0; i < dim->Size(); i++) {
    int64_t index = GetPosDimWithStd(dim->operator[](i), selfShape.GetDimNum());
    // 前序已检查， 此处如果dim不会重复
    dimMask.set(index);
  }

  for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
    if (!dimMask[i]) {
      reduceShape.AppendDim(selfShape.GetDim(i));
    } else if (keepDim) {
      reduceShape.AppendDim(1);
    }
  }
}
static bool CheckShape(const aclTensor* self, const aclIntArray* dim, const bool keepDim, const aclTensor* out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
  // dim可以为空指针
  if (dim == nullptr || dim->Size() == 0) {
    return true;
  }
  op::Shape reduceShape;
  StdInferShape(self->GetViewShape(), dim, keepDim, reduceShape);

  // out的shape必须满足Infer shape
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, reduceShape, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim, const bool keepdim,
                               aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查reduce的轴是否合理
  CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足推导
  CHECK_RET(CheckShape(self, dim, keepdim, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclIntArray* CalcDim(const aclTensor *self, aclOpExecutor *executor){
  FVector<int64_t> dimVector;
  auto selfViewShape = self->GetViewShape();
  size_t selfDimNum = selfViewShape.GetDimNum();
  for (size_t i = 0; i < selfDimNum; i++){
    dimVector.push_back(static_cast<int64_t>(i));
  }
  return executor->AllocIntArray(dimVector.data(), dimVector.size());
}

static aclIntArray* ConvToNotNegDim(const aclTensor *self, const aclIntArray *dim, aclOpExecutor *executor) {
  FVector<int64_t> dimVector;
  auto selfViewShape = self->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
  int64_t dimValue;
  for (size_t i = 0; i < dim->Size(); i++) {
    dimValue = static_cast<int64_t>(dim->operator[](i));
    if (dimValue < 0) {
      dimVector.push_back(dimValue + selfDimNum);
    } else {
      dimVector.push_back(dimValue);
    }
  }
  return executor->AllocIntArray(dimVector.data(), dimVector.size());
}

static aclnnStatus aclnnStdV2ImplUnify(const aclTensor *self, const aclIntArray *dim, int64_t correction,
    bool keepdim, aclTensor *out, uint64_t* workspaceSize, UniqueExecutor &uniqueExecutor, aclOpExecutor **executor)
{
    bool isMeanOut = false;
    auto reduceStdV2Out = l0op::ReduceStdV2(self, dim, correction, keepdim, isMeanOut, uniqueExecutor.get());

    auto stdOut = std::get<0>(reduceStdV2Out);
    CHECK_RET(stdOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut = l0op::Cast(stdOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStdGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, const int64_t correction,
                                     bool keepdim, aclTensor *out, uint64_t *workspaceSize,
                                     aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnStd, DFX_IN(self, dim, correction, keepdim), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto selfReshape = self;
  auto ret = CheckParams(selfReshape, dim, keepdim, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->GetViewShape().GetDimNum() == 0) {
    int64_t selfShapeValue[1] = {1};
    aclIntArray *selfShape = uniqueExecutor.get()->AllocIntArray(selfShapeValue, 1);
    CHECK_RET(selfShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfReshapeI = l0op::Reshape(self, selfShape, uniqueExecutor.get());
    CHECK_RET(selfReshapeI != nullptr, ACLNN_ERR_INNER_NULLPTR);
    selfReshape = selfReshapeI;
  }

  const aclIntArray* dimArray;
  if (dim == nullptr || dim->Size() == 0) {
    // dim为空指针
    dimArray = CalcDim(self, uniqueExecutor.get());
  } else {
    // 负dim转成正值
    dimArray = ConvToNotNegDim(selfReshape, dim, uniqueExecutor.get());
  }
  CHECK_RET(dimArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // Std算子的空tensor在kernel中支持
  if (selfReshape->IsEmpty()) {
    // 空tensor填充NAN
    ret = CheckFillScalarShapeStdAndVar(out, NAN, uniqueExecutor.get());
    *workspaceSize = 0UL;
    uniqueExecutor.ReleaseTo(executor);
    return ret;
  }

  // shapeProd小于等于correction场景
  int64_t shapeProd = 1;
  shapeProd = CalcShapeProdStdAndVarMean(selfReshape, dimArray);
  if ((shapeProd == 1) && (shapeProd <= correction)) {
    // 返回NAN
    ret = CheckFillScalarShapeStdAndVar(out, NAN, uniqueExecutor.get());
    *workspaceSize = 0UL;
    uniqueExecutor.ReleaseTo(executor);
    return ret;
  }
  if ((correction > 1) && (shapeProd <= correction)) {
    // 返回INF
    ret = CheckFillScalarShapeStdAndVar(out, INFINITY, uniqueExecutor.get());
    *workspaceSize = 0UL;
    uniqueExecutor.ReleaseTo(executor);
    return ret;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(selfReshape, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfReformat = l0op::ReFormat(selfContiguous, Format::FORMAT_ND);
  CHECK_RET(selfReformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (IsRegBase()) {
    return aclnnStdV2ImplUnify(selfReformat, dimArray, correction, keepdim, out, workspaceSize,
                               uniqueExecutor, executor);
  }

  // 调用Mean算子kernel
  auto meanOpOut = l0op::ReduceMean(selfContiguous, dimArray, true, uniqueExecutor.get());
  CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Expand算子kernel
  FVector<int64_t> shapeVector;
  auto selfShape = selfReshape->GetViewShape();
  size_t selfDimNum = selfShape.GetDimNum();
  for (size_t i = 0; i < selfDimNum; i++) {
    shapeVector.emplace_back(selfShape[i]);
  }
  auto shapeArray = uniqueExecutor.get()->AllocIntArray(shapeVector.data(), selfDimNum);
  CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto expandOpOut = l0op::Expand(meanOpOut, shapeArray, uniqueExecutor.get());
  CHECK_RET(expandOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用StdWithMean算子kernel
  const aclTensor *stdWithMeanOpOut = nullptr;
  bool invert = false;
  float eps = 0.001f;
  stdWithMeanOpOut = l0op::ReduceStdWithMean(selfContiguous, expandOpOut, dimArray, correction,
                                             keepdim, invert, eps, uniqueExecutor.get());
  CHECK_RET(stdWithMeanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(stdWithMeanOpOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnStd(void *workspace, uint64_t workspaceSize,
                     aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnStd);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif