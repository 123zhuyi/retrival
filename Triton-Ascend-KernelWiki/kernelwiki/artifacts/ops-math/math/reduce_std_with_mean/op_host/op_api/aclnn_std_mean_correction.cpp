/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_std_mean_correction.h"
#include <bitset>
#include "reduce_std_with_mean.h"
#include "math/reduce_mean/op_api/reduce_mean.h"
#include "math/reduce_std_v2/op_api/reduce_std_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "math/expand/op_api/expand.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "op_api/level2_base_caculation.h"
#include "op_api/aclnn_check.h"

using namespace op;
using std::bitset;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_MASK_NUM = 64;
static const bool invert = false;
static const float eps = 0.001f;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *stdOut, const aclTensor *meanOut) {
  // 检查self的数据类型是否在算子的支持列表内
  auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查stdOut、meanOut的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(stdOut, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(meanOut, supportList, return false);

  // Ascend950支持输入、输出数据类型不一致
  if (IsRegBase()) {
    return true;
  }

  // 检查self的数据类型是否输出的数据类型一致
  OP_CHECK_DTYPE_NOT_MATCH(self, stdOut->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(self, meanOut->GetDataType(), return false);
  return true;
}

static bool CheckDimValid(const aclTensor *self, const aclIntArray *dim) {
  auto selfViewShape = self->GetViewShape();
  auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
  int64_t dimMin = std::min(-1 * selfDimNum, selfDimNum - 1);
  int64_t dimMax = std::max(-1 * selfDimNum, selfDimNum - 1);
  // 0维tensor
  if (selfDimNum == 0) {
    selfDimNum = 1;
  }
  bool dimMask[64] = {false};
  // dim可以为空指针
  if (dim == nullptr) {
    return true;
  }
  // 获取dim元素
  for (size_t i = 0; i < dim->Size(); i++) {
    // dim值不能超出范围
    auto currentDim = dim->operator[](i);
    if (currentDim > dimMax || currentDim < dimMin) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)",
              dimMin, dimMax, selfDimNum);
      return false;
    }
    // dim值可以为负
    if (currentDim < 0) {
      currentDim = currentDim + selfDimNum;
    }
    // dim值不能重复
    if (dimMask[currentDim]) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims.", currentDim);
      return false;
    }
    dimMask[currentDim] = true;
  }
  return true;
}

static void OutInferShape(const op::Shape& selfShape, const aclIntArray* dim, bool keepDim, op::Shape& reduceShape) {
  bitset<MAX_MASK_NUM> dimTmp = bitset<MAX_MASK_NUM>();
  for (size_t i = 0; i < dim->Size(); i++) {
    int64_t index = GetPosDimWithStd(dim->operator[](i), selfShape.GetDimNum());
    // 前序已检查， 此处如果dim不会重复
    dimTmp.set(index);
  }

  for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
    if (!dimTmp[i]) {
      reduceShape.AppendDim(selfShape.GetDim(i));
    } else if (keepDim) {
      reduceShape.AppendDim(1);
    }
  }
}

static bool CheckShape(const aclTensor *self, const aclIntArray* dim, const bool keepDim, const aclTensor *stdOut, const aclTensor *meanOut) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(stdOut, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(meanOut, MAX_SUPPORT_DIMS_NUMS, return false);
  // dim可以为空指针
  if (dim == nullptr || dim->Size() == 0) {
    return true;
  }
  op::Shape reduceShape;
  OutInferShape(self->GetViewShape(), dim, keepDim, reduceShape);
  // stdOut的shape必须满足Infer shape
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(stdOut, reduceShape, return false);
  // meanOut的shape必须满足Infer shape
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(meanOut, reduceShape, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *dim, const bool keepdim,
                               const aclTensor *stdOut, const aclTensor *meanOut) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull3Tensor(self, stdOut, meanOut), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, stdOut, meanOut), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查reduce的轴是否合理
  CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足推导
  CHECK_RET(CheckShape(self, dim, keepdim, stdOut, meanOut), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclIntArray* CalcDim(const aclTensor *self, aclOpExecutor *executor) {
  FVector<int64_t> dimVector;
  auto selfViewShape = self->GetViewShape();
  size_t selfDimNum = selfViewShape.GetDimNum();
  for (size_t i = 0; i < selfDimNum; i++) {
    dimVector.push_back(static_cast<int64_t>(i));
  }
  return executor->AllocIntArray(dimVector.data(), dimVector.size());
}

static const aclIntArray* GetDimArray(const aclTensor *self, const aclIntArray *dim, aclOpExecutor *executor) {
  const aclIntArray* dimArray = dim;
  if (dim == nullptr || dim->Size() == 0) {
    dimArray = CalcDim(self, executor);
  }
  CHECK_RET(dimArray != nullptr, nullptr);
  return dimArray;
}

static int64_t CalcShapeProd(const aclTensor *self, const aclIntArray *dim) {
  auto selfViewShape = self->GetViewShape();
  int64_t shapeProd = 1;
  size_t selfDimNum = selfViewShape.GetDimNum();
  if (selfDimNum != 0) {
    // dim为all reduce
    if (dim->Size() == 0) {
        for (size_t i = 0; i < selfDimNum; i++) {
          shapeProd *= selfViewShape.GetDim(i);
        }
    } else {
      for (size_t i = 0; i < dim->Size(); i++) {
        auto oriDim = dim->operator[](i);
        auto realDim = oriDim < 0 ? oriDim + selfDimNum : oriDim;
        shapeProd *= selfViewShape.GetDim(realDim);
      }
    }
  }
  return shapeProd;
}

static const aclTensor* GetExpandMean(const aclTensor *self, const aclTensor *meanOpOut, const aclIntArray* dimArray,
                                      bool keepdim, aclOpExecutor *executor) {
  auto meanOpOutTmp = keepdim ? meanOpOut : l0op::UnsqueezeNd(meanOpOut, dimArray, executor);
  CHECK_RET(meanOpOutTmp != nullptr, nullptr);
  // 调用Expand算子kernel
  FVector<int64_t> shapeVector;
  auto selfShape = self->GetViewShape();
  size_t selfDimNum = selfShape.GetDimNum();
  if(!selfDimNum){
    return meanOpOutTmp;
  }
  for (size_t i = 0; i < selfDimNum; i++) {
    shapeVector.emplace_back(selfShape[i]);
  }
  auto shapeArray = executor->AllocIntArray(shapeVector.data(), selfDimNum);
  CHECK_RET(shapeArray != nullptr, nullptr);
  auto expandOpOut = l0op::Expand(meanOpOutTmp, shapeArray, executor);
  CHECK_RET(expandOpOut != nullptr, nullptr);
  return expandOpOut;
}

static aclnnStatus DealEmpty(aclTensor *stdOut, aclTensor *meanOut, aclOpExecutor *executor) {
  auto ret = CheckFillScalarShapeStdAndVar(stdOut, NAN, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  ret = CheckFillScalarShapeStdAndVar(meanOut, NAN, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  return ACLNN_SUCCESS;
}

static aclnnStatus DealshapeProdNAN(const aclTensor *meanOpOut, aclTensor *stdOut, aclTensor *meanOut,
                                    aclOpExecutor *executor) {
  // stdOut返回NAN
  auto ret = CheckFillScalarShapeStdAndVar(stdOut, NAN, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  // 固定写法，将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
  auto viewCopyMeanResult = l0op::ViewCopy(meanOpOut, meanOut, executor);
  CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus DealshapeProdINF(const aclTensor *meanOpOut, aclTensor *stdOut, aclTensor *meanOut,
                                    aclOpExecutor *executor) {
  // stdOut返回INF
  auto ret = CheckFillScalarShapeStdAndVar(stdOut, INFINITY, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  // 固定写法，将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
  auto viewCopyMeanResult = l0op::ViewCopy(meanOpOut, meanOut, executor);
  CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus aclnnStdMeanCorrectionImplUnify(const aclTensor *self, const aclIntArray *dim, int64_t correction,
    bool keepdim, aclTensor *stdOut, aclTensor *meanOut, uint64_t* workspaceSize,
    UniqueExecutor &uniqueExecutor, aclOpExecutor **executor)
{
    bool isMeanOut = true;
    auto reduceStdOut = l0op::ReduceStdV2(self, dim, correction, keepdim, isMeanOut, uniqueExecutor.get());

    auto stdOpOut = std::get<0>(reduceStdOut);
    CHECK_RET(stdOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut = l0op::Cast(stdOpOut, stdOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(castOut, stdOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanOpOut = std::get<1>(reduceStdOut);
    CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut1 = l0op::Cast(meanOpOut, meanOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(castOut1, meanOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStdMeanCorrectionGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, int64_t correction,
                                                   bool keepdim, aclTensor *stdOut, aclTensor *meanOut,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnStdMeanCorrection, DFX_IN(self, dim, correction, keepdim), DFX_OUT(stdOut, meanOut));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, dim, keepdim, stdOut, meanOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // dim为空指针
  auto dimArray = GetDimArray(self, dim, uniqueExecutor.get());
  CHECK_RET(dimArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 空tensor在kernel中支持
  if (self->IsEmpty()) {
    // 空tensor填充NAN
    ret = DealEmpty(stdOut, meanOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ret;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto selfReformat = l0op::ReFormat(selfContiguous, Format::FORMAT_ND);
  CHECK_RET(selfReformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (IsRegBase()) {
      return aclnnStdMeanCorrectionImplUnify(selfReformat, dimArray, correction, keepdim, stdOut, meanOut,
                                             workspaceSize, uniqueExecutor, executor);
  }

  // 调用Mean算子kernel
  auto meanOpOut = l0op::ReduceMean(selfContiguous, dimArray, keepdim, uniqueExecutor.get());
  CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // shapeProd小于等于correction场景
  int64_t shapeProd = 1;
  shapeProd = CalcShapeProd(self, dimArray);
  if ((shapeProd == 1) && (shapeProd <= correction)) {
    ret = DealshapeProdNAN(meanOpOut, stdOut, meanOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ret;
  }
  if ((correction > 1) && (shapeProd <= correction)) {
    ret = DealshapeProdINF(meanOpOut, stdOut, meanOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ret;
  }
  // 固定写法，将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
  auto viewCopyMeanResult = l0op::ViewCopy(meanOpOut, meanOut, uniqueExecutor.get());
  CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto expandOpOut = GetExpandMean(self, meanOpOut, dimArray, keepdim, uniqueExecutor.get());
  CHECK_RET(expandOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用StdWithMean算子kernel
  auto stdWithMeanOpOut = l0op::ReduceStdWithMean(selfContiguous, expandOpOut, dimArray, correction,
                                                  keepdim, invert, eps, uniqueExecutor.get());
  CHECK_RET(stdWithMeanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出stdOut上，stdOut可能是非连续的tensor
  auto viewCopyStdResult = l0op::ViewCopy(stdWithMeanOpOut, stdOut, uniqueExecutor.get());
  CHECK_RET(viewCopyStdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnStdMeanCorrection(void *workspace, uint64_t workspaceSize,
                                   aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnStdMeanCorrection);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

