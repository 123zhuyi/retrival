/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_split_with_size.h"
#include "split_v.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;
constexpr int64_t SPLIT_LOOP_SIZE = 32;
constexpr int64_t SPLITV2_LOOP_SIZE = 64;
constexpr int64_t SPLIT_LOOP_SIZE_512 = 512;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16,
    DataType::DT_INT64, DataType::DT_INT32, DataType::DT_INT16, DataType::DT_INT8,
    DataType::DT_UINT8, DataType::DT_BOOL, DataType::DT_COMPLEX128, DataType::DT_COMPLEX64};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_950 = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16,
    DataType::DT_INT64, DataType::DT_UINT64, DataType::DT_INT32, DataType::DT_UINT32, 
    DataType::DT_INT16, DataType::DT_UINT16, DataType::DT_INT8, DataType::DT_UINT8,
    DataType::DT_BOOL, DataType::DT_COMPLEX128, DataType::DT_COMPLEX64};

inline static bool CheckNotNull(const aclTensor *self, const aclIntArray *splitSize, const aclTensorList *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(splitSize, return false);
  OP_CHECK_NULL(out, return false);

  for (size_t index = 0; index < out->Size(); index++) {
    OP_CHECK_NULL((*out)[index], return false);
  }
  return true;
}

inline static bool CheckDtypeValid(const aclTensor *self, const aclTensorList *out) {
  // 检查self的数据类型是否在API支持列表内
  if (!IsRegBase()) {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    // 检查每一个输出tensor的数据类型是否在API支持列表内
    for (size_t index = 0; index < out->Size(); index++) {
      OP_CHECK_DTYPE_NOT_SUPPORT((*out)[index], DTYPE_SUPPORT_LIST, return false);
    }
  } else {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_950, return false);
    // 检查每一个输出tensor的数据类型是否在API支持列表内
    for (size_t index = 0; index < out->Size(); index++) {
      OP_CHECK_DTYPE_NOT_SUPPORT((*out)[index], DTYPE_SUPPORT_LIST_950, return false);
    }
  }
  return true;
}

static bool CheckShape(const aclTensor *self, const aclIntArray *splitSize, int64_t dim, const aclTensorList *out) {
  // 校验输入的长度
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MIN_DIM(self, 1, return false);
  // 校验每一个输出的长度
  for (size_t index = 0; index < out->Size(); index++) {
    OP_CHECK_MAX_DIM((*out)[index], MAX_DIM_LEN, return false);
  }
  // 校验输入self与dim间关系
  int64_t selfDim = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if ((dim >= 0 && dim >= selfDim) || (dim < 0 && dim < -selfDim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnSplitWithSize dim value [%ld] to be in range [%ld, %ld) but check failed.",
            dim, -selfDim, selfDim);
    return false;
  }
  size_t dimIndex = dim >= 0 ? static_cast<size_t>(dim) : static_cast<size_t>(dim + selfDim);
  // 判断splitSize所有元素之和是否与被split的维度大小相等
  int64_t len = 0;
  for (size_t index = 0; index < splitSize->Size(); index++) {
    len += *(splitSize->GetData() + index);
  }
  if (len != self->GetViewShape().GetDim(dimIndex)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnSplitWithSize all split value sum [%ld] to be equal to split dimIndex [%zu] value [%ld] but "
            "check failed.",
            len, dimIndex, self->GetViewShape().GetDim(dimIndex));
    return false;
  }
  // 判断splitSize的个数是否与输出个数相等
  if (splitSize->Size() != out->Size()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnSplitWithSize splitSize num [%zu] to be equal to output num [%zu] but check failed.",
            splitSize->Size(), out->Size());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *splitSize, int64_t dim,
                               const aclTensorList *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, splitSize, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入与输出的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入输出的shape支持能力
  CHECK_RET(CheckShape(self, splitSize, dim, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

inline static aclnnStatus SplitZeroCalculation(const aclTensor *self, aclTensorList *out, aclOpExecutor *executor) {
  auto selfCast = l0op::Cast(self, (*out)[0]->GetDataType(), executor);
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto selfViewCopy = l0op::ViewCopy(selfCast, (*out)[0], executor);
  CHECK_RET(selfViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus SplitOnceCalculation(const aclTensor *self, const aclIntArray *splitSize, int64_t dim,
                                        aclTensorList *out, aclOpExecutor *executor) {
  // 调用SplitV算子
  auto splitRes = l0op::SplitV(self, splitSize, dim, executor);
  CHECK_RET(splitRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 循环调用Cast和ViewCopy
  for (size_t index = 0; index < splitSize->Size(); index++) {
    CHECK_RET(CheckShapeAndScalarSame((*splitRes)[index], (*out)[index]), ACLNN_ERR_PARAM_INVALID);
    auto splitCast = l0op::Cast((*splitRes)[index], (*out)[index]->GetDataType(), executor);
    CHECK_RET(splitCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto splitViewCopy = l0op::ViewCopy(splitCast, (*out)[index], executor);
    CHECK_RET(splitViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus SplitLoopCalculation(const aclTensor *self, const aclIntArray *splitSize, int64_t dim,
                                        aclTensorList *out, aclOpExecutor *executor) {
  const int64_t numSplit = splitSize->Size();
  const int64_t splitLoopSize = (!IsRegBase()) ?
                                SPLIT_LOOP_SIZE : SPLIT_LOOP_SIZE_512;
  const int64_t loopSize = (numSplit + splitLoopSize - 1) / splitLoopSize;
  const int64_t lastSize = (numSplit % splitLoopSize == 0) ? splitLoopSize : numSplit % splitLoopSize;
  // 1. 根据loopSize和lastSize, 将splitSize构造为新的SplitSize
  op::Shape selfShape = self->GetViewShape();
  const size_t selfDim = selfShape.GetDimNum();
  FVector<int64_t> newSplitSize;
  FVector<aclIntArray *> splitList;
  for (int64_t loopIndex = 0; loopIndex < loopSize; loopIndex++) {
    int64_t newSplit = 0;
    FVector<int64_t> chunkVector;
    int64_t currentSplitValue = 0;
    if (loopIndex != loopSize - 1) {
      for (int64_t noLastIndex = 0; noLastIndex < splitLoopSize; noLastIndex++) {
        currentSplitValue = *(splitSize->GetData() + loopIndex * splitLoopSize + noLastIndex);
        chunkVector.emplace_back(currentSplitValue);
        newSplit += currentSplitValue;
      }
    } else {
      for (int64_t lastIndex = 0; lastIndex < lastSize; lastIndex++) {
        currentSplitValue = *(splitSize->GetData() + loopIndex * splitLoopSize + lastIndex);
        chunkVector.emplace_back(currentSplitValue);
        newSplit += currentSplitValue;
      }
    }
    splitList.emplace_back(executor->AllocIntArray(chunkVector.data(), chunkVector.size()));
    newSplitSize.emplace_back(newSplit);
  }

  // 2. 循环调用Slice将self切成N个大块, 并对每个大块使用SplitV再次切分
  FVector<const aclTensor *> splitTensorList;
  int64_t offsetValue = 0;
  for (size_t sliceIndex = 0; sliceIndex < newSplitSize.size(); sliceIndex++) {
    // 计算offset, offset逐块递增
    FVector<int64_t> offsetVector(selfDim, 0);
    offsetValue += sliceIndex == 0 ? 0 : newSplitSize[sliceIndex - 1];
    offsetVector[static_cast<size_t>(dim)] = offsetValue;
    aclIntArray *offsetArray = executor->AllocIntArray(offsetVector.data(), offsetVector.size());

    // 计算size, size与输出块大小保持一致
    FVector<int64_t> sizeVector;
    for (size_t selfIndex = 0; selfIndex < selfDim; selfIndex++) {
      int64_t sizeValue =
          selfIndex == static_cast<size_t>(dim) ? newSplitSize[sliceIndex] : selfShape.GetDim(selfIndex);
      sizeVector.emplace_back(sizeValue);
    }
    aclIntArray *sizeArray = executor->AllocIntArray(sizeVector.data(), sizeVector.size());

    // 调用l0op::Slice对每一块进行处理
    auto sliceRes = l0op::Slice(self, offsetArray, sizeArray, executor);
    CHECK_RET(sliceRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0op::SPlitV将sliceRes进行切分
    auto splitRes = l0op::SplitV(sliceRes, splitList[sliceIndex], dim, executor);
    CHECK_RET(splitRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    for (int64_t resIndex = 0; resIndex < static_cast<int64_t>(splitRes->Size()); resIndex++) {
      auto splitCast = l0op::Cast((*splitRes)[resIndex], (*out)[resIndex + sliceIndex * splitLoopSize]->GetDataType(),
                                  executor);
      CHECK_RET(splitCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto splitViewCopy = l0op::ViewCopy(splitCast, (*out)[resIndex + sliceIndex * splitLoopSize], executor);
      CHECK_RET(splitViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  }
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSplitWithSizeGetWorkspaceSize(const aclTensor *self, const aclIntArray *splitSize, int64_t dim,
                                               aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnSplitWithSize, DFX_IN(self, splitSize, dim), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, splitSize, dim, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 修正dim取值
  if (dim < 0) {
    dim += static_cast<int64_t>(self->GetViewShape().GetDimNum());
  }

  // 空tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 根据数据类型和输出个数判断调用对应的实现函数
  if (splitSize->Size() == 1) {
    // 无需调用SplitV,直接进行Cast和ViewCopy即可
    ret = SplitZeroCalculation(selfContiguous, out, uniqueExecutor.get());
  } else if (l0op::IsSplitV2AiCoreSupport(selfContiguous, splitSize, dim) &&
             splitSize->Size() > SPLIT_LOOP_SIZE && splitSize->Size() <= SPLITV2_LOOP_SIZE) {
    // 在SplitV2算子的AICore场景且输出个数超过32但不超过64，直接切分，不需要slice
    ret = SplitOnceCalculation(selfContiguous, splitSize, dim, out, uniqueExecutor.get());
  } else if (l0op::SplitVAiCoreSupport(selfContiguous) && splitSize->Size() > SPLIT_LOOP_SIZE &&
    !IsRegBase()) {
    // 在SplitV算子的AiCore场景或者输出个数超过32个时,使用以32为基数的循环切分
    ret = SplitLoopCalculation(selfContiguous, splitSize, dim, out, uniqueExecutor.get());
  } else if (splitSize->Size() > SPLIT_LOOP_SIZE_512) {
    ret = SplitLoopCalculation(selfContiguous, splitSize, dim, out, uniqueExecutor.get());
  } else {
    ret = SplitOnceCalculation(selfContiguous, splitSize, dim, out, uniqueExecutor.get());
  }
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSplitWithSize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSplitWithSize);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
