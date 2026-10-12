/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_v.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SplitV);
OP_TYPE_REGISTER(SplitV2);

static constexpr int64_t FP16_BLOCK_NUM = 16;
static constexpr int64_t FP32_BLOCK_NUM = 8;
static constexpr int64_t SPLITV2_NUM_THRESHOLD = 65536;

static const std::initializer_list<op::DataType> SPLITV2_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_ASCEND910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT8,
    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_ASCEND950 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_INT16,
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_UINT8, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_BOOL, op::DataType::DT_BF16
};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_ASCEND310P = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT8,
    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> AICPU_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_BOOL, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
  op::DataType::DT_INT16, op::DataType::DT_UINT16, op::DataType::DT_UINT8, op::DataType::DT_INT32,
  op::DataType::DT_INT64, op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_DOUBLE,
  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

bool IsSplitV2AiCoreSupport(const aclTensor *self, const aclIntArray *splitSize, int64_t dim) {
  int64_t numSplit = static_cast<int64_t>(splitSize->Size());
  bool isSupport = false;
  auto selfDimNum = self->GetViewShape().GetDimNum();
  int64_t totalLen = 1, curXDim = 0, blockNum = 1, firstDim = 1;
  for (size_t i = 0; i < selfDimNum; ++i) {
    curXDim = static_cast<int64_t>(self->GetViewShape().GetDim(i));
    totalLen *= curXDim;
    int64_t idx = static_cast<int64_t>(i);
    if (idx < dim) {
      firstDim *= curXDim;
    }
  }
  blockNum = self->GetDataType() == DataType::DT_FLOAT ? FP32_BLOCK_NUM : FP16_BLOCK_NUM;
  // 切分轴大于0, 输入数据大于65536, 输入数据合轴后首维小于等于8(fp32时为8，bf16/fp16为16), 切分块数大于32
  isSupport = dim > 0 && totalLen > SPLITV2_NUM_THRESHOLD && firstDim <= blockNum && numSplit > 32;
  if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B ||
      op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_93) {
    return (isSupport && op::CheckType(self->GetDataType(), SPLITV2_AICORE_DTYPE_SUPPORT_LIST));
  }
  return false;
}

bool SplitVAiCoreSupport(const aclTensor *self) {
  if (IsRegBase()) {
    return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ASCEND950);
  } else if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B ||
             op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_93) {
    return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ASCEND910B);
  } else if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P) {
    return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ASCEND310P);
  }
  return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ASCEND910);
}

inline static bool IsAiCpuSupport(const aclTensor *self) {
  return op::CheckType(self->GetDataType(), AICPU_DTYPE_SUPPORT_LIST);
}

// SplitV2 AICORE算子kernel
inline static const aclTensorList *SplitV2AiCore(const aclTensor *self, const aclTensor *splitTensor,
                                                 const aclTensor *dimTensor, int64_t numSplit,
                                                 const aclTensorList *splitVOut, aclOpExecutor *executor) {
  L0_DFX(SplitV2AiCore, self, splitTensor, dimTensor, numSplit, splitVOut);
  ADD_TO_LAUNCHER_LIST_AICORE(SplitV2,
    OP_INPUT(self, splitTensor, dimTensor),
    OP_OUTPUT(splitVOut),
    OP_ATTR(numSplit));
  return splitVOut;
}

// AICORE算子kernel
inline static const aclTensorList *SplitVAiCore(const aclTensor *self, const aclTensor *splitTensor,
                                                const aclTensor *dimTensor, int64_t numSplit,
                                                const aclTensorList *splitVOut, aclOpExecutor *executor) {
  L0_DFX(SplitVAiCore, self, splitTensor, dimTensor, numSplit, splitVOut);
  ADD_TO_LAUNCHER_LIST_AICORE(SplitV,
    OP_INPUT(self, splitTensor, dimTensor),
    OP_OUTPUT(splitVOut),
    OP_ATTR(numSplit));
  return splitVOut;
}

// AICPU算子kernel
inline static const aclTensorList *SplitVAiCpu(const aclTensor *self, const aclTensor *splitTensor,
                                               const aclTensor *dimTensor, int64_t numSplit,
                                               const aclTensorList *splitVOut, aclOpExecutor *executor) {
  L0_DFX(SplitVAiCpu, self, splitTensor, dimTensor, numSplit, splitVOut);
  if (IsComplexType(self->GetDataType()) || (self->GetDataType() == ge::DataType::DT_BF16)) {
    static internal::AicpuTaskSpace space("SplitV", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SplitV, OP_ATTR_NAMES({"num_split", "T", "Tlen"}),
                                          OP_INPUT(self, splitTensor, dimTensor), OP_OUTPUT(splitVOut),
                                          OP_ATTR(numSplit, self->GetDataType(), splitTensor->GetDataType()));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return splitVOut;
  }
  static internal::AicpuTaskSpace space("SplitV");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SplitV,
                                        OP_ATTR_NAMES({"num_split"}),
                                        OP_INPUT(self, splitTensor, dimTensor),
                                        OP_OUTPUT(splitVOut),
                                        OP_ATTR(numSplit));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return splitVOut;
}

const aclTensorList *SplitV(const aclTensor *self, const aclIntArray *splitSize, int64_t dim, aclOpExecutor *executor) {
  L0_DFX(SplitV, self, splitSize, dim);
  // 算子原型仅支持splitV后的输出个数属性的数据类型为有符号整型, dim的数据类型为int32
  int64_t numSplit = splitSize->Size();
  int64_t selfDim = static_cast<int64_t>(self->GetViewShape().GetDimNum());

  // 将splitSize转换成算子依赖的Tensor类型输入, 算子支持int32，int64
  auto splitTensor = executor->ConvertToTensor(splitSize, op::DataType::DT_INT32);
  if (self->GetDataType() == op::DataType::DT_DOUBLE) {
    splitTensor = executor->ConvertToTensor(splitSize, op::DataType::DT_INT64);
  }

  FVector<const aclTensor *> splitVector;
  for (int64_t index = 0; index < numSplit; index++) {
    // 更新每一个输出的shape
    op::Shape indexShape = self->GetViewShape();
    indexShape.SetDim(dim, *(splitSize->GetData() + index));
    // 构造每一个输出tensor
    auto outTensor = executor->AllocTensor(indexShape, self->GetDataType());
    splitVector.emplace_back(outTensor);
  }
  auto splitVOut = executor->AllocTensorList(splitVector.data(), numSplit);

  int32_t dimRefine = (dim >= 0) ? static_cast<int32_t>(dim) : static_cast<int32_t>(dim + selfDim);
  if (IsSplitV2AiCoreSupport(self, splitSize, dimRefine)) {
    // 将dimRefine转换为算子依赖的Tensor类型输入，SplitV2中splitTensor和dimTensor数据类型保持一致
    auto dimScalar = executor->AllocScalar(&dimRefine, op::DataType::DT_INT32);
    auto dimTensor = executor->ConvertToTensor(dimScalar, op::DataType::DT_INT32);
    return SplitV2AiCore(self, splitTensor, dimTensor, numSplit, splitVOut, executor);
  } else if (SplitVAiCoreSupport(self)) {
    int64_t dimCast = (dim >= 0) ? dim : (dim + selfDim);
    // 将dimCast转换为算子依赖的Tensor类型输入
    auto dimScalar = executor->AllocScalar(&dimCast, op::DataType::DT_INT64);
    auto dimTensor = executor->ConvertToTensor(dimScalar, op::DataType::DT_INT64);
    return SplitVAiCore(self, splitTensor, dimTensor, numSplit, splitVOut, executor);
  } else {
    CHECK_RET(IsAiCpuSupport(self), nullptr);
    int32_t dimCast = (dim >= 0) ? static_cast<int32_t>(dim) : static_cast<int32_t>(dim + selfDim);
    // 将dimCast转换为算子依赖的Tensor类型输入
    auto dimScalar = executor->AllocScalar(&dimCast, op::DataType::DT_INT32);
    auto dimTensor = executor->ConvertToTensor(dimScalar, op::DataType::DT_INT32);
    return SplitVAiCpu(self, splitTensor, dimTensor, numSplit, splitVOut, executor);
  }
}
} // namespace l0op
