/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_segsum.h"
#include "aclnn_kernels/contiguous.h"
#include "segsum.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t THREEDIMS = 3;
static const size_t FOURDIMS = 4;

static const std::initializer_list<op::DataType> AICORE_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self) {
  OP_CHECK_DTYPE_NOT_SUPPORT(self, AICORE_DTYPE_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckDtypeEqual(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  auto selfShape = self->GetViewShape();
  auto outShape = out->GetViewShape();
  auto selfDimSize = selfShape.GetDimNum();
  auto outDimSize = outShape.GetDimNum();
  if (selfDimSize != THREEDIMS && selfDimSize != FOURDIMS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected input size equals to %zu or %zu, but got sizes %zu.",
            THREEDIMS, FOURDIMS, selfDimSize);
    return false;
  }
  if (outDimSize != selfDimSize + 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "It is expected output size is 1 large than input size, but got input size %zu, output size %zu.",
            selfDimSize, outDimSize); 
    return false;
  }
  for (size_t i = 0; i < outDimSize  - 1; i++) {
    if (outShape.GetDim(i) != selfShape.GetDim(i)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "It is expected output size equals to input size at index %zu , but got input %s, output %s.",
              i, op::ToString(selfShape).GetString(), op::ToString(outShape).GetString());
      return false;
    }
  }
  if (selfShape.GetDim(selfDimSize - 1) != outShape.GetDim(outDimSize - 1)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "It is expected output size at index %zu equals to input size at index %zu, \
            but got input size %s, output size %s.",
            selfDimSize - 1 , outDimSize - 1, op::ToString(selfShape).GetString(), op::ToString(outShape).GetString());
    return false;
  }
  return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out) {
  Format selfFormat = self->GetStorageFormat();
  Format outFormat = out->GetStorageFormat();
  if (selfFormat != outFormat) {
  OP_LOGE(ACLNN_ERR_PARAM_INVALID,
          "Format of input and output should be equal, input[%s], out[%s].",
          op::ToString(selfFormat).GetString(), op::ToString(outFormat).GetString());
  return false;
  }
  if (selfFormat != Format::FORMAT_ND) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input format should be ND. Actual: input[%s].",
          op::ToString(selfFormat).GetString());
      return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和out数据类型是否相同
  CHECK_RET(CheckDtypeEqual(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否支持
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查format是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnExpSegsumGetWorkspaceSize(const aclTensor *self,
                                           aclTensor *out,
                                           uint64_t *workspaceSize,
                                           aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnExpSegsum, DFX_IN(self), DFX_OUT(out));
  
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto result = l0op::Segsum(selfContiguous, out, uniqueExecutor.get());
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnExpSegsum(void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnExpSegsum);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif