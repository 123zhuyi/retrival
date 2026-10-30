/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_precision_compare.h"
#include "precision_compare.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "precision_compare_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
static const uint32_t OFFLINE_DETECT_TYPE = 2;

ACLNN_API aclnnStatus aclnnPrecisionCompareGetWorkspaceSize(const aclTensor *golden, const aclTensor *realdata,
                                                            aclTensor *out, uint64_t *workspaceSize,
                                                            aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnPrecisionCompare, DFX_IN(golden, realdata), DFX_OUT(out));
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = PrecisionCompareCheckParams(golden, realdata, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  if (golden->IsEmpty() || realdata->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入golden转换成连续的tensor
  auto benchMarkContiguous = l0op::Contiguous(golden, uniqueExecutor.get());
  CHECK_RET(benchMarkContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入realdata转换成连续的tensor
  auto realDataContiguous = l0op::Contiguous(realdata, uniqueExecutor.get());
  CHECK_RET(realDataContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto compOut =
      l0op::PrecisionCompare(benchMarkContiguous, realDataContiguous, out, OFFLINE_DETECT_TYPE, uniqueExecutor.get());
  CHECK_RET(compOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnPrecisionCompare(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnPrecisionCompare);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif