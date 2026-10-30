/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sign_bits_unpack.h"
#include "sign_bits_unpack.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* SignBitsUnpack 算子的完整计算流程如下:
 *          self               dtype    size
 *            \                 /       /
 *  Contiguous(workspace_0)    /       /
 *               \            /       /
 *          SignBitsUnpack(workspace_1)
 *                      |
 *                   ViewCopy
 *                      |
 *                    result
 */

static const size_t DIM_NUM_1D = 1;
static const size_t DIM_NUM_2D = 2;

static const int64_t PACK_SIZE = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST =  {
        op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST =  {
        op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
    // 检查输入和输出是否是空指针
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out, aclDataType dtype) {
    // 检查芯片类型是否支持
    if(GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P && 
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SignBitsUnpack is not supported on this device.");
        return false;
    }

    // 检查self的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);

    // 检查out的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);

    // 检查out与dtype的数据类型是否一致
    op::DataType dtypeOP = op::ToOpDataType(dtype);
    OP_CHECK_DTYPE_NOT_MATCH(out, dtypeOP, return false);

    return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out) {
    // 检查self的format是否为ND
    if(self->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self fromat should be ND. Actual: self is [%s].",
                op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    // 检查out的format是否为ND
    if(out->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out fromat should be ND. Actual: out is [%s].",
                op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if(op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support format [%s].",
                op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *out) {
    // 检查输入shape
    size_t selfDimNum = self->GetViewShape().GetDimNum();
    if(selfDimNum != DIM_NUM_1D) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dims %zu should be 1.", selfDimNum);
        return false;
    }

    // 检查输出shape
    size_t outDimNum = out->GetViewShape().GetDimNum();
    if(outDimNum != DIM_NUM_2D) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out dims %zu should be 2.", outDimNum);
        return false;
    }
    return true;
}

static inline bool CheckValue(const aclTensor *self, int64_t size, const aclTensor *out) {
    // 检查size的值
    if(size <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size value must bigger than zero.");
        return false;
    }

    for(size_t i = 0; i < out->GetViewShape().GetDimNum(); i++) {
        if(out->GetViewShape().GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim Value of out is negative.");
            return false;
        }
    }

    // size可被uint8的拆包输出整除。输出大小为（self的元素个数) * 8
    int64_t selfDim = self->GetViewShape().GetDim(0);
    if((selfDim * PACK_SIZE) % size != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of elements in self * 8 cannot be divided by size.");
        return false;
    }

    // 检查out第一维度是否等于size
    int64_t outDimOneNum = out->GetViewShape().GetDim(0);  
    if(size != outDimOneNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of the first dimension of 'out' is incorrect and should be equal to size.");
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor *self, aclDataType dtype, int64_t size, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out, dtype), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查入参数值是否合法
  CHECK_RET(CheckValue(self, size, out), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查输入和输出tensor的shape是否为异常，输入必须1维
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSignBitsUnpackGetWorkspaceSize(const aclTensor* self, int64_t size, aclDataType dtype, aclTensor* out,
                                                uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnSignBitsUnpack, DFX_IN(self, size, dtype), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, dtype, size, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 输入为空tensor时，直接返回dtype类型的空tensor
  if (self->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用SignBitsUnpack算子kernel,将输入self的数据类型转换成指定的数据类型
  auto castOut = l0op::SignBitsUnpack(selfContiguous, size, op::ToOpDataType(dtype), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSignBitsUnpack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSignBitsUnpack);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
