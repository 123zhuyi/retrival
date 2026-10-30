/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to.cpp
 * \brief
 */

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BroadcastTo);

static const std::initializer_list<DataType> ASCEND310P_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,  DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_INT64,
    DataType::DT_UINT32, DataType::DT_UINT8,   DataType::DT_INT8,  DataType::DT_BOOL};

static const std::initializer_list<DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,  DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_INT64,
    DataType::DT_UINT32, DataType::DT_UINT8,   DataType::DT_INT8,  DataType::DT_BOOL};

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT32,
    DataType::DT_BF16,  DataType::DT_UINT8,   DataType::DT_INT8,  DataType::DT_BOOL};

static const std::initializer_list<DataType> REGBASE_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8,    DataType::DT_INT32,    DataType::DT_INT64,       DataType::DT_UINT8,
    DataType::DT_FLOAT16, DataType::DT_FLOAT,    DataType::DT_BF16,        DataType::DT_UINT32,
    DataType::DT_BOOL,    DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN};

static bool IsAiCoreSupport(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    // Cast只需要根据npuArch信息判断对应芯片的dtype支持
    if (curArch == NpuArch::DAV_2201) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    } else if (curArch == NpuArch::DAV_2002) {
        return CheckType(self->GetDataType(), ASCEND310P_AICORE_DTYPE_SUPPORT_LIST);
    } else if (IsRegBase(curArch)) {
        return CheckType(self->GetDataType(), REGBASE_AICORE_DTYPE_SUPPORT_LIST);
    } else {
        return CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
    }
}

const aclTensor* BroadcastToAiCore(const aclTensor* x, const aclTensor* y, const aclTensor* shape,
                                   aclOpExecutor* executor)
{
    L0_DFX(BroadcastToAiCore, x, y, shape);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BroadcastTo, OP_INPUT(x, shape), OP_OUTPUT(y));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BroadcastToAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return y;
}

const aclTensor* BroadcastToAiCpu(const aclTensor* x, const aclTensor* y, const aclTensor* shape,
                                  aclOpExecutor* executor)
{
    L0_DFX(BroadcastToAiCpu, x, y, shape);
    static op::internal::AicpuTaskSpace space("BroadcastTo", ge::DEPEND_CONST_VALUE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(BroadcastTo, OP_ATTR_NAMES({"T", "Tidx"}), OP_INPUT(x, shape), OP_OUTPUT(y),
                                          OP_ATTR(x->GetDataType(), shape->GetDataType()));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BroadcastToAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return y;
}

const aclTensor* BroadcastTo(const aclTensor* x, const aclTensor* y, const aclTensor* shape, aclOpExecutor* executor)
{
    if (IsAiCoreSupport(x)) {
        return BroadcastToAiCore(x, y, shape, executor);
    }
    return BroadcastToAiCpu(x, y, shape, executor);
}

const aclTensor* BroadcastTo(const aclTensor* x, const aclIntArray* shape, aclOpExecutor* executor)
{
    auto storageFormat = (x->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) ? op::Format::FORMAT_ND :
                                                                                    x->GetStorageFormat();
    auto originalFormat = (x->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) ? op::Format::FORMAT_ND :
                                                                                     x->GetOriginalFormat();
    auto out = executor->AllocTensor(x->GetDataType(), storageFormat, originalFormat);
    auto shapeTensor = executor->ConvertToTensor(shape, ToOpDataType(ACL_INT64));
    INFER_SHAPE(BroadcastTo, OP_INPUT(x, shapeTensor), OP_OUTPUT(out));
    return BroadcastTo(x, out, shapeTensor, executor);
}
} // namespace l0op
