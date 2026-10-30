/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "opdev/framework_op.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

namespace l0op {

OP_TYPE_REGISTER(TensorMove);

static const int64_t DATA_LIMIT_910 = 200000 * 4;
static const int64_t DATA_LIMIT_910B = 100000 * 4;
static const int64_t DATA_LIMIT_310P = 32;

// float,float16,int32,uint32,int8,int16,uint16,uint8,bool,int64,uint64,bfloat16,double,hifloat8,float8_e5m2,float8_e4m3fn,complex32,complex64
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT,     op::DataType::DT_FLOAT16,  op::DataType::DT_INT32,       op::DataType::DT_UINT32,
    op::DataType::DT_INT8,      op::DataType::DT_INT16,    op::DataType::DT_UINT16,      op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,      op::DataType::DT_INT64,    op::DataType::DT_UINT64,      op::DataType::DT_BF16,
    op::DataType::DT_DOUBLE,    op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_COMPLEX32, op::DataType::DT_COMPLEX64};

// 910B float16,float,int32,uint32,int8,int16,uint16,uint8,bool,int64,uint64,bfloat16,double
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,  op::DataType::DT_UINT32,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_UINT16, op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,  op::DataType::DT_INT64,   op::DataType::DT_UINT64, op::DataType::DT_BF16,
    op::DataType::DT_DOUBLE};

// 910  float16,float,int32,uint32,int8,int16,uint16,uint8,bool,int64,uint64,double
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,  op::DataType::DT_UINT32,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_UINT16, op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,  op::DataType::DT_INT64,   op::DataType::DT_UINT64, op::DataType::DT_DOUBLE};

static bool IsAiCoreSupport(const aclTensor* self)
{
    auto dataType = self->GetDataType();
    auto curArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST_910B);
    }

    if (op::IsRegBase(curArch)) {
        return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST_REGBASE);
    }
    return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* TensorMoveAiCore(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(TensorMoveAiCore, x, y);
    ADD_TO_LAUNCHER_LIST_AICORE(TensorMove, op::AI_CORE, OP_INPUT(x), OP_OUTPUT(y));
    return y;
}

const aclTensor* TensorMoveAiCpu(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(TensorMoveAiCpu, x, y);
    static op::internal::AicpuTaskSpace space("Identity");
    ADD_TO_LAUNCHER_LIST_AICPU(TensorMove, OP_ATTR_NAMES(), OP_INPUT(x), OP_OUTPUT(y));
    return y;
}

bool IsCopyNpuToNpu(const aclTensor* x)
{
    auto dataSize = static_cast<uint64_t>(x->GetViewShape().GetShapeSize()) * op::TypeSize(x->GetDataType());
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == op::SocVersion::ASCEND910B || socVersion == op::SocVersion::ASCEND910_93
        || op::IsRegBase()) {
        return static_cast<int64_t>(dataSize) <= DATA_LIMIT_910B;
    }
    if (socVersion == op::SocVersion::ASCEND910) {
        return static_cast<int64_t>(dataSize) <= DATA_LIMIT_910;
    }
    if (socVersion == op::SocVersion::ASCEND310P) {
        return static_cast<int64_t>(dataSize) < DATA_LIMIT_310P;
    }
    return false;
}

const aclTensor* TensorMove(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    if (IsCopyNpuToNpu(x)) {
        auto status = op::CopyNpuToNpu(x, y, executor);
        if (status == ACLNN_SUCCESS) {
            return y;
        }
    }
    if (IsAiCoreSupport(x)) {
        return TensorMoveAiCore(x, y, executor);
    }
    return TensorMoveAiCpu(x, y, executor);
}

} // namespace l0op
