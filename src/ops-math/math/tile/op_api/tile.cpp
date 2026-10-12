/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile.cpp
 * \brief
 */

#include "tile.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Tile);

static inline const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_BOOL,    op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_310P_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16,    op::DataType::DT_INT8,   op::DataType::DT_UINT8,
    op::DataType::DT_INT16, op::DataType::DT_UINT16,  op::DataType::DT_UINT32, op::DataType::DT_UINT64};

static inline const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT,    op::DataType::DT_FLOAT16, op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_BOOL,     op::DataType::DT_BF16,    op::DataType::DT_INT8,   op::DataType::DT_UINT8,
    op::DataType::DT_INT16,    op::DataType::DT_UINT16,  op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_COMPLEX64};

static inline const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BOOL};

static const std::initializer_list<DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,        op::DataType::DT_FLOAT16,  op::DataType::DT_INT32,
    op::DataType::DT_INT64,        op::DataType::DT_BOOL,     op::DataType::DT_BF16,
    op::DataType::DT_INT8,         op::DataType::DT_UINT8,    op::DataType::DT_INT16,
    op::DataType::DT_UINT16,       op::DataType::DT_UINT32,   op::DataType::DT_UINT64,
    op::DataType::DT_COMPLEX64,    op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2,
    op::DataType::DT_FLOAT8_E4M3FN};

static inline bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_910B_LIST);
    }
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_310P_LIST);
    }
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND610LITE) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    if (IsRegBase()) {
        return CheckType(self->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static inline const aclTensor* TileAiCore(
    const aclTensor* self, const aclTensor* repeats, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(TileAiCore, self, repeats, out);
    ADD_TO_LAUNCHER_LIST_AICORE(Tile, OP_INPUT(self, repeats), OP_OUTPUT(out));
    return out;
}

static inline const aclTensor* TileAiCpu(
    const aclTensor* self, const aclTensor* repeats, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(TileAiCpu, self, repeats, out);
    static internal::AicpuTaskSpace space("Tile", ge::DEPEND_IN_SHAPE, true);
    op::DataType dtype = repeats->GetDataType();
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Tile, OP_ATTR_NAMES({"Tmultiples"}), OP_INPUT(self, repeats), OP_OUTPUT(out), OP_ATTR(dtype));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* Tile(const aclTensor* self, const aclIntArray* repeats, aclOpExecutor* executor)
{
    int64_t expandDim = static_cast<int64_t>(repeats->Size() - self->GetViewShape().GetDimNum());
    op::Shape broadcastShape;
    if (expandDim < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "repeats size should not be smaller than input tensor dim");
        return nullptr;
    } else {
        for (int64_t i = 0; i < expandDim; i++) {
            broadcastShape.AppendDim((*repeats)[i]);
        }
        for (size_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
            broadcastShape.AppendDim(self->GetViewShape()[i] * (*repeats)[i + expandDim]);
        }
    }
    auto out = executor->AllocTensor(broadcastShape, self->GetDataType());
    auto repeatTensor = executor->ConvertToTensor(repeats, op::DataType::DT_INT64);

    if (IsAiCoreSupport(self)) {
        return TileAiCore(self, repeatTensor, out, executor);
    } else {
        return TileAiCpu(self, repeatTensor, out, executor);
    }
}
} // namespace l0op
