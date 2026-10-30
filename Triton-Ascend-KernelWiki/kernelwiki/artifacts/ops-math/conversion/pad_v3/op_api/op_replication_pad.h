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
 * \file op_replication_pad.h
 * \brief
 */

#ifndef OP_REPLICATION_PAD_H_
#define OP_REPLICATION_PAD_H_

#include <string>
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

namespace op {
static const std::string REPLICATION_MODE = "edge";
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_DOUBLE,     op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_REGBASE_LIST = {
    op::DataType::DT_INT64,      op::DataType::DT_UINT64,      op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64,  op::DataType::DT_INT32,       op::DataType::DT_UINT32,
    op::DataType::DT_FLOAT,      op::DataType::DT_COMPLEX128,  op::DataType::DT_INT16,
    op::DataType::DT_UINT16,     op::DataType::DT_FLOAT16,     op::DataType::DT_BF16,
    op::DataType::DT_INT8,       op::DataType::DT_UINT8,       op::DataType::DT_BOOL,
    op::DataType::DT_HIFLOAT8,   op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_FLOAT8_E8M0};

inline static bool CheckNotNull(const aclTensor* self, const aclIntArray* padding, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK(
        self->GetViewFormat() == out->GetViewFormat(),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], gradInoutput [%s].",
                op::ToString(self->GetViewFormat()).GetString(), op::ToString(out->GetViewFormat()).GetString()),
        return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    if (IsRegBase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_REGBASE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_REGBASE_LIST, return false);
        OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    } else {
        // 检查self的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

        // 检查out的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);

        // self和out数据类型必须一样
        OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    }
    return true;
}

static const aclTensor* GetPaddingTensor(int64_t dim, const aclIntArray* padding, aclOpExecutor* executor)
{
    FVector<int64_t, op::MAX_DIM_NUM> paddingsVector;
    // 2 is the magnification
    for (size_t i = 2 * dim; i > 0; i -= 2) {
        if (i <= static_cast<size_t>(padding->Size())) {
            // 2 and 1 indicate the element of padding is put into paddingsVector from the back to the front
            paddingsVector.emplace_back((*padding)[i - 2]);
            paddingsVector.emplace_back((*padding)[i - 1]);
        } else {
            paddingsVector.emplace_back(0);
            paddingsVector.emplace_back(0);
        }
    }
    // 2 is the magnification
    auto newpadding = executor->AllocIntArray(paddingsVector.data(), 2 * dim);
    auto paddingsTensor = executor->ConvertToTensor(newpadding, static_cast<op::DataType>(ACL_INT64));
    return paddingsTensor;
}
} // namespace op
#endif // OP_REPLICATION_PAD_H_
