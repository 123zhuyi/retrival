/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_GCD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_GCD_OP_H_

#include "opdev/common_types.h"
#include "opdev/op_executor.h"

namespace gcd {
// These direct mixed signatures must stay synchronized with op_host/gcd_def.cpp.
inline bool IsRegisteredMixedKernelSignature(op::DataType selfType, op::DataType otherType, op::DataType outType)
{
    bool uint8Bf16ToUint8 = outType == op::DataType::DT_UINT8 &&
                            ((selfType == op::DataType::DT_UINT8 && otherType == op::DataType::DT_BF16) ||
                             (selfType == op::DataType::DT_BF16 && otherType == op::DataType::DT_UINT8));
    bool int8FloatToInt8 = outType == op::DataType::DT_INT8 &&
                           ((selfType == op::DataType::DT_INT8 && otherType == op::DataType::DT_FLOAT) ||
                            (selfType == op::DataType::DT_FLOAT && otherType == op::DataType::DT_INT8));
    bool int16Fp16ToInt16 = outType == op::DataType::DT_INT16 &&
                            ((selfType == op::DataType::DT_INT16 && otherType == op::DataType::DT_FLOAT16) ||
                             (selfType == op::DataType::DT_FLOAT16 && otherType == op::DataType::DT_INT16));
    return uint8Bf16ToUint8 || int8FloatToInt8 || int16Fp16ToInt16;
}
} // namespace gcd

namespace l0op {
const aclTensor* Gcd(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
// Directly writes a registered same-dtype Gcd signature into a same-dtype output tensor.
const aclTensor* GcdToOutput(const aclTensor* self, const aclTensor* other, aclTensor* out, aclOpExecutor* executor);
// Supports only the mixed dtype signatures explicitly registered in op_host/gcd_def.cpp.
const aclTensor* GcdWithOutputType(const aclTensor* self, const aclTensor* other, op::DataType outputType,
                                   aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_GCD_OP_H_
