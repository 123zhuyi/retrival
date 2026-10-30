/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rfft1d.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Rfft1D);

static const int UPPER_BORDER = 262144;

bool IsRfft1DAiCoreSupported(const aclTensor* self, int64_t n)
{
    bool res = false;

    if ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         IsRegBase() ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) &&
        (n <= UPPER_BORDER) && self->GetDataType() == op::DataType::DT_FLOAT) {
        res = true;
    }
    return res;
}

static const aclTensor* Rfft1DAiCore(
    const aclTensor* self, const aclTensor* dft, aclTensor* out, int64_t n, int64_t norm, aclOpExecutor* executor)
{
    L0_DFX(Rfft1DAiCore, self, dft, n, norm, out);

    ADD_TO_LAUNCHER_LIST_AICORE(
        Rfft1D, OP_ATTR_NAMES({"n", "norm"}), OP_INPUT(self, dft), OP_OUTPUT(out), OP_ATTR(n, norm));
    return out;
}

static const aclTensor* Rfft1DAiCpu(
    const aclTensor* self, const aclTensor* dft, aclTensor* out, int64_t n, int64_t norm, aclOpExecutor* executor)
{
    L0_DFX(Rfft1DAiCpu, self, dft, n, norm, out);

    static internal::AicpuTaskSpace space("Rfft1D");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Rfft1D, OP_ATTR_NAMES({"n", "norm"}), OP_INPUT(self, dft), OP_OUTPUT(out), OP_ATTR(n, norm));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

static op::Shape GetOutputShape(const aclTensor* self, int64_t n)
{
    op::Shape input_shape = self->GetViewShape();

    int64_t dims = input_shape.GetDimNum();
    op::Shape outputShape;
    outputShape.SetDimNum(dims + 1);
    for (int i = 0; i < dims - 1; i++) {
        outputShape.SetDim(i, input_shape.GetDim(i));
    }

    uint8_t complexPart = 2;
    outputShape.SetDim(dims - 1, n / complexPart + 1);
    outputShape.SetDim(dims, complexPart);

    return outputShape;
}

const aclTensor* Rfft1D(const aclTensor* self, const aclTensor* dft, int64_t n, int64_t norm, aclOpExecutor* executor)
{
    op::Shape outShape = GetOutputShape(self, n);
    op::DataType outType = self->GetDataType();
    auto out = executor->AllocTensor(outShape, outType);
    CHECK_RET(out != nullptr, nullptr);

    if (IsRfft1DAiCoreSupported(self, n)) {
        return Rfft1DAiCore(self, dft, out, n, norm, executor);
    } else {
        return Rfft1DAiCpu(self, dft, out, n, norm, executor);
    }
}

} // namespace l0op