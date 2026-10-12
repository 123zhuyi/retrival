/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "../../../common/inc/aicpu/aicpu_op_def.h"

namespace ops {
class SparseSegmentMeanGrad : public OpDef {
public:
    explicit SparseSegmentMeanGrad(const char* name) : OpDef(name)
    {
        // Keep this list strictly aligned with sparse_segment_mean_grad_aicpu.cpp Compute() switch-case.
        const std::vector<ge::DataType> gradDataTypes = {ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT16};
        const std::vector<ge::DataType> indexDataTypes = {ge::DT_INT32, ge::DT_INT64};
        this->Input("x").ParamType(REQUIRED).DataType(gradDataTypes);
        this->Input("indices").ParamType(REQUIRED).DataType(indexDataTypes);
        this->Input("segment_ids").ParamType(REQUIRED).DataType(indexDataTypes);
        this->Input("output_dim0").ParamType(REQUIRED).DataType({ge::DT_INT32});
        this->Output("y").ParamType(REQUIRED).DataType(gradDataTypes);

        ApplyNnAicpuDefaultCfg(*this);
        // canndev aicpu_kernel.ini declares opInfo.opsFlag=OPS_FLAG_OPEN for this operator.
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
    }
};

OP_ADD(SparseSegmentMeanGrad);
} // namespace ops
