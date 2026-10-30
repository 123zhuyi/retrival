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
class InplaceTopKDistance : public OpDef {
public:
    explicit InplaceTopKDistance(const char* name) : OpDef(name)
    {
        this->Input("topk_pq_distance").DataType({ge::DT_FLOAT16, ge::DT_FLOAT});
        this->Input("topk_pq_index").DataType({ge::DT_INT32});
        this->Input("topk_pq_ivf").DataType({ge::DT_INT32});
        this->Input("pq_distance").DataType({ge::DT_FLOAT16, ge::DT_FLOAT});
        this->Input("pq_index").DataType({ge::DT_INT32});
        this->Input("pq_ivf").DataType({ge::DT_INT32});
        this->Attr("order").AttrType(OPTIONAL).String("asc");

        ApplyMathAicpuDefaultCfg(*this);
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
        this->AICPU().ExtendCfgInfo(OP_INFO_FORMAT_AGNOSTIC.c_str(), TRUE_FORMAT_AGNOSTIC.c_str());
    }
};

OP_ADD(InplaceTopKDistance);
} // namespace ops
