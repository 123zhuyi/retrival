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
class SliceWrite : public OpDef {
public:
    explicit SliceWrite(const char* name) : OpDef(name)
    {
        // I/O definition (参照 canndev/ops/built-in/op_proto/inc/vector_search.h:289-297)
        this->Input("x")
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_INT32, ge::DT_INT64});
        
        this->Input("begin")
            .DataType({ge::DT_INT32, ge::DT_INT64});
        
        this->Input("value")
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_INT32, ge::DT_INT64});
        
        this->Output("x")
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_INT32, ge::DT_INT64});
        
        // 应用 ops-math 默认配置
        ApplyMathAicpuDefaultCfg(*this);
        
        // 差异配置（与默认值不同的字段）
        // canndev ini 中 opsFlag=OPS_FLAG_OPEN，默认配置为 OPS_FLAG_CLOSE
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
    }
};

OP_ADD(SliceWrite);

}  // namespace ops