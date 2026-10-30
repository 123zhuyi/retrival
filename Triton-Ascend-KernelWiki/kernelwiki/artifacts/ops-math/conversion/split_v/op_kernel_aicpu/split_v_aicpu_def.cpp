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
class SplitV : public OpDef {
public:
    explicit SplitV(const char* name) : OpDef(name)
    {
        // I/O definition (参照ops-math/conversion/split_v/op_host/split_v_def.cpp)
        // Note: first input name is "x", not "value"
        this->Input("x")
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT16,
                       ge::DT_UINT16, ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_UINT32, ge::DT_UINT64, ge::DT_BOOL, ge::DT_DOUBLE});
        
        this->Input("size_splits")
            .DataType({ge::DT_INT32, ge::DT_INT64});
        
        this->Input("split_dim")
            .DataType({ge::DT_INT32});
        
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT16,
                       ge::DT_UINT16, ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_UINT32, ge::DT_UINT64, ge::DT_BOOL, ge::DT_DOUBLE});
        
        ApplyMathAicpuDefaultCfg(*this);
        
        // 差异配置（与默认值不同的字段）
        // canndev ini中opsFlag=OPS_FLAG_OPEN，默认配置为OPS_FLAG_CLOSE
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
    }
};

OP_ADD(SplitV);

}  // namespace ops