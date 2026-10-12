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
 * \file all_gather_add_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class AllGatherAdd : public OpDef {
 public:
  explicit AllGatherAdd(const char *name) : OpDef(name) {
    this->Input("a")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Input("b")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    
    this->Output("c")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Output("gather_out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->Attr("group").AttrType(REQUIRED).String(); // 通算融合算子属性，表示通信域名称
    this->Attr("rank_size").AttrType(REQUIRED).Int(0);

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
    this->MC2().HcclGroup("group"); // group 属性配置为该算子的通信域名称
  }
};

OP_ADD(AllGatherAdd);
}  // namespace ops
