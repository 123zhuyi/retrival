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
class TanhGrad : public OpDef {
 public:
  explicit TanhGrad(const char *name) : OpDef(name)
  {
    this->Input("y").DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE,
                                ge::DT_COMPLEX64, ge::DT_COMPLEX128});
    this->Input("dy").DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE,
                                 ge::DT_COMPLEX64, ge::DT_COMPLEX128});
    this->Output("z").DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE,
                                 ge::DT_COMPLEX64, ge::DT_COMPLEX128});
    this->Attr("complex_conj").AttrType(OPTIONAL).Bool(false);

    ApplyMathAicpuDefaultCfg(*this);
  }
};

OP_ADD(TanhGrad);
}  // namespace ops
