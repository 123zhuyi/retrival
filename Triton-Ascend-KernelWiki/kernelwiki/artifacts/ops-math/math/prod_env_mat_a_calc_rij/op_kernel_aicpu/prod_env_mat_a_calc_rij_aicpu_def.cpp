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
class ProdEnvMatACalcRij : public OpDef {
public:
    explicit ProdEnvMatACalcRij(const char* name) : OpDef(name)
    {
        this->Input("coord").DataType({ge::DT_FLOAT, ge::DT_DOUBLE});
        this->Input("type").DataType({ge::DT_INT32});
        this->Input("natoms").DataType({ge::DT_INT32});
        this->Input("box").DataType({ge::DT_FLOAT, ge::DT_DOUBLE});
        this->Input("mesh").DataType({ge::DT_INT32});
        this->Output("rij").DataType({ge::DT_FLOAT, ge::DT_DOUBLE});
        this->Output("nlist").DataType({ge::DT_INT32});
        this->Output("distance").DataType({ge::DT_FLOAT, ge::DT_DOUBLE});
        this->Output("rij_x").DataType({ge::DT_FLOAT, ge::DT_DOUBLE});
        this->Output("rij_y").DataType({ge::DT_FLOAT, ge::DT_DOUBLE});
        this->Output("rij_z").DataType({ge::DT_FLOAT, ge::DT_DOUBLE});
        this->Attr("rcut_a").AttrType(OPTIONAL).Float(1.0);
        this->Attr("rcut_r").AttrType(OPTIONAL).Float(1.0);
        this->Attr("rcut_r_smth").AttrType(OPTIONAL).Float(1.0);
        this->Attr("sel_a").AttrType(OPTIONAL).ListInt({});
        this->Attr("sel_r").AttrType(OPTIONAL).ListInt({});

        ApplyMathAicpuDefaultCfg(*this);
        // opsFlag=CLOSE and formatAgnostic=False are the applied defaults; subTypeOfInferShape defaults to 1 and must
        // be overridden to 2 because the output shapes are inferred from the const value of natoms.
        this->AICPU().ExtendCfgInfo(OP_INFO_SUB_TYPE_OF_INFERSHAPE.c_str(), DEFAULT_SUB_TYPE_OF_INFERSHAPE_2.c_str());
    }
};

OP_ADD(ProdEnvMatACalcRij);
} // namespace ops
