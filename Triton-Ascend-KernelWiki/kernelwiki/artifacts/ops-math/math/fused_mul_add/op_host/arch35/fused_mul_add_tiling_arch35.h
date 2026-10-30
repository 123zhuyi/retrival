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
 * \file fused_mul_add_tiling_arch35.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_MUL_ADD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_MUL_ADD_TILING_H

#include "register/op_def_registry.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"

namespace optiling {

class FusedMulAddTiling : public Ops::Base::TilingBaseClass {
public:
    explicit FusedMulAddTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t tilingKey = 0;
    bool CheckDtype(
        const ge::DataType& x1Dtype, const ge::DataType& x2Dtype, const ge::DataType& x3Dtype,
        const ge::DataType& outputDtype) const;
};

struct FusedMulAddCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_MUL_ADD_TILING_H
