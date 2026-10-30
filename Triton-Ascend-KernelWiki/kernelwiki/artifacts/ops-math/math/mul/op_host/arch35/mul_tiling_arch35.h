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
 * \file mul_tiling_arch35.h
 * \brief mul_tiling head file
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MUL_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MUL_TILING_H

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling {

class MulTiling : public Ops::Base::TilingBaseClass {
public:
    explicit MulTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}
    gert::TilingContext* GetContext() const
    {
        return context_;
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

public:
    uint64_t tilingKey_ = 0;
    int64_t ubSize_ = 0;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_MUL_TILING_H