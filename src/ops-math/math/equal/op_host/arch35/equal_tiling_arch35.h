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
 * \file equal_tiling_arch35.h
 * \brief equal_tiling_arch35
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_EQUAL_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_EQUAL_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base_class.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
using namespace Ops::Base;
class EqualTiling : public Ops::Base::TilingBaseClass {
public:
    explicit EqualTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    bool IsCapable();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus DoLibApiTiling();
    uint64_t GetTilingKey() const;
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();

private:
    uint64_t tilingKey = 0;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_EQUAL_TILING_H