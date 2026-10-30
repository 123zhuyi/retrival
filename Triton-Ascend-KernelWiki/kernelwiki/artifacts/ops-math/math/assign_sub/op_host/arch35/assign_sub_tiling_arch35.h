/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ASSIGN_SUB_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ASSIGN_SUB_H
#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/arch35/assign_sub_tiling_struct.h"

namespace optiling {
using namespace AssignSubNs;
using namespace Ops::Base;
struct AssignSubCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class AssignSubTiling {
public:
    explicit AssignSubTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    AssignSubTilingData* tiling = nullptr;

protected:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape() const;
    void SetTilingData();

private:
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_ASSIGN_SUB_H
