/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file grouped_bias_add_grad_ARA_tiling.h
\brief
*/
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GROUPED_BIAS_ADD_GRAD_ARA_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GROUPED_BIAS_ADD_GRAD_ARA_H_
#include "register/tilingdata_base.h"
#include "op_common/atvoss/reduce/reduce_tiling.h"
#include "grouped_bias_add_grad_tiling_arch35.h"

namespace optiling {

ge::graphStatus Tiling4GroupedBiasAddGradARA(
    gert::TilingContext* context, const GroupedBiasAddGradCompileInfoArch35* compileInfo);

bool IsGroupedBiasAddGradARA(const gert::TilingContext* context);

struct groupedBiasAddGradARATilingKey {
    Ops::Base::ReduceTilingKey ReduceTiling;
    GroupedBiasAddGradTilingModeArch35 templateNum;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_GROUPED_BIAS_ADD_GRAD_ARA_H_