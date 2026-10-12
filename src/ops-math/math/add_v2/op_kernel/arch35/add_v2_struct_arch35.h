/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_v2_struct_arch35.h
 * \brief add_v2 tiling key struct for ascend950 (arch35)
 */

#ifndef ADD_V2_STRUCT_ARCH35_H
#define ADD_V2_STRUCT_ARCH35_H

#include "atvoss/broadcast/broadcast_base_struct.h"

namespace AddV2Op {
// 空 Tensor 分支的 tiling data。内容其实用不上（无元素可搬），但**必须存在**：
// opc 依赖 kernel 里的 GET_TILING_DATA_WITH_STRUCT 反推每个模板实例的 tiling 结构体
// 大小，自定义分支若完全不引用任何结构体，会报
// "UnboundLocalError: cannot access local variable 'tiling_struct_size'"。
struct AddV2EmptyTilingData {
    int64_t numel; // 恒为 0，仅作占位与调试
};

// userDef = 0：常规 broadcast 通路，schMode 由 BroadcastBaseTiling 决定；
// userDef = 1：空 Tensor 通路，走自定义 schMode 999。ATVOSS 的 BroadcastBaseTiling
//              在合轴后显式拒绝 0 元素（broadcast_tiling.h 的 "tensor check is empty"），
//              所以空 Tensor 不能复用常规 schMode，必须另开一条自定义模板分支。
//              这里的写法与 math/select 的 SIMT 自定义分支一致。
ASCENDC_TPL_ARGS_DECL(AddV2, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
                      ASCENDC_TPL_UINT_DECL(userDef, 8, ASCENDC_TPL_UI_LIST, 0, 1));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_UINT_SEL(userDef, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_CUSTOM_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_UINT_SEL(userDef, ASCENDC_TPL_UI_LIST, 1)));
} // namespace AddV2Op

#endif // ADD_V2_STRUCT_ARCH35_H
