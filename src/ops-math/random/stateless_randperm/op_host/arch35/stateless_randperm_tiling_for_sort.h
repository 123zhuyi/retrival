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
 * \file stateless_randperm_tiling_for_sort.h
 * \brief sort tiling API for stateless_randperm
 */
#ifndef STATELESS_RANDPERM_TILING_FOR_SORT_H
#define STATELESS_RANDPERM_TILING_FOR_SORT_H
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "../../../../math/sort/op_kernel/arch35/sort_tiling_data.h"  // 侵入修改：头文件路径
#include "../../../../math/sort/op_kernel/arch35/sort_tiling_key.h"   // 侵入修改：头文件路径

namespace optiling {
namespace statelessRandpermTiling{
ge::graphStatus  SortTilingSimt(gert::TilingContext* context, int32_t maxCoreNum);
}
}
#endif
