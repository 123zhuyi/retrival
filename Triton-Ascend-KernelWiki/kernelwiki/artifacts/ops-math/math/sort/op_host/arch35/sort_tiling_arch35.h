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
 * \file sort_tiling_arch35.h
 * \brief sort ac tiling
 */
#ifndef SORT_TILING_ARCH35_H
#define SORT_TILING_ARCH35_H
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
ge::graphStatus SortTilingSimt(gert::TilingContext* context, int32_t maxCoreNum);
}
#endif // SORT_TILING_ARCH35_H
