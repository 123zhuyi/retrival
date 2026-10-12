/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file im2_col_tiling_def.h
 * \brief
 */

#ifndef IM_2_COL_TILING_DEF_H
#define IM_2_COL_TILING_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {
struct Im2ColCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};
} // namespace optiling

#endif // IM_2_COL_TILING_DEF_H
