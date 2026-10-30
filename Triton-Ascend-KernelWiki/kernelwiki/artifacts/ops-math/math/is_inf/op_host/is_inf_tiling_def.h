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
 * \file is_inf_tiling_def.h
 * \brief
 */

#ifndef IS_INF_TILING_DEF_H
#define IS_INF_TILING_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {
struct IsInfCompileInfo {
};

BEGIN_TILING_DATA_DEF(IsInfTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalDataCount);
TILING_DATA_FIELD_DEF(uint32_t, usableUbSize);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, perCoreDataCount);
TILING_DATA_FIELD_DEF(uint32_t, tailDataCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreDataCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsInf, IsInfTilingData)
} // namespace optiling

#endif // IS_INF_TILING_DEF_H
