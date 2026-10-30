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
 * \file hans_encode_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_ENCODE_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_HANS_ENCODE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
struct HansEncodeCompileInfo {
};

BEGIN_TILING_DATA_DEF(HansEncodeTilingData)
TILING_DATA_FIELD_DEF(int64_t, processCoreDim);
TILING_DATA_FIELD_DEF(int64_t, processLoopPerCore);
TILING_DATA_FIELD_DEF(int64_t, processLoopLastCore);
TILING_DATA_FIELD_DEF(int64_t, fixedLengthPerCore);
TILING_DATA_FIELD_DEF(int64_t, fixedLengthLastCore);
TILING_DATA_FIELD_DEF(int64_t, varLength);
TILING_DATA_FIELD_DEF(bool, statistic);
TILING_DATA_FIELD_DEF(bool, reshuff);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HansEncode, HansEncodeTilingData)
} // namespace optiling

#endif
