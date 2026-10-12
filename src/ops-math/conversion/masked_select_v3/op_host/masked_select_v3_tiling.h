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
 * \file masked_select_v3_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MASKED_SELECT_V3_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MASKED_SELECT_V3_H

#include <cstdint>
#include <vector>
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaskedSelectV3TilingData) 
  TILING_DATA_FIELD_DEF(uint64_t, formerNum);
  TILING_DATA_FIELD_DEF(uint64_t, formerLength);
  TILING_DATA_FIELD_DEF(uint64_t, formertileNum);
  TILING_DATA_FIELD_DEF(uint64_t, formertileLength);
  TILING_DATA_FIELD_DEF(uint64_t, formerlasttileLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailtileNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailtileLength);
  TILING_DATA_FIELD_DEF(uint64_t, taillasttileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaskedSelectV3, MaskedSelectV3TilingData)
struct MaskedSelectV3CompileInfo {
  uint64_t aivNum = 0;
  uint64_t ubSize = 0;
  uint64_t workSpaceSize = 0;
  bool isRegbase = false;
};

ge::graphStatus TilingForMaskedSelectV3IsRegbaseSocVersion(gert::TilingContext* context);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_MASKED_SELECT_V3_H