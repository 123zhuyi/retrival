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
 * \file moe_update_expert_tiling.h
 * \brief
 */

#ifndef __MOE_UPDATE_EXPERT_TILING_H__
#define __MOE_UPDATE_EXPERT_TILING_H__

#include <cstdint>

namespace MoeUpdateExpertNamespace {

struct MoeUpdateExpertTilingData {
    int32_t bs;
    int32_t k;
    int32_t moeExpertNum;
    int32_t f;
    uint32_t aivCoreNum;
    int64_t localRankId;
    int64_t worldSize;
    int64_t balanceMode;
    int32_t isActiveMask;
};

} // namespace MoeUpdateExpertNamespace

#endif // __MOE_UPDATE_EXPERT_TILING_H__