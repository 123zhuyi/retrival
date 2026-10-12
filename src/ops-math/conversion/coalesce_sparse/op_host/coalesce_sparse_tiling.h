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
 * \file coalesce_sparse_tiling.h
 * \brief
 * 
 * 
 * 
 * 
 * 
 * 
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_CoalesceSparse_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_CoalesceSparse_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CoalesceSparseTilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, m);
TILING_DATA_FIELD_DEF(uint64_t, valueSize);
TILING_DATA_FIELD_DEF(uint64_t, taskNum);
TILING_DATA_FIELD_DEF(uint64_t, taskTail);
TILING_DATA_FIELD_DEF(uint64_t, moveOneSize);
TILING_DATA_FIELD_DEF(uint64_t, taskRepeatTimes);
TILING_DATA_FIELD_DEF(uint64_t, taskRepeatTail);
TILING_DATA_FIELD_DEF(uint64_t, taskTailRepeatTimes);
TILING_DATA_FIELD_DEF(uint64_t, taskTailRepeatTail);
TILING_DATA_FIELD_DEF(uint64_t, moveValueTimes);
TILING_DATA_FIELD_DEF(uint64_t, moveValueLen);
TILING_DATA_FIELD_DEF(uint64_t, moveValueTail);
TILING_DATA_FIELD_DEF(uint64_t, deterministicFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CoalesceSparse, CoalesceSparseTilingData)
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_CoalesceSparse_H