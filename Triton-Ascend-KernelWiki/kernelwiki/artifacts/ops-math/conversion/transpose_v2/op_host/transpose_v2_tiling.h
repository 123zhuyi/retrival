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
 * \file transpose_v2_tiling.h
 * \brief
 */

#ifndef TRANSPOSE_V2_TILING_H
#define TRANSPOSE_V2_TILING_H

#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"

namespace optiling {
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t TRANS_BLOCK = 16;
constexpr uint64_t NO_TRANS_KEY = 1000000;
constexpr uint64_t TYPE_KEY = 10;
constexpr uint64_t LIMIT_H = 128;
constexpr uint64_t LIMIT_W = 128;
constexpr uint64_t BUFFER_NUM = 4;
constexpr uint64_t COPY_KEY = 1;
constexpr uint64_t PERM_KEY = 100;

BEGIN_TILING_DATA_DEF(TransposeV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, dim1Len);
TILING_DATA_FIELD_DEF(uint64_t, dim2Len);
TILING_DATA_FIELD_DEF(uint64_t, dim3Len);
TILING_DATA_FIELD_DEF(uint64_t, dim3LenAlign);
TILING_DATA_FIELD_DEF(uint64_t, tasksPerCore);
TILING_DATA_FIELD_DEF(uint64_t, tailCore);
TILING_DATA_FIELD_DEF(uint64_t, subMode);
TILING_DATA_FIELD_DEF(uint64_t, dim1OnceMax);
TILING_DATA_FIELD_DEF(uint64_t, dim2OnceMax);
TILING_DATA_FIELD_DEF(uint64_t, doubleBuffer);

TILING_DATA_FIELD_DEF(int64_t, tasksTail);
TILING_DATA_FIELD_DEF(int64_t, inputH);
TILING_DATA_FIELD_DEF(int64_t, inputW);
TILING_DATA_FIELD_DEF(int64_t, inputH16Align);
TILING_DATA_FIELD_DEF(int64_t, inputWAlign);
TILING_DATA_FIELD_DEF(int64_t, hOnce);
TILING_DATA_FIELD_DEF(int64_t, tasksOnceMax);
TILING_DATA_FIELD_DEF(int64_t, transLoop);
TILING_DATA_FIELD_DEF(int64_t, repeatH);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TransposeV2, TransposeV2TilingData);

struct Tiling4TransposeV2CompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
    uint32_t sysWorkspaceSize;
};
} // namespace optiling
#endif
