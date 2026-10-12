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
 * \file chunk_cat_tiling.h
 * \brief
 */

#ifndef CHUNK_CAT_TILING_H
#define CHUNK_CAT_TILING_H

#include <string>
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "../op_kernel/chunk_cat_tiling_data.h"

namespace optiling {

struct ChunkCatCompileInfo {};

class ChunkCatTiling {
public:
    explicit ChunkCatTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetInputInfo();
    ge::graphStatus CalculateOutputInfo();
    void DoUbSplit();
    void DoBlockSplit();
    void SetTilingData(ChunkCatTilingData* tilingData);
    std::string TilingDataToString() const;

private:
    gert::TilingContext* context_;
    bool isRegBase{false};
    int32_t coreNum_{0};
    uint64_t ubSize_{0};
    uint64_t sysWorkspaceSize_{0};
    int32_t usedCoreNum_{0};
    uint32_t srcEleUbBlock_{0};
    uint32_t srcDtypeSize_{0};

    bool isAllAlign_{true};
    bool isHalfAlign_{true};
    bool isOneConcat_{false};
    int64_t inputNum_{0};
    int64_t dim_{0};
    int64_t numChunk_{0};
    int64_t outputRow_{0};
    int64_t outputCol_{0};
    int64_t inUbSize_{0};
    int64_t outUbSize_{0};
    int64_t blockRowNum_{0};
    int64_t blockColNum_{0};
    int64_t ubRowFactor_{0};
    int64_t ubColFactor_{0};
    int64_t blockRowFactor_{0};
    int64_t blockColFactor_{0};
    int64_t tailBlockRowFactor_{0};
    int64_t tailBlockColFactor_{0};
};
} // namespace optiling
#endif
