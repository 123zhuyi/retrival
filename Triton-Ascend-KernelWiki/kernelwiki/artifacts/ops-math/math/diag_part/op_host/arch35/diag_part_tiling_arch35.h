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
 * \file diag_part_tiling.h
 * \brief tiling for DiagPart
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_PART_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_PART_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t B8_SPLIT_ELEMENT = 128;
constexpr int64_t B16_SPLIT_ELEMENT = 64;
constexpr int64_t B32_SPLIT_ELEMENT = 32;
constexpr int64_t B64_SPLIT_ELEMENT = 16;
constexpr int32_t B8_BYTES = 1;
constexpr int32_t B16_BYTES = 2;
constexpr int32_t B32_BYTES = 4;
constexpr int32_t B64_BYTES = 8;
constexpr uint64_t TWO = 2;
constexpr int64_t WORK_SPACE_SIZE = static_cast<int64_t>(16 * 1024 * 1024);
constexpr int64_t TILING_KEY_GATHER = 1000;
constexpr int64_t TILING_KEY_SIMT = 2000;
constexpr int64_t DCACHE_SIZE = 32 * 1024;

BEGIN_TILING_DATA_DEF(DiagPartTilingData)
TILING_DATA_FIELD_DEF(int64_t, sideLength);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, tailNum);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DiagPart, DiagPartTilingData)

ge::graphStatus DiagPartTilingForAscendC(gert::TilingContext* context);

class DiagPartTiling {
public:
    explicit DiagPartTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunDiagPartTiling();
    ge::graphStatus DiagPartVerifying();
    ge::graphStatus RunDiagPartGatherTiling();
    ge::graphStatus SetTilingData();
    void PrintTilingData();

private:
    gert::TilingContext* tilingContext_ = nullptr;
    DiagPartTilingData tilingData_;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t sideLength_ = 1;
    int64_t blockNum_ = 0;
};

struct DiagCompileInfo {
    int64_t core_num;
    int64_t ub_size;
};

} // namespace optiling
#endif