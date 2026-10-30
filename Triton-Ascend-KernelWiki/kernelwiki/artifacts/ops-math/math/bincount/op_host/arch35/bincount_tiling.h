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
 * \file bincount_tiling.h
 * \brief bincount_tiling head file
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADD_TILING_H

#include "register/op_def_registry.h"
#include "op_host/tiling_base_class.h"
#include "../../op_kernel/arch35/bincount_tiling_key.h"
#include "../../op_kernel/arch35/bincount_tiling_data.h"

namespace optiling {

constexpr size_t WORK_SPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_3 = 3;
constexpr uint64_t DIM_2 = 2;
constexpr uint64_t DIM_4 = 4;
constexpr int32_t EVEN_FACTOR = 2;
constexpr int64_t INPUT_IDX_ARRAY = 0;
constexpr int64_t INPUT_IDX_SIZE = 1;
constexpr int64_t INPUT_IDX_WEIGHTS = 2;
constexpr int64_t OUTPUT_IDX_BINS = 0;
constexpr uint64_t SIZE_DTYPE_FLOAT = 4;
constexpr uint64_t SIZE_DTYPE_INT32 = 4;
constexpr uint64_t SIZE_DTYPE_INT64 = 8;

constexpr uint64_t EMPTY_WEIGHT = 0;
constexpr uint64_t WEIGHT = 1;
constexpr uint64_t OUTPUT_DTYPE_FLOAT = 0;
constexpr uint64_t OUTPUT_DTYPE_INT32 = 1;
constexpr uint64_t OUTPUT_DTYPE_INT64 = 2;

constexpr uint64_t SCH_ID_SIMT_FULL_LOAD = 0;     // full load to UB
constexpr uint64_t SCH_ID_SIMT_BATCH_LOAD = 1;    // batch load to UB
constexpr uint64_t SCH_ID_SIMT_NOT_FULL_LOAD = 2; // not load to UB
constexpr uint64_t SCH_ID_SIMT_DETERMIN = 3;

constexpr int64_t GM_ATOMIC_ADD_FACTOR = 100;
constexpr uint64_t SIMD_SIMT_DCACHE_SIZE = static_cast<uint64_t>(32 * 1024); // 32K for DCACHE and 8K for AscendC
constexpr uint32_t BATCH_MODE = 1;
constexpr int64_t MAX_UB_LOOP_FOR_BATH = 8;

struct AscendCBincountCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
};

class BincountTiling {
public:
    explicit BincountTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus Init();
    ge::graphStatus BincountGetPlatformData(const AscendCBincountCompileInfo* compileInfo);
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckInputParams();

    inline bool IsMatchSimtBatchLoadMode();
    ge::graphStatus ComputeTilingStrategy();
    ge::graphStatus ComputeTilingSimtDetermine();
    ge::graphStatus ComputeTilingSimtNotDetermine();
    ge::graphStatus SetTilingData();
    void PrintTilingData();

private:
    gert::TilingContext* context_ = nullptr;
    BincountTilingData* tilingData_{nullptr};
    // tiling key template param
    uint64_t schId_ = 0;
    uint64_t isWeight_ = EMPTY_WEIGHT;
    uint64_t outputDtype_ = 0;
    // configed info
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t localMemorySize_ = 0;
    int64_t sizeOfBinsDtype_ = 0;
    ge::DataType binsDataType_ = ge::DataType::DT_MAX;
    int64_t isDetermine_ = 0;

    // tiling info
    int64_t arrayShapeSize_ = 0;
    int64_t inputSize_ = 0;
    int64_t weightsShapeSize_ = 0;
    int64_t binsShapeSize_ = 0;
    int64_t needCoreNum_ = 0;

    int64_t ubNumCanUse_ = 0;
    int64_t ubLoopNum_ = 0;
    int64_t needXCoreNum_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;
    int64_t clearYCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYTail_ = 0;
    int64_t binsFormerLength_ = 0;
    int64_t needBinsCoreNum_ = 0;
    int64_t binsTailLength_ = 0;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADD_TILING_H
