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
 * \file sim_thread_exponential_tiling_arch32.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_SIM_THREAD_EXPONENTIAL_ARCH32_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_SIM_THREAD_EXPONENTIAL_ARCH32_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SimThreadExponentialTilingData)
TILING_DATA_FIELD_DEF(uint32_t, key0);
TILING_DATA_FIELD_DEF(uint32_t, key1);
TILING_DATA_FIELD_DEF(uint32_t, offset_t_low);
TILING_DATA_FIELD_DEF(uint32_t, offset_t_high);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, batchNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, batchNumTailCore);
TILING_DATA_FIELD_DEF(uint32_t, batchNumTotal);
TILING_DATA_FIELD_DEF(int64_t, numel);
TILING_DATA_FIELD_DEF(uint32_t, stepBlock);
TILING_DATA_FIELD_DEF(uint32_t, roundedSizeBlock);
TILING_DATA_FIELD_DEF(float, range);
TILING_DATA_FIELD_DEF(uint32_t, handleNumLoop);
TILING_DATA_FIELD_DEF(uint32_t, handleNumTail);
TILING_DATA_FIELD_DEF(uint64_t, state);
TILING_DATA_FIELD_DEF(float, start);
TILING_DATA_FIELD_DEF(float, end);
TILING_DATA_FIELD_DEF(float, lambda);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SimThreadExponential, SimThreadExponentialTilingData)

struct Tiling4SimThreadExponentialCompileInfo {};

class SimThreadExponentialTiling {
public:
    explicit SimThreadExponentialTiling(gert::TilingContext* context_) : context(context_) {};
    virtual ~SimThreadExponentialTiling() = default;
    ge::graphStatus DoTiling();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetInputTensorInfo();
    ge::graphStatus SetAttrParams();
    ge::graphStatus Tiling4Block();
    uint64_t GetTilingKey();
    bool GetDataTypeKey(ge::DataType dataType);

    void PrintInfo();
    void SetTilingKey();
    void SetTilingData();

private:
    SimThreadExponentialTilingData tiling;
    gert::TilingContext* context = nullptr;

    template <typename T>
    inline auto Min(T x, T y) const -> T
    {
        return x <= y ? x : y;
    }

    int64_t usrWorkspaceSize = 1;
    int64_t dataTypeTilingKey = 0;
    uint64_t tilingKey_{0};

    uint32_t key0 = 5;
    uint32_t key1 = 0;
    uint32_t offset_t_low = 0;
    uint32_t offset_t_high = 0;
    uint32_t useCoreNum = 48;
    uint32_t batchNumPerCore = 18;
    uint32_t batchNumTailCore = 18;
    uint32_t batchNumTotal = 864;
    int64_t numel = 250000;

    uint32_t stepNum = 0;
    uint32_t stepBlock = 0;
    uint32_t roundedSizeNum = 0;
    uint32_t roundedSizeBlock = 0;
    float range = 0;
    uint32_t handleNumLoop = 0;
    uint32_t handleNumTail = 0;
    uint32_t totalCoreNum = 0;
    uint32_t ubSize = 0;

    uint64_t state = 0;

    float start = 0;
    float end = 1;

    uint32_t dataSizeType = 3;
    int64_t count = 0;
    float lambda = 0;
    uint64_t seed = 0;
    uint64_t offset = 0;
    int threadPerProcessor = 0;
    int streamProcessorCount = 0;

    ge::DataType selfDType = ge::DT_UNDEFINED;
};
} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_SIM_THREAD_EXPONENTIAL_ARCH32_H_
