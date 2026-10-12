/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_tiling.h
 * \brief
 */
#ifndef MATH_HISTOGRAM_V2_H
#define MATH_HISTOGRAM_V2_H
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"
#include "op_host/tiling_base_class.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HistogramV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, bins);
TILING_DATA_FIELD_DEF(int64_t, ubBinsLength);

TILING_DATA_FIELD_DEF(int64_t, formerNum);
TILING_DATA_FIELD_DEF(int64_t, formerLength);
TILING_DATA_FIELD_DEF(int64_t, formerLengthAligned);
TILING_DATA_FIELD_DEF(int64_t, tailLength);
TILING_DATA_FIELD_DEF(int64_t, tailLengthAligned);

TILING_DATA_FIELD_DEF(int64_t, formerTileNum);
TILING_DATA_FIELD_DEF(int64_t, formerTileDataLength);
TILING_DATA_FIELD_DEF(int64_t, formerTileLeftDataLength);
TILING_DATA_FIELD_DEF(int64_t, formerTileLeftDataLengthAligned);

TILING_DATA_FIELD_DEF(int64_t, tailTileNum);
TILING_DATA_FIELD_DEF(int64_t, tailTileDataLength);
TILING_DATA_FIELD_DEF(int64_t, tailTileLeftDataLength);
TILING_DATA_FIELD_DEF(int64_t, tailTileLeftDataLengthAligned);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HistogramV2, HistogramV2TilingData)

BEGIN_TILING_DATA_DEF(HistogramV2SimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, bins);
TILING_DATA_FIELD_DEF(int64_t, ubNumCanUse);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNum);
TILING_DATA_FIELD_DEF(int64_t, needXCoreNum);
TILING_DATA_FIELD_DEF(int64_t, formerLength);
TILING_DATA_FIELD_DEF(int64_t, tailLength);
TILING_DATA_FIELD_DEF(int64_t, clearYCoreNum);
TILING_DATA_FIELD_DEF(int64_t, clearYFactor);
TILING_DATA_FIELD_DEF(int64_t, clearYTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HistogramV2_101, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_102, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_103, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_104, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_105, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_106, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_107, HistogramV2SimtTilingData)

REGISTER_TILING_DATA_CLASS(HistogramV2_201, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_202, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_203, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_204, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_205, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_206, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_207, HistogramV2SimtTilingData)

REGISTER_TILING_DATA_CLASS(HistogramV2_301, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_302, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_303, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_304, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_305, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_306, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_307, HistogramV2SimtTilingData)

// fp32 output (non-deterministic): tens digit = 1 (base + 10 + inputDtypeVal); only fp32(1) and fp16(7) used
// UB_FULL fp32-out: 100 + 10 + 1/7 = 111/117
// UB_NOT_FULL fp32-out: 200 + 10 + 1/7 = 211/217
// UB_NOT_FULL_SIMT fp32-out: 300 + 10 + 1/7 = 311/317
REGISTER_TILING_DATA_CLASS(HistogramV2_111, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_117, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_211, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_217, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_311, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_317, HistogramV2SimtTilingData)

// fp32 output + deterministic: thousands digit = 1 (fp32-out-key + 1000); only fp32(1) and fp16(7) used
// UB_FULL det fp32-out: 111 + 1000 = 1111 / 117 + 1000 = 1117
// UB_NOT_FULL det fp32-out: 211 + 1000 = 1211 / 217 + 1000 = 1217
// UB_NOT_FULL_SIMT det fp32-out: 311 + 1000 = 1311 / 317 + 1000 = 1317
REGISTER_TILING_DATA_CLASS(HistogramV2_1111, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_1117, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_1211, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_1217, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_1311, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_1317, HistogramV2SimtTilingData)

struct HistogramV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t sysWorkspaceSize = 0;
    NpuArch npuArch = NpuArch::DAV_2201;
};

class HistogramV2BaseClass : public Ops::Base::TilingBaseClass
{
public:
    explicit HistogramV2BaseClass(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;

    NpuArch npuArch = NpuArch::DAV_2201;
};
} // namespace optiling
#endif // MATH_HISTOGRAM_V2_H