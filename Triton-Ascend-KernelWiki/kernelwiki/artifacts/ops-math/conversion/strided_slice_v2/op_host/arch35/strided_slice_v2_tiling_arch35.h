/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file strided_slice_v2_tiling.h
 * \brief
 */
#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_STRIDED_SLICE_V2_TILING_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_STRIDED_SLICE_V2_TILING_H_

#include <string>
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"
#include "../../../strided_slice/op_host/arch35/strided_slice_tiling_arch35.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(StridedSliceV2, StridedSliceTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_100, StridedSliceMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_101, StridedSliceMALastDimTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_102, StridedSliceNDDMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_103, StridedSliceNDDMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_302, StridedSliceNDDMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_303, StridedSliceNDDMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_150, StridedSliceMALast2DimTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_200, StridedSliceSIMTTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_201, StridedSliceSIMTTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_300, StridedSliceMAGatherTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV2_301, StridedSliceMAUB2UBTilingData)

struct StridedSliceV2CompileInfo {
    int32_t blockDim {-1};
    int32_t ubSize {0};
    int32_t coreNum {0};
    uint32_t cacheLineSize {0};
    std::string to_string() const
    {
        std::string str = "blockDim: " + std::to_string(blockDim);
        str += " ubSize: " + std::to_string(ubSize);
        return str;
    }
};

class StridedSliceV2Tiling : public StrideSliceTiling {
    public:
        explicit StridedSliceV2Tiling(gert::TilingContext* context) : StrideSliceTiling(context) {};
};
}

#endif // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_STRIDED_SLICE_V2_TILING_H_
