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
 * \file strided_slice_v3.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDED_SLICE_V3_RUNTIME2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDED_SLICE_V3_RUNTIME2_H

#include <string>
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"
#include "../../../strided_slice/op_host/arch35/strided_slice_tiling_arch35.h"

namespace optiling {
struct StridedSliceV3CompileInfo {
  int32_t block_dim;
  int32_t ub_size;
  uint32_t cacheLineSize{0};
};

struct SliceParameters {
    std::vector<int64_t> input;
    std::vector<int64_t> output_shape;
    std::vector<int64_t> begin_list;
    std::vector<int64_t> end_list;
    std::vector<int64_t> stride_list;
    int64_t tiling_mode = 0;
    int64_t core_num = 0;

    std::string to_string() const;
};

REGISTER_TILING_DATA_CLASS(StridedSliceV3, StridedSliceTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_100, StridedSliceMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_101, StridedSliceMALastDimTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_102, StridedSliceNDDMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_103, StridedSliceNDDMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_302, StridedSliceNDDMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_303, StridedSliceNDDMATilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_150, StridedSliceMALast2DimTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_200, StridedSliceSIMTTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_201, StridedSliceSIMTTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_300, StridedSliceMAGatherTilingData)
REGISTER_TILING_DATA_CLASS(StridedSliceV3_301, StridedSliceMAUB2UBTilingData)

class StridedSliceV3Tiling : public StrideSliceTiling {
public:
    explicit StridedSliceV3Tiling(gert::TilingContext* context) : StrideSliceTiling(context) {};
};

ge::graphStatus StridedSliceV3TilingForAscendC(gert::TilingContext* context, int64_t coreNum, int64_t ubSize,
                                               int64_t cacheLineSize, SliceParameters& param,
                                               const ge::DataType dtype);

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDED_SLICE_V3_RUNTIME2_H