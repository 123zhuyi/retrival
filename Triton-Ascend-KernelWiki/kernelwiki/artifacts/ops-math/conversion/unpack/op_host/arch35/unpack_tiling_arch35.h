/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNPACK_RUNTIME2_H_
#define UNPACK_RUNTIME2_H_

#include "conversion/split_v/op_host/arch35/split_v_tiling_arch35.h"
#include "register/op_impl_registry.h"

namespace optiling {
class UnpackTiling: public SplitVTiling{
public:
    explicit UnpackTiling(gert::TilingContext* tilingContext) : SplitVTiling(tilingContext) {};

protected:
    ge::graphStatus InitParamsSameLen(int32_t maxCoreNum, uint32_t ubSize) override;
    ge::graphStatus GetInputParamsSameLen() override;
};

REGISTER_TILING_DATA_CLASS(Unpack, SplitVTilingData);
constexpr int32_t kBlockSize{32};
constexpr int32_t kRegBufSize{192};
constexpr int32_t kCalcMemSize{1024};
constexpr int32_t RightIndex{2};
enum class TilingStrategy : uint32_t {
  SINGLE_OUTPUT = 0,
  SMALL_SHAPE,
  BIG_SHAPE,
  LESS_32B,
  LAST_DIM_SMALL,
  SMALL_SHAPE_MULTI_COEXISTING_QUANTITIES
};

const std::map<ge::DataType, int32_t> kDtypeSizeMap {
    {ge::DT_INT8, 1},   {ge::DT_UINT8, 1},   {ge::DT_BOOL, 1},  {ge::DT_INT64, 8},
    {ge::DT_UINT64, 8}, {ge::DT_FLOAT16, 2}, {ge::DT_INT16, 2}, {ge::DT_UINT16, 2},
    {ge::DT_FLOAT, 4},  {ge::DT_INT32, 4},   {ge::DT_UINT32, 4}};

struct TilingInfo {
  int32_t axis;
  int32_t output_num;
  int32_t ub_size;
  int32_t core_num;
  int32_t dtype_size;
  bool is_special_tiling;
  int32_t multi_coexisting_quantities;
};

struct UnpackCompileInfo {
  uint32_t maxCoreNum{0};
  uint32_t ubSizePlatform{0};
};

struct ConstParams {
  int32_t ub_limit;
  TilingInfo tiling_info;
};
}  // namespace optiling
#endif  // UNPACK_RUNTIME2_H_