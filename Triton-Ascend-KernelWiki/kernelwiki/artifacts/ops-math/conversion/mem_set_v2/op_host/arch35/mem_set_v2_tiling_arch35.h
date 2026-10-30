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
 * \file mem_set_v2_tiling_arch35.h
 * \brief tiling for mem set v2
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MEM_SET_V2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MEM_SET_V2_H

#include <vector>
#include <set>
#include <string>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {

constexpr int64_t TENSOR_LIST_SIZE_2 = 2;
constexpr int64_t TENSOR_LIST_SIZE_4 = 4;
constexpr int64_t TENSOR_LIST_SIZE_8 = 8;
constexpr int64_t TENSOR_LIST_SIZE_16 = 16;
constexpr int64_t TENSOR_LIST_SIZE_32 = 32;
constexpr int64_t TENSOR_LIST_SIZE_64 = 64;
constexpr int64_t TENSOR_LIST_SIZE_128 = 128;
constexpr int64_t TENSOR_LIST_SIZE_256 = 256;

// 为memsetv2注册一个tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2NoListTilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MemSetV2, MemSetV2NoListTilingData);

// 处理个数为(0,2]的tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2List2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, processGMNum); // 待处理的GM个数
TILING_DATA_FIELD_DEF(int64_t, valuesIntListSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_2, dtypesList);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_2, valuesIntList);
TILING_DATA_FIELD_DEF_ARR(float, TENSOR_LIST_SIZE_2, valuesFloatList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_2, blockNumList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_2, tailSizeList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_2, startIndexList);
END_TILING_DATA_DEF;

// 处理个数为(2,4]的tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2List4TilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, processGMNum); // 待处理的GM个数
TILING_DATA_FIELD_DEF(int64_t, valuesIntListSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_4, dtypesList);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_4, valuesIntList);
TILING_DATA_FIELD_DEF_ARR(float, TENSOR_LIST_SIZE_4, valuesFloatList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_4, blockNumList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_4, tailSizeList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_4, startIndexList);
END_TILING_DATA_DEF;

// 处理个数为(4,8]的tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2List8TilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, processGMNum); // 待处理的GM个数
TILING_DATA_FIELD_DEF(int64_t, valuesIntListSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_8, dtypesList);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_8, valuesIntList);
TILING_DATA_FIELD_DEF_ARR(float, TENSOR_LIST_SIZE_8, valuesFloatList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_8, blockNumList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_8, tailSizeList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_8, startIndexList);
END_TILING_DATA_DEF;

// 处理个数为(8,16]的tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2List16TilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, processGMNum); // 待处理的GM个数
TILING_DATA_FIELD_DEF(int64_t, valuesIntListSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_16, dtypesList);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_16, valuesIntList);
TILING_DATA_FIELD_DEF_ARR(float, TENSOR_LIST_SIZE_16, valuesFloatList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_16, blockNumList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_16, tailSizeList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_16, startIndexList);
END_TILING_DATA_DEF;

// 处理个数为(16,32]的tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2List32TilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, processGMNum); // 待处理的GM个数
TILING_DATA_FIELD_DEF(int64_t, valuesIntListSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_32, dtypesList);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_32, valuesIntList);
TILING_DATA_FIELD_DEF_ARR(float, TENSOR_LIST_SIZE_32, valuesFloatList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_32, blockNumList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_32, tailSizeList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_32, startIndexList);
END_TILING_DATA_DEF;

// 处理个数为(32,64]的tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2List64TilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, processGMNum); // 待处理的GM个数
TILING_DATA_FIELD_DEF(int64_t, valuesIntListSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_64, dtypesList);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_64, valuesIntList);
TILING_DATA_FIELD_DEF_ARR(float, TENSOR_LIST_SIZE_64, valuesFloatList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_64, blockNumList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_64, tailSizeList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_64, startIndexList);
END_TILING_DATA_DEF;

// 处理个数为(64,128]的tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2List128TilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, processGMNum); // 待处理的GM个数
TILING_DATA_FIELD_DEF(int64_t, valuesIntListSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_128, dtypesList);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_128, valuesIntList);
TILING_DATA_FIELD_DEF_ARR(float, TENSOR_LIST_SIZE_128, valuesFloatList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_128, blockNumList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_128, tailSizeList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_128, startIndexList);
END_TILING_DATA_DEF;

// 处理个数为(128,256]的tiling数据结构
BEGIN_TILING_DATA_DEF(MemSetV2List256TilingData)
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, processGMNum); // 待处理的GM个数
TILING_DATA_FIELD_DEF(int64_t, valuesIntListSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_256, dtypesList);
TILING_DATA_FIELD_DEF_ARR(int64_t, TENSOR_LIST_SIZE_256, valuesIntList);
TILING_DATA_FIELD_DEF_ARR(float, TENSOR_LIST_SIZE_256, valuesFloatList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_256, blockNumList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_256, tailSizeList);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TENSOR_LIST_SIZE_256, startIndexList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MemSetV2_10002, MemSetV2List2TilingData);
REGISTER_TILING_DATA_CLASS(MemSetV2_10004, MemSetV2List4TilingData);
REGISTER_TILING_DATA_CLASS(MemSetV2_10008, MemSetV2List8TilingData);
REGISTER_TILING_DATA_CLASS(MemSetV2_10016, MemSetV2List16TilingData);
REGISTER_TILING_DATA_CLASS(MemSetV2_10032, MemSetV2List32TilingData);
REGISTER_TILING_DATA_CLASS(MemSetV2_10064, MemSetV2List64TilingData);
REGISTER_TILING_DATA_CLASS(MemSetV2_10128, MemSetV2List128TilingData);
REGISTER_TILING_DATA_CLASS(MemSetV2_10256, MemSetV2List256TilingData);

struct MemSetV2CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

constexpr uint64_t INDEX_VALUESINT = 0;
constexpr uint64_t INDEX_VALUESFLOAT = 1;

constexpr uint64_t TILINGKEY_10002 = 10002;
constexpr uint64_t TILINGKEY_10004 = 10004;
constexpr uint64_t TILINGKEY_10008 = 10008;
constexpr uint64_t TILINGKEY_10016 = 10016;
constexpr uint64_t TILINGKEY_10032 = 10032;
constexpr uint64_t TILINGKEY_10064 = 10064;
constexpr uint64_t TILINGKEY_10128 = 10128;
constexpr uint64_t TILINGKEY_10256 = 10256;

constexpr uint64_t SIZE_ZERO = 0;
} // namespace optiling
#endif