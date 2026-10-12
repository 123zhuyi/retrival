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
 * \file group_norm_silu_quant_tiling_arch35.h
 * \brief GroupNormSiluQuant arch35(Ascend950) regbase tiling data + compile info。
 *        类型全部 Regbase 前缀, 与 A2(arch22/) 完全按名隔开; 仅 same-type(无 mix-type)。
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_QUANT_TILING_ARCH35_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_QUANT_TILING_ARCH35_H_

#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
// 注意: 数据结构名 GroupNormSiluQuantRegbaseTilingData 供 kernel GET_TILING_DATA_WITH_STRUCT 使用, 保持不变。
BEGIN_TILING_DATA_DEF(GroupNormSiluQuantRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, numGroups);
TILING_DATA_FIELD_DEF(int64_t, hwNum);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF(int64_t, shapeN);
TILING_DATA_FIELD_DEF(int64_t, shapeC);
TILING_DATA_FIELD_DEF(int64_t, shapeD);
TILING_DATA_FIELD_DEF(int64_t, shapeQuantScale);
// 多核切分
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, numLastCore);
// 主循环/分块
TILING_DATA_FIELD_DEF(int64_t, processSize);
TILING_DATA_FIELD_DEF(int64_t, loopNum);
TILING_DATA_FIELD_DEF(int64_t, loopTail);
TILING_DATA_FIELD_DEF(int64_t, innerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);
// tilingKey 与算法参数
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, activateSilu);
TILING_DATA_FIELD_DEF(int64_t, parallelN);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
// 二分累加参数
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddPower);
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddK);
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddLastNum);
// 归约/分组参数
TILING_DATA_FIELD_DEF(int64_t, powerOfTwoForReduce);
TILING_DATA_FIELD_DEF(int64_t, coresPerGroup);
TILING_DATA_FIELD_DEF(float, groupInvCnt);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupNormSiluQuant_1000, GroupNormSiluQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSiluQuant_1100, GroupNormSiluQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSiluQuant_1110, GroupNormSiluQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSiluQuant_1120, GroupNormSiluQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSiluQuant_1130, GroupNormSiluQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSiluQuant_1140, GroupNormSiluQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSiluQuant_1150, GroupNormSiluQuantRegbaseTilingData)

struct GroupNormSiluQuantRegbaseCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int32_t is310P = 0;
    int32_t isRegbase = 0;
    uint32_t vectorLength = 0;
    uint32_t blockSizePlatform = 0;
};

// 仅 same-type(x/gamma/beta 同 dtype),与 A2 一致,无 mix-type。
enum class GroupNormSiluQuantRegbaseTilingKey : int64_t {
    TILINGKEY_EMPTY_TENSOR = 1000,
    TILINGKEY_R_PARTIAL_LOAD = 1100,
    TILINGKEY_R_FULL_LOAD = 1110,
    TILINGKEY_R_PARTIAL_LOAD_GENERALIZED = 1120,
    TILINGKEY_R_FULL_LOAD_GENERALIZED = 1130,
    TILINGKEY_R_SPLIT_REDUCE = 1140, // 少组+大reduce:按shapeD通道把单组切到多核,两阶段跨核Welford
    TILINGKEY_MANY_TINY_GROUPS = 1150 // 多小组+tiny reduce:把"组"放到向量lane,一条VF批量算多组,摊薄per-group开销
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_QUANT_TILING_ARCH35_H_
