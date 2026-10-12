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
 * \file group_norm_silu_quant_tiling_data.h
 * \brief GroupNormSiluQuant arch35(Ascend950) kernel 侧 plain tiling-data 结构。
 *        字段顺序/类型必须与 host op_host/arch35 的 BEGIN_TILING_DATA_DEF(GroupNormSiluQuantRegbaseTilingData)
 * 完全一致。
 */
#ifndef GROUP_NORM_SILU_QUANT_TILING_DATA_H_
#define GROUP_NORM_SILU_QUANT_TILING_DATA_H_

#include <cstdint>

struct GroupNormSiluQuantRegbaseTilingData {
    int64_t numGroups;
    int64_t hwNum;
    int64_t elemNum;
    int64_t shapeN;
    int64_t shapeC;
    int64_t shapeD;
    int64_t shapeQuantScale;
    int64_t realCoreNum;
    int64_t numPerCore;
    int64_t numLastCore;
    int64_t processSize;
    int64_t loopNum;
    int64_t loopTail;
    int64_t innerLoopNum;
    int64_t innerLoopTail;
    int64_t tilingKey;
    float epsilon;
    int64_t activateSilu;
    int64_t parallelN;
    int64_t ubSize;
    int64_t dichotomyAddPower;
    int64_t dichotomyAddK;
    int64_t dichotomyAddLastNum;
    int64_t powerOfTwoForReduce;
    int64_t coresPerGroup; // split-reduce(1140): 每个 group 分到的核数(按 shapeD 通道切); 其它 tilingKey 恒为 1
    float groupInvCnt; // split-reduce(1140): 1.0/(shapeD*hwNum), 主机预算避免设备标量浮点除法
};

#endif // GROUP_NORM_SILU_QUANT_TILING_DATA_H_
