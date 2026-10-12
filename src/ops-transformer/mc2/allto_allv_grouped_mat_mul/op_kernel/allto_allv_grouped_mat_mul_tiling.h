/* *
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file allto_allv_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__
#define __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "mc2_templates/common/a2av_common_tiling.h"

constexpr uint32_t MAX_EXPERT_SIZE = 384U;

struct AlltoAllvGmmCommonTilingInfo {
    uint64_t BSK;
    uint64_t BS;
    uint64_t K;
    uint64_t H1;
    uint64_t H2;
    uint64_t A;
    uint64_t N1;
    uint64_t N2;
    uint64_t epWorldSize;
    uint64_t stepSize;
    uint64_t E_ep; // 单卡专家数量
    uint64_t commOut;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    uint64_t totalUbSize;
    bool isGmmWeightTrans;
    bool isMmWeightTrans;
    bool isSendCntsTensor;
    bool isRecvCntsTensor;
    bool isPermuteOut;
    bool isNeedMM;
    bool isFp16;
};

struct AlltoAllvGmmAicpuTiling {
    int64_t sendCnt[MAX_EXPERT_SIZE];
    int64_t recvCnt[MAX_EXPERT_SIZE];
};

class AlltoAllvGmmTilingData {
public:
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling allGatherCcTiling;
    Mc2CcTiling alltoAllvCcTiling;
    AlltoAllvGmmCommonTilingInfo commonTilingInfo;
    TCubeTiling gmmTilingData;
    TCubeTiling mmTilingData;
    AlltoAllvGmmAicpuTiling aicpuTiling;
};

#pragma pack(push, 8)
struct QuantAlltoAllvGroupedMatmulTilingData {
    MC2KernelTemplate::HcclA2avTilingInfo hcclA2avTilingInfo;
    MC2KernelTemplate::TaskTilingInfo taskTilingInfo;
    bool isPermuteOut = false;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData gmmQuantTilingData;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData mmQuantTilingData;
};
#pragma pack(pop)

#endif // __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__