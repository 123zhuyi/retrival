/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file a2av_common_tiling.h
 * \brief
 */

#ifndef A2AV_COMMON_H
#define A2AV_COMMON_H

#if __has_include("../../../3rd/grouped_matmul/op_kernel/grouped_matmul_tiling_data_apt.h")
#include "../../../3rd/grouped_matmul/op_kernel/grouped_matmul_tiling_data_apt.h"
#else
#include "../../../../3rd/grouped_matmul/op_kernel/grouped_matmul_tiling_data_apt.h"
#endif

namespace MC2KernelTemplate {
static constexpr uint32_t MAX_EP_RANK_SIZE = 8U;
static constexpr uint32_t MAX_EXPERT_PER_EP = 1U;
static constexpr uint32_t MAX_EXPERT_SIZE = 256U;

// 类型复用声明
using GMMQuantTilingData = Mc2GroupedMatmulTilingData::GMMQuantTilingData;
using GMMArray = Mc2GroupedMatmulTilingData::GMMArray;

struct HcclA2avTilingInfo {
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling a2avCcTiling;
};

struct TaskTilingInfo {
    // Tensor维度参数（对应aclnn接口的输入输出shape）
    uint64_t BSK;           // 参考各个算子aclnn接口
    uint64_t BS;            // mmXOptional的第一维: (BS, H)，batch*seq
    uint64_t H1;            // gmmX和gmmWeight的隐藏层维度H
    uint64_t H2;            // mmWeightOptional的隐藏层维度H
    uint64_t A;             // 参考各个算子aclnn接口
    uint64_t N1;            // gmmWeight/gmmY的输出维度N1
    uint64_t N2;            // mmWeightOptional/mmYOptional的输出维度N2
    
    // Expert并行参数
    uint64_t epWorldSize;   // expert parallel world size (EP并行域大小)
    uint64_t e;             // 单卡上的专家数量
    
    // 循环调度参数
    uint32_t mainLoopExpertNum;  // 主循环每次处理的expert数量
    uint32_t tailLoopExpertNum;  // 尾循环处理的expert数量
    uint32_t totalLoopCount;     // 总循环次数
    
    // 通信参数（对应sendCounts和recvCounts）
    int32_t sendCnt[MAX_EXPERT_SIZE];  // 每个expert的发送计数
    int32_t recvCnt[MAX_EXPERT_SIZE];  // 每个expert的接收计数
};

}
#endif // A2AV_COMMON_H
