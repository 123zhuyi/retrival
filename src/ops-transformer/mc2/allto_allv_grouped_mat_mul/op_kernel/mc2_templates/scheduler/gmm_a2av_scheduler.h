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
* \file gmm_a2av_scheduler.h
* \brief
*/

#ifndef MC2_GMMA2AV_PIPELINE_TEMPLATE_COMM_COMPUTE_H
#define MC2_GMMA2AV_PIPELINE_TEMPLATE_COMM_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "basic_api/kernel_basic_intf.h"

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename CommOpType, typename ComputationOpType, typename SharedComputationOpType, bool IsNeedMM>
class GmmA2avScheduler {
public:
    __aicore__ inline GmmA2avScheduler(const CommOpType& hcclOp, const ComputationOpType& computeOp,
        const SharedComputationOpType& shareComputeOp, TaskTilingInfo* taskTilingInfo):hcclOp_(hcclOp),
        computeOp_(computeOp), shareComputeOp_(shareComputeOp), taskTilingInfo_(taskTilingInfo){};
    __aicore__ inline void Init();

    __aicore__ inline void Process()
    {
        uint32_t expertNumInOneRank = taskTilingInfo_->e;

        for (uint32_t expertIdx = 0U; expertIdx < expertNumInOneRank; expertIdx++) {
            computeOp_.Process(expertIdx);
            SyncAll<false>();
            hcclOp_.Launch(expertIdx, 1);   // 每次专家数量设置为1
        }
        SyncAll<false>();

        if (IsNeedMM) {
            shareComputeOp_.Process(0);
            SyncAll<false>();
        }

        hcclOp_.WaitAll(expertNumInOneRank);
        End();
    }

    __aicore__ inline void End()
    {
        hcclOp_.End();
    }

private:
    CommOpType hcclOp_;
    ComputationOpType computeOp_;
    SharedComputationOpType shareComputeOp_;
    const TaskTilingInfo *taskTilingInfo_;
};
}
#endif