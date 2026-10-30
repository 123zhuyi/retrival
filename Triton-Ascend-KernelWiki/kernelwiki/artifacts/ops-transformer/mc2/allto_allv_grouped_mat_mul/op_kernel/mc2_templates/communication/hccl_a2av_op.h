/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

#ifndef MC2_HCCL_IMPL_H
#define MC2_HCCL_IMPL_H

#include "kernel_operator.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/hccl/hccl.h"
#include "../common/a2av_common_tiling.h"

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename hcclDataType, bool commBeforeComputeFlag> class HcclA2avOp {
public:
     __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset,
        const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer, GM_ADDR recvBuffer)
    {
        taskTilingInfo_ = taskTilingInfo;
        GM_ADDR hcclContextGm = GetHcclContext<HCCL_GROUP_ID_0>();
        hccl_.InitV2(hcclContextGm, hcclInitTiling);
        hccl_.SetCcTilingV2(hcclCcTilingOffset); 
        rankId_ = hccl_.GetRankId();	 
        rankDim_ = hccl_.GetRankDim();	 
        e_ = taskTilingInfo_->e;
        H1_ = taskTilingInfo_->H1;
        N1_ = taskTilingInfo_->N1;
        sendGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)sendBuffer);
        recvGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)recvBuffer);
    }

    __aicore__ inline void Launch(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        if constexpr (AscendC::IsSameType<hcclDataType, bfloat16_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_BFP16;
        } else if constexpr (AscendC::IsSameType<hcclDataType, hifloat8_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_HIF8;
        } else {
            hcclDataType_ = HCCL_DATA_TYPE_FP16;
        }
        if constexpr (commBeforeComputeFlag) {
            LaunchCommBeforeCompute(startExpertIdx, expertNum);
        } else {
            LaunchCommAfterCompute(startExpertIdx, expertNum);
        }
    }

    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(alltoAllvHandleId_[startExpertIdx]);
    }

    __aicore__ inline void WaitAll(uint32_t allNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        for (uint32_t i = 0U; i < allNum; i++) {
            hccl_.Wait(alltoAllvHandleId_[i]);
        }
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            return;
        }
        hccl_.Finalize();
    }

private:
    __aicore__ inline void LaunchCommBeforeCompute(uint32_t startExpertIdx, uint32_t expertNum)
    {
        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        const auto *recvCnt = &taskTilingInfo_->recvCnt[0];
        uint64_t axis = H1_;
        for (uint64_t i = 0UL; i < rankDim_; i++) {
            alltoAllvSendCnt[i] = 0UL;
            alltoAllvRecvCnt[i] = 0UL;
            for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                alltoAllvSendCnt[i] += static_cast<uint64_t>(sendCnt[expertIdx + i * e_]) * axis;
                alltoAllvRecvCnt[i] += static_cast<uint64_t>(recvCnt[expertIdx + i * e_]) * axis;
            }
        }
        alltoAllvSendOffset[0] = 0UL;
        for (uint32_t j = 0U; j < startExpertIdx; j++) {
            alltoAllvSendOffset[0] += static_cast<uint64_t>(sendCnt[j]) * H1_;
        }
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
            for (uint32_t j = 0U; j < e_; j++) {
                alltoAllvSendOffset[i] += static_cast<uint64_t>(sendCnt[startExpertIdx + (i - 1U) * e_ + j]) * H1_;
            }
        }

        for (uint32_t i = 0U; i < rankDim_; i++) {
            if ((startExpertIdx == 0U) && (i == 0U)) {
                alltoAllvRecvOffset[i] = 0UL;
                alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[0];
            } else {
                alltoAllvRecvOffset[i] = alltoAllvRecvOffsetLastSum;
                alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[i];
            }
        }
        alltoAllvHandleId_[startExpertIdx] =
        hccl_.AlltoAllV<true>((__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
            (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
    }

    __aicore__ inline void LaunchCommAfterCompute(uint32_t startExpertIdx, uint32_t expertNum)
    {
        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        const auto *recvCnt = &taskTilingInfo_->recvCnt[0];
        uint64_t axis = N1_;
        for (uint64_t i = 0UL; i < rankDim_; i++) {
            alltoAllvSendCnt[i] = 0UL;
            alltoAllvRecvCnt[i] = 0UL;
            for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                alltoAllvSendCnt[i] += static_cast<uint64_t>(sendCnt[expertIdx + i * e_]) * axis;
                alltoAllvRecvCnt[i] += static_cast<uint64_t>(recvCnt[expertIdx + i * e_]) * axis;
            }
        }
        uint64_t expertOffset = 0UL;
        for (uint64_t i = 0UL; i < startExpertIdx; i++) {
            for (uint64_t j = 0UL; j < rankDim_; j++) {
                expertOffset += static_cast<uint64_t>(sendCnt[i + j * e_]);
            }
        }
        alltoAllvSendOffset[0] = expertOffset * N1_;
        for (uint64_t i = 1UL; i < rankDim_; i++) {
            alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1];
            for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                alltoAllvSendOffset[i] += static_cast<uint64_t>(sendCnt[expertIdx + (i - 1) * e_]) * N1_;
            }
        }
        alltoAllvRecvOffset[0] = 0UL;
        for (uint64_t i = 0UL; i < startExpertIdx; i++) {
            alltoAllvRecvOffset[0] += static_cast<uint64_t>(recvCnt[i]) * N1_;
        }
        for (uint64_t i = 1UL; i < rankDim_; i++) {
            alltoAllvRecvOffset[i] = alltoAllvRecvOffset[i - 1];
            for (uint64_t j = 0UL; j < e_; j++) {
                alltoAllvRecvOffset[i] += static_cast<uint64_t>(recvCnt[startExpertIdx + (i - 1) * e_ + j]) * N1_;
            }
        }
        alltoAllvHandleId_[startExpertIdx] =
        hccl_.AlltoAllV<true>((__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
            (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;
#endif

    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;

    const TaskTilingInfo *taskTilingInfo_;

    uint32_t rankId_ = 0U;
    uint32_t rankDim_ = 0U;
    uint32_t e_ = 0U;
    uint64_t H1_ = 0UL;
    uint64_t N1_ = 0UL;

    GlobalTensor<hcclDataType> sendGlobalBuffer_;
    GlobalTensor<hcclDataType> recvGlobalBuffer_;

    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;

    uint64_t alltoAllvRecvOffsetLastSum = 0UL;
    uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};
};
};

#endif