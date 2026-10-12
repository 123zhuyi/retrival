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
 * \file hccl_impl.h 
 * \brief
 */

#ifndef MC2_HCCL_IMPL_H
#define MC2_HCCL_IMPL_H

#include "../primitives/hccl_primitives.h"

namespace MC2KernelTemplate {
using namespace AscendC;

/**
 * AICSync:通信控制核，true代表cube核，false代表vector核
 * ServerType:通信控制方式，ccu/mte/aicpu等
 * ContextType:上下文数据类型
 * TilingDataType:tilingdata具体数据类型
 * Primitive:通信原语的通信实现
 * SendCnt:每轮发送的次数
 * RecvCnt:每轮等待发送完毕的次数
 */
template <bool AICSync, HcclServerType ServerType, typename ContextType, typename TilingDataType,
    template<typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
class HcclCommunication
{
public:
    __aicore__ inline HcclCommunication(TilingDataType* tiling) : tiling_(tiling){};
    __aicore__ inline void Init();
    __aicore__ inline void PrepareAll(uint32_t taskCnt);
    __aicore__ inline ContextType* GetContextPtr();
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void End();

private:
    enum Communicationtype{
        COMMUNICATION_WAIT_ONE,
        COMMUNICATION_SEND_ONE
    };
    TilingDataType* tiling_;
    Hccl<ServerType> hccl_;
    ContextType context_;
    Primitive<Hccl<ServerType>> primitive_;
    uint32_t startIndex_ = 0;
    uint32_t endIndex_ = 0;
    bool notifyFlag_ = false;
    Communicationtype communicationType_ = COMMUNICATION_WAIT_ONE;
    static constexpr uint8_t MAX_HCCL_HANDLE_ = 63; //hccl只支持最多63个任务并行
    AscendC::HcclHandle hTasks_[MAX_HCCL_HANDLE_];
    bool taskSuccess_[MAX_HCCL_HANDLE_];
};

template <bool AICSync, HcclServerType ServerType, typename ContextType, typename TilingDataType,
    template<typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void HcclCommunication<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::Init()
{
    notifyFlag_ = false;   
    if ASCEND_IS_AIV {
        if (!AICSync && AscendC::GetBlockIdx() == 0) {
            notifyFlag_ = true; 
        }
    } else if ASCEND_IS_AIC {
        if (AICSync && AscendC::GetBlockIdx() == 0) {
            notifyFlag_ = true; 
        }
    }

    hccl_.InitV2(GetHcclContext<0>(), &(tiling_->mc2InitTiling));
    hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));
    if constexpr (SendCnt == 1U && RecvCnt == 0U) {
        communicationType_ = Communicationtype::COMMUNICATION_SEND_ONE;
    } else if constexpr (SendCnt == 0U && RecvCnt == 1U) {
        communicationType_ = Communicationtype::COMMUNICATION_WAIT_ONE;
    }

    for (uint8_t i = 0;i < MAX_HCCL_HANDLE_; ++i) {
        taskSuccess_[i] = true;
    }
}

template <bool AICSync, HcclServerType ServerType, typename ContextType, typename TilingDataType,
    template<typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void HcclCommunication<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::PrepareAll(uint32_t taskCnt)
{
    // 只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    for (uint32_t i = 0; i < taskCnt; i++) {
        hTasks_[endIndex_ + i] = primitive_.Prepare(&hccl_, &context_, i);
    }
    // 更新全局变量
    startIndex_ = endIndex_;
    endIndex_ += taskCnt;
    // 如果是先通后算就全量启动通信
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        for (uint32_t i = 0; i < taskCnt; i++) {
            hccl_.Commit(hTasks_[startIndex_ + i]);
            taskSuccess_[startIndex_ + i] = false;
        }
    }
}

template <bool AICSync, HcclServerType ServerType, typename ContextType, typename TilingDataType,
    template<typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline ContextType*
HcclCommunication<AICSync, ServerType, ContextType, TilingDataType,Primitive, SendCnt, RecvCnt>::GetContextPtr()
{
    return &context_;
}

template <bool AICSync, HcclServerType ServerType, typename ContextType, typename TilingDataType,
    template<typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void HcclCommunication<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::Process(uint32_t taskIndex)
{
    // 只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        hccl_.Wait(hTasks_[startIndex_ + taskIndex]);
        taskSuccess_[startIndex_ + taskIndex] = true;
    } else if (communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
        hccl_.Commit(hTasks_[startIndex_ + taskIndex]);
        taskSuccess_[startIndex_ + taskIndex] = false;
    }
}

template <bool AICSync, HcclServerType ServerType, typename ContextType, typename TilingDataType,
    template<typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void HcclCommunication<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::End()
{
    // 如果是先算后通就全量等待通信
    if (notifyFlag_ && communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
        for (uint32_t i = 0;i < endIndex_; ++i) {
            if (!taskSuccess_[i]) {
                hccl_.Wait(hTasks_[i]);
                taskSuccess_[i] = true;
            }
        }
    }

    if (notifyFlag_) {
        hccl_.Finalize();
    }
}
}; // namespace MC2KernelTemplate

#endif