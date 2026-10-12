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
 * \file pipeline_template_compute_trans_comm.h
 * \brief
 */

#ifndef MC2_PIPELINE_TEMPLATE_COMPUTE_TRANS_COMM_H
#define MC2_PIPELINE_TEMPLATE_COMPUTE_TRANS_COMM_H

#include "pipeline_context.h"

// 流水线模板
namespace MC2KernelTemplate {

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType>
class MC2KernelPipelineTemplate {
public:
    __aicore__ inline MC2KernelPipelineTemplate(ComputationType* computeStage, TransposeType* transStage, CommunicationType* commStage) : computeStage_(computeStage), transStage_(transStage), commStage_(commStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType* context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    ComputationType* computeStage_; // 矩阵乘的计算节点
    TransposeType* transStage_; // 转置计算的计算节点
    CommunicationType* commStage_; // 通信节点
};

// 初始化各节点
template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType>
__aicore__ inline void MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, ContextType>::Init()
{
    computeStage_->Init();
    commStage_->Init();
}

// 变更各节点在流水线中的运行规格，主要是输入输出和每轮流水的偏移
template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType>
__aicore__ inline void MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, ContextType>::GetContext(ContextType* context)
{
    context->computationContext = computeStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->communicationContext = commStage_->GetContextPtr();
}

//执行流水线
template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType>
__aicore__ inline void MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, ContextType>::Process(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    uint32_t index;
    for (index = 0 ; index < taskCnt; index++) {
        computeStage_->Process(index);
        //后续流水需要使用计算节点的结果
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            transStage_->Process(index);
            //后续通信需要使用转置后的结果
            AscendC::SyncAll<true>();
            commStage_->Process(index);
        }
        AscendC::SyncAll<false>();
    }
}

//释放流水线各节点资源
template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType>
__aicore__ inline void MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, ContextType>::End()
{
    computeStage_->End();
    commStage_->End();
}

};

#endif