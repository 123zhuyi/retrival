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
 * \file allto_all_mx_quant_matmul_pipeline.h
 * \brief
 */

#ifndef ALLTO_ALL_MX_QUANT_MATMUL_PIPELINE
#define ALLTO_ALL_MX_QUANT_MATMUL_PIPELINE

#include "../../matmul_allto_all/mc2_templates/scheduler/template/pipeline_context.h"

// 流水线模板
namespace MC2KernelTemplate {
// 通信转置计算模板
template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, 
          typename ComputationType, typename ContextType>
class AlltoAllMxQuantMatmulPipeLine {
public:
    __aicore__ inline AlltoAllMxQuantMatmulPipeLine(CommunicationType* commStage, 
                                                    TransposeType* transStage,
                                                    ScaleTransposeType* scaleTransStage, 
                                                    ComputationType* computeStage) 
        : commStage_(commStage), transStage_(transStage), scaleTransStage_(scaleTransStage),
          computeStage_(computeStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType* context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void ProcessScale();

    __aicore__ inline void End();

private:
    CommunicationType* commStage_; // 通信节点
    TransposeType* transStage_; // 转置计算的计算节点
    ScaleTransposeType* scaleTransStage_; // 缩放转置的计算节点
    ComputationType* computeStage_; // 矩阵乘的计算节点
};

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, 
          typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMxQuantMatmulPipeLine<CommunicationType, TransposeType,
                                                     ScaleTransposeType, ComputationType,
                                                     ContextType>::Init()
{   
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, 
          typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMxQuantMatmulPipeLine<CommunicationType, TransposeType,
                                                     ScaleTransposeType, ComputationType,
                                                     ContextType>::GetContext(ContextType* context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->scaleTransposeContext = scaleTransStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, 
          typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMxQuantMatmulPipeLine<CommunicationType, TransposeType,
                                                     ScaleTransposeType, ComputationType,
                                                     ContextType>::Process(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    uint32_t index;
    for (index = 0 ; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            transStage_->Process(index);
        }
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIC {
            computeStage_->Process(index);
        }
    }
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, 
          typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMxQuantMatmulPipeLine<CommunicationType, TransposeType,
                                                     ScaleTransposeType, ComputationType,
                                                     ContextType>::ProcessScale()
{
    if ASCEND_IS_AIV {
        commStage_->PrepareAll(1);
        commStage_->Process(0);
        AscendC::SyncAll<true>();
        scaleTransStage_->Process(0);
    }
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, 
          typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMxQuantMatmulPipeLine<CommunicationType, TransposeType,
                                                     ScaleTransposeType, ComputationType,
                                                     ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
};

#endif