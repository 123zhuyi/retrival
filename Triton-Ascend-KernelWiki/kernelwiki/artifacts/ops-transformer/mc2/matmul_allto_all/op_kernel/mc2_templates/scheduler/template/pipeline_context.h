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
 * \file pipeline_context.h
 * \brief
 */

#ifndef MC2_PIPELINE_CONTEXT_H
#define MC2_PIPELINE_CONTEXT_H

namespace MC2KernelTemplate {
struct MC2TransposeContext;
struct MC2AlltoAllContext;
struct MC2PertokenDQuantContext;
// 后续可以按节点拆成对应的上下文复用
template <typename ComputationContextType>
struct PipelineContext {
    // computation
    ComputationContextType* computationContext;
    // transpose
    MC2TransposeContext* transposeContext;
    // scaleTranspose
    MC2TransposeContext* scaleTransposeContext;
    // communication
    MC2AlltoAllContext* communicationContext;
    // quantization
    MC2PertokenDQuantContext* quantizationContext;
};
}; // namespace MC2KernelTemplate

#endif