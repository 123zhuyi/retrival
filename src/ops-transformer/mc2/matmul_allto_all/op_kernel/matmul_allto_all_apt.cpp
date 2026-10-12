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
 * \file matmul_allto_all.cpp
 * \brief
 */
#include <cstring>
#include <lib/matmul_intf.h>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "common.h"
#include "./arch35/template_head.h"
#include "./arch35/matmul_allto_all_arch35.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "./arch32/matmul_allto_all_tiling_data_910_93.h"
#include "./arch32/matmul_allto_all_tiling_key_910_93.h"
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#include "./arch35/matmul_allto_all_tiling_key.h"
#include "./arch35/kc_quant_matmul_allto_all_arch35.h"
#include "./arch35/mx_quant_matmul_allto_all_arch35.h"
#endif

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace MatmulAlltoAllImpl;

#ifndef MATMUL_ALLTO_ALL_A3_FP_IMPL
#define MATMUL_ALLTO_ALL_A3_FP_IMPL(tilingData, pipe)  \
    do {    \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_A3_FP(ComputationContextType);\
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_A3_FP(ComputationType); \
        ComputationType matmulImplName(&pipe); \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);    \
        TransposeType transposeImplName(&pipe);    \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_AICPU, MC2AlltoAllContext,\
            MatmulAlltoAllTilingDataA3, MC2AlltoAllPrimitives, 1, 0, CommunicationType); \
        CommunicationType commImplName(&tilingData);  \
        using SchedulerContextType = PipelineContext<ComputationContextType>;  \
        using SchedulerType = MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, SchedulerContextType>;   \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);    \
        MatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataA3> op(&SchedulerImpl); \
        op.Init(x1, x2, bias, y, workspaceGM, &tilingData, &pipe);  \
        op.Process();   \
    } while (0)
#endif

#ifndef MATMUL_ALLTO_ALL_APT_FP_IMPL
#define MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe)  \
    do {    \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP(ComputationContextType);\
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(ComputationType); \
        ComputationType matmulImplName(&pipe); \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);    \
        TransposeType transposeImplName(&pipe);    \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,\
            MatmulAlltoAllTilingData, MC2AlltoAllPrimitives, 1, 0, CommunicationType); \
        CommunicationType commImplName(&tilingData);  \
        using SchedulerContextType = PipelineContext<ComputationContextType>;  \
        using SchedulerType = MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, SchedulerContextType>;   \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);    \
        MatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingData> op(&SchedulerImpl); \
        op.Init(x1, x2, bias, y, workspaceGM, &tilingData, &pipe);  \
        op.Process();   \
    } while (0)
#endif

#ifndef MATMUL_ALLTO_ALL_A3_BODY
#define MATMUL_ALLTO_ALL_A3_BODY(x1, x2, bias, y, workspaceGM, tilingGM) \
    do { \
        TPipe pipe; \
        REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingDataA3); \
        GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingDataA3, tilingData, tilingGM); \
        \
        if constexpr (DTYPEBIAS == DTYPE_BIAS_SAME_WITH_X) { \
            using DtypeBias = DTYPE_X1; \
            MATMUL_ALLTO_ALL_A3_FP_IMPL(tilingData, pipe); \
        } else if constexpr (DTYPEBIAS == DTYPE_BIAS_FP32) { \
            using DtypeBias = float; \
            MATMUL_ALLTO_ALL_A3_FP_IMPL(tilingData, pipe); \
        } \
    } while (0)
#endif

#ifndef MATMUL_ALLTO_ALL_APT_BODY
#define MATMUL_ALLTO_ALL_APT_BODY(x1, x2, bias, y, x1Scale, x2Scale, x2Offset, workspaceGM, tilingGM) \
    do { \
        TPipe pipe; \
        /* 注册默认的tilingdata，需要保证有且只有一个默认tilingdata被注册 */ \
        REGISTER_TILING_DEFAULT(QuantMatmulAlltoAllTilingData); \
        GET_TILING_DATA_WITH_STRUCT(QuantMatmulAlltoAllTilingData, tilingData, tilingGM); \
        \
        if constexpr (QUANTMODE == KC_QUANT_MODE) { \
            DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_QUANT(ComputationContextType); \
            DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(ComputationType, DTYPE_X1, DTYPE_X2); \
            ComputationType matmulImplName(&pipe); \
            DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType); \
            TransposeType transposeImplName(&pipe); \
            DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,\
                QuantMatmulAlltoAllTilingData, MC2AlltoAllPrimitives, 1, 0, CommunicationType); \
            CommunicationType commImplName(&tilingData); \
            using SchedulerContextType = PipelineContext<ComputationContextType>; \
            using SchedulerType = MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, SchedulerContextType>; \
            SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName); \
            KcQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, QuantMatmulAlltoAllTilingData> op(&SchedulerImpl); \
            op.Init(x1, x2, bias, y, x1Scale, x2Scale, x2Offset, workspaceGM, &tilingData, &pipe); \
            op.Process(); \
        } \
        else if constexpr (QUANTMODE == MX_QUANT_MODE) { \
            DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_MX_QUANT(ComputationContextType); \
            DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_MX_QUANT(ComputationType); \
            ComputationType matmulImplName(&pipe); \
            DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType); \
            TransposeType transposeImplName(&pipe); \
            DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,\
                QuantMatmulAlltoAllTilingData, MC2AlltoAllPrimitives, 1, 0, CommunicationType); \
            CommunicationType commImplName(&tilingData); \
            using SchedulerContextType = PipelineContext<ComputationContextType>; \
            using SchedulerType = MC2KernelPipelineTemplate<ComputationType, TransposeType, CommunicationType, SchedulerContextType>; \
            SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName); \
            MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, QuantMatmulAlltoAllTilingData> op(&SchedulerImpl); \
            op.Init(x1, x2, bias, y, x1Scale, x2Scale, workspaceGM, &tilingData, &pipe); \
            op.Process(); \
        } \
    } while (0)
#endif

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS>
__global__ __aicore__ void matmul_allto_all(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                            GM_ADDR commScale, GM_ADDR x1Offset, GM_ADDR x2Offset, GM_ADDR y,
                                            GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    //kernel的使用类型，这里是cube和vic混用，cube是主核，cube:vec=1:2
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MATMUL_ALLTO_ALL_A3_BODY(x1, x2, bias, y, workspaceGM, tilingGM);
#endif

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    TPipe pipe;
    // 注册默认的tilingdata，需要保证有且只有一个默认tilingdata被注册
    REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);

    if constexpr (DTYPEBIAS == DTYPE_BIAS_SAME_WITH_X) {
        using DtypeBias = DTYPE_X1;
        MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe);
    } else if constexpr (DTYPEBIAS == DTYPE_BIAS_FP32) {
        using DtypeBias = float;
        MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe);
    }
#else
    MATMUL_ALLTO_ALL_APT_BODY(x1, x2, bias, y, x1Scale, x2Scale, x2Offset, workspaceGM, tilingGM);
#endif
#endif
}