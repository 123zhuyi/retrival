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
 * \file prompt_flash_attention_entry_regbase.h
 * \brief
 */

#ifndef PROMPT_FLASH_ATTENTION_ENTRY_310_H_
#define PROMPT_FLASH_ATTENTION_ENTRY_310_H_
#include "kernel/prompt_flash_attention_normal_bns1_preload.h"
#include "matmul_modules/pfa_policy_data.h"
#include "prompt_flash_attention_zero_output.h"
#include "prompt_flash_attention_tiling_regbase.h"
#include "prompt_flash_attention_template_tiling_key_enum.h"
#if __has_include("../../../common/op_kernel/arch35/flash_attention_score_kernel_infer.h")
#include "../../../common/op_kernel/arch35/flash_attention_score_kernel_infer.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_kernel_infer_mla_fullquant.h"
#include "../../../common/op_kernel/arch35/flash_attention_kernel_noquant_mla.h"
#else
#include "../../common/arch35/flash_attention_score_kernel_infer.h"
#include "../../common/arch35/flash_attention_score_kernel_infer_mla_fullquant.h"
#include "../../common/arch35/flash_attention_kernel_noquant_mla.h"
#endif

using namespace regbaseutil;

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define REGBASE_COPY_TILING_DATA_ASCEND950_KVSAME_BASEAPI(tiling)                                                    \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling);                           \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn

#define INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(templateClass, ...)                                               \
    do {                                                                                                                \
        TPipe tPipe;                                                                                                    \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                               \
        REGBASE_COPY_TILING_DATA_ASCEND950_KVSAME_BASEAPI(tiling);                                                   \
        using CubeBlockType = FABlockCubeNoquantMla<__VA_ARGS__>;                                                   \
        using VecBlockType = BaseApi::FABlockVecInfer<__VA_ARGS__>;                                                     \
        templateClass<CubeBlockType, VecBlockType> op;                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths,                                               \
                actualSeqLengthsKV, blocktable, postQuantScale, postQuantOffset, queryRope, keyRope, softmaxLse, attentionOut,                           \
                user, tilingData, &tPipe);                                                                              \
        op.Process();                                                                                                   \
    } while(0)

#if defined(__DAV_C310_CUBE__) || (defined __DAV_310R6_CUBE__)
#define INVOKE_PFA_TILING_DATA_95(tiling)                                                                               \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingDataV2, bmm1TilingDataRect, bmm1TilingData, tiling);                 \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingDataV2, bmm2TilingDataRect, bmm2TilingData, tiling);                 \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingData;                                                         \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingData;                                                         \
    const PromptFlashAttentionTilingDataV2* __restrict tiling_data = nullptr;                                             \
    AscendC::Impl::Detail::PFAGlobalTscmArray tscmArray;                                                                   \
    AscendC::Impl::Detail::tscmGlobalPFA = &tscmArray;                                                                     \
    TSCM<QuePosition::VECIN, 1, 0x4> bmm2Scm[2];                                                                        \
    tPipe.InitBuffer(bmm2Scm[0], 1, 65536);                                                                             \
    tPipe.InitBuffer(bmm2Scm[1], 1, 65536);                                                                             \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[0], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[1], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[2], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[3], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[4], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[5], 1, L1BUFSIZE)

#define INVOKE_PFA_TILING_DATA_55(tiling)                                                                               \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);                                                                  \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingDataV2, bmm1TilingDataRect, bmm1TilingData, tiling);                 \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingDataV2, bmm2TilingDataRect, bmm2TilingData, tiling);                 \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingData;                                                         \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingData;                                                         \
    const PromptFlashAttentionTilingDataV2* __restrict tiling_data = nullptr;                                             \
    AscendC::Impl::Detail::PFAGlobalTscmArray tscmArray;                                                                   \
    AscendC::Impl::Detail::tscmGlobalPFA = &tscmArray;                                                                     \
    TSCM<QuePosition::VECIN, 1, 0x4> bmm2Scm[2];                                                                        \
    tPipe.InitBuffer(bmm2Scm[0], 1, 32768);                                                                             \
    tPipe.InitBuffer(bmm2Scm[1], 1, 32768);                                                                             \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[0], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[1], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[2], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[3], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[4], 1, L1BUFSIZE);                                     \
    tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobalPFA->localScm[5], 1, L1BUFSIZE)

#else
#define INVOKE_PFA_TILING_DATA_V2(tiling)                                                                             \
    PromptFlashAttentionTilingDataV2 tiling_data_in;                                                                    \
    GET_TILING_DATA_WITH_STRUCT(PFAFullQuantTilingData, tiling_data_in_new, tiling);                       \
    TilingDataCopy(tiling_data_in, tiling_data_in_new);                                                               \
    const PromptFlashAttentionTilingDataV2* __restrict tiling_data = &tiling_data_in;                                   \
    const TCubeTiling* __restrict bmm1tiling = &(tiling_data->bmm1TilingDataRect);                                    \
    const TCubeTiling* __restrict bmm2tiling = &(tiling_data->bmm2TilingDataRect)

#define INVOKE_PFA_TILING_DATA_95(tiling)           \
    INVOKE_PFA_TILING_DATA_V2(tiling)

#define INVOKE_PFA_TILING_DATA_55(tiling)           \
    INVOKE_PFA_TILING_DATA_V2(tiling)
#endif

#ifdef __DAV_C310_CUBE__ // CUBE 实现
#define PFA_REGBASE_COPY_TILING_DATA(tiling)                                                                     \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = nullptr

#define INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA(templateClass, vec1ResultSize, qkvSize, ...)                                     \
    do {                                                                                                     \
        templateClass<__VA_ARGS__> op;                                                                       \
        TPipe tPipe;                                                                                         \
        AscendC::Impl::Detail::GlobalTscmArray tscmArray;                                                    \
        AscendC::Impl::Detail::tscmGlobal = &tscmArray;                                                      \
        TSCM<QuePosition::VECIN, 1, 0x4> vec1ScmPing;                                                        \
        TSCM<QuePosition::VECIN, 1, 0x4> vec1ScmPong;                                                        \
        tPipe.InitBuffer(vec1ScmPing, 1, vec1ResultSize);                                                    \
        tPipe.InitBuffer(vec1ScmPong, 1, vec1ResultSize);                                                    \
        tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobal->localQue[0], 1, qkvSize);                        \
        tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobal->localQue[1], 1, qkvSize);                        \
        tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobal->localQue[2], 1, qkvSize);                        \
        tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobal->localQue[3], 1, qkvSize);                        \
        tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobal->localQue[4], 1, qkvSize);                        \
        tPipe.InitBuffer(AscendC::Impl::Detail::tscmGlobal->localQue[5], 1, qkvSize);                        \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, (TCubeTiling *)nullptr, op.bmm2, (TCubeTiling *)nullptr);           \
    } while (0)

#define INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(templateClass, vec1ResultSize, qkvSize, ...)                                 \
    do {                                                                                                                                \
        if (query == nullptr) {return;}                                                                                                 \
        TPipe tPipe;                                                                                                                    \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockCube<__VA_ARGS__>, BaseApi::FABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockVecDummy<__VA_ARGS__>, BaseApi::FABlockVecInfer<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op;                                                                                  \
        op.InitBaseAPI(query, key, value, pseShift, nullptr, nullptr, attenMask, nullptr, actualSeqLengths,                             \
            actualSeqLengthsKV, blocktable, queryPaddingSize, kvPaddingSize, dequantScaleQuery, key_antiquant_scale, value_antiquant_scale, nullptr, postQuantScale,                 \
            postQuantOffset, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, learnableSink, nullptr, nullptr, nullptr, softmaxLse, attentionOut, user, nullptr, &tPipe);            \
        op.Process();                                                                                                                   \
    } while (0)
#else // VECTOR 实现
#define PFA_REGBASE_COPY_TILING_DATA(tiling)                                                                                                \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling);                                           \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn

#define INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA(templateClass, vec1ResultSize, qkvSize, ...)\
    do {                                                                                                 \
        if (query == nullptr) {return;}                                                                  \
        PFA_REGBASE_COPY_TILING_DATA(tiling);                                                            \
        TPipe tPipe;                                                                                     \
        templateClass<__VA_ARGS__> op;                                                                   \
        tPipe.InitBuffer(op.scm[0], 1, vec1ResultSize);                                                \
        tPipe.InitBuffer(op.scm[1], 1, vec1ResultSize);                                                \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, (TCubeTiling *)nullptr, op.bmm2, (TCubeTiling *)nullptr);              \
        op.Init(query, key, value, pseShift, nullptr, nullptr, attenMask, nullptr, actualSeqLengths,                                    \
            actualSeqLengthsKV, blocktable, queryPaddingSize, kvPaddingSize, nullptr, nullptr, nullptr, postQuantScale, postQuantOffset, queryRope, keyRope,             \
            nullptr, nullptr, nullptr, softmaxLse, attentionOut, user, tilingData, &tPipe);                   \
        op.Process();                                                                                                                   \
    } while (0)

#define INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(templateClass, vec1ResultSize, qkvSize, ...)                                 \
    do {                                                                                                                                \
        if (query == nullptr) {return;}                                                                                                 \
        PFA_REGBASE_COPY_TILING_DATA(tiling);                                                                                           \
        TPipe tPipe;                                                                                                                    \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockCube<__VA_ARGS__>, BaseApi::FABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockVecDummy<__VA_ARGS__>, BaseApi::FABlockVecInfer<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op;                                                                                  \
        op.InitBaseAPI(query, key, value, pseShift, nullptr, nullptr, attenMask, nullptr, actualSeqLengths,                             \
            actualSeqLengthsKV, blocktable, queryPaddingSize, kvPaddingSize, dequantScaleQuery, key_antiquant_scale, value_antiquant_scale, nullptr, postQuantScale,                 \
            postQuantOffset, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, learnableSink, nullptr, nullptr, nullptr, softmaxLse, attentionOut, user, tilingData, &tPipe);        \
        op.Process();                                                                                                                   \
    } while (0)
#endif

#define INVOKE_PFA_GENERAL_OP_IMPL_V2(templateClass, ...)                                                                  \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        TPipe tPipe;                                                                                                    \
        INVOKE_PFA_TILING_DATA_95(tiling);                                                                              \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm1.mm, (TCubeTiling*)nullptr, op.mm2.mm, (TCubeTiling*)nullptr);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, queryRope, keyRope, attentionOut, softmaxLse, user, tiling_data, &tPipe);                                    \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_INT8_OP_IMPL_V2(templateClass, ...)                                                                  \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        TPipe tPipe;                                                                                                    \
        INVOKE_PFA_TILING_DATA_95(tiling);                                                                              \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm1.mm, (TCubeTiling*)nullptr, op.mm2.mm, (TCubeTiling*)nullptr);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, queryRope, keyRope, attentionOut, softmaxLse, user, tiling_data, &tPipe);                                    \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, postQuantScale, postQuantOffset);                                \
        op.Process();                                                                                                   \
    } while (0)
// kv is empty tensor, return zero output
#define INVOKE_PFA_ZERO_OP_IMPL_V2(T)                                                                   \
    TPipe tPipe;                                                                                        \
    PFA_REGBASE_COPY_TILING_DATA(tiling);                                                               \
    PromptFlashAttentionZeroOutPut<T> op;                                                               \
    op.Init(attentionOut, softmaxLse, tilingData);                                                      \
    op.Process();                                                                                       \
    return
#define INVOKE_PFA_DUMMY(templateClass, ...)                                                            \
    TPipe tPipe;                                                                                        \
    PFA_REGBASE_COPY_TILING_DATA(tiling);                                                               \
    PromptFlashAttentionDummy<half> op;                                                                 \
    op.Init(attentionOut, tilingData);                                                                  \
    op.Process();                                                                                       \
    return

constexpr uint32_t L1BUFSIZE = 65536; // D最大支持256, 65536: 128 * 256 * 2

template<uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope, 
  bool isPa, bool isFd, bool emptyTensor, uint8_t pFAMask, uint8_t pFAMatMulType, bool enableKVPrefix>
inline __aicore__ void prompt_flash_attention_FIAS_regbase(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
    __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
    __gm__ uint8_t* deq_scale2, __gm__ uint8_t* postQuantScale, __gm__ uint8_t* postQuantOffset,
    __gm__ uint8_t* antiquant_scale, __gm__ uint8_t* antiquant_offset, __gm__ uint8_t* blocktable,
    __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* key_antiquant_scale,
    __gm__ uint8_t* key_antiquant_offset, __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset,
    __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
    __gm__ uint8_t * queryRope, __gm__ uint8_t * keyRope, __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
#if (__CCE_AICORE__ == 310) && (!defined (__DAV_310R6__))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(PFAFullQuantTilingData);
    REGISTER_TILING_FOR_TILINGKEY("((TILING_KEY_VAR >> 22) & 0x1f) != 30", FlashAttentionScoreSimplifiedTilingData);
    REGISTER_TILING_FOR_TILINGKEY("((TILING_KEY_VAR >> 22) & 0x1f) == 30", PFAFullQuantTilingData);
    if constexpr (emptyTensor == true) {
        # if (ORIG_DTYPE_ATTENTION_OUT != DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT != DT_BF16)
            INVOKE_PFA_ZERO_OP_IMPL_V2(fp8_e4m3fn_t);
        #else
            INVOKE_PFA_ZERO_OP_IMPL_V2(half);
        #endif
        return;
    }
    // 非量化用新模板
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
        // 解析两个合并字段
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        // 计算参数，这个地方必须先用constexpr将表达式的值计算出来，否则INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI会报结构体的某些变量不存在
        // 原因：不使用constexpr，所有组合都会在编译阶段进入该函数，因此会出现hasRope字段为false的情况导致变量不存在
        if constexpr(dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(FAKernelNoquantMla, half, float, half, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true, true, isPa, false); //实际模板参数hasRope为false，但模板需要其为true，选择在kernel直接写入，tiling不做修改
            return;
        }
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, half,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, half,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }    
    #endif
    #if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
        // 解析两个合并字段
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        // 计算参数，这个地方必须先用constexpr将表达式的值计算出来，否则INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI会报结构体的某些变量不存在
        // 原因：不使用constexpr，所有组合都会在编译阶段进入该函数，因此会出现hasRope字段为false的情况导致变量不存在    
        if constexpr(dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(FAKernelNoquantMla, bfloat16_t, float, bfloat16_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true, true, isPa, false);
            return;
        }

        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }    
    #endif

    #if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
        // 解析两个合并字段
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        // 计算参数，这个地方必须先用constexpr将表达式的值计算出来，否则INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI会报结构体的某些变量不存在
        // 原因：不使用constexpr，所有组合都会在编译阶段进入该函数，因此会出现hasRope字段为false的情况导致变量不存在    
        if constexpr(dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(FAKernelNoquantMla, half, float, int8_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true, true, isPa, false);
            return;
            }
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, int8_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, int8_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }    
    #endif

    #if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
        // 解析两个合并字段
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        if constexpr(dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(FAKernelNoquantMla, half, float, hifloat8_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true, true, isPa, false);
            return;
        }
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, hifloat8_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, hifloat8_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }    
    #endif

    #if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
        // 解析两个合并字段
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        if constexpr(dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(FAKernelNoquantMla, half, float, fp8_e4m3fn_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true, true, isPa, false);
            return;
        }
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, fp8_e4m3fn_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, fp8_e4m3fn_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }    
    #endif

    #if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
        // 解析两个合并字段
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        if constexpr(dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(FAKernelNoquantMla, bfloat16_t, float, int8_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true, true, isPa, false);
            return;
        }
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, int8_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, int8_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }    
    #endif

    #if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
        // 解析两个合并字段
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        if constexpr(dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(FAKernelNoquantMla, bfloat16_t, float, hifloat8_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
                s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true, true, isPa, false);
            return;
        }
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, hifloat8_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, hifloat8_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }
    #endif

    #if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
        // 解析两个合并字段
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        if constexpr(dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(FAKernelNoquantMla, bfloat16_t, float, fp8_e4m3fn_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
                s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true, true, isPa, false);
            return;
        }
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, fp8_e4m3fn_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, fp8_e4m3fn_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }
    #endif

    #if (ORIG_DTYPE_QUERY == DT_INT8 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
        PARSE_PARAMS_FullQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType);
        constexpr PFAPse pFaPse = (pseMode == PSE_MODE_PSE_NONE_TYPE) ? PFAPse::DISABLE_PSE : PFAPse::ENABLE_PSE;
        INVOKE_PFA_INT8_OP_IMPL_V2(PromptFlashAttentionNormalBNS1Preload, PFAType<static_cast<PFALayout>(inputLayoutType), int8_t, uint8_t, half, int8_t,
                                    static_cast<PFAMask>(pFAMask), pFaPse, RunMode::HighPrecision, SplitCoreMode::SPLIT_NBS_CUBE,
                                    static_cast<uint32_t>(s1TemplateType), static_cast<uint32_t>(s2TemplateType), static_cast<uint32_t>(dTemplateType), static_cast<uint32_t>(dVTemplateType), static_cast<PFAMatMulType>(pFAMatMulType)>);
    #endif

    #if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, fp8_e4m3fn_t, float, half,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, fp8_e4m3fn_t, float, half,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }
    #endif

    #if (ORIG_DTYPE_QUERY == DT_HIFLOAT8 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType);
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, hifloat8_t, float, half,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, hifloat8_t, float, half,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd);
        }
    #endif

    #if (ORIG_DTYPE_QUERY == DT_HIFLOAT8 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType);
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, hifloat8_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, hifloat8_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd);
        }
    #endif

    #if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, pFAMatMulType, enableKVPrefix);
        constexpr uint64_t vec1ResultSize = static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr(dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, fp8_e4m3fn_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else if constexpr (quantMode == FULLQUANT_MODE_PER_TOKEN_HEAD) { // mla fullquant
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * (static_cast<uint64_t>(dVTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dVTemplateType) >> 1)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInferMlaFullquant, vec1ResultSize, qkvSizeRsv2, fp8_e4m3fn_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        } else {
            constexpr uint64_t qkvSizeRsv2 = MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) * static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) * 2;
            INVOKE_PFA_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(BaseApi::FlashAttentionScoreKernelInfer, vec1ResultSize, qkvSizeRsv2, fp8_e4m3fn_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd, enableKVPrefix);
        }
    #endif
#endif
}
#endif // end of PROMPT_FLASH_ATTENTION_ENTRY_310_H_