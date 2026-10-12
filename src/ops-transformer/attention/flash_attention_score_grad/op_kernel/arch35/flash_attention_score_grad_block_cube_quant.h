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
 * \file flash_attention_score_grad_block_cube_fp8.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_FP8_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_FP8_H
 
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"

namespace FagBaseApi {
 
 
TEMPLATES_DEF
class FAGBlockCubeQuant {
public:
    constexpr static bool IS_FP8_INPUT =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value || IsSameType<INPUT_TYPE, hifloat8_t>::value;
    constexpr static uint32_t CUBE_BASEM = 128;
    constexpr static uint32_t CUBE_BASEN = 128;
    constexpr static uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType);
    constexpr static uint32_t K_SIZE = 256;
    constexpr static uint32_t BASEK = 128;
    constexpr static uint32_t L0_OFFSET = 128 * 128;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    constexpr static uint32_t L0C_SINGLE_BUFFER_SIZE = 64 * 1024;
    constexpr static uint32_t DUAL_DST_NUM = 2;
    constexpr static uint8_t L1_BUFFER_NUM = 4;

    static constexpr uint32_t EVENT_ID0 = 0;
    static constexpr uint32_t EVENT_ID1 = 1;
    static constexpr uint32_t EVENT_ID2 = 2;
    static constexpr uint32_t EVENT_ID3 = 3;
    static constexpr uint32_t EVENT_ID4 = 4;
    static constexpr uint32_t EVENT_ID5 = 5;
    static constexpr uint32_t EVENT_ID6 = 6;
    static constexpr uint32_t EVENT_ID7 = 7;

    uint32_t commonBufferId = 0;
    uint32_t l0BufferId = 0;
    uint32_t l0cBufferId = 0;

    // input global mmemory
    GlobalTensor<INPUT_TYPE> queryGm, keyGm, valueGm;
    GlobalTensor<INPUT_TYPE> dyGm;
 
    TPipe *pipe;
    FagTilingType tilingData;
    // l1 buffer manage
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> kL1Buf[4];
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> vL1Buf[4];
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> commonL1Buf[4];
 
    // l0ab buffer manage, double buffer
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicySingleBuffer<BufferType::L0A, SyncType::NO_SYNC> l0aBuf[2];
    BuffersPolicySingleBuffer<BufferType::L0B, SyncType::NO_SYNC> l0bBuf[2];

    // l0c buffer manage
    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicySingleBuffer<BufferType::L0C, SyncType::NO_SYNC> mm1L0CBuf;
    BuffersPolicySingleBuffer<BufferType::L0C, SyncType::NO_SYNC> mm2L0CBuf;
    BuffersPolicySingleBuffer<BufferType::L0C, SyncType::NO_SYNC> dqkvL0CBuf[2];

    __aicore__ inline FAGBlockCubeQuant(){};
    __aicore__ inline ~FAGBlockCubeQuant();
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                              BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy);
    __aicore__ inline void InitCubeBuffer(FagConstInfo &constInfo);
    __aicore__ inline void IterateMmDsP(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void IterateMmDsK(LocalTensor<CALC_TYPE> &outTensor,
                                        LocalTensor<INPUT_TYPE> dSL1Tensor0, LocalTensor<INPUT_TYPE> dSL1Tensor1,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void IterateMmDsQ(LocalTensor<CALC_TYPE> &outTensor,
                                        LocalTensor<INPUT_TYPE> dSL1Tensor0, LocalTensor<INPUT_TYPE> dSL1Tensor1,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void IterateMmPDy(LocalTensor<CALC_TYPE> &outTensor,
                                        LocalTensor<INPUT_TYPE> pL1Tensor0, LocalTensor<INPUT_TYPE> pL1Tensor1,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void CopyOutDkDvResult(LocalTensor<CALC_TYPE> &dvOutTensor, LocalTensor<CALC_TYPE> &dkOutTensor, FagConstInfo &constInfo);
    __aicore__ inline void CopyOutDqResult(LocalTensor<CALC_TYPE> &dqOutTensor, FagConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

private:
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_TYPE> &l1Tensor, GlobalTensor<INPUT_TYPE> &gmSrcTensor, 
                                                            uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInQueryToL1(LocalTensor<INPUT_TYPE> &l1Tensor, int64_t offset,
                                                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInKeyToL1(LocalTensor<INPUT_TYPE> &l1Tensor, int64_t offset,
                                                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInValueToL1(LocalTensor<INPUT_TYPE> &l1Tensor, int64_t offset,
                                                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInDYToL1(LocalTensor<INPUT_TYPE> &l1Tensor, int64_t offset,
                                                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
};
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline FAGBlockCubeQuant<TEMPLATE_ARGS>::~FAGBlockCubeQuant()
{
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                                                       BufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->l1BufferManagerPtr = l1BuffMgr;
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy)
{
    queryGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dy);
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::InitCubeBuffer(FagConstInfo &constInfo)
{
    // init l1 buffer
    for (uint8_t idx = 0; idx < L1_BUFFER_NUM; idx++)
    {
        commonL1Buf[idx].Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
    }
    for (uint8_t idx = 0; idx < L1_BUFFER_NUM; idx++)
    {
        vL1Buf[idx].Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
    }
    for (uint8_t idx = 0; idx < L1_BUFFER_NUM; idx++)
    {
        kL1Buf[idx].Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
    }

    // init l0a l0b buffer
    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf[0].Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0aBuf[1].Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf[0].Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf[1].Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);
 
    // init l0c buffer
    l0cBufferManager.Init(pipe, L0C_MAX_SIZE);
    mm1L0CBuf.Init(l0cBufferManager, L0C_SINGLE_BUFFER_SIZE);
    mm2L0CBuf.Init(l0cBufferManager, L0C_SINGLE_BUFFER_SIZE);
    dqkvL0CBuf[0].Init(l0cBufferManager, L0C_SINGLE_BUFFER_SIZE);
    dqkvL0CBuf[1].Init(l0cBufferManager, L0C_SINGLE_BUFFER_SIZE);
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);

    SetFlag<HardEvent::FIX_M>(EVENT_ID7);
    SetFlag<HardEvent::FIX_M>(EVENT_ID6);
    SetFlag<HardEvent::FIX_M>(EVENT_ID5);
    SetFlag<HardEvent::FIX_M>(EVENT_ID4);

    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);

    WaitFlag<HardEvent::FIX_M>(EVENT_ID7);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID6);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID5);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID4);

    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::CopyGmToL1(LocalTensor<INPUT_TYPE> &l1Tensor,
                                    GlobalTensor<INPUT_TYPE> &gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN;
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = AlignTo32(srcN);
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::CopyInQueryToL1(LocalTensor<INPUT_TYPE> &l1Tensor, int64_t offset,
                                                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    auto srcGm = queryGm[offset];
    CopyGmToL1(l1Tensor, srcGm, srcN, srcD, srcDstride);
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::CopyInKeyToL1(LocalTensor<INPUT_TYPE> &l1Tensor, int64_t offset,
                                                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    auto srcGm = keyGm[offset];
    CopyGmToL1(l1Tensor, srcGm, srcN, srcD, srcDstride);
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::CopyInValueToL1(LocalTensor<INPUT_TYPE> &l1Tensor, int64_t offset,
                                                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    auto srcGm = valueGm[offset];
    CopyGmToL1(l1Tensor, srcGm, srcN, srcD, srcDstride);
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::CopyInDYToL1(LocalTensor<INPUT_TYPE> &l1Tensor, int64_t offset,
                                                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    auto srcGm = dyGm[offset];
    CopyGmToL1(l1Tensor, srcGm, srcN, srcD, srcDstride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::CopyOutDkDvResult(LocalTensor<CALC_TYPE> &dvOutTensor, LocalTensor<CALC_TYPE> &dkOutTensor, FagConstInfo &constInfo)
{
    uint32_t l0cId = l0cBufferId;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.mSize = CUBE_BASEM;
    fixpipeParams.nSize = HEAD_DIM_ALIGN;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = AlignTo16(HEAD_DIM_ALIGN);
    fixpipeParams.dualDstCtl = DUAL_DST_NUM;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    Fixpipe<CALC_TYPE, CALC_TYPE, DK_FIXPIPE_CONFIG>(dvOutTensor, dqkvL0CBuf[l0cId].Get().GetTensor<CALC_TYPE>(), fixpipeParams);
    Fixpipe<CALC_TYPE, CALC_TYPE, DK_FIXPIPE_CONFIG>(dkOutTensor, dqkvL0CBuf[(l0cId + 1) & 1].Get().GetTensor<CALC_TYPE>(), fixpipeParams);
    SetFlag<HardEvent::FIX_M>(l0cId + EVENT_ID4);
    SetFlag<HardEvent::FIX_M>(EVENT_ID4 + ((l0cId + 1) & 1));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::CopyOutDqResult(LocalTensor<CALC_TYPE> &dqOutTensor, FagConstInfo &constInfo)
{
    uint32_t l0cId = l0cBufferId;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.mSize = CUBE_BASEM;
    fixpipeParams.nSize = HEAD_DIM_ALIGN;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = AlignTo16(HEAD_DIM_ALIGN);
    fixpipeParams.dualDstCtl = DUAL_DST_NUM;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    Fixpipe<CALC_TYPE, CALC_TYPE, DQ_FIXPIPE_CONFIG>(dqOutTensor, dqkvL0CBuf[l0cId].Get().GetTensor<CALC_TYPE>(), fixpipeParams);
    SetFlag<HardEvent::FIX_M>(EVENT_ID4 + l0cId);
    l0cBufferId = (l0cBufferId + 1) & 1;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::IterateMmDsP(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                                 FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    uint32_t realM = runInfo.quantRunInfo.innerS2RealSize[runInfo.quantRunInfo.s2Idx];
    uint32_t realN = runInfo.quantRunInfo.innerS1RealSize[runInfo.quantRunInfo.s1Idx];
    uint32_t queryId = commonBufferId;
    uint32_t dyId = commonBufferId + 1;
    uint32_t l0Id = l0BufferId;

    int64_t dYOffset = runInfo.dyOffset + runInfo.quantRunInfo.s1Idx * constInfo.commonConstInfo.n2GDv * CUBE_BASEM;
    int64_t vOffset = runInfo.commonRunInfo.valueOffset + runInfo.quantRunInfo.s2Idx * constInfo.commonConstInfo.n2Dv * CUBE_BASEN;
    int64_t qOffset = runInfo.commonRunInfo.queryOffset + runInfo.quantRunInfo.s1Idx * constInfo.commonConstInfo.n2D * CUBE_BASEM;
    int64_t kOffset = runInfo.commonRunInfo.keyOffset + runInfo.quantRunInfo.s2Idx * constInfo.commonConstInfo.n2GD * CUBE_BASEN;
    
    LocalTensor<INPUT_TYPE> vL1Tensor = vL1Buf[runInfo.quantRunInfo.s2Idx].Get().GetTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> kL1Tensor = kL1Buf[runInfo.quantRunInfo.s2Idx].Get().GetTensor<INPUT_TYPE>();
    if (!runInfo.isValueReuse && runInfo.quantRunInfo.s1Idx == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(runInfo.quantRunInfo.s2Idx);
        CopyInKeyToL1(kL1Tensor, kOffset, realM, HEAD_DIM_ALIGN, constInfo.commonConstInfo.n2GD);
        CopyInValueToL1(vL1Tensor, vOffset, realM, HEAD_DIM_ALIGN, constInfo.commonConstInfo.n2Dv);
        SetFlag<HardEvent::MTE2_MTE1>(runInfo.quantRunInfo.s2Idx);
    }

    LocalTensor<INPUT_TYPE> qL1Tensor = commonL1Buf[queryId].Get().GetTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> dyL1Tensor = commonL1Buf[dyId].Get().GetTensor<INPUT_TYPE>();
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + queryId);
    CopyInQueryToL1(qL1Tensor, qOffset, realN, HEAD_DIM_ALIGN, constInfo.commonConstInfo.n2D);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + queryId);

    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + dyId);
    CopyInDYToL1(dyL1Tensor, dYOffset, realN, HEAD_DIM_ALIGN, constInfo.commonConstInfo.n2GDv);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + dyId);

    MMParam param = {
        realM,                                       // singleM
        realN,                                       // singleN
        HEAD_DIM_ALIGN,                              // singleK
        false,                                       // isLeftTranspose
        true                                         // isRightTranspose
    };

    Buffer<BufferType::L0A, SyncType::NO_SYNC> l0aBuffer = l0aBuf[l0Id].Get();
    LocalTensor<INPUT_TYPE> L0ATensor = l0aBuffer.GetTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> L0ATensorSecond = L0ATensor[L0_OFFSET];

    WaitFlag<HardEvent::M_MTE1>(l0Id + EVENT_ID4);
    if (!runInfo.isValueReuse && runInfo.quantRunInfo.s1Idx == 0) {
        WaitFlag<HardEvent::MTE2_MTE1>(runInfo.quantRunInfo.s2Idx);
    }
    LoadDataToL0A<INPUT_TYPE>(L0ATensor, kL1Tensor, param, 0, HEAD_DIM_ALIGN, realM);
    LoadDataToL0A<INPUT_TYPE>(L0ATensorSecond, vL1Tensor, param, 0, HEAD_DIM_ALIGN, realM);
    if ((!runInfo.isKeyReuse || runInfo.isLastProcessBlock) && (runInfo.quantRunInfo.s1Idx == runInfo.quantRunInfo.innerS1LoopNum - 1) && (runInfo.quantRunInfo.s2Idx == NUM_THREE))
    {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
    }
    if ((!runInfo.isKeyReuse && runInfo.isFirstProcessBlock) && (runInfo.quantRunInfo.s1Idx == runInfo.quantRunInfo.innerS1LoopNum - 1) && (runInfo.quantRunInfo.s2Idx <= NUM_TWO))
    {
        SetFlag<HardEvent::MTE1_MTE2>(runInfo.quantRunInfo.s2Idx);
    }
    Buffer<BufferType::L0B, SyncType::NO_SYNC> l0bBuffer = l0bBuf[l0Id].Get();
    LocalTensor<INPUT_TYPE> L0BTensor = l0bBuffer.GetTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> L0BTensorSecond = L0BTensor[L0_OFFSET];

    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + queryId);
    LoadDataToL0B<INPUT_TYPE>(L0BTensor, qL1Tensor, param, 0, HEAD_DIM_ALIGN, realN);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + queryId);

    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + dyId);
    LoadDataToL0B<INPUT_TYPE>(L0BTensorSecond, dyL1Tensor, param, 0, HEAD_DIM_ALIGN, realN);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + dyId);

    SetFlag<HardEvent::MTE1_M>(l0Id);
    WaitFlag<HardEvent::MTE1_M>(l0Id);
    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;

    Buffer<BufferType::L0C, SyncType::NO_SYNC> mm1L0CBuffer = mm1L0CBuf.Get();
    WaitFlag<HardEvent::FIX_M>(EVENT_ID7);
    Mmad(mm1L0CBuffer.GetTensor<CALC_TYPE>(), L0ATensor, L0BTensor, mmadParams);
    SetFlag<HardEvent::M_FIX>(EVENT_ID7);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID7);

    // fixp2ub
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = AlignTo64(realN);
    fixpipeParams.mSize = realM;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = CUBE_BASEN;
    fixpipeParams.dualDstCtl = DUAL_DST_NUM;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm1ResTensor, mm1L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    SetFlag<HardEvent::FIX_M>(EVENT_ID7);

    Buffer<BufferType::L0C, SyncType::NO_SYNC> mm2L0CBuffer = mm2L0CBuf.Get();
    WaitFlag<HardEvent::FIX_M>(EVENT_ID6);
    Mmad(mm2L0CBuffer.GetTensor<CALC_TYPE>(), L0ATensorSecond, L0BTensorSecond, mmadParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4 + l0Id);
    SetFlag<HardEvent::M_FIX>(EVENT_ID6);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID6);
    Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm2ResTensor, mm2L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
    SetFlag<HardEvent::FIX_M>(EVENT_ID6);
    commonBufferId = (commonBufferId + NUM_TWO) & NUM_THREE; 
    l0BufferId = (l0BufferId + 1) & 1;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::IterateMmDsK(LocalTensor<CALC_TYPE> &outTensor,
                                                LocalTensor<INPUT_TYPE> dSL1Tensor0, LocalTensor<INPUT_TYPE> dSL1Tensor1,
                                                FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    uint32_t l0Id = l0BufferId;
    uint32_t l0cId = l0cBufferId;
    bool isTailK = (runInfo.quantRunInfo.s2Idx * BASEK < runInfo.commonRunInfo.s2RealSize) && ((runInfo.quantRunInfo.s2Idx + NUM_TWO) * BASEK > runInfo.commonRunInfo.s2RealSize);
    uint32_t realM = runInfo.quantRunInfo.innerS1RealSize[runInfo.quantRunInfo.s1Idx];
    uint32_t realK = isTailK ? (runInfo.commonRunInfo.s2RealSize % K_SIZE) : K_SIZE;
    uint32_t kSizeFirst = BASEK;
    uint32_t kSizeSecond = realK - BASEK;
    if (realK < BASEK) {
        kSizeFirst = realK;
        kSizeSecond = 0;
    }
    Buffer<BufferType::L1, SyncType::NO_SYNC> kL1Buffer;
    Buffer<BufferType::L1, SyncType::NO_SYNC> kL1BufferSecond;
    LocalTensor<INPUT_TYPE> kL1Tensor;
    LocalTensor<INPUT_TYPE> kL1TensorSecond;

    MMParam param = {
        CUBE_BASEM,                                 // singleM
        HEAD_DIM_ALIGN,                             // singleN
        realK,                                      // singleK
        true,                                       // isLeftTranspose
        false                                       // isRightTranspose
    };

    Buffer<BufferType::L0A, SyncType::NO_SYNC> l0aBuffer = l0aBuf[l0Id].Get();
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4 + l0Id);
    LocalTensor<INPUT_TYPE> L0ATensor = l0aBuffer.GetTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> L0ATensorSecond = L0ATensor[L0_OFFSET];
    LoadDataToL0A<INPUT_TYPE>(L0ATensor, dSL1Tensor0, param, 0, BASEK, CUBE_BASEM);
    LoadDataToL0A<INPUT_TYPE>(L0ATensorSecond, dSL1Tensor1, param, 0, BASEK, CUBE_BASEM);

    Buffer<BufferType::L0B, SyncType::NO_SYNC> l0bBuffer = l0bBuf[l0Id].Get();
    LocalTensor<INPUT_TYPE> L0BTensor = l0bBuffer.GetTensor<INPUT_TYPE>();

    if (runInfo.isKeyReuse) {
        kL1Tensor = kL1Buf[runInfo.quantRunInfo.s2Idx].Get().GetTensor<INPUT_TYPE>();
        kL1TensorSecond = kL1Buf[runInfo.quantRunInfo.s2Idx + 1].Get().GetTensor<INPUT_TYPE>();
        LocalTensor<INPUT_TYPE> L0BTensorSecond = L0BTensor[L0_OFFSET];
        LoadDataToL0B<INPUT_TYPE>(L0BTensor, kL1Tensor, param, 0, kSizeFirst, HEAD_DIM_ALIGN);
        LoadDataToL0B<INPUT_TYPE>(L0BTensorSecond, kL1TensorSecond, param, 0, kSizeSecond, HEAD_DIM_ALIGN);

        if (!runInfo.isNextKeyReuse || runInfo.isLastProcessBlock) {
            if ((runInfo.quantRunInfo.s1Idx == runInfo.quantRunInfo.innerS1LoopNum - 1) && runInfo.quantRunInfo.s2Idx == 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
            } else if ((runInfo.quantRunInfo.s1Idx == runInfo.quantRunInfo.innerS1LoopNum - 1) && runInfo.quantRunInfo.s2Idx == NUM_TWO){
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            }
        }
    } else {
        uint32_t kBufferId = commonBufferId;
        kL1Tensor = commonL1Buf[kBufferId].Get().GetTensor<INPUT_TYPE>();
        int64_t keyOffset = runInfo.commonRunInfo.keyOffset + runInfo.quantRunInfo.s2Idx * constInfo.commonConstInfo.n2D * CUBE_BASEN;
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + kBufferId);
        CopyInKeyToL1(kL1Tensor, keyOffset, realK, HEAD_DIM_ALIGN, constInfo.commonConstInfo.n2D);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + kBufferId);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + kBufferId);
        LoadDataToL0B<INPUT_TYPE>(L0BTensor, kL1Tensor, param, 0, realK, HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + kBufferId);
        commonBufferId = (commonBufferId + NUM_TWO) & NUM_THREE;
    }

    SetFlag<HardEvent::MTE1_M>(l0Id);
    WaitFlag<HardEvent::MTE1_M>(l0Id);
    Buffer<BufferType::L0C, SyncType::NO_SYNC> dqL0CBuffer = dqkvL0CBuf[l0cId].Get();
    if (!runInfo.quantRunInfo.isDqFixOut) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID4 + l0cId);
    }
    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = !runInfo.quantRunInfo.isDqFixOut;
    mmadParams.cmatrixSource = false;
    Mmad(dqL0CBuffer.GetTensor<CALC_TYPE>(), L0ATensor, L0BTensor, mmadParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4 + l0Id);
    SetFlag<HardEvent::M_FIX>(EVENT_ID4);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID4);

    l0BufferId = (l0BufferId + 1) & 1;
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::IterateMmDsQ(LocalTensor<CALC_TYPE> &outTensor,
                                                LocalTensor<INPUT_TYPE> dSL1Tensor0, LocalTensor<INPUT_TYPE> dSL1Tensor1,
                                                FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1, SyncType::NO_SYNC> qL1Buffer;
    Buffer<BufferType::L1, SyncType::NO_SYNC> qL1BufferSecond;
 
    bool isTailK = (runInfo.quantRunInfo.s1Idx * CUBE_BASEM < runInfo.commonRunInfo.s1RealSize) && ((runInfo.quantRunInfo.s1Idx + NUM_TWO) * CUBE_BASEM > runInfo.commonRunInfo.s1RealSize);
    uint32_t realM = runInfo.quantRunInfo.innerS2RealSize[runInfo.quantRunInfo.s2Idx];
    uint32_t realK = isTailK ? (runInfo.commonRunInfo.s1RealSize % K_SIZE) : K_SIZE;
    uint32_t qBufferId = commonBufferId;
    uint32_t l0Id = l0BufferId;
    uint32_t l0cId = (l0cBufferId + 1) & 1;
    LocalTensor<INPUT_TYPE> qL1Tensor = commonL1Buf[qBufferId].Get().GetTensor<INPUT_TYPE>();
    int64_t queryOffset = runInfo.commonRunInfo.queryOffset + runInfo.quantRunInfo.s1Idx * constInfo.commonConstInfo.n2GD * CUBE_BASEM;
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + qBufferId);
    CopyInQueryToL1(qL1Tensor, queryOffset, realK, HEAD_DIM_ALIGN, constInfo.commonConstInfo.n2GD);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + qBufferId);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + qBufferId);

    MMParam param = {
        CUBE_BASEM,                                 // singleM
        HEAD_DIM_ALIGN,                             // singleN
        realK,                                      // singleK
        false,                                      // isLeftTranspose
        false                                       // isRightTranspose
    };

    Buffer<BufferType::L0A, SyncType::NO_SYNC> l0aBuffer = l0aBuf[l0Id].Get();
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4 + l0Id);
    LocalTensor<INPUT_TYPE> L0ATensor = l0aBuffer.GetTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> L0ATensorSecond = L0ATensor[L0_OFFSET];
    LoadDataToL0A<INPUT_TYPE>(L0ATensor, dSL1Tensor0, param, 0, BASEK, CUBE_BASEM);
    LoadDataToL0A<INPUT_TYPE>(L0ATensorSecond, dSL1Tensor1, param, 0, BASEK, CUBE_BASEM);

    Buffer<BufferType::L0B, SyncType::NO_SYNC> l0bBuffer = l0bBuf[l0Id].Get();
    LocalTensor<INPUT_TYPE> L0BTensor = l0bBuffer.GetTensor<INPUT_TYPE>();
    LoadDataToL0B<INPUT_TYPE>(L0BTensor, qL1Tensor, param, 0, realK, HEAD_DIM_ALIGN);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + qBufferId);
    SetFlag<HardEvent::MTE1_M>(l0Id);
    WaitFlag<HardEvent::MTE1_M>(l0Id);

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = !runInfo.quantRunInfo.isDkFixOut;
    mmadParams.cmatrixSource = false;
    Buffer<BufferType::L0C, SyncType::NO_SYNC> dkL0CBuffer;
    dkL0CBuffer = dqkvL0CBuf[l0cId].Get();
    if (!runInfo.quantRunInfo.isDkFixOut) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID4 + l0cId);
    }
    Mmad(dkL0CBuffer.GetTensor<CALC_TYPE>(), L0ATensor, L0BTensor, mmadParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4 + l0Id);
    SetFlag<HardEvent::M_FIX>(EVENT_ID5);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID5);
    commonBufferId = (commonBufferId + NUM_TWO) & NUM_THREE;
    l0BufferId = (l0BufferId + 1) & 1;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeQuant<TEMPLATE_ARGS>::IterateMmPDy(LocalTensor<CALC_TYPE> &outTensor,
                                                LocalTensor<INPUT_TYPE> pL1Tensor0, LocalTensor<INPUT_TYPE> pL1Tensor1,
                                                FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1, SyncType::NO_SYNC> dyL1Buffer;
    Buffer<BufferType::L1, SyncType::NO_SYNC> dyL1BufferSecond;
 
    bool isTailK = (runInfo.quantRunInfo.s1Idx * CUBE_BASEM < runInfo.commonRunInfo.s1RealSize) && ((runInfo.quantRunInfo.s1Idx + NUM_TWO) * CUBE_BASEM > runInfo.commonRunInfo.s1RealSize);
    uint32_t realM = runInfo.quantRunInfo.innerS2RealSize[runInfo.quantRunInfo.s2Idx];
    uint32_t realK = isTailK ? (runInfo.commonRunInfo.s1RealSize % K_SIZE) : K_SIZE;
    uint32_t dyBufferId = commonBufferId;
    uint32_t l0Id = l0BufferId;
    uint32_t l0cId = l0cBufferId;
    LocalTensor<INPUT_TYPE> dyL1Tensor = commonL1Buf[dyBufferId].Get().GetTensor<INPUT_TYPE>();
    int64_t dyOffset = runInfo.dyOffset + runInfo.quantRunInfo.s1Idx * constInfo.commonConstInfo.n2GDv * CUBE_BASEM;
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + dyBufferId);
    CopyInDYToL1(dyL1Tensor, dyOffset, realK, HEAD_DIM_ALIGN, constInfo.commonConstInfo.n2GDv);   
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + dyBufferId);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + dyBufferId);
    MMParam param = {
        CUBE_BASEM,                                 // singleM
        HEAD_DIM_ALIGN,                             // singleN
        realK,                                      // singleK
        false,                                      // isLeftTranspose
        false                                       // isRightTranspose
    };

    Buffer<BufferType::L0A, SyncType::NO_SYNC> l0aBuffer = l0aBuf[l0Id].Get();
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4 + l0Id);
    LocalTensor<INPUT_TYPE> L0ATensor = l0aBuffer.GetTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> L0ATensorSecond = L0ATensor[L0_OFFSET];
    LoadDataToL0A<INPUT_TYPE>(L0ATensor, pL1Tensor0, param, 0, BASEK, CUBE_BASEM);
    LoadDataToL0A<INPUT_TYPE>(L0ATensorSecond, pL1Tensor1, param, 0, BASEK, CUBE_BASEM);

    Buffer<BufferType::L0B, SyncType::NO_SYNC> l0bBuffer = l0bBuf[l0Id].Get();
    LocalTensor<INPUT_TYPE> L0BTensor = l0bBuffer.GetTensor<INPUT_TYPE>();
    LoadDataToL0B<INPUT_TYPE>(L0BTensor, dyL1Tensor, param, 0, realK, HEAD_DIM_ALIGN);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + dyBufferId);
    SetFlag<HardEvent::MTE1_M>(l0Id);
    WaitFlag<HardEvent::MTE1_M>(l0Id);

    Buffer<BufferType::L0C, SyncType::NO_SYNC> dvL0CBuffer = dqkvL0CBuf[l0cId].Get();
    if (!runInfo.quantRunInfo.isDvFixOut) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID4 + l0cId);
    }
    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = !runInfo.quantRunInfo.isDvFixOut;
    mmadParams.cmatrixSource = false;
    Mmad(dvL0CBuffer.GetTensor<CALC_TYPE>(), L0ATensor, L0BTensor, mmadParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4 + l0Id);
    SetFlag<HardEvent::M_FIX>(EVENT_ID4);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID4);
    commonBufferId = (commonBufferId + NUM_TWO) & NUM_THREE;
    l0BufferId = (l0BufferId + 1) & 1;
}

 
TEMPLATES_DEF
class FAGBlockCubeQuantDummy {
public:
    __aicore__ inline FAGBlockCubeQuantDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                              BufferManager<BufferType::L1> *l1BuffMgr){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy){};
    __aicore__ inline void InitCubeBuffer(FagConstInfo &constInfo){};
    __aicore__ inline void IterateMmDsP(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void IterateMmDsK(LocalTensor<CALC_TYPE> &outTensor,
                                        LocalTensor<INPUT_TYPE> dSL1Tensor0, LocalTensor<INPUT_TYPE> dSL1Tensor1,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo){}; // mm3 dq
    __aicore__ inline void IterateMmDsQ(LocalTensor<CALC_TYPE> &outTensor,
                                        LocalTensor<INPUT_TYPE> dSL1Tensor0, LocalTensor<INPUT_TYPE> dSL1Tensor1,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo){}; // mm4 dk
    __aicore__ inline void IterateMmPDy(LocalTensor<CALC_TYPE> &outTensor,
                                        LocalTensor<INPUT_TYPE> pL1Tensor0, LocalTensor<INPUT_TYPE> pL1Tensor1,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo){}; // mm5 dv
    __aicore__ inline void CopyOutDkDvResult(LocalTensor<CALC_TYPE> &dvOutTensor, LocalTensor<CALC_TYPE> &dkOutTensor, FagConstInfo &constInfo){};
    __aicore__ inline void CopyOutDqResult(LocalTensor<CALC_TYPE> &dqOutTensor, FagConstInfo &constInfo){};
    __aicore__ inline void AllocEventID(){};
    __aicore__ inline void FreeEventID(){};
};
 
template <typename T>
struct CubeBlockTraits; // 声明
 
/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;
 
#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS)                                                                     \
    TEMPLATES_DEF_NO_DEFAULT                                                                                           \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> {                                                          \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE)                                                                  \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST)                                                                \
    };
 
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeQuant);
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeQuantDummy);
 
// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS                                                                                                    \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE)                                                                       \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
 
 
} // namespace FagBaseApi
#endif
