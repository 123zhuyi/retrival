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
 * \file sparse_lightning_indexer_grad_kl_loss_vector_block.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_BLOCK_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_BLOCK_H 
#include "sparse_lightning_indexer_grad_kl_loss_regbase_common.h"
#include "vf/vf_process_vec0.h"
#include "vf/vf_process_vec1.h"
#include "vf/vf_process_vec2.h"
#include "vf/vf_process_vec4.h"
#include "vf/vf_process_vec5.h"
#include "vf/vf_process_vec6.h"
#include "vf/vf_cast_dup.h"

namespace SligKlLoss {

TEMPLATES_DEF
class SligKlLossBlockVec {
public:
    __aicore__ inline SligKlLossBlockVec(){};
    __aicore__ inline void InitParams(const SLIGradKLLossConstInfo &vecConstInfo, const optiling::SparseLightningIndexerGradKLLossRegBaseTilingData *tiling);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR keyIndex, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
        GM_ADDR keyRope, GM_ADDR dKeyIndex, GM_ADDR dWeight, GM_ADDR loss, GlobalTensor<INPUT_T> &gatherSYRes, GlobalTensor<T> &relu,
        GlobalTensor<T> &reduceSumRes, GlobalTensor<T> &scatterAddRes);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void ProcessVector0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline void CopyInWeight(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void CopyInMaxSum(SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void ProcessVector1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline void ProcessVector2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, SLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void ProcessVector3(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline void ProcessVector4(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline void ProcessVector5(SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline void ProcessVector6(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline void ProcessVector7(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3ResBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline void ReInitBuffers(TPipe *pipe);
    __aicore__ inline void ProcessVector8();

    // gm addr
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> keyIndexGm;
    GlobalTensor<INPUT_T> keyRopeGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<INPUT_T> weightGm;
    GlobalTensor<int32_t> topKGm;
    GlobalTensor<T> softmaxMaxGm; 
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<T> lossGm;
    GlobalTensor<OUT_T> dWeightGm;
    GlobalTensor<OUT_T> dKeyIndexGm;

    // workspace
    GlobalTensor<INPUT_T> gatherSYResGm;
    GlobalTensor<T> reduceSumResGm;
    GlobalTensor<T> reluGm;
    GlobalTensor<T> scatterAddResGm;

    const optiling::SparseLightningIndexerGradKLLossRegBaseTilingData *tilingData;
    TPipe *pipe;

private:
    static constexpr int64_t UB_ROW_SIZE = 8;
    static constexpr event_t EVENT_ID0 = (event_t)4;
    static constexpr event_t EVENT_ID2 = (event_t)6;

    template <bool gatherRope>
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
        int64_t realS2Idx2, const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo,
        GlobalTensor<INPUT_T> srcTensor, GlobalTensor<INPUT_T> srcRopeTensor);
    template <bool gatherRope>
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
        int64_t keyBNBOffset,int64_t s2IdLimit, const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo,
        GlobalTensor<INPUT_T> srcTensor, GlobalTensor<INPUT_T> srcRopeTensor);
    __aicore__ inline int64_t GetKeyRopeGmOffset(int64_t realS2Idx, const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit, SLIGradKLLossKRunInfo &kRunInfo);
    template <event_t IdStart, bool gatherRope, bool syGmEn>
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset, int64_t mergeMte3Idx, const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo);
    template <event_t IdStart, bool gatherRope, bool syGmEn>
    __aicore__ inline void MergeKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo,
        GlobalTensor<INPUT_T> srcTensor, GlobalTensor<INPUT_T> srcRopeTensor);
    SLIGradKLLossConstInfo constInfo;
    TBuf<> gatherTbuf;
    TQue<QuePosition::VECOUT, 1> reduceSumOutQue;
    TQue<QuePosition::VECIN, 1> weightInQue;
    TQue<QuePosition::VECOUT, 1> gatherKeyQue[2];
    TQue<QuePosition::VECOUT, 1> gatherKeyIndexQue[2];
    TQue<QuePosition::VECIN, 1> maxInQue;
    TQue<QuePosition::VECIN, 1> sumInQue;
    TQue<QuePosition::VECOUT, 1> mulsResQue;
    TQue<QuePosition::VECIN, 1> mulsInQue;
    TQue<QuePosition::VECOUT, 1> lossSumQue;
    TQue<QuePosition::VECIN, 1> reluQue;
    TQue<QuePosition::VECOUT, 1> reluGradQue;
    TQue<QuePosition::VECOUT, 1> dwQue;
    TQue<QuePosition::VECOUT, 1> dwOutQue;
    TQue<QuePosition::VECOUT, 1> gatherOutQue;
    TQue<QuePosition::VECIN, 1> scatterAddInQue[2];
    TQue<QuePosition::VECOUT, 1> dKeyOutQue[2];

    LocalTensor<INPUT_T> gatherKeyUb;
    LocalTensor<INPUT_T> gatherKeyIndexUb;

    LocalTensor<INPUT_T> kvMergUb_;
    LocalTensor<INPUT_T> ropeMergUb_;
    LocalTensor<INPUT_T> mm1AL1Tensor;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::InitParams(
    const SLIGradKLLossConstInfo &vecConstInfo, const optiling::SparseLightningIndexerGradKLLossRegBaseTilingData *tiling)
{
    this->constInfo = vecConstInfo;
    tilingData = tiling;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR key, GM_ADDR keyIndex, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
    GM_ADDR keyRope, GM_ADDR dKeyIndex, GM_ADDR dWeight, GM_ADDR loss, GlobalTensor<INPUT_T> &gatherSYRes,
    GlobalTensor<T> &relu, GlobalTensor<T> &reduceSumRes, GlobalTensor<T> &scatterAddRes)
{
    keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    keyIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)keyIndex);
    weightGm.SetGlobalBuffer((__gm__ INPUT_T *)weight);
    topKGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    softmaxMaxGm.SetGlobalBuffer((__gm__ T *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ T *)softmaxSum);
    if constexpr (HAS_ROPE) {
        keyRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)keyRope);
    }

    dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
    dWeightGm.SetGlobalBuffer((__gm__ OUT_T *)dWeight);
    lossGm.SetGlobalBuffer((__gm__ T *)loss);
    if (constInfo.aivIdx == 0) {
        lossGm.SetValue(0, 0.0F);
        AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(lossGm);
    }

    this->gatherSYResGm = gatherSYRes;
    this->reluGm = relu;
    this->reduceSumResGm = reduceSumRes;
    this->scatterAddResGm= scatterAddRes;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(this->gatherTbuf, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_9K * 2);
    pipe->InitBuffer(this->gatherOutQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_9K * 2);
    pipe->InitBuffer(this->reduceSumOutQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->weightInQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_64);
    pipe->InitBuffer(this->maxInQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->sumInQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->mulsResQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(this->mulsInQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(this->reluQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->reluGradQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->dwQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_128);
    pipe->InitBuffer(this->dwOutQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_64);
    pipe->InitBuffer(this->lossSumQue, 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_32);
    gatherKeyUb = gatherTbuf.Get<INPUT_T>();
    gatherKeyIndexUb = gatherKeyUb;
    auto lossSumUb = this->lossSumQue.template AllocTensor<T>();
    auto dwUb = this->dwQue.template AllocTensor<T>();
    Duplicate(lossSumUb, 0.0f, 8);
    Duplicate(dwUb, 0.0f, 32);
    this->lossSumQue.template FreeTensor(lossSumUb);
    this->dwQue.template FreeTensor(dwUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[kRunInfo.kTaskIdMod2]);
    this->mm1AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    this->kvMergUb_ = gatherKeyIndexUb;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
    MergeKv<EVENT_ID0, false, true>(outputBuf, runInfo, kRunInfo, keyIndexGm, keyRopeGm);
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[kRunInfo.kTaskIdMod2]);

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
}

TEMPLATES_DEF_NO_DEFAULT
template <bool gatherRope>
__aicore__ inline void
SligKlLossBlockVec<TEMPLATE_ARGS>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
    int64_t keyBNBOffset, int64_t s2IdLimit, const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo,
    GlobalTensor<INPUT_T> srcTensor, GlobalTensor<INPUT_T> srcRopeTensor)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * kRunInfo.dValue * sizeof(INPUT_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<INPUT_T> padParams;
    DataCopyPad(kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue + (mte2Size - mte3Size) * kRunInfo.dValue],
                srcTensor[keyBNBOffset * kRunInfo.dValue], intriParams, padParams);
    
    if constexpr (gatherRope) {
        intriParams.blockLen = validS2Count * kRunInfo.dRopeValue * sizeof(INPUT_T);
        DataCopyPad(ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dRopeValue + (mte2Size - mte3Size) * kRunInfo.dRopeValue],
                    srcRopeTensor[keyBNBOffset * kRunInfo.dRopeValue], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SligKlLossBlockVec<TEMPLATE_ARGS>::GetKeyRopeGmOffset(int64_t realS2Idx,
    const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit, SLIGradKLLossKRunInfo &kRunInfo)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t tensorBRopeOffset = runInfo.accumS2Idx * kRunInfo.dRopeValue;
    int64_t realKeyRopeGmOffset = (tensorBRopeOffset + realS2Idx * kRunInfo.dRopeValue) / kRunInfo.dRopeValue;
    return realKeyRopeGmOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SligKlLossBlockVec<TEMPLATE_ARGS>::GetKeyGmOffset(int64_t realS2Idx,
    const SLIGradKLLossRunInfo &runInfo, int64_t s2IdLimit, SLIGradKLLossKRunInfo &kRunInfo)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t tensorBOffset = runInfo.accumS2Idx * kRunInfo.dValue;
    int64_t realKeyGmOffset = (tensorBOffset + realS2Idx * kRunInfo.dValue) / kRunInfo.dValue;
    return realKeyGmOffset;
}

TEMPLATES_DEF_NO_DEFAULT
template <bool gatherRope>
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
    int64_t realS2Idx1, int64_t realS2Idx2, const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo,
    GlobalTensor<INPUT_T> srcTensor, GlobalTensor<INPUT_T> srcRopeTensor)
{
    int64_t s2IdLimit = runInfo.s2SparseLen;
    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, kRunInfo);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, kRunInfo);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                    (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * kRunInfo.dValue * sizeof(INPUT_T);
    if constexpr (gatherRope) {
        int64_t keyRopeOffset1 = GetKeyRopeGmOffset(realS2Idx1, runInfo, s2IdLimit, kRunInfo);
        int64_t keyRopeOffset2 = GetKeyRopeGmOffset(realS2Idx2, runInfo, s2IdLimit, kRunInfo);
        keyRopeSrcStride = ((keyRopeOffset1 > keyRopeOffset2 ? (keyRopeOffset1 - keyRopeOffset2) :
                            (keyRopeOffset2 - keyRopeOffset1)) - constInfo.sparseBlockSize) *
                                kRunInfo.dRopeValue * sizeof(INPUT_T);
    }

    bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);
    bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0) ||
        (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0);
    bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);

    if (key1LessThankey2 || strideInvalid || copyOutOfRange) {
        // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
        CopyInSingleKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1,
            s2IdLimit, runInfo, kRunInfo, srcTensor, srcRopeTensor);
        CopyInSingleKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2,
            s2IdLimit, runInfo, kRunInfo, srcTensor, srcRopeTensor);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = constInfo.sparseBlockSize * kRunInfo.dValue * sizeof(INPUT_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<INPUT_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue + (mte2Size - mte3Size) * kRunInfo.dValue],
                    srcTensor[startGmOffset * kRunInfo.dValue], intriParams, padParams);

        if constexpr (gatherRope) {
            intriParams.blockLen = kRunInfo.dRopeValue * sizeof(INPUT_T);
            intriParams.srcStride = keyRopeSrcStride;
            DataCopyPad(ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dRopeValue + (mte2Size - mte3Size) * kRunInfo.dRopeValue],
                        srcRopeTensor[startGmOffset * kRunInfo.dRopeValue], intriParams, padParams);
        }

        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1));
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <event_t IdStart, bool gatherRope, bool syGmEn>
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
    int64_t s2GmStartOffset, int64_t mergeMte3Idx, const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<HardEvent::MTE2_MTE3>(mergeMte3Idx + IdStart);
    WaitFlag<HardEvent::MTE2_MTE3>(mergeMte3Idx + IdStart);
    uint32_t dstNzC0Stride = gatherRope ? VEC_P_BASESIZE : VEC_SY_BASESIZE;
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    LocalTensor<INPUT_T> kvMergUbCp_ = this->gatherOutQue.template AllocTensor<INPUT_T>();
    Ub2UbNd2Nz<INPUT_T>(kvMergUbCp_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue],
                        kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue], UB_ROW_SIZE, kRunInfo.dValue);
    this->gatherOutQue.template EnQue(kvMergUbCp_);
    this->gatherOutQue.template DeQue<INPUT_T>();
    uint16_t valid_row = (mte2Size % UB_ROW_SIZE == 0) ? UB_ROW_SIZE : (mte2Size % UB_ROW_SIZE);
    DataCopy(mm1AL1Tensor[(s2GmStartOffset + mte3Size) << 4], kvMergUbCp_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue],
             {static_cast<uint16_t>(kRunInfo.dValue >> 4), valid_row, static_cast<uint16_t>(UB_ROW_SIZE - valid_row),
              static_cast<uint16_t>(dstNzC0Stride - valid_row)});
    this->gatherOutQue.template FreeTensor(kvMergUbCp_);

    if constexpr (gatherRope) {
        eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        LocalTensor<INPUT_T> kvMergUbCp_ = this->gatherOutQue.template AllocTensor<INPUT_T>();
        Ub2UbNd2Nz<INPUT_T>(kvMergUbCp_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dRopeValue],
                            ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dRopeValue], UB_ROW_SIZE, kRunInfo.dRopeValue);
        this->gatherOutQue.template EnQue(kvMergUbCp_);
        this->gatherOutQue.template DeQue<INPUT_T>();
        valid_row = (mte2Size % UB_ROW_SIZE == 0) ? UB_ROW_SIZE : (mte2Size % UB_ROW_SIZE);
        DataCopy(mm1AL1Tensor[(s2GmStartOffset + mte3Size) * BLOCK_SINGLE_LEN + runInfo.s2BaseSize * kRunInfo.dValue],
                 kvMergUbCp_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dRopeValue],
                 {static_cast<uint16_t>(kRunInfo.dRopeValue / BLOCK_SINGLE_LEN), valid_row, static_cast<uint16_t>(UB_ROW_SIZE - valid_row),
                  static_cast<uint16_t>(dstNzC0Stride - valid_row)});
        this->gatherOutQue.template FreeTensor(kvMergUbCp_);
    }
    if constexpr (syGmEn) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mte2Size - mte3Size;
        dataCopyParams.blockLen = kRunInfo.dValue * sizeof(INPUT_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        int64_t gmStartOffset = kRunInfo.s2Idx * runInfo.s2BaseSize * (kRunInfo.dValue + kRunInfo.dRopeValue);
        DataCopyPad(gatherSYResGm[gmStartOffset + (s2GmStartOffset + mte3Size) * kRunInfo.dValue],
                kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue], dataCopyParams);
        if constexpr (gatherRope) {
            dataCopyParams.blockLen = kRunInfo.dRopeValue * sizeof(INPUT_T);
            DataCopyPad(gatherSYResGm[gmStartOffset + runInfo.s2BaseSize * kRunInfo.dValue + (s2GmStartOffset + mte3Size) *
                kRunInfo.dRopeValue], ropeMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dRopeValue], dataCopyParams);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
    const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    int64_t topkGmIdx = (s2GmOffset + kRunInfo.s2Idx * runInfo.s2BaseSize);
    realS2Idx = topKGm.GetValue(runInfo.topkGmBaseOffset + topkGmIdx);
}

TEMPLATES_DEF_NO_DEFAULT
template <event_t IdStart, bool gatherRope, bool syGmEn>
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::MergeKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    const SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo, GlobalTensor<INPUT_T> srcTensor,
    GlobalTensor<INPUT_T> srcRopeTensor)
{
    int64_t s2ProcessSize = kRunInfo.s2RealBaseSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L);
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2: s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2) {
        if (needWaitMte3ToMte2) {
            WaitFlag<HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, runInfo, kRunInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult<IdStart, gatherRope, syGmEn>(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo, kRunInfo);
            SetFlag<HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, runInfo, kRunInfo);
        CopyInKv<gatherRope>(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo, kRunInfo, srcTensor, srcRopeTensor);
        if ((mte2Size - mte3Size + 2 > UB_ROW_SIZE) ||
            s2GmOffsetArray + 2 >= s2GmLimit) {
            CopyOutMrgeResult<IdStart, gatherRope, syGmEn>(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo, kRunInfo);
            mte3Size = mte2Size;
            SetFlag<HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::CopyInWeight(SLIGradKLLossRunInfo &runInfo)
{
    LocalTensor<INPUT_T> weightInUb = weightInQue.template AllocTensor<INPUT_T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.nIndexSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<INPUT_T> padParams;
    DataCopyPad(weightInUb, weightGm[runInfo.weightOffset], dataCopyParams, padParams);
    this->weightInQue.template EnQue(weightInUb);
    this->weightInQue.template DeQue<INPUT_T>();
    this->weightInQue.template FreeTensor(weightInUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    if (kRunInfo.s2SingleIdx == 0) {
        CrossCoreWaitFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
    }
    LocalTensor<INPUT_T> weightInUb = weightInQue.template AllocTensor<INPUT_T>();
    auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();

    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    ProcessVec1Vf<T, INPUT_T>(reduceSumTensor[runInfo.s2CurSize], mmRes[kRunInfo.s2SingleCurSize], weightInUb, runInfo.nIndexSize, constInfo.syKBaseSize);
    if (kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1) {
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
    }
    if (kRunInfo.isS2end) {
        this->reduceSumOutQue.template EnQue(reduceSumTensor);
        this->reduceSumOutQue.template DeQue<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = runInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(reduceSumResGm[constInfo.subBlockIdx * constInfo.kSize], reduceSumTensor, dataCopyParams);
        CrossCoreSetFlag<1, PIPE_MTE3>(EVENT_ID2);
    }
    this->weightInQue.template FreeTensor(weightInUb);
    this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, SLIGradKLLossRunInfo &runInfo)
{
    auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();
    LocalTensor<T> reduceOtherSumTensor = bmm1ResBuf.template GetTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.s2RealSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    CrossCoreWaitFlag<1, PIPE_MTE2>(EVENT_ID2);
    DataCopyPad(reduceOtherSumTensor, reduceSumResGm[(1 - constInfo.subBlockIdx) * constInfo.kSize], dataCopyParams, padParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (runInfo.s2TailSize == VEC_SY_BASESIZE) {
        ProcessVec2Vf<T, true>(reduceSumTensor, reduceSumTensor, reduceOtherSumTensor, runInfo.s2RealSize);
    } else {
        ProcessVec2Vf<T, false>(reduceSumTensor, reduceSumTensor, reduceOtherSumTensor, runInfo.s2RealSize);
    }
    this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector3(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[kRunInfo.kTaskIdMod2]);
    mm1AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    this->kvMergUb_ = gatherKeyUb;
    this->ropeMergUb_ = gatherKeyUb[UB_ROW_SIZE * constInfo.dSizeQuery * 2];

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
    MergeKv<EVENT_ID0, HAS_ROPE, false>(outputBuf, runInfo, kRunInfo, keyGm, keyRopeGm);
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[kRunInfo.kTaskIdMod2]);

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::CopyInMaxSum(SLIGradKLLossRunInfo &runInfo)
{
    auto maxTensor = this->maxInQue.template AllocTensor<T>();
    auto sumTensor = this->sumInQue.template AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.nVecSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(maxTensor, softmaxMaxGm[runInfo.softmaxInputOffset], dataCopyParams, padParams);
    DataCopyPad(sumTensor, softmaxSumGm[runInfo.softmaxInputOffset], dataCopyParams, padParams);
    this->maxInQue.template EnQue(maxTensor);
    this->maxInQue.template DeQue<T>();
    this->sumInQue.template EnQue(sumTensor);
    this->sumInQue.template DeQue<T>();
    this->maxInQue.template FreeTensor(maxTensor);
    this->sumInQue.template FreeTensor(sumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector4(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
    SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    if (kRunInfo.s2SingleIdx == 0) {
        CrossCoreWaitFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
    }
    auto mulsResUb = this->mulsResQue.template AllocTensor<T>();
    auto maxTensor = this->maxInQue.template AllocTensor<T>();
    auto sumTensor = this->sumInQue.template AllocTensor<T>();

    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    ProcessVec4Vf<T>(mulsResUb, mmRes[kRunInfo.s2SingleCurSize], maxTensor, sumTensor, constInfo.scaleValue,
        constInfo.pScaler, runInfo.nVecSize, constInfo.pKBaseSize);
    this->mulsResQue.template EnQue(mulsResUb);
    this->mulsResQue.template DeQue<T>();
    if (kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1) {
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
    }
    DataCopyExtParams dataCopyGmParams;
    dataCopyGmParams.blockCount = 1;
    dataCopyGmParams.blockLen = kRunInfo.s2RealBaseSize * sizeof(T);
    dataCopyGmParams.srcStride = 0;
    dataCopyGmParams.dstStride = 0;
    DataCopyPad(reduceSumResGm[constInfo.subBlockIdx * constInfo.pKBaseSize], mulsResUb, dataCopyGmParams);
    CrossCoreSetFlag<1, PIPE_MTE3>(EVENT_ID2);
    this->maxInQue.template FreeTensor(maxTensor);
    this->sumInQue.template FreeTensor(sumTensor);
    this->mulsResQue.template FreeTensor(mulsResUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector5(SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    auto mulsInUb = this->mulsInQue.template AllocTensor<T>();
    auto mulsResUb = this->mulsResQue.template AllocTensor<T>();
    auto lossSumUb = this->lossSumQue.template AllocTensor<T>();
    auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = kRunInfo.s2RealBaseSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    CrossCoreWaitFlag<1, PIPE_MTE2>(EVENT_ID2);
    DataCopyPad(mulsInUb, reduceSumResGm[(1 - constInfo.subBlockIdx) * constInfo.pKBaseSize], dataCopyParams, padParams);
    this->mulsInQue.template EnQue(mulsInUb);
    this->mulsInQue.template DeQue<T>();
    if (kRunInfo.isAlign64) {
        ProcessVec5Vf<T, true>(lossSumUb, mulsResUb, mulsInUb, reduceSumTensor[runInfo.s2CurSize], kRunInfo.s2RealBaseSize);
    } else {
        ProcessVec5Vf<T, false>(lossSumUb, mulsResUb, mulsInUb, reduceSumTensor[runInfo.s2CurSize], kRunInfo.s2RealBaseSize);
    }
    if (kRunInfo.isS2end && runInfo.isLastK && constInfo.subBlockIdx == 0) {
        this->lossSumQue.template EnQue(lossSumUb);
        this->lossSumQue.template DeQue<T>();
        DataCopyExtParams dataCopyGmParams;
        dataCopyGmParams.blockCount = 1;
        dataCopyGmParams.blockLen = sizeof(T);
        dataCopyGmParams.srcStride = 0;
        dataCopyGmParams.dstStride = 0;
        SetAtomicAdd<float>();
        DataCopyPad(lossGm, lossSumUb, dataCopyGmParams);
        SetAtomicNone();
    }
    this->mulsInQue.template FreeTensor(mulsInUb);
    this->mulsResQue.template FreeTensor(mulsResUb);
    this->lossSumQue.template FreeTensor(lossSumUb);
    this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector6(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    if (kRunInfo.s2SingleIdx == 0) {
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
    }
    auto mulsResUb = this->mulsResQue.template AllocTensor<T>();
    auto reluUb = this->reluQue.template AllocTensor<T>();
    auto reluGradUb = this->reluGradQue.template AllocTensor<INPUT_T>();
    LocalTensor<INPUT_T> weightInUb = weightInQue.template AllocTensor<INPUT_T>();
    auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();
    auto dwTensor = this->dwQue.template AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = runInfo.nIndexSize;
    dataCopyParams.blockLen = VEC_P_BASESIZE * sizeof(T);
    dataCopyParams.srcStride = (constInfo.kSize - VEC_P_BASESIZE) * sizeof(T);
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(reluUb, reluGm[runInfo.s2CurSize + runInfo.nIndexSize * constInfo.subBlockIdx * constInfo.kSize], dataCopyParams, padParams);
    this->reluQue.template EnQue(reluUb);
    this->reluQue.template DeQue<T>();
    if (kRunInfo.isAlign64) {
        ProcessVec6Vf<T, INPUT_T, true>(reluGradUb[kRunInfo.s2SingleCurSize * runInfo.nIndexSize], dwTensor, mulsResUb,
            weightInUb, reduceSumTensor[runInfo.s2CurSize], reluUb, kRunInfo.s2RealBaseSize, runInfo.nIndexSize, VEC_P_BASESIZE);
    } else {
        ProcessVec6Vf<T, INPUT_T, false>(reluGradUb[kRunInfo.s2SingleCurSize * runInfo.nIndexSize], dwTensor, mulsResUb, weightInUb,
            reduceSumTensor[runInfo.s2CurSize], reluUb, kRunInfo.s2RealBaseSize, runInfo.nIndexSize, VEC_P_BASESIZE);
    }
    if (kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1) {
        this->reluGradQue.template EnQue(reluGradUb);
        this->reluGradQue.template DeQue<INPUT_T>();
        LocalTensor<INPUT_T> mm3AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        DataCopy(mm3AL1Tensor[constInfo.subBlockIdx * runInfo.nIndexSize * BLOCK_SINGLE_LEN], reluGradUb,
            {static_cast<uint16_t>(constInfo.pKBaseSize / BLOCK_SINGLE_LEN),  static_cast<uint16_t>(runInfo.nIndexSize), 0, static_cast<uint16_t>(runInfo.nIndexSize)});
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
    }
    if (kRunInfo.isS2end) {
        this->dwQue.template EnQue(dwTensor);
        this->dwQue.template DeQue<T>();
        LocalTensor<OUT_T> dwOutTensor = dwOutQue.template AllocTensor<OUT_T>();
        CastDupVf<T, OUT_T>(dwOutTensor, dwTensor, runInfo.nIndexSize);
        this->dwOutQue.template EnQue(dwOutTensor);
        this->dwOutQue.template DeQue<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockLen = runInfo.nIndexSize * sizeof(OUT_T);
        dataCopyParams.srcStride = 1;
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        DataCopyPad(dWeightGm[runInfo.weightOffset], dwOutTensor, dataCopyParams);
        this->dwOutQue.template FreeTensor(dwOutTensor);
    }
    this->mulsResQue.template FreeTensor(mulsResUb);
    this->reluGradQue.template FreeTensor(reluGradUb);
    this->reluQue.template FreeTensor(reluUb);
    this->weightInQue.template FreeTensor(weightInUb);
    this->dwQue.template FreeTensor(dwTensor);
    this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector7(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3ResBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
    int32_t vRealKSize;
    int64_t coreKOffset;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
        coreKOffset = 0;
    } else {
        vRealKSize = kRunInfo.s2RealBaseSize - AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
        coreKOffset = AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
    }
    if (vRealKSize <= 0) {
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
        return;
    }

    LocalTensor<T> scatterAddTmpUb = bmm3ResBuf.template GetTensor<T>();
    int64_t realS2Idx1, realS2Idx2, s2GmOffset;
    SetAtomicAdd<T>();
    constexpr int32_t topKSplitSize = 2;
    int32_t topKTailSize = (vRealKSize % topKSplitSize == 0) ? topKSplitSize : (vRealKSize % topKSplitSize);
    int32_t topKProcessSize = topKSplitSize;
    int32_t topKLoopTimes = CeilDiv(vRealKSize, topKSplitSize);
    for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
        if (topKIdx >= topKLoopTimes - 1) {
            topKProcessSize = topKTailSize;
        }
        s2GmOffset = coreKOffset + topKIdx * topKSplitSize;
        GetRealS2Idx(s2GmOffset, realS2Idx1, runInfo, kRunInfo);
        GetRealS2Idx(s2GmOffset + 1, realS2Idx2, runInfo, kRunInfo);
        if (topKProcessSize % topKSplitSize != 0) {
            realS2Idx2 = -1;
        }

        int64_t s2IdLimit = runInfo.s2SparseLen;
        int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, kRunInfo);
        int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, kRunInfo);
        if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
            break;
        }

        int64_t keySrcStride = 0;
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                        (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * constInfo.dSizeQueryIndex * sizeof(T);

        bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0);
        bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
            realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);
        bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);

        int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
        int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;
        if (strideInvalid || copyOutOfRange || key1LessThankey2) {
            // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
            if (keyOffset1 < 0) {
                break;
            }
            LocalTensor<T> srcTmpUb1 = scatterAddTmpUb[ub1Offset].template ReinterpretCast<T>();
            DataCopy(scatterAddResGm[keyOffset1 * constInfo.dSizeQueryIndex], srcTmpUb1, constInfo.dSizeQueryIndex);
            if (keyOffset2 < 0) {
                break;
            }
            LocalTensor<T> srcTmpUb2 = scatterAddTmpUb[ub2Offset].template ReinterpretCast<T>();
            DataCopy(scatterAddResGm[keyOffset2 * constInfo.dSizeQueryIndex], srcTmpUb2, constInfo.dSizeQueryIndex);
        } else {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockLen = constInfo.dSizeQueryIndex * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
            dataCopyParams.dstStride = keySrcStride;
            int64_t keyStartOffset = (keyOffset1 >= 0) ? keyOffset1 : keyOffset2;
            int64_t ubStartOffset = (keyOffset1 >= 0) ? ub1Offset : ub2Offset;
            LocalTensor<T> srcTmpUb = scatterAddTmpUb.template ReinterpretCast<T>();
            DataCopyPad(scatterAddResGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset], dataCopyParams);
        }
    }
    SetAtomicNone();
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ReInitBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(this->scatterAddInQue[0], 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->scatterAddInQue[1], 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->dKeyOutQue[0], 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->dKeyOutQue[1], 1, SLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligKlLossBlockVec<TEMPLATE_ARGS>::ProcessVector8()
{
    int64_t avgCost = CeilDiv(constInfo.totalCost, GetBlockNum() * GetTaskRation());
    int32_t t2Start = Min(constInfo.aivIdx * avgCost, constInfo.totalCost);
    int32_t t2End = Min(t2Start + avgCost, constInfo.totalCost);

    int32_t t2ProcessSize = UB_ROW_SIZE;
    int32_t t2TailSize = ((t2End - t2Start) % UB_ROW_SIZE == 0) ? UB_ROW_SIZE : ((t2End - t2Start) % UB_ROW_SIZE);
    int32_t pingPongIdx = 0;

    LocalTensor<T> copyInUb;
    LocalTensor<OUT_T> copyOutUb;

    for (int32_t t2Idx = t2Start; t2Idx < t2End; t2Idx += UB_ROW_SIZE) {
        if (t2Idx + UB_ROW_SIZE >= t2End) {
            t2ProcessSize = t2TailSize;
        }
        copyInUb = this->scatterAddInQue[pingPongIdx].template AllocTensor<T>();
        DataCopy(copyInUb, scatterAddResGm[t2Idx * constInfo.dSizeQueryIndex], t2ProcessSize * constInfo.dSizeQueryIndex);
        this->scatterAddInQue[pingPongIdx].template EnQue(copyInUb);
        this->scatterAddInQue[pingPongIdx].template DeQue<T>();
        copyOutUb = this->dKeyOutQue[pingPongIdx].template AllocTensor<OUT_T>();
        Cast(copyOutUb, copyInUb, RoundMode::CAST_ROUND, t2ProcessSize * constInfo.dSizeQueryIndex);
        this->dKeyOutQue[pingPongIdx].template EnQue(copyOutUb);
        this->dKeyOutQue[pingPongIdx].template DeQue<OUT_T>();
        DataCopy(dKeyIndexGm[t2Idx * constInfo.dSizeQueryIndex], copyOutUb, t2ProcessSize * constInfo.dSizeQueryIndex);
        this->scatterAddInQue[pingPongIdx].template FreeTensor(copyInUb);
        this->dKeyOutQue[pingPongIdx].template FreeTensor(copyOutUb);
        pingPongIdx = 1 - pingPongIdx;
    }
}

TEMPLATES_DEF
class SligKlLossBlockVecDummy {
public:
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR keyIndex, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
        GM_ADDR keyRope, GM_ADDR dKeyIndex, GM_ADDR dWeight, GM_ADDR loss, GlobalTensor<INPUT_T> &gatherSYRes, GlobalTensor<T> &relu,
        GlobalTensor<T> &reduceSumRes, GlobalTensor<T> &scatterAddRes){};
    __aicore__ inline void InitBuffers(TPipe *pipe){};
    __aicore__ inline void ProcessVector0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo){};
    __aicore__ inline void CopyInWeight(SLIGradKLLossRunInfo &runInfo){};
    __aicore__ inline void ProcessVector1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo){};
    __aicore__ inline void ProcessVector2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, SLIGradKLLossRunInfo &runInfo){};
    __aicore__ inline void ProcessVector3(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo){};
    __aicore__ inline void CopyInMaxSum(SLIGradKLLossRunInfo &runInfo){};
    __aicore__ inline void ProcessVector4(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo){};
    __aicore__ inline void ProcessVector5(SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo){};
    __aicore__ inline void ProcessVector6(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo){};
    __aicore__ inline void ProcessVector7(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3ResBuf, SLIGradKLLossRunInfo &runInfo, SLIGradKLLossKRunInfo &kRunInfo){};
    __aicore__ inline void ReInitBuffers(TPipe *pipe){};
    __aicore__ inline void ProcessVector8(){};
};
} // namespace SligKlLoss
#endif