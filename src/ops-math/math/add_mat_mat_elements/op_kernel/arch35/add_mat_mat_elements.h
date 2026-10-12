/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_MAT_MAT_ELEMENTS_H_
#define ADD_MAT_MAT_ELEMENTS_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_mat_mat_elements_tiling_data.h"
#include "add_mat_mat_elements_tiling_key.h"

namespace NsAddMatMatElements {

using namespace AscendC;

static constexpr int64_t kMaxInputSlots = ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS;
static constexpr int64_t kMaxOutputSlots = ADD_MAT_MAT_ELEMENTS_MAX_OUTPUT_SLOTS;
static constexpr int64_t kPhysNodes = ADD_MAT_MAT_ELEMENTS_PHYS_NODES;

// Non-broadcast VF declarations
template <typename T>
__simd_vf__ inline void MulAlphaVF(__ubuf__ T* dstAddr, __ubuf__ T* aAddr, __ubuf__ T* bAddr, T alpha, uint32_t count,
                                   uint32_t oneRepeatSize, uint16_t repeatTimes);

template <typename T>
__simd_vf__ inline void BetaAddVF(__ubuf__ T* dstAddr, __ubuf__ T* cAddr, __ubuf__ T* t2Addr, T beta, uint32_t count,
                                  uint32_t oneRepeatSize, uint16_t repeatTimes);

__aicore__ inline void GetCoreRange(int64_t coreId, int64_t tilesMain, int64_t coresTail, int64_t& start, int64_t& end)
{
    if (coreId < coresTail) {
        start = coreId * (tilesMain + 1);
        end = start + tilesMain + 1;
    } else {
        start = coresTail * (tilesMain + 1) + (coreId - coresTail) * tilesMain;
        end = start + tilesMain;
    }
}

template <typename T, int64_t RANK>
class AddMatMatElementsKernel {
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;
    static constexpr uint32_t VL_T = AscendC::GetVecLen() / sizeof(T);

    AscendC::TPipe pipe_;
    const AddMatMatElementsTilingData* td_;
    AscendC::GlobalTensor<T> gmIn_[kMaxInputSlots];
    AscendC::GlobalTensor<T> gmOut_[kMaxOutputSlots];
    AscendC::GlobalTensor<T> gmBeta_;
    AscendC::GlobalTensor<T> gmAlpha_;
    T alphaVal_;
    T betaVal_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];
    AscendC::MultiCopyParams<T, ND> nddmaParams_[kMaxInputSlots];
    int64_t nddmaDims_;

public:
    __aicore__ inline AddMatMatElementsKernel() {}
    __aicore__ inline void Init(GM_ADDR c, GM_ADDR a, GM_ADDR b, GM_ADDR beta, GM_ADDR alpha, GM_ADDR cOut,
                                const AddMatMatElementsTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBrc(const int64_t* coord, int64_t inputIdx, int64_t slot, int64_t aISeg);
    __aicore__ inline void CopyBroadcastSource(const int64_t* coord, int64_t inputIdx, int64_t slot,
                                               const uint32_t* srcShape);
    __aicore__ inline void ProcessUnifiedBroadcastTile(int64_t tileBaseFlat, int64_t count, const int64_t* tileCoord,
                                                       int64_t aISeg, int32_t evMte2V, int32_t evVMte2,
                                                       int32_t evVMte3);
    __aicore__ inline void Compute(int64_t count);
    __aicore__ inline void CopyOut(const int64_t* coord, int64_t count);
    __aicore__ inline int64_t ComputeInnerCount() const;
    __aicore__ inline int64_t TiledAxis() const;
    __aicore__ inline bool HasBroadcast(int64_t inputIdx) const;
    __aicore__ inline void FlatToEffectiveCoord(int64_t flatIdx, const int64_t* broShape, int64_t rank, int64_t* coord);
    __aicore__ inline int64_t CalcInputOffset(const int64_t* coord, const int64_t* strides, int64_t rank);
};

template <typename T, int64_t RANK>
__aicore__ inline void AddMatMatElementsKernel<T, RANK>::Init(GM_ADDR c, GM_ADDR a, GM_ADDR b, GM_ADDR beta,
                                                              GM_ADDR alpha, GM_ADDR cOut,
                                                              const AddMatMatElementsTilingData& tilingData)
{
    td_ = &tilingData;
    if (td_->multicore.total_tiles <= 0)
        return;
    gmIn_[0].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(c));
    gmIn_[1].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));
    gmIn_[2].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));
    gmOut_[0].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cOut));
    gmBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta), 1);
    gmAlpha_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alpha), 1);
    betaVal_ = gmBeta_.GetValue(0);
    alphaVal_ = gmAlpha_.GetValue(0);
    for (int i = 0; i < kPhysNodes; i++) {
        pipe_.InitBuffer(buf_[i], td_->per_buf_bytes);
    }
    int64_t k = TiledAxis();
    nddmaDims_ = (RANK - k <= ND) ? (RANK - k) : ND;
    const int64_t* dstShape = td_->max_bro_shape;
    for (int inp = 0; inp < kMaxInputSlots; inp++) {
        int64_t inner = 1;
        int64_t nd = 0;
        for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
            nddmaParams_[inp].loopInfo.loopSize[nd] = (d == k) ? 0 : dstShape[d];
            nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->input_strides[inp][d];
            nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
            nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
            nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
            inner *= (d == k) ? td_->split.a_i : dstShape[d];
            nd++;
        }
        for (; nd < ND; nd++) {
            nddmaParams_[inp].loopInfo.loopSize[nd] = 1;
            nddmaParams_[inp].loopInfo.loopSrcStride[nd] = 0;
            nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
            nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
            nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
        }
    }
}

template <typename T, int64_t RANK>
__aicore__ inline int64_t AddMatMatElementsKernel<T, RANK>::TiledAxis() const
{
    return (td_->split.axis == 0) ? 0 : (td_->split.axis - 1);
}

template <typename T, int64_t RANK>
__aicore__ inline int64_t AddMatMatElementsKernel<T, RANK>::ComputeInnerCount() const
{
    int64_t k = TiledAxis();
    int64_t inner = 1;
    for (int64_t d = k + 1; d < RANK; d++) {
        inner *= td_->max_bro_shape[d];
    }
    return inner;
}

template <typename T, int64_t RANK>
__aicore__ inline bool AddMatMatElementsKernel<T, RANK>::HasBroadcast(int64_t inputIdx) const
{
    int64_t k = TiledAxis();
    const int64_t* strides = td_->input_strides[inputIdx];
    const int64_t* dstShape = td_->max_bro_shape;
    for (int64_t d = k; d < RANK; d++) {
        if (strides[d] == 0 && dstShape[d] > 1)
            return true;
    }
    return false;
}

// Process
template <typename T, int64_t RANK>
__aicore__ inline void AddMatMatElementsKernel<T, RANK>::Process()
{
    if (td_->multicore.total_tiles <= 0)
        return;
    int64_t start, end;
    GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tiles_main, td_->multicore.cores_tail, start, end);
    if (end <= start)
        return;

    int64_t innerCount = ComputeInnerCount();
    int64_t aI = td_->split.a_i;
    int64_t aTail = td_->split.a_i_tail;
    int64_t k = TiledAxis();
    int64_t fullTileElems = aI * innerCount;
    int64_t tailTileElems = aTail * innerCount;
    if (fullTileElems <= 0)
        return;

    int64_t numTilesAlongAxis = (aTail > 0) ? (td_->split.a_o + 1) : td_->split.a_o;
    if (numTilesAlongAxis <= 0)
        numTilesAlongAxis = 1;
    int64_t tiledAxisDim = td_->max_bro_shape[k];

    int32_t evMte2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    int32_t evVMte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
    int32_t evVMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    int32_t evMte3Mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

    bool aBrc = HasBroadcast(1);
    bool bBrc = HasBroadcast(2);
    bool anyBrc = aBrc || bBrc;

    int64_t coord[ADD_MAT_MAT_ELEMENTS_RANK_MAX] = {0};

    for (int64_t flat = start; flat < end; flat++) {
        int64_t outerIdx = flat / numTilesAlongAxis;
        int64_t tileIdx = flat % numTilesAlongAxis;
        bool isTail = (aTail > 0) && (tileIdx == numTilesAlongAxis - 1);
        int64_t aISeg = isTail ? aTail : aI;
        int64_t count = isTail ? tailTileElems : fullTileElems;
        if (count <= 0)
            continue;

        int64_t aBegin = outerIdx * (tiledAxisDim * innerCount) + tileIdx * fullTileElems;
        FlatToEffectiveCoord(aBegin, td_->max_bro_shape, RANK, coord);

        if (flat != start)
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);

        if (anyBrc) {
            ProcessUnifiedBroadcastTile(aBegin, count, coord, aISeg, evMte2V, evVMte2, evVMte3);
        } else {
            CopyInBrc(coord, 1, 0, aISeg);
            CopyInBrc(coord, 2, 1, aISeg);
            CopyInBrc(coord, 0, 2, aISeg);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
            Compute(count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
            CopyOut(coord, count);
        }

        if (flat != end - 1)
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
    }

    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evMte2V);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evVMte2);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(evVMte3);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
}

// Copy the unique values needed by one broadcast tile into a compact UB
// tensor. Broadcast dimensions have srcShape[d] == 1, so NDDMA never relies
// on the unsupported loopSize > 1 plus loopSrcStride == 0 combination.
template <typename T, int64_t RANK>
__aicore__ inline void AddMatMatElementsKernel<T, RANK>::CopyBroadcastSource(const int64_t* coord, int64_t inputIdx,
                                                                             int64_t slot, const uint32_t* srcShape)
{
    int64_t k = TiledAxis();
    const int64_t* strides = td_->input_strides[inputIdx];
    int64_t offset = CalcInputOffset(coord, strides, RANK);

    auto params = nddmaParams_[inputIdx];
    int64_t coveredDims = RANK - k;
    if (coveredDims > ND)
        coveredDims = ND;

    int64_t compactInner = 1;
    for (int64_t nd = 0; nd < coveredDims; nd++) {
        int64_t d = RANK - 1 - nd;
        int64_t dim = static_cast<int64_t>(srcShape[d]);
        params.loopInfo.loopSize[nd] = dim;
        params.loopInfo.loopSrcStride[nd] = (strides[d] == 0) ? 1 : strides[d];
        params.loopInfo.loopDstStride[nd] = compactInner;
        params.loopInfo.loopLpSize[nd] = 0;
        params.loopInfo.loopRpSize[nd] = 0;
        compactInner *= dim;
    }
    for (int64_t nd = coveredDims; nd < ND; nd++) {
        params.loopInfo.loopSize[nd] = 1;
        params.loopInfo.loopSrcStride[nd] = 1;
        params.loopInfo.loopDstStride[nd] = compactInner;
        params.loopInfo.loopLpSize[nd] = 0;
        params.loopInfo.loopRpSize[nd] = 0;
    }

    static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad,
                                                 false};
    int64_t coveredStart = RANK - coveredDims;
    int64_t outerIters = 1;
    for (int64_t d = k; d < coveredStart; d++) {
        outerIters *= static_cast<int64_t>(srcShape[d]);
    }

    AscendC::LocalTensor<T> dst = buf_[slot].template Get<T>();
    for (int64_t oi = 0; oi < outerIters; oi++) {
        int64_t elemAdj = 0;
        int64_t tmp = oi;
        for (int64_t d = coveredStart - 1; d >= k; d--) {
            int64_t dim = static_cast<int64_t>(srcShape[d]);
            elemAdj += (tmp % dim) * strides[d];
            tmp /= dim;
        }
        AscendC::DataCopy<T, ND, cfg>(dst[oi * compactInner], gmIn_[inputIdx][offset + elemAdj], params);
    }
}

// All in-tile broadcast patterns use the same representation: compact source
// data in B3 followed by one complete-rank Broadcast into the input's compute
// buffer. Axes above the tiled axis are fixed for this tile and therefore have
// extent 1. The tiled axis uses aISeg, including tail tiles.
template <typename T, int64_t RANK>
__aicore__ inline void AddMatMatElementsKernel<T, RANK>::ProcessUnifiedBroadcastTile(int64_t tileBaseFlat,
                                                                                     int64_t count,
                                                                                     const int64_t* tileCoord,
                                                                                     int64_t aISeg, int32_t evMte2V,
                                                                                     int32_t evVMte2, int32_t evVMte3)
{
    constexpr int B0 = 0, B1 = 1, B2 = 2, B3 = 3;
    int64_t k = TiledAxis();
    uint32_t dstShape[ADD_MAT_MAT_ELEMENTS_RANK_MAX] = {0};
    uint32_t aSrcShape[ADD_MAT_MAT_ELEMENTS_RANK_MAX] = {0};
    uint32_t bSrcShape[ADD_MAT_MAT_ELEMENTS_RANK_MAX] = {0};
    bool expandA = false;
    bool expandB = false;

    for (int64_t d = 0; d < RANK; d++) {
        int64_t tileDim = (d < k) ? 1 : ((d == k) ? aISeg : td_->max_bro_shape[d]);
        dstShape[d] = static_cast<uint32_t>(tileDim);
        bool aBroadcastDim = td_->input_strides[1][d] == 0 && tileDim > 1;
        bool bBroadcastDim = td_->input_strides[2][d] == 0 && tileDim > 1;
        aSrcShape[d] = aBroadcastDim ? 1U : dstShape[d];
        bSrcShape[d] = bBroadcastDim ? 1U : dstShape[d];
        expandA = expandA || aBroadcastDim;
        expandB = expandB || bBroadcastDim;
    }

    // c never broadcasts. Non-broadcast a/b keep the already-validated NDDMA
    // path; only expanded inputs pass through the compact B3 workspace.
    CopyInBrc(tileCoord, 0, B2, aISeg);
    if (expandA) {
        CopyBroadcastSource(tileCoord, 1, B3, aSrcShape);
    } else {
        CopyInBrc(tileCoord, 1, B0, aISeg);
    }
    if (!expandB) {
        CopyInBrc(tileCoord, 2, B1, aISeg);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);

    if (expandA) {
        AscendC::BroadcastTiling tiling;
        AscendC::GetBroadcastTilingInfo<T>(static_cast<uint32_t>(RANK), dstShape, aSrcShape, true, tiling);
        AscendC::Broadcast<T>(buf_[B0].template Get<T>(), buf_[B3].template Get<T>(), dstShape, aSrcShape, &tiling);
        AscendC::PipeBarrier<PIPE_V>();
    }

    if (expandB) {
        if (expandA) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
        }
        CopyBroadcastSource(tileCoord, 2, B3, bSrcShape);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);

        AscendC::BroadcastTiling tiling;
        AscendC::GetBroadcastTilingInfo<T>(static_cast<uint32_t>(RANK), dstShape, bSrcShape, true, tiling);
        AscendC::Broadcast<T>(buf_[B1].template Get<T>(), buf_[B3].template Get<T>(), dstShape, bSrcShape, &tiling);
        AscendC::PipeBarrier<PIPE_V>();
    }

    Compute(count);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
    CopyOut(tileCoord, count);
}

// CopyInBrc — NDDMA (non-broadcast fast path)
template <typename T, int64_t RANK>
__aicore__ inline void AddMatMatElementsKernel<T, RANK>::CopyInBrc(const int64_t* coord, int64_t inputIdx, int64_t slot,
                                                                   int64_t aISeg)
{
    int64_t k = TiledAxis();
    int64_t off = CalcInputOffset(coord, td_->input_strides[inputIdx], RANK);
    const int64_t* dstShape = td_->max_bro_shape;
    const int64_t* strides = td_->input_strides[inputIdx];

    auto params = nddmaParams_[inputIdx];
    int64_t kNd = RANK - 1 - k;
    int64_t inner = 1;
    for (int64_t nd = 0; nd < ND; nd++) {
        if (nd == kNd)
            params.loopInfo.loopSize[nd] = aISeg;
        params.loopInfo.loopDstStride[nd] = inner;
        inner *= params.loopInfo.loopSize[nd];
    }

    static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad,
                                                 false};
    if constexpr (RANK <= 5) {
        AscendC::DataCopy<T, ND, cfg>(buf_[slot].template Get<T>(), gmIn_[inputIdx][off], params);
    } else {
        int64_t outerIters = 1;
        for (int64_t d = k; d < RANK - nddmaDims_; d++)
            outerIters *= (d == k) ? aISeg : dstShape[d];
        AscendC::LocalTensor<T> buf = buf_[slot].template Get<T>();
        for (int64_t oi = 0; oi < outerIters; oi++) {
            int64_t elemAdj = 0, tmp = oi;
            for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                int64_t sz = (d == k) ? aISeg : dstShape[d];
                elemAdj += (tmp % sz) * strides[d];
                tmp /= sz;
            }
            AscendC::DataCopy<T, ND, cfg>(buf[oi * inner], gmIn_[inputIdx][off + elemAdj], params);
        }
    }
}

// Compute — VF chains (non-broadcast fast path)
template <typename T, int64_t RANK>
__aicore__ inline void AddMatMatElementsKernel<T, RANK>::Compute(int64_t count)
{
    constexpr int B0 = 0, B1 = 1, B2 = 2, B3 = 3;
    uint32_t cnt = static_cast<uint32_t>(count);
    uint16_t rep = AscendC::CeilDivision(cnt, VL_T);
    asc_vf_call<MulAlphaVF<T>>((__ubuf__ T*)buf_[B3].template Get<T>().GetPhyAddr(),
                               (__ubuf__ T*)buf_[B0].template Get<T>().GetPhyAddr(),
                               (__ubuf__ T*)buf_[B1].template Get<T>().GetPhyAddr(), alphaVal_, cnt, VL_T, rep);
    asc_vf_call<BetaAddVF<T>>((__ubuf__ T*)buf_[B0].template Get<T>().GetPhyAddr(),
                              (__ubuf__ T*)buf_[B2].template Get<T>().GetPhyAddr(),
                              (__ubuf__ T*)buf_[B3].template Get<T>().GetPhyAddr(), betaVal_, cnt, VL_T, rep);
}

template <typename T, int64_t RANK>
__aicore__ inline void AddMatMatElementsKernel<T, RANK>::CopyOut(const int64_t* coord, int64_t count)
{
    int64_t off = CalcInputOffset(coord, td_->output_strides[0], RANK);
    AscendC::DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = count * sizeof(T);
    p.srcStride = 0;
    p.dstStride = 0;
    AscendC::DataCopyPad(gmOut_[0][off], buf_[0].template Get<T>(), p);
}

template <typename T, int64_t RANK>
__aicore__ inline void AddMatMatElementsKernel<T, RANK>::FlatToEffectiveCoord(int64_t flatIdx, const int64_t* broShape,
                                                                              int64_t rank, int64_t* coord)
{
    for (int64_t d = rank - 1; d >= 0; d--) {
        coord[d] = flatIdx % broShape[d];
        flatIdx /= broShape[d];
    }
}

template <typename T, int64_t RANK>
__aicore__ inline int64_t AddMatMatElementsKernel<T, RANK>::CalcInputOffset(const int64_t* coord,
                                                                            const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += coord[d] * strides[d];
    return offset;
}

} // namespace NsAddMatMatElements

// VF definitions
template <typename T>
__simd_vf__ inline void NsAddMatMatElements::MulAlphaVF(__ubuf__ T* dstAddr, __ubuf__ T* aAddr, __ubuf__ T* bAddr,
                                                        T alpha, uint32_t count, uint32_t oneRepeatSize,
                                                        uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> aReg, bReg, tmpReg, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aRegIdx;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aRegIdx = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        uint32_t remain = count - (uint32_t)i * oneRepeatSize;
        mask = AscendC::Reg::UpdateMask<T>(remain);
        AscendC::Reg::LoadAlign(aReg, aAddr, aRegIdx);
        AscendC::Reg::LoadAlign(bReg, bAddr, aRegIdx);
        AscendC::Reg::Mul(tmpReg, aReg, bReg, mask);
        AscendC::Reg::Muls(dstReg, tmpReg, alpha, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aRegIdx, mask);
    }
}

template <typename T>
__simd_vf__ inline void NsAddMatMatElements::BetaAddVF(__ubuf__ T* dstAddr, __ubuf__ T* cAddr, __ubuf__ T* t2Addr,
                                                       T beta, uint32_t count, uint32_t oneRepeatSize,
                                                       uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> cReg, t2Reg, tmpReg, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aRegIdx;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aRegIdx = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        uint32_t remain = count - (uint32_t)i * oneRepeatSize;
        mask = AscendC::Reg::UpdateMask<T>(remain);
        AscendC::Reg::LoadAlign(cReg, cAddr, aRegIdx);
        AscendC::Reg::LoadAlign(t2Reg, t2Addr, aRegIdx);
        AscendC::Reg::Muls(tmpReg, cReg, beta, mask);
        AscendC::Reg::Add(dstReg, tmpReg, t2Reg, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aRegIdx, mask);
    }
}

#endif // ADD_MAT_MAT_ELEMENTS_H_
