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
 * \file pad_v3_grad_circular_huge_width.h
 * \brief pad v3 grad circular huge width.h
 */

#ifndef PAD_V3_GRAD_CIRCULAR_HUGE_WIDTH_H
#define PAD_V3_GRAD_CIRCULAR_HUGE_WIDTH_H

#include "kernel_operator.h"
#include "pad_v3_grad_struct.h"
#include "pad_v3_grad_common.h"

namespace PadV3Grad {
using namespace AscendC;

template <typename T>
class KernelPadV3GradCircularHugeWidth {
private:
    constexpr static size_t DIM_IDX_4 = 4U;
    constexpr static size_t GM_ADDR_IDX_CENTER = 0U;
    constexpr static size_t GM_ADDR_IDX_LEFT = 1U;
    constexpr static size_t GM_ADDR_IDX_RIGHT = 2U;

    uint32_t inResStart_;
    uint32_t inSrcStart_;

    struct IdxAndTimes {
        uint64_t inGmIdx[3]{0};
        uint8_t cnt = 1;
    };

    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;

    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> inQueue_;
    TBuf<TPosition::VECCALC> resQueue_;
    uint32_t blockIdx_;

    const PadV3GradACTilingData* tilingData_ = nullptr;

    uint64_t inIndex_[PAD_GRAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_GRAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t ubFactor_{0};
    uint8_t dimNum_{0};
    uint8_t ubAxis_{0};
    uint64_t factorOfubAxis_{0};
    uint64_t dataLen_{0};

    DataCopyExtParams copyInParams_;
    DataCopyPadExtParams<T> PadParams_{false, 0, 0, 0};

    uint32_t leftUbAddLen_{0};
    uint32_t leftUbStartIdx_{0};

    uint32_t rightUbAddLen_{0};

    uint16_t oneRepeatSize_{0};
    uint64_t leftStartOnInner_{0};

    using CalType = std::conditional_t<std::is_same_v<T, bfloat16_t>, float32_t,
                                       std::conditional_t<std::is_same_v<T, float16_t>, float32_t, T>>;

public:
    __aicore__ inline KernelPadV3GradCircularHugeWidth(TPipe* pipe, const PadV3GradACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;

        inResStart_ = 0;
        inSrcStart_ = 0;

        copyInParams_.blockCount = 1;
        copyInParams_.srcStride = 0;
        copyInParams_.dstStride = 0;

        oneRepeatSize_ = GetVecLen() / sizeof(CalType);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();

        input_gm.SetGlobalBuffer((__gm__ T*)x);
        output_gm.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQueue_, BUFFER_NUM * tilingData_->outTileSize * sizeof(T));
        pipe_->InitBuffer(resQueue_, BUFFER_NUM * tilingData_->outTileSize * sizeof(CalType));

        ubFactor_ = tilingData_->ubFactor;
        dimNum_ = tilingData_->dimNum;
        ubAxis_ = tilingData_->ubAxis;
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdx = blockIdx_ * tilingData_->ubPerCount;
        if (startIdx >= tilingData_->ubTotalCount) {
            return;
        }
        uint32_t endIdx = min(startIdx + tilingData_->ubPerCount, tilingData_->ubTotalCount);
        // 尾轴需要分几次处理完成
        factorOfubAxis_ = CeilDiv(tilingData_->outShape[ubAxis_], ubFactor_);
        leftStartOnInner_ = tilingData_->outShape[ubAxis_] - tilingData_->leftPad[ubAxis_];

        // 每个核一次只能处理尾轴的一块数据，以尾轴的一块为单位循环
        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            uint32_t curIdx = idx;
            uint64_t inSelfAddr = 0;
            uint64_t outSelfAddr = 0;

            for (int32_t i = ubAxis_; i >= 0; i--) {
                uint64_t factor = tilingData_->outShape[i];
                if (i == ubAxis_) {
                    factor = factorOfubAxis_;
                }
                // 在第i维的索引
                outIndex_[i] = (i == ubAxis_ ? curIdx % factor * ubFactor_ : curIdx % factor);
                inIndex_[i] = outIndex_[i] + tilingData_->leftPad[i];
                curIdx = curIdx / factor;
                // 本块正文在input中的索引
                inSelfAddr += inIndex_[i] * tilingData_->inStride[i];
                outSelfAddr += outIndex_[i] * tilingData_->outStride[i];
            }
            // 当前块的数据长度
            dataLen_ = (outIndex_[ubAxis_] + ubFactor_ <= tilingData_->outShape[ubAxis_] ?
                            ubFactor_ :
                            tilingData_->outShape[ubAxis_] - outIndex_[ubAxis_]);
            ProcessOneStep(idx - startIdx, inSelfAddr, outSelfAddr);
        }
    }

private:
    __aicore__ inline void ProcessOneStep(uint32_t idx, uint64_t inSelfAddr, uint64_t outSelfAddr)
    {
        // 双buffer，获取指定长度的tensor
        LocalTensor<T> srcLocal = inQueue_.Get<T>();
        LocalTensor<T> src = srcLocal[(idx & 1) * tilingData_->outTileSize];

        LocalTensor<CalType> resLocal = resQueue_.Get<CalType>();
        LocalTensor<CalType> res = resLocal[(idx & 1) * tilingData_->outTileSize];

        // 数据拷贝参数
        copyInParams_.blockLen = dataLen_ * sizeof(T);
        float32_t initVal{0.0f};

        // ub地址
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ CalType* resAddr = (__ubuf__ CalType*)res.GetPhyAddr();

        // 同步：V等上一块从res搬出
        if constexpr (sizeof(T) == sizeof(float32_t)) {
            if (idx > 1) {
                WaitEvent<HardEvent::MTE3_V>(idx);
            }
        }

        Duplicate<CalType>(res[inResStart_], initVal, static_cast<int32_t>(dataLen_));

        // 同步：MTE2等MTE3
        if constexpr (sizeof(T) != sizeof(float32_t)) {
            if (idx > 1) {
                WaitEvent<HardEvent::MTE3_MTE2>(idx);
            }
        }

        // inSelfAddr 正文数据在输入中的地址
        DataCopyPad<T>(src[inSrcStart_], input_gm[inSelfAddr], copyInParams_, PadParams_);

        SetEvent<HardEvent::MTE2_V>(idx);
        WaitEvent<HardEvent::MTE2_V>(idx);

        // 同步：V等MTE2
        ProcessSelfAdd(srcAddr, resAddr);
        // 本身（首块）

        SetEvent<HardEvent::V_MTE2>(idx);
        WaitEvent<HardEvent::V_MTE2>(idx);

        // 计算线性索引
        IdxAndTimes inIdxCnt[5];
        CalculateOffsetParams(inIdxCnt);

        CopyAndCal(srcAddr, resAddr, src, idx, outSelfAddr, inIdxCnt);
        CopyOutputToGM(srcAddr, resAddr, src, res, idx, outSelfAddr);
    }
    __aicore__ inline void CalculateOffsetParams(IdxAndTimes* inIdxCnt)
    {
        if (tilingData_->leftPad[ubAxis_] == 0 || outIndex_[ubAxis_] + dataLen_ <= leftStartOnInner_) {
            leftUbAddLen_ = 0;   // 需要参与加法计算的长度
            leftUbStartIdx_ = 0; // 在输出上参与pad的起始索引
        } else {
            leftUbStartIdx_ = (outIndex_[ubAxis_] < leftStartOnInner_) ? (leftStartOnInner_ - outIndex_[ubAxis_]) : 0;
            leftUbAddLen_ = dataLen_ - leftUbStartIdx_;
            inIdxCnt[DIM_IDX_4].inGmIdx[GM_ADDR_IDX_LEFT] = outIndex_[ubAxis_] + leftUbStartIdx_ - leftStartOnInner_;
        }

        if (tilingData_->rightPad[ubAxis_] == 0 || outIndex_[ubAxis_] >= tilingData_->rightPad[ubAxis_]) {
            rightUbAddLen_ = 0;
        } else {
            rightUbAddLen_ = (outIndex_[ubAxis_] + dataLen_ <= tilingData_->rightPad[ubAxis_]) ?
                                 dataLen_ :
                                 tilingData_->rightPad[ubAxis_] - outIndex_[ubAxis_];
            inIdxCnt[DIM_IDX_4].inGmIdx[GM_ADDR_IDX_RIGHT] = outIndex_[ubAxis_] + tilingData_->leftPad[ubAxis_] +
                                                             tilingData_->outShape[ubAxis_];
            // 当前块的右pad在输入中的索引
        }

        for (uint8_t i = 0; i < dimNum_ - 1; ++i) {
            // self
            inIdxCnt[i].inGmIdx[GM_ADDR_IDX_CENTER] = inIndex_[i] * tilingData_->inStride[i];
            // left
            if (tilingData_->leftPad[i] != 0 && outIndex_[i] >= tilingData_->outShape[i] - tilingData_->leftPad[i]) {
                inIdxCnt[i].inGmIdx[inIdxCnt[i].cnt++] = (outIndex_[i] + tilingData_->leftPad[i] -
                                                          tilingData_->outShape[i]) *
                                                         tilingData_->inStride[i];
            }
            // right
            if (tilingData_->rightPad[i] != 0 && outIndex_[i] < tilingData_->rightPad[i]) {
                inIdxCnt[i].inGmIdx[inIdxCnt[i].cnt++] = (outIndex_[i] + tilingData_->leftPad[i] +
                                                          tilingData_->outShape[i]) *
                                                         tilingData_->inStride[i];
            }
        }
    }

    __aicore__ inline void CopyAndCal(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, LocalTensor<T> src, uint32_t idx,
                                      uint64_t outSelfAddr, IdxAndTimes* inIdxCnt)
    {
        // 搬入
        for (uint8_t a0 = 0; a0 < inIdxCnt[0].cnt; ++a0) {
            uint64_t a0Offset = inIdxCnt[0].inGmIdx[a0];
            for (uint8_t a1 = 0; a1 < inIdxCnt[1].cnt; ++a1) {
                uint64_t a1Offset = a0Offset + inIdxCnt[1].inGmIdx[a1];
                for (uint8_t a2 = 0; a2 < inIdxCnt[2].cnt; ++a2) {
                    uint64_t a2Offset = a1Offset + inIdxCnt[2].inGmIdx[a2];
                    for (uint8_t a3 = 0; a3 < inIdxCnt[3].cnt; ++a3) {
                        uint64_t a3Offset = a2Offset + inIdxCnt[3].inGmIdx[a3];

                        // 中间
                        if (a0 != 0 || a1 != 0 || a2 != 0 || a3 != 0) {
                            ProcessMiddleData(srcAddr, resAddr, src, idx, a3Offset);
                        }
                        // 左
                        if (leftUbAddLen_ != 0) {
                            ProcessLeftData(srcAddr, resAddr, src, idx, a3Offset, inIdxCnt);
                        }

                        // 右边
                        if (rightUbAddLen_ != 0) {
                            ProcessRightData(srcAddr, resAddr, src, idx, a3Offset, inIdxCnt);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessMiddleData(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, LocalTensor<T> src,
                                             uint32_t idx, uint64_t a3Offset)
    {
        uint32_t inMidAddr = a3Offset + inIndex_[ubAxis_];
        copyInParams_.blockLen = dataLen_ * sizeof(T);
        // 同步：MTE2等V

        DataCopyPad(src[inSrcStart_], input_gm[inMidAddr], copyInParams_, PadParams_);

        SetEvent<HardEvent::MTE2_V>(idx);
        WaitEvent<HardEvent::MTE2_V>(idx);
        // 同步：V等MTE2
        // 本身（非首块）
        ProcessSelfAdd(srcAddr, resAddr);

        SetEvent<HardEvent::V_MTE2>(idx);
        WaitEvent<HardEvent::V_MTE2>(idx);
    }

    __aicore__ inline void ProcessSelfAdd(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr)
    {
        uint16_t repeatSelfTimes = CeilDiv(dataLen_, oneRepeatSize_);
        uint32_t midUbAddLenVF = dataLen_;
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> srcReg;
            Reg::RegTensor<CalType> tempRegB32;
            Reg::RegTensor<CalType> resReg;
            Reg::MaskReg maskReg;

            for (uint16_t k = 0; k < repeatSelfTimes; k++) {
                maskReg = AscendC::Reg::UpdateMask<CalType>(midUbAddLenVF);
                if constexpr (sizeof(T) != sizeof(float32_t)) {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + k * oneRepeatSize_);
                } else {
                    Reg::LoadAlign(srcReg, srcAddr + k * oneRepeatSize_);
                }
                Reg::LoadAlign(resReg, resAddr + k * oneRepeatSize_);
                if constexpr (sizeof(T) != sizeof(float32_t)) {
                    Reg::Cast<CalType, T, CAST_TRAIT_0>(tempRegB32, srcReg, maskReg);
                    Reg::Add(resReg, tempRegB32, resReg, maskReg);
                } else {
                    Reg::Add(resReg, srcReg, resReg, maskReg);
                }
                Reg::StoreAlign(resAddr + k * oneRepeatSize_, resReg, maskReg);
            }
        }
    }

    __aicore__ inline void ProcessLeftData(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, LocalTensor<T> src,
                                           uint32_t idx, uint64_t a3Offset, IdxAndTimes* inIdxCnt)
    {
        uint64_t inLeftAddr = a3Offset + inIdxCnt[DIM_IDX_4].inGmIdx[GM_ADDR_IDX_LEFT];
        copyInParams_.blockLen = leftUbAddLen_ * sizeof(T);

        uint16_t leftMainTimes = leftUbAddLen_ / oneRepeatSize_;
        uint32_t leftTailLen = leftUbAddLen_ - leftMainTimes * oneRepeatSize_;
        uint32_t leftTailLenStroe = leftTailLen;
        uint16_t leftTailTimes = (leftTailLen != 0);

        uint32_t leftUbAddLenVF = leftUbAddLen_;
        uint16_t oneRepeatSizeVF = oneRepeatSize_;

        // ---------- MTE2 搬入 ----------
        DataCopyPad(src[inSrcStart_], input_gm[inLeftAddr], copyInParams_, PadParams_);

        SetEvent<HardEvent::MTE2_V>(idx);
        WaitEvent<HardEvent::MTE2_V>(idx);

        resAddr += leftUbStartIdx_;
        __ubuf__ CalType* dstAddr = resAddr;
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> inReg16;
            Reg::RegTensor<CalType> inReg32;
            Reg::RegTensor<CalType> resReg;

            AscendC::Reg::UnalignRegForLoad ureg0;
            AscendC::Reg::UnalignRegForStore ureg1;

            AscendC::Reg::MaskReg mask;

            mask = Reg::UpdateMask<CalType>(leftUbAddLenVF);
            for (uint16_t r = 0; r < leftMainTimes; ++r) {
                if constexpr (sizeof(T) == sizeof(float32_t)) {
                    Reg::LoadAlign<T>(inReg32, srcAddr + r * oneRepeatSizeVF);
                } else {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inReg16, srcAddr + r * oneRepeatSizeVF);
                    Reg::Cast<CalType, T, CAST_TRAIT_0>(inReg32, inReg16, mask);
                }

                AscendC::Reg::LoadUnAlignPre(ureg0, resAddr + r * oneRepeatSizeVF);
                AscendC::Reg::LoadUnAlign(resReg, ureg0, resAddr + r * oneRepeatSizeVF);

                Reg::Add(resReg, inReg32, resReg, mask);

                AscendC::Reg::StoreUnAlign(dstAddr, resReg, ureg1, oneRepeatSizeVF);
            }
            for (uint16_t r = 0; r < leftTailTimes; ++r) {
                mask = Reg::UpdateMask<CalType>(leftTailLen);

                if constexpr (sizeof(T) == sizeof(float32_t)) {
                    Reg::LoadAlign<T>(inReg32, srcAddr + leftMainTimes * oneRepeatSizeVF);
                } else {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inReg16,
                                                                      srcAddr + leftMainTimes * oneRepeatSizeVF);
                    Reg::Cast<CalType, T, CAST_TRAIT_0>(inReg32, inReg16, mask);
                }

                AscendC::Reg::LoadUnAlignPre(ureg0, resAddr + leftMainTimes * oneRepeatSizeVF);
                AscendC::Reg::LoadUnAlign(resReg, ureg0, resAddr + leftMainTimes * oneRepeatSizeVF);

                Reg::Add(resReg, inReg32, resReg, mask);

                AscendC::Reg::StoreUnAlign(dstAddr, resReg, ureg1, leftTailLenStroe);
            }
            AscendC::Reg::StoreUnAlignPost(dstAddr, ureg1, 0);
        }
        // ---- V→MTE2 同步 ----
        SetEvent<HardEvent::V_MTE2>(idx);
        WaitEvent<HardEvent::V_MTE2>(idx);
    }

    __aicore__ inline void ProcessRightData(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, LocalTensor<T> src,
                                            uint32_t idx, uint64_t a3Offset, IdxAndTimes* inIdxCnt)
    {
        uint64_t inRightAddr = a3Offset + inIdxCnt[DIM_IDX_4].inGmIdx[GM_ADDR_IDX_RIGHT];
        copyInParams_.blockLen = rightUbAddLen_ * sizeof(T);

        uint16_t rightMainTimes = CeilDiv(rightUbAddLen_, oneRepeatSize_);
        uint32_t rightUbAddLenVF = rightUbAddLen_;
        uint16_t oneRepeatSizeVF = oneRepeatSize_;

        // ---- MTE2 搬入 ----
        DataCopyPad(src[inSrcStart_], input_gm[inRightAddr], copyInParams_, PadParams_);

        SetEvent<HardEvent::MTE2_V>(idx);
        WaitEvent<HardEvent::MTE2_V>(idx);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> inReg16;
            Reg::RegTensor<CalType> inReg32;
            Reg::RegTensor<CalType> resReg;
            AscendC::Reg::MaskReg mask;

            for (uint16_t r = 0; r < rightMainTimes; ++r) {
                mask = Reg::UpdateMask<CalType>(rightUbAddLenVF);
                if constexpr (sizeof(T) == sizeof(float32_t)) {
                    Reg::LoadAlign<T>(inReg32, srcAddr + r * oneRepeatSizeVF);
                } else {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inReg16, srcAddr + r * oneRepeatSizeVF);
                    Reg::Cast<CalType, T, CAST_TRAIT_0>(inReg32, inReg16, mask);
                }

                Reg::LoadAlign<CalType>(resReg, resAddr + r * oneRepeatSizeVF);
                Reg::Add(resReg, inReg32, resReg, mask);
                Reg::StoreAlign<CalType>(resAddr + r * oneRepeatSizeVF, resReg, mask);
            }
        }
        // ---- V→MTE2 同步 ----
        SetEvent<HardEvent::V_MTE2>(idx);
        WaitEvent<HardEvent::V_MTE2>(idx);
    }

    __aicore__ inline void CopyOutputToGM(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, LocalTensor<T> src,
                                          LocalTensor<CalType> res, uint32_t idx, uint64_t outSelfAddr)
    {
        copyInParams_.blockLen = dataLen_ * sizeof(T);
        if constexpr (sizeof(T) != sizeof(float32_t)) {
            uint32_t midUbAddLenVF = dataLen_;

            Cast<T, CalType>(src, res, RoundMode::CAST_RINT, midUbAddLenVF);

            SetEvent<HardEvent::V_MTE3>(idx);
            WaitEvent<HardEvent::V_MTE3>(idx);
            // 同步：MTE3_V
            DataCopyPad(output_gm[outSelfAddr], src[inSrcStart_], copyInParams_);

            SetEvent<HardEvent::MTE3_MTE2>(idx);
        } else {
            // 如果是fp32就可以直接往外搬
            SetEvent<HardEvent::V_MTE3>(idx);
            WaitEvent<HardEvent::V_MTE3>(idx);

            DataCopyPad(output_gm[outSelfAddr], res[inResStart_], copyInParams_);

            SetEvent<HardEvent::MTE3_V>(idx);
        }
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetEvent(uint32_t bufIdx)
    {
        if (bufIdx & 1) {
            SetFlag<EVENT>(EVENT_ID1);
        } else {
            SetFlag<EVENT>(EVENT_ID0);
        }
    }

    template <HardEvent EVENT>
    __aicore__ inline void WaitEvent(uint32_t bufIdx)
    {
        if (bufIdx & 1) {
            WaitFlag<EVENT>(EVENT_ID1);
        } else {
            WaitFlag<EVENT>(EVENT_ID0);
        }
    }
}; // 类

} // namespace PadV3Grad
#endif
