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
 * \file pad_v3_grad_mirror_huge_width.h
 * \brief pad v3 grad mirror huge width.h
 */

#ifndef PAD_V3_GRAD_MIRROR_HUGE_WIDTH_H
#define PAD_V3_GRAD_MIRROR_HUGE_WIDTH_H

#include "kernel_operator.h"
#include "pad_v3_grad_struct.h"

namespace PadV3Grad {
constexpr uint8_t UB_AXIS_DATA_IDX = 4;
constexpr uint8_t LEFT_PAD_IDX = 1;
constexpr uint8_t RIGHT_PAD_IDX = 2;
constexpr uint8_t MIRROR_SHAPE_MULTIPLIER = 2;
using namespace AscendC;

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

template <typename T, uint8_t Mode>
class KernelPadV3GradMirrorHugeWidth {
private:
    uint32_t inResStart_;
    uint32_t inSrcStart_;
    uint32_t inTempStart_;

    struct IdxAndTimes {
        uint64_t inGmIdx[3]{0};
        uint8_t cnt = 1;
    };

    GlobalTensor<T> input_gm;
    GlobalTensor<T> output_gm;

    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> inQueue_;
    TBuf<TPosition::VECCALC> resQueue_;
    TBuf<TPosition::VECCALC> tempQueue_;
    uint32_t blockIdx_;

    const PadV3GradACTilingData* tilingData_ = nullptr;

    uint64_t inIndex_[PAD_GRAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_GRAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t mUbFactor_{0};
    uint8_t mDim_{0};
    uint8_t mUbAxis_{0};

    uint8_t mode_{0};

    uint64_t factorOfmUbAxis_{0};
    uint64_t originRightPadStartIndex_{0};
    uint64_t mDataLen_{0};

    DataCopyExtParams copyInParams_;
    DataCopyPadExtParams<T> PadParams_{false, 0, 0, 0};

    uint32_t leftUbAddLen_{0};
    uint32_t leftUbStartIdx_{0};

    uint32_t rightUbAddLen_{0};
    uint32_t rightUbStartIdx_{0};

    uint16_t oneRepeatSize_{0};

    using RangeType = int32_t;
    using IdxType = uint32_t;
    using CalType = std::conditional_t<std::is_same_v<T, bfloat16_t>, float32_t,
                                       std::conditional_t<std::is_same_v<T, float16_t>, float32_t, T>>;

public:
    __aicore__ inline KernelPadV3GradMirrorHugeWidth(TPipe* pipe, const PadV3GradACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;

        mode_ = (Mode == TPL_MODE_REFLECT); // reflect=1, symmetric=0

        inResStart_ = 0;
        inSrcStart_ = 0;
        inTempStart_ = 0;

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
        pipe_->InitBuffer(tempQueue_, BUFFER_NUM * tilingData_->outTileSize * sizeof(CalType));

        mUbFactor_ = tilingData_->ubFactor;
        mDim_ = tilingData_->dimNum;
        mUbAxis_ = tilingData_->ubAxis;
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdx = blockIdx_ * tilingData_->ubPerCount;
        if (startIdx >= tilingData_->ubTotalCount) {
            return;
        }
        uint32_t endIdx = min(startIdx + tilingData_->ubPerCount, tilingData_->ubTotalCount);
        // 尾轴需要分几次处理完成
        factorOfmUbAxis_ = CeilDiv(tilingData_->outShape[mUbAxis_], mUbFactor_);
        // 右pad在输出中的起始索引
        originRightPadStartIndex_ = tilingData_->outShape[mUbAxis_] - tilingData_->rightPad[mUbAxis_] - mode_;

        // 每个核一次只能处理尾轴的一块数据，以尾轴的一块为单位循环
        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            uint32_t curIdx = idx;
            uint64_t inSelfAddr = 0;
            uint64_t outSelfAddr = 0;

            for (int32_t i = mUbAxis_; i >= 0; i--) {
                uint64_t factor = tilingData_->outShape[i];
                if (i == mUbAxis_) {
                    factor = factorOfmUbAxis_;
                }
                // 在第i维的索引
                outIndex_[i] = (i == mUbAxis_ ? curIdx % factor * mUbFactor_ : curIdx % factor);
                inIndex_[i] = outIndex_[i] + tilingData_->leftPad[i];
                curIdx = curIdx / factor;
                // 本块正文在input中的索引
                inSelfAddr += inIndex_[i] * tilingData_->inStride[i];
                outSelfAddr += outIndex_[i] * tilingData_->outStride[i];
            }
            // 当前块的数据长度
            mDataLen_ = (outIndex_[mUbAxis_] + mUbFactor_ <= tilingData_->outShape[mUbAxis_] ?
                             mUbFactor_ :
                             tilingData_->outShape[mUbAxis_] - outIndex_[mUbAxis_]);
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

        LocalTensor<CalType> tempLocal = tempQueue_.Get<CalType>();
        LocalTensor<CalType> temp = tempLocal[(idx & 1) * tilingData_->outTileSize];

        // 数据拷贝参数
        copyInParams_.blockLen = mDataLen_ * sizeof(T);
        float32_t initVal{0.0f};

        // ub地址
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ CalType* resAddr = (__ubuf__ CalType*)res.GetPhyAddr();
        __ubuf__ CalType* tempAddr = (__ubuf__ CalType*)temp.GetPhyAddr();

        // 同步：V等上一块从res搬出
        if constexpr (sizeof(T) == sizeof(float32_t)) {
            if (idx > 1) {
                WaitEvent<HardEvent::MTE3_V>(idx);
            }
        }

        Duplicate<CalType>(res[inResStart_], initVal, static_cast<int32_t>(mDataLen_));

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

        CopyAndCal(srcAddr, resAddr, tempAddr, src, temp, idx, outSelfAddr, inIdxCnt);
        CopyOutputToGM(srcAddr, resAddr, src, res, idx, outSelfAddr);
    }
    __aicore__ inline void CalculateOffsetParams(IdxAndTimes* inIdxCnt)
    {
        if (tilingData_->leftPad[mUbAxis_] == 0 || outIndex_[mUbAxis_] >= (tilingData_->leftPad[mUbAxis_] + mode_)) {
            leftUbAddLen_ = 0;   // 需要参与加法计算的长度
            leftUbStartIdx_ = 0; // 在输出上参与pad的起始索引
        } else {
            leftUbStartIdx_ = (mode_ && outIndex_[mUbAxis_] == 0);
            leftUbAddLen_ = min(mDataLen_, tilingData_->leftPad[mUbAxis_] + mode_ - outIndex_[mUbAxis_]) -
                            leftUbStartIdx_;
            // 当前块的左pad在输入中的索引
            inIdxCnt[UB_AXIS_DATA_IDX].inGmIdx[LEFT_PAD_IDX] = tilingData_->leftPad[mUbAxis_] + mode_ -
                                                               (outIndex_[mUbAxis_] + leftUbStartIdx_ + leftUbAddLen_);
        }

        if (tilingData_->rightPad[mUbAxis_] == 0 || outIndex_[mUbAxis_] + mDataLen_ <= originRightPadStartIndex_) {
            rightUbAddLen_ = 0;
            rightUbStartIdx_ = 0;
        } else {
            rightUbStartIdx_ = (originRightPadStartIndex_ <= outIndex_[mUbAxis_]) ?
                                   0 :
                                   originRightPadStartIndex_ - outIndex_[mUbAxis_];
            rightUbAddLen_ = mDataLen_ - rightUbStartIdx_ -
                             (mode_ && outIndex_[mUbAxis_] + mDataLen_ == tilingData_->outShape[mUbAxis_]);
            inIdxCnt[UB_AXIS_DATA_IDX].inGmIdx[RIGHT_PAD_IDX] = MIRROR_SHAPE_MULTIPLIER *
                                                                    tilingData_->outShape[mUbAxis_] +
                                                                tilingData_->leftPad[mUbAxis_] -
                                                                (outIndex_[mUbAxis_] + rightUbStartIdx_ +
                                                                 rightUbAddLen_ + mode_);
            // 当前块的右pad在输入中的索引
        }

        for (uint8_t i = 0; i < mDim_ - 1; ++i) {
            // self
            inIdxCnt[i].inGmIdx[0] = inIndex_[i] * tilingData_->inStride[i];
            // left
            if (tilingData_->leftPad[i] != 0 && outIndex_[i] >= mode_ &&
                outIndex_[i] < tilingData_->leftPad[i] + mode_) {
                inIdxCnt[i].inGmIdx[inIdxCnt[i].cnt++] = (tilingData_->leftPad[i] - outIndex_[i] - !mode_) *
                                                         tilingData_->inStride[i]; // symm -1    reflect -0
            }
            // right
            if (tilingData_->rightPad[i] != 0 &&
                tilingData_->outShape[i] - outIndex_[i] - mode_ <= tilingData_->rightPad[i] &&
                tilingData_->outShape[i] - outIndex_[i] - mode_ > 0) {
                inIdxCnt[i].inGmIdx[inIdxCnt[i].cnt++] = (MIRROR_SHAPE_MULTIPLIER * tilingData_->outShape[i] -
                                                          outIndex_[i] + tilingData_->leftPad[i] - 1 - mode_) *
                                                         tilingData_->inStride[i];
            }
        }
    }

    __aicore__ inline void CopyAndCal(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, __ubuf__ CalType* tempAddr,
                                      LocalTensor<T> src, LocalTensor<CalType> temp, uint32_t idx, uint64_t outSelfAddr,
                                      IdxAndTimes* inIdxCnt)
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
                            ProcessLeftData(srcAddr, resAddr, tempAddr, src, temp, idx, a3Offset, inIdxCnt);
                        }

                        // 右边
                        if (rightUbAddLen_ != 0) {
                            ProcessRightData(srcAddr, resAddr, tempAddr, src, temp, idx, a3Offset, inIdxCnt);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessMiddleData(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, LocalTensor<T> src,
                                             uint32_t idx, uint64_t a3Offset)
    {
        uint32_t inMidAddr = a3Offset + inIndex_[mUbAxis_];
        copyInParams_.blockLen = mDataLen_ * sizeof(T);
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
        uint16_t repeatSelfTimes = CeilDiv(mDataLen_, oneRepeatSize_);
        uint32_t midUbAddLenVF = mDataLen_;
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> srcReg;
            Reg::RegTensor<CalType> tempRegB32;
            Reg::RegTensor<CalType> resReg;
            Reg::MaskReg maskReg;

            static constexpr Reg::CastTrait castTrait16to32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

            for (uint16_t k = 0; k < repeatSelfTimes; k++) {
                maskReg = AscendC::Reg::UpdateMask<CalType>(midUbAddLenVF);
                if constexpr (sizeof(T) != sizeof(float32_t)) {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + k * oneRepeatSize_);
                } else {
                    Reg::LoadAlign(srcReg, srcAddr + k * oneRepeatSize_);
                }
                Reg::LoadAlign(resReg, resAddr + k * oneRepeatSize_);
                if constexpr (sizeof(T) != sizeof(float32_t)) {
                    Reg::Cast<CalType, T, castTrait16to32>(tempRegB32, srcReg, maskReg);
                    Reg::Add(resReg, tempRegB32, resReg, maskReg);
                } else {
                    Reg::Add(resReg, srcReg, resReg, maskReg);
                }
                Reg::StoreAlign(resAddr + k * oneRepeatSize_, resReg, maskReg);
            }
        }
    }

    __aicore__ inline void ProcessLeftData(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, __ubuf__ CalType* tempAddr,
                                           LocalTensor<T> src, LocalTensor<CalType> temp, uint32_t idx,
                                           uint64_t a3Offset, IdxAndTimes* inIdxCnt)
    {
        uint64_t inLeftAddr = a3Offset + inIdxCnt[4].inGmIdx[1];
        copyInParams_.blockLen = leftUbAddLen_ * sizeof(T);

        uint16_t leftMainTimes = leftUbAddLen_ / oneRepeatSize_;
        uint16_t leftTailLen = leftUbAddLen_ - leftMainTimes * oneRepeatSize_;
        // 有尾块时，主体只跑前 (repeatLeftTimes-1) 次；无尾块时跑全部
        uint16_t leftTailTimes = (leftTailLen != 0);

        uint32_t leftMainLenVF = oneRepeatSize_;
        uint32_t leftTailLenVF = leftTailLen;

        __ubuf__ CalType* src_l_resAddr = resAddr + leftUbStartIdx_;
        __ubuf__ CalType* dst_l_resAddr = resAddr + leftUbStartIdx_;

        // ---------- MTE2 搬入 ----------
        DataCopyPad(src[inSrcStart_], input_gm[inLeftAddr], copyInParams_, PadParams_);

        SetEvent<HardEvent::MTE2_V>(idx);
        WaitEvent<HardEvent::MTE2_V>(idx);

        // ---------- 类型转换（编译期分支，合法）----------
        if constexpr (sizeof(T) != sizeof(float32_t)) {
            Cast<CalType, T>(temp, src, RoundMode::CAST_NONE, leftUbAddLen_);
        }

        __VEC_SCOPE__
        {
            Reg::UnalignRegForLoad ureg0;
            Reg::UnalignRegForStore ureg1;
            Reg::RegTensor<CalType> tempReg;
            Reg::RegTensor<CalType> resReg;
            Reg::RegTensor<RangeType> idxReg;
            Reg::MaskReg maskRegMain = Reg::UpdateMask<CalType>(leftMainLenVF);
            Reg::MaskReg maskRegTail = Reg::UpdateMask<CalType>(leftTailLenVF);

            for (uint16_t k = 0; k < leftMainTimes; k++) {
                // 1. res → reg
                if constexpr (Mode == 2) {
                    Reg::LoadUnAlignPre(ureg0, src_l_resAddr + k * oneRepeatSize_);
                    Reg::LoadUnAlign(resReg, ureg0, src_l_resAddr + k * oneRepeatSize_);
                } else {
                    Reg::LoadAlign(resReg, resAddr + k * oneRepeatSize_);
                }

                // 2. 逆序 Gather left → reg
                Reg::Arange<RangeType, Reg::IndexOrder::DECREASE_ORDER>(
                    idxReg, (RangeType)((leftUbAddLen_ - 1) - (oneRepeatSize_ - 1) - k * oneRepeatSize_));

                if constexpr (sizeof(T) != sizeof(float32_t)) {
                    Reg::Gather(tempReg, tempAddr, (Reg::RegTensor<IdxType>&)idxReg, maskRegMain);
                } else {
                    Reg::Gather(tempReg, srcAddr, (Reg::RegTensor<IdxType>&)idxReg, maskRegMain);
                }

                // 3. Add
                Reg::Add(resReg, tempReg, resReg, maskRegMain);

                // 4. Store —— 主体块固定用 oneRepeatSize_，无运行时分支
                if constexpr (Mode == 2) {
                    Reg::StoreUnAlign(dst_l_resAddr, resReg, ureg1, oneRepeatSize_);
                } else {
                    Reg::StoreAlign(resAddr + k * oneRepeatSize_, resReg, maskRegMain);
                }
            }
            // 尾块
            for (uint16_t k = 0; k < leftTailTimes; k++) {
                if constexpr (Mode == 2) {
                    Reg::LoadUnAlignPre(ureg0, src_l_resAddr + leftMainTimes * oneRepeatSize_);
                    Reg::LoadUnAlign(resReg, ureg0, src_l_resAddr + leftMainTimes * oneRepeatSize_);
                } else {
                    Reg::LoadAlign(resReg, resAddr + leftMainTimes * oneRepeatSize_);
                }

                // 2. 逆序 Gather（尾块起始索引同样在域外算好）
                Reg::Arange<RangeType, Reg::IndexOrder::DECREASE_ORDER>(
                    idxReg, (RangeType)((leftUbAddLen_ - 1) - (oneRepeatSize_ - 1) - leftMainTimes * oneRepeatSize_));

                if constexpr (sizeof(T) != sizeof(float32_t)) {
                    Reg::Gather(tempReg, tempAddr, (Reg::RegTensor<IdxType>&)idxReg, maskRegTail);
                } else {
                    Reg::Gather(tempReg, srcAddr, (Reg::RegTensor<IdxType>&)idxReg, maskRegTail);
                }

                // 3. Add
                Reg::Add(resReg, tempReg, resReg, maskRegTail);

                // 4. Store —— 尾块用 leftTailLen，值在域外已确定
                if constexpr (Mode == 2) {
                    Reg::StoreUnAlign(dst_l_resAddr, resReg, ureg1, leftTailLen);
                } else {
                    Reg::StoreAlign(resAddr + leftMainTimes * oneRepeatSize_, resReg, maskRegTail);
                }
            }
            if constexpr (Mode == 2) {
                Reg::StoreUnAlignPost(dst_l_resAddr, ureg1, 0);
            }
        }
        // ---------- V → MTE2 反向同步 ----------
        SetEvent<HardEvent::V_MTE2>(idx);
        WaitEvent<HardEvent::V_MTE2>(idx);
    }

    __aicore__ inline void ProcessRightData(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, __ubuf__ CalType* tempAddr,
                                            LocalTensor<T> src, LocalTensor<CalType> temp, uint32_t idx,
                                            uint64_t a3Offset, IdxAndTimes* inIdxCnt)
    {
        uint64_t inRightAddr = a3Offset + inIdxCnt[4].inGmIdx[2];
        copyInParams_.blockLen = rightUbAddLen_ * sizeof(T);

        uint16_t rightMainTimes = rightUbAddLen_ / oneRepeatSize_; // 向下取整
        uint16_t rightTailLen = rightUbAddLen_ - rightMainTimes * oneRepeatSize_;
        uint16_t rightTailTimes = (rightTailLen != 0);
        uint32_t rightMainLenVF = oneRepeatSize_;
        uint32_t rightTailLenVF = rightTailLen;

        __ubuf__ CalType* src_r_resAddr = resAddr + rightUbStartIdx_;
        __ubuf__ CalType* dst_r_resAddr = resAddr + rightUbStartIdx_;

        // ---- MTE2 搬入 ----
        DataCopyPad(src[inSrcStart_], input_gm[inRightAddr], copyInParams_, PadParams_);

        SetEvent<HardEvent::MTE2_V>(idx);
        WaitEvent<HardEvent::MTE2_V>(idx);

        // ---- 类型转换（非 float32 路径）----
        if constexpr (sizeof(T) != sizeof(float32_t)) {
            Cast<CalType, T>(temp, src, RoundMode::CAST_NONE, rightUbAddLen_);
        }

        __VEC_SCOPE__
        {
            Reg::UnalignRegForLoad ureg0;
            Reg::UnalignRegForStore ureg1;

            Reg::RegTensor<CalType> tempReg;
            Reg::RegTensor<CalType> resReg;
            Reg::RegTensor<RangeType> idxReg;
            Reg::MaskReg maskRegMain = Reg::UpdateMask<CalType>(rightMainLenVF);
            Reg::MaskReg maskRegTail = Reg::UpdateMask<CalType>(rightTailLenVF);

            // —— 主体：每块恰好 oneRepeatSize_ 个元素 ——
            for (uint16_t k = 0; k < rightMainTimes; k++) {
                Reg::LoadUnAlignPre(ureg0, src_r_resAddr + k * oneRepeatSize_);
                Reg::LoadUnAlign(resReg, ureg0, src_r_resAddr + k * oneRepeatSize_);

                // DECREASE_ORDER: 第k块对应的反向起始索引
                // 第0块最高索引 = rightUbAddLen_-1，每块递减 oneRepeatSize_
                RangeType mainStartIdx = (RangeType)((rightUbAddLen_ - 1) - (oneRepeatSize_ - 1) - k * oneRepeatSize_);

                Reg::Arange<RangeType, Reg::IndexOrder::DECREASE_ORDER>(idxReg, mainStartIdx);

                if constexpr (sizeof(T) != sizeof(float32_t)) {
                    Reg::Gather(tempReg, tempAddr, (Reg::RegTensor<IdxType>&)idxReg, maskRegMain);
                } else {
                    Reg::Gather(tempReg, srcAddr, (Reg::RegTensor<IdxType>&)idxReg, maskRegMain);
                }
                Reg::Add(resReg, tempReg, resReg, maskRegMain);
                Reg::StoreUnAlign(dst_r_resAddr, resReg, ureg1, oneRepeatSize_);
            }
            for (uint16_t k = 0; k < rightTailTimes; k++) {
                // —— 尾块：仅 rightTailLen 个有效元素 ——
                Reg::LoadUnAlignPre(ureg0, src_r_resAddr + rightMainTimes * oneRepeatSize_);
                Reg::LoadUnAlign(resReg, ureg0, src_r_resAddr + rightMainTimes * oneRepeatSize_);

                // 尾块反向：最高索引 = rightTailLen - 1（从0到rightTailLen-1倒序）
                RangeType tailStartIdx = (RangeType)((rightUbAddLen_ - 1) - (oneRepeatSize_ - 1) -
                                                     rightMainTimes * oneRepeatSize_);

                Reg::Arange<RangeType, Reg::IndexOrder::DECREASE_ORDER>(idxReg, tailStartIdx);
                if constexpr (sizeof(T) != sizeof(float32_t)) {
                    Reg::Gather(tempReg, tempAddr, (Reg::RegTensor<IdxType>&)idxReg, maskRegTail);
                } else {
                    Reg::Gather(tempReg, srcAddr, (Reg::RegTensor<IdxType>&)idxReg, maskRegTail);
                }
                Reg::Add(resReg, tempReg, resReg, maskRegTail);
                Reg::StoreUnAlign(dst_r_resAddr, resReg, ureg1, rightTailLen);
            }
            Reg::StoreUnAlignPost(dst_r_resAddr, ureg1, 0);
        }
        // ---- V→MTE2 同步 ----
        SetEvent<HardEvent::V_MTE2>(idx);
        WaitEvent<HardEvent::V_MTE2>(idx);
    }

    __aicore__ inline void CopyOutputToGM(__ubuf__ T* srcAddr, __ubuf__ CalType* resAddr, LocalTensor<T> src,
                                          LocalTensor<CalType> res, uint32_t idx, uint64_t outSelfAddr)
    {
        copyInParams_.blockLen = mDataLen_ * sizeof(T);
        if constexpr (sizeof(T) != sizeof(float32_t)) {
            uint32_t midUbAddLenVF = mDataLen_;
            uint16_t repeatSelfTimes = CeilDiv(mDataLen_, oneRepeatSize_);

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
