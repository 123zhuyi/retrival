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
 * \file batch_to_space_smallc.h
 * \brief batch_to_space_smallc
 */

#ifndef _BATCH_TO_SPACE_N_D_SMALL_C_H_
#define _BATCH_TO_SPACE_N_D_SMALL_C_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "batch_to_space_nd_tiling_data.h"

namespace B2SND {
using namespace AscendC;
using namespace Ops::Base;

template <typename T, uint8_t BLOCK_DIM_NUM>
class BatchToSpaceSmallC {
private:
    constexpr static uint32_t LOG_OFFSET[]{1, 2, 4, 8, 16};
    constexpr static uint16_t LOG_OFFSET_NUM = static_cast<uint16_t>(sizeof(LOG_OFFSET) / sizeof(LOG_OFFSET[0]));

    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint32_t BLK_ELEMS = Ops::Base::GetUbBlockSize() / sizeof(T);
    constexpr static uint32_t SUB_BASE = uint32_t(4294967296);
    constexpr static int16_t DIV_BASE = 32;
    constexpr static uint8_t MAX_DIMS_NUM = 8;
    constexpr static uint8_t MAX_CROP_NUM = 3;
    constexpr static uint8_t MAX_CROP_DIM_NUM = 2;
    constexpr static uint8_t N_C_NUM = 2;
    constexpr static uint8_t COPYPARAM_IDX = 0;
    constexpr static uint8_t LOOP1PARAM_IDX = 1;
    constexpr static uint8_t LOOP2PARAM_IDX = 2;
    constexpr static uint8_t OUTLOOP1_IDX = 3;
    constexpr static uint8_t OUTLOOP2_IDX = 4;
    constexpr static uint8_t LEFT_CROP = 0;
    constexpr static uint8_t RIGHT_CROP = 1;
    constexpr static uint8_t TWO_DIMENSION = 2;
    constexpr static uint8_t THIRD_DIMENSION = 3;
    constexpr static uint8_t OUTER_STRIDE_AXIS_OFFSET = 2; // 外层stride的轴偏移量(跨2级轴获取stride)
    constexpr static int8_t BS_PIXEL_MAP[4][3] = {{-1, -1, -1}, {2, -1, -1}, {3, 4, -1}, {4, 5, 6}};
    constexpr static uint8_t SHAPE_DIM_NUM = MAX_CROP_DIM_NUM * BLOCK_DIM_NUM + N_C_NUM;
    constexpr static uint32_t UINT64_PROCESS_FACTOR = 2; // uint64类型数据处理时元素计数翻倍因子
    const B2SNDSmallCTilingData* td_ = nullptr;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int64_t blockIdx_;
    uint64_t outShape_[MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t tiledInShape_[MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t tiledOutShape_[MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t cropOffset_[MAX_CROP_NUM][MAX_CROP_DIM_NUM] = {{0, 0}, {0, 0}, {0, 0}};
    bool copyMode_[MAX_CROP_NUM] = {false, false, false};
    uint8_t inUbAxis_{0};
    uint8_t outUbAxis_{0};
    uint32_t inUbFactor_{0};
    uint32_t outUbFactor_{0};
    uint32_t ubTotalCount_{0};
    uint32_t ubPerCount_{0};
    uint8_t outIndexs_{0};
    int8_t indexMap_[MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint8_t outdexMap_[MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint8_t copyInAxis{0};
    uint64_t inIndex_[MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t cropIndex_[MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t srcStride_[MAX_DIMS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t outStride_[MAX_DIMS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t ubInStride_[MAX_DIMS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t ubOutStride_[MAX_DIMS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t ubTileSize_{0};
    bool noCut_{false};
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    using RangeType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using FloatType_ = float;
    using CastType_ = std::conditional_t<sizeof(T) == 1,
                                         std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    uint32_t vlSize_ = static_cast<uint32_t>(Ops::Base::GetVRegSize() / sizeof(CastType_));

public:
    __aicore__ inline BatchToSpaceSmallC() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const B2SNDSmallCTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        td_ = tilingData;
        ubTileSize_ = tilingData->ubTileSize;
        inUbAxis_ = tilingData->inUbAxis;
        outUbAxis_ = tilingData->outUbAxis;
        inUbFactor_ = tilingData->inUbFactor;
        outUbFactor_ = tilingData->outUbFactor;
        ubTotalCount_ = tilingData->ubTotalCount;
        ubPerCount_ = tilingData->ubPerCount;
        if (inUbAxis_ == 0 && td_->oriInShape[0] == inUbFactor_) {
            noCut_ = true;
        }
        pipe_->InitBuffer(inQueue_, BUFFER_NUM, ubTileSize_);
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, ubTileSize_);
        blockIdx_ = GetBlockIdx();
        inputGM_.SetGlobalBuffer((__gm__ T*)x);
        outputGM_.SetGlobalBuffer((__gm__ T*)y);
        CalIndexMap();
        uint64_t srcShapeSize = 1UL;
        uint64_t outShapeSize = 1UL;
        for (int8_t i = SHAPE_DIM_NUM - 1; i >= 0; --i) {
            int8_t outI = indexMap_[i];
            outShape_[i] = td_->oriInShape[outI];
            srcStride_[i] = srcShapeSize;
            srcShapeSize *= td_->oriInShape[i];
            outStride_[i] = outShapeSize;
            outShapeSize *= outShape_[i];
            if (outI > BLOCK_DIM_NUM && outI <= MAX_CROP_DIM_NUM * BLOCK_DIM_NUM) {
                outShapeSize = outShapeSize -
                               (td_->crops[outI - BLOCK_DIM_NUM - 1][0] + td_->crops[outI - BLOCK_DIM_NUM - 1][1]) *
                                   outStride_[i + 1];
                outStride_[i] = (outStride_[i] > outShapeSize) ? outShapeSize : outStride_[i];
            }
        }
        if (!noCut_) {
            GetOuterAxes(inUbAxis_, outUbAxis_, outIndexs_);
        }
        CalcCropIndex(cropIndex_);
    }

    __aicore__ inline void Process()
    {
        uint64_t startIdx = blockIdx_ * ubPerCount_;
        if (startIdx >= ubTotalCount_) {
            return;
        }

        uint64_t endIdx = (blockIdx_ + 1UL) * ubPerCount_;
        endIdx = (endIdx < ubTotalCount_ ? endIdx : ubTotalCount_);

        for (uint64_t idx = startIdx; idx < endIdx; idx++) {
            uint64_t curIdx = idx;
            uint64_t gmStart{0};
            for (uint8_t i = 0; i < MAX_CROP_NUM; ++i) {
                for (uint8_t j = 0; j < MAX_CROP_DIM_NUM; ++j) {
                    cropOffset_[i][j] = 0;
                }
            }
            for (int8_t i = SHAPE_DIM_NUM - 1; i >= 0; --i) {
                uint64_t factor = td_->croppedInShape[i];
                if ((outIndexs_ & (1 << i)) != 0) {
                    if (i == inUbAxis_ || i == outUbAxis_) {
                        factor = Ops::Base::CeilDiv(td_->croppedInShape[i],
                                                    static_cast<uint64_t>(i == inUbAxis_ ? inUbFactor_ : outUbFactor_));
                    }
                    inIndex_[i] = curIdx % factor *
                                  (i == inUbAxis_ ? inUbFactor_ : (i == outUbAxis_ ? outUbFactor_ : 1));
                    curIdx /= factor;
                } else {
                    inIndex_[i] = 0;
                }
                tiledInShape_[i] = CalUbFactor(i);
                gmStart += (inIndex_[i] + cropIndex_[i]) * srcStride_[i];
            }
            // 计算预截取后还需要crop多少
            for (uint8_t i = 0; i < MAX_CROP_NUM; ++i) {
                copyMode_[i] = false;
            }
            copyInAxis = 0;
            for (int8_t i = 0; i < BLOCK_DIM_NUM; ++i) {
                int8_t index1 = i + 1 + BLOCK_DIM_NUM;
                if (td_->croppedInShape[index1] == 1) {
                    continue;
                }
                uint64_t crop_i0 = td_->crops[i][0] % td_->croppedInShape[i];
                uint64_t crop_i1 = td_->crops[i][1] % td_->croppedInShape[i];
                if (inIndex_[index1] == 0) {
                    cropOffset_[i][0] = inIndex_[i] > crop_i0 ?
                                            0 :
                                            (inIndex_[i] + tiledInShape_[i] < crop_i0 ? tiledInShape_[i] :
                                                                                        crop_i0 - inIndex_[i]);
                }
                if (inIndex_[index1] + tiledInShape_[index1] >= td_->croppedInShape[index1]) {
                    cropOffset_[i][1] = inIndex_[i] + tiledInShape_[i] < td_->croppedInShape[i] - crop_i1 ?
                                            0 :
                                            (inIndex_[i] < td_->croppedInShape[i] - crop_i1 ?
                                                 inIndex_[i] + tiledInShape_[i] + crop_i1 - td_->croppedInShape[i] :
                                                 tiledInShape_[i]);
                }
                if ((i == inUbAxis_ || i == outUbAxis_) && !noCut_) {
                    if ((cropOffset_[i][0] > 0) && cropOffset_[i][0] < tiledInShape_[i]) {
                        copyMode_[i] = true;
                    }
                }
            }
            CalUbStride();
            if (ubOutStride_[0] == 0) {
                continue;
            }
            CopyIn(gmStart, idx);
            Compute(idx);
            CopyOut(idx);
        }
    }

    __aicore__ inline void CopyIn(uint64_t gmStart, uint64_t idx)
    {
        LocalTensor<T> src = inQueue_.AllocTensor<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        LoopModeParams loopParams = {1, 1, 0, 0, 0, 0};
        uint64_t outerLoop1 = 1;
        uint64_t outerLoop2 = 1;
        uint64_t outSrcStride1 = 0;
        uint64_t outSrcStride2 = 0;
        uint64_t outDstStride1 = 0;
        uint64_t outDstStride2 = 0;
        uint64_t tempFactor = 1;
        int8_t tempIndex = 0;
        for (int8_t i = SHAPE_DIM_NUM - 1; i >= 0; i--) {
            tempFactor = tempFactor * tiledInShape_[i];
            if ((copyInAxis & (1 << i)) != 0 || i == 0) {
                if (tempIndex == COPYPARAM_IDX) {
                    copyInParams.blockLen = tempFactor * sizeof(T);
                    copyInParams.dstStride = 0;
                    copyInParams.srcStride = (i == 0) ? 0 : (srcStride_[i - 1] - tempFactor) * sizeof(T);
                } else if (tempIndex == LOOP1PARAM_IDX) {
                    if (tiledInShape_[i] > 1 || i == 0 || tempFactor > 1) {
                        copyInParams.blockCount = tempFactor;
                        loopParams.loop1SrcStride = (i == 0) ? 0 : srcStride_[i - 1] * sizeof(T);
                        loopParams.loop1DstStride = (i == 0) ? 0 : ubInStride_[i - 1] * sizeof(T);
                    } else {
                        copyInParams.srcStride = (i == 0) ? 0 : srcStride_[i - 1] * sizeof(T) - copyInParams.blockLen;
                    }
                } else if (tempIndex == LOOP2PARAM_IDX) {
                    if (tiledInShape_[i] > 1 || i == 0 || tempFactor > 1) {
                        loopParams.loop1Size = tempFactor;
                        loopParams.loop2SrcStride = (i == 0) ? 0 : srcStride_[i - 1] * sizeof(T);
                        loopParams.loop2DstStride = (i == 0) ? 0 : ubInStride_[i - 1] * sizeof(T);
                    } else {
                        loopParams.loop1SrcStride = (i == 0) ? 0 : srcStride_[i - 1] * sizeof(T);
                    }
                } else if (tempIndex == OUTLOOP1_IDX) {
                    if (tiledInShape_[i] > 1 || i == 0 || tempFactor > 1) {
                        loopParams.loop2Size = tempFactor;
                        outSrcStride1 = (i == 0) ? 0 : srcStride_[i - 1];
                        outDstStride1 = (i == 0) ? 0 : ubInStride_[i - 1];
                    } else {
                        loopParams.loop2SrcStride = (i == 0) ? 0 : srcStride_[i - 1] * sizeof(T);
                    }
                } else if (tempIndex == OUTLOOP2_IDX) {
                    if (tiledInShape_[i] > 1 || i == 0 || tempFactor > 1) {
                        outerLoop1 = tempFactor;
                        outSrcStride2 = (i == 0) ? 0 : srcStride_[i - 1];
                        outDstStride2 = (i == 0) ? 0 : ubInStride_[i - 1];
                    } else {
                        outSrcStride1 = (i == 0) ? 0 : srcStride_[i - 1];
                    }
                } else if (tempIndex == OUTLOOP2_IDX + 1) {
                    outerLoop2 = tempFactor;
                }
                if (tiledInShape_[i] > 1 || tempIndex == COPYPARAM_IDX || tempFactor > 1) {
                    tempIndex = tempIndex + 1;
                    tempFactor = 1;
                }
            }
        }
        for (uint64_t a = 0; a < outerLoop2; a++) {
            for (uint64_t b = 0; b < outerLoop1; b++) {
                SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
                DataCopyPad<T, PaddingMode::Compact>(src[a * outDstStride2 + b * outDstStride1],
                                                     inputGM_[gmStart + a * outSrcStride2 + b * outSrcStride1],
                                                     copyInParams, padParams_);
                ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            }
        }
        inQueue_.EnQue(src);
    }

    __aicore__ inline void Compute(uint64_t idx)
    {
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        __ubuf__ T* inputAddr = (__ubuf__ T*)input.GetPhyAddr();
        __ubuf__ T* outputAddr = (__ubuf__ T*)output.GetPhyAddr();
        uint32_t tiledProcessSize = ubOutStride_[0] * tiledInShape_[indexMap_[0]];
        uint16_t loopNum = (tiledProcessSize + vlSize_ - 1) / vlSize_;
        uint64_t tiledOutShape[SHAPE_DIM_NUM];
        for (int8_t i = SHAPE_DIM_NUM - 1; i >= 0; --i) {
            tiledOutShape[i] = tiledInShape_[indexMap_[i]];
        }
        VFProcess(inputAddr, outputAddr, loopNum, tiledProcessSize, tiledOutShape);
        outQueue_.EnQue<T>(output);
        inQueue_.FreeTensor(input);
    }

    __aicore__ inline void VFProcess(__ubuf__ T* inputAddr, __ubuf__ T* outputAddr, uint16_t loopNum,
                                     uint32_t tiledProcessSize, uint64_t tiledOutShape[SHAPE_DIM_NUM])
    {
        __ubuf__ T* outputAddrTmp = outputAddr;
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
            tiledProcessSize *= UINT64_PROCESS_FACTOR;
        }
        uint32_t vlSize = vlSize_;
        uint64_t offset0 = cropOffset_[0][0];
        uint64_t offset1;
        uint64_t offset2;
        uint64_t outStride0 = ubOutStride_[0];
        uint64_t outStride1 = ubOutStride_[1];
        uint64_t outStride2 = ubOutStride_[2];
        uint64_t outStride3 = ubOutStride_[3];
        uint64_t outStride4;
        uint64_t outStride5;
        uint64_t outStride6;
        uint64_t outStride7;
        uint16_t bs0Axis = 0;
        uint16_t noBs0Axis = 0;
        uint16_t bs1Axis = 0;
        uint16_t noBs1Axis = 0;
        uint16_t bs2Axis = 0;
        uint16_t noBs2Axis = 0;
        uint64_t shape2 = tiledOutShape[2];
        uint64_t shape4;
        uint64_t shape6;
        uint64_t tiledInStride0 = ubInStride_[indexMap_[0]];
        uint64_t tiledInStride1 = ubInStride_[indexMap_[1]];
        uint64_t tiledInStride2 = ubInStride_[indexMap_[2]];
        uint64_t tiledInStride3 = ubInStride_[indexMap_[3]];
        uint64_t tiledInStride4;
        uint64_t tiledInStride5;
        uint64_t tiledInStride6;
        uint64_t tiledInStride7;
        FloatType_ epsilon = 1e-5f;
        FloatType_ invStride0 = 1.0f / outStride0;
        FloatType_ invStride1 = 1.0f / outStride1;
        FloatType_ invStride2 = 1.0f / outStride2;
        FloatType_ invStride3 = 1.0f / outStride3;
        FloatType_ invStride4;
        FloatType_ invStride5;
        FloatType_ invStride6;
        static constexpr Reg::CastTrait castTrait3 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        static constexpr Reg::CastTrait castTrait4 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait5 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        static constexpr Reg::CastTrait castTrait6 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        int16_t k0;
        int16_t k1;
        int16_t k2;
        int16_t k3;
        int16_t k4;
        int16_t k5;
        int16_t k6;
        IdxType_ m0;
        IdxType_ m1;
        IdxType_ m2;
        IdxType_ m3;
        IdxType_ m4;
        IdxType_ m5;
        IdxType_ m6;

        if constexpr (sizeof(T) > sizeof(uint16_t)) {
            k0 = DIV_BASE + CeilLog2(outStride0);
            m0 = CeilDiv((1UL << k0), outStride0) - SUB_BASE;
            k0 -= DIV_BASE;
            k1 = DIV_BASE + CeilLog2(outStride1);
            m1 = CeilDiv((1UL << k1), outStride1) - SUB_BASE;
            k1 -= DIV_BASE;
            k2 = DIV_BASE + CeilLog2(outStride2);
            m2 = CeilDiv((1UL << k2), outStride2) - SUB_BASE;
            k2 -= DIV_BASE;
            k3 = DIV_BASE + CeilLog2(outStride3);
            m3 = CeilDiv((1UL << k3), outStride3) - SUB_BASE;
            k3 -= DIV_BASE;
        }

        if ((inUbAxis_ == 0 || outUbAxis_ == 0) && !noCut_ && copyMode_[0]) {
            bs0Axis = 1;
        } else if (offset0 > 0) {
            noBs0Axis = 1;
        }
        if constexpr (BLOCK_DIM_NUM >= TWO_DIMENSION) {
            offset1 = cropOffset_[1][0];
            outStride4 = ubOutStride_[4];
            outStride5 = ubOutStride_[5];
            shape4 = tiledOutShape[4];
            tiledInStride4 = ubInStride_[indexMap_[4]];
            tiledInStride5 = ubInStride_[indexMap_[5]];
            invStride4 = 1.0f / outStride4;
            invStride5 = 1.0f / outStride5;

            if ((inUbAxis_ == 1 || outUbAxis_ == 1) && !noCut_ && copyMode_[1]) {
                bs1Axis = 1;
            } else if (offset1 > 0) {
                noBs1Axis = 1;
            }

            if constexpr (sizeof(T) > sizeof(uint16_t)) {
                k4 = DIV_BASE + CeilLog2(outStride4);
                m4 = CeilDiv((1UL << k4), outStride4) - SUB_BASE;
                k4 -= DIV_BASE;
                k5 = DIV_BASE + CeilLog2(outStride5);
                m5 = CeilDiv((1UL << k5), outStride5) - SUB_BASE;
                k5 -= DIV_BASE;
            }
        }
        if constexpr (BLOCK_DIM_NUM == THIRD_DIMENSION) {
            offset2 = cropOffset_[2][0];
            outStride6 = ubOutStride_[6];
            outStride7 = ubOutStride_[7];
            bs2Axis = 0;
            noBs2Axis = 0;
            shape6 = tiledOutShape[6];
            tiledInStride6 = ubInStride_[indexMap_[6]];
            tiledInStride7 = ubInStride_[indexMap_[7]];
            invStride6 = 1.0f / outStride6;

            if ((inUbAxis_ == 2 || outUbAxis_ == 2) && !noCut_ && copyMode_[2]) {
                bs2Axis = 1;
            } else if (offset2 > 0) {
                noBs2Axis = 1;
            }

            if constexpr (sizeof(T) > sizeof(uint16_t)) {
                k6 = DIV_BASE + CeilLog2(outStride6);
                m6 = CeilDiv((1UL << k6), outStride6) - SUB_BASE;
                k6 -= DIV_BASE;
            }
        }

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType_> tmpIdxReg;
            Reg::RegTensor<IdxType_> idxReg;
            Reg::RegTensor<FloatType_> floatReg;
            Reg::RegTensor<RangeType_> temReg;
            Reg::RegTensor<RangeType_> temReg2;
            Reg::RegTensor<IdxType_> outIdx0Reg;
            Reg::RegTensor<IdxType_> outIdx1Reg;
            Reg::RegTensor<IdxType_> outIdx2Reg;
            Reg::RegTensor<IdxType_> outIdx3Reg;
            Reg::RegTensor<IdxType_> outIdx4Reg;
            Reg::RegTensor<IdxType_> outIdx5Reg;
            Reg::RegTensor<IdxType_> outIdx6Reg;
            Reg::RegTensor<IdxType_> outIdx7Reg;
            Reg::RegTensor<IdxType_> outStride0Reg;
            Reg::RegTensor<IdxType_> outStride1Reg;
            Reg::RegTensor<IdxType_> outStride2Reg;
            Reg::RegTensor<IdxType_> outStride3Reg;
            Reg::RegTensor<IdxType_> outStride4Reg;
            Reg::RegTensor<IdxType_> outStride5Reg;
            Reg::RegTensor<IdxType_> outStride6Reg;
            Reg::RegTensor<IdxType_> tmpCalReg;
            Reg::RegTensor<IdxType_> tmpModReg;
            Reg::RegTensor<IdxType_> tmpCompareReg;
            Reg::RegTensor<T> dstReg;
            Reg::RegTensor<T> dstRegT;
            Reg::UnalignReg uReg;
            Reg::MaskReg mask;
            Reg::MaskReg vmask;
            Reg::MaskReg offsetMask;
            Reg::MaskReg allMask;

            if constexpr (sizeof(T) > sizeof(uint16_t)) {
                Reg::Duplicate(outStride0Reg, m0);
                Reg::Duplicate(outStride1Reg, m1);
                Reg::Duplicate(outStride2Reg, m2);
                Reg::Duplicate(outStride3Reg, m3);
            }
            if constexpr (BLOCK_DIM_NUM >= TWO_DIMENSION && sizeof(T) > sizeof(uint16_t)) {
                Reg::Duplicate(outStride4Reg, m4);
                Reg::Duplicate(outStride5Reg, m5);
            }
            if constexpr (BLOCK_DIM_NUM == THIRD_DIMENSION && sizeof(T) > sizeof(uint16_t)) {
                Reg::Duplicate(outStride6Reg, m6);
            }

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                Reg::Arange(tmpIdxReg, loopIdx * vlSize);
                idxReg = (Reg::RegTensor<IdxType_>&)tmpIdxReg;
                mask = Reg::UpdateMask<IdxType_>(tiledProcessSize);
                if constexpr (sizeof(T) == sizeof(uint16_t)) {
                    allMask = Reg::CreateMask<IdxType_, AscendC::Reg::MaskPattern::ALLF>();
                    Reg::MaskInterleave<IdxType_>(allMask, vmask, allMask, mask);
                }

                if constexpr (sizeof(T) > sizeof(uint16_t)) {
                    Reg::Mull(tmpCalReg, tmpCompareReg, idxReg, outStride0Reg, mask);
                    Reg::Add(tmpCalReg, idxReg, tmpCompareReg, mask);
                    Reg::ShiftRights(outIdx0Reg, tmpCalReg, k0, mask);
                } else {
                    Reg::Cast<FloatType_, RangeType_, castTrait4>(floatReg, tmpIdxReg, mask);
                    Reg::Muls(floatReg, floatReg, invStride0, mask);
                    Reg::Adds(floatReg, floatReg, epsilon, mask);
                    Reg::Cast<RangeType_, FloatType_, castTrait3>(temReg, floatReg, mask);
                    Reg::Cast<FloatType_, RangeType_, castTrait6>(floatReg, tmpIdxReg, mask);
                    Reg::Muls(floatReg, floatReg, invStride0, mask);
                    Reg::Adds(floatReg, floatReg, epsilon, mask);
                    Reg::Cast<RangeType_, FloatType_, castTrait5>(temReg2, floatReg, mask);
                    Reg::Select(outIdx0Reg, (Reg::RegTensor<IdxType_>&)temReg2, (Reg::RegTensor<IdxType_>&)temReg,
                                allMask);
                }

                Reg::Muls(tmpCalReg, outIdx0Reg, outStride0, mask);
                Reg::Sub(tmpModReg, idxReg, tmpCalReg, mask);
                if constexpr (sizeof(T) > sizeof(uint16_t)) {
                    Reg::Mull(tmpCalReg, tmpCompareReg, tmpModReg, outStride1Reg, mask);
                    Reg::Add(tmpCalReg, tmpModReg, tmpCompareReg, mask);
                    Reg::ShiftRights(outIdx1Reg, tmpCalReg, k1, mask);
                } else {
                    Reg::Cast<FloatType_, RangeType_, castTrait4>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                  mask);
                    Reg::Muls(floatReg, floatReg, invStride1, mask);
                    Reg::Adds(floatReg, floatReg, epsilon, mask);
                    Reg::Cast<RangeType_, FloatType_, castTrait3>(temReg, floatReg, mask);
                    Reg::Cast<FloatType_, RangeType_, castTrait6>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                  mask);
                    Reg::Muls(floatReg, floatReg, invStride1, mask);
                    Reg::Adds(floatReg, floatReg, epsilon, mask);
                    Reg::Cast<RangeType_, FloatType_, castTrait5>(temReg2, floatReg, mask);
                    Reg::Select(outIdx1Reg, (Reg::RegTensor<IdxType_>&)temReg2, (Reg::RegTensor<IdxType_>&)temReg,
                                allMask);
                }

                Reg::Muls(tmpCalReg, outIdx1Reg, outStride1, mask);
                Reg::Sub(tmpModReg, tmpModReg, tmpCalReg, mask);
                if constexpr (sizeof(T) > sizeof(uint16_t)) {
                    Reg::Mull(tmpCalReg, tmpCompareReg, tmpModReg, outStride2Reg, mask);
                    Reg::Add(tmpCalReg, tmpModReg, tmpCompareReg, mask);
                    Reg::ShiftRights(outIdx2Reg, tmpCalReg, k2, mask);
                } else {
                    Reg::Cast<FloatType_, RangeType_, castTrait4>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                  mask);
                    Reg::Muls(floatReg, floatReg, invStride2, mask);
                    Reg::Adds(floatReg, floatReg, epsilon, mask);
                    Reg::Cast<RangeType_, FloatType_, castTrait3>(temReg, floatReg, mask);
                    Reg::Cast<FloatType_, RangeType_, castTrait6>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                  mask);
                    Reg::Muls(floatReg, floatReg, invStride2, mask);
                    Reg::Adds(floatReg, floatReg, epsilon, mask);
                    Reg::Cast<RangeType_, FloatType_, castTrait5>(temReg2, floatReg, mask);
                    Reg::Select(outIdx2Reg, (Reg::RegTensor<IdxType_>&)temReg2, (Reg::RegTensor<IdxType_>&)temReg,
                                allMask);
                }

                Reg::Muls(tmpCalReg, outIdx2Reg, outStride2, mask);
                if constexpr (BLOCK_DIM_NUM < TWO_DIMENSION) {
                    Reg::Sub(outIdx3Reg, tmpModReg, tmpCalReg, mask);
                }

                if constexpr (BLOCK_DIM_NUM >= TWO_DIMENSION) {
                    Reg::Sub(tmpModReg, tmpModReg, tmpCalReg, mask);
                    if constexpr (sizeof(T) > sizeof(uint16_t)) {
                        Reg::Mull(tmpCalReg, tmpCompareReg, tmpModReg, outStride3Reg, mask);
                        Reg::Add(tmpCalReg, tmpModReg, tmpCompareReg, mask);
                        Reg::ShiftRights(outIdx3Reg, tmpCalReg, k3, mask);
                    } else {
                        Reg::Cast<FloatType_, RangeType_, castTrait4>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                      mask);
                        Reg::Muls(floatReg, floatReg, invStride3, mask);
                        Reg::Adds(floatReg, floatReg, epsilon, mask);
                        Reg::Cast<RangeType_, FloatType_, castTrait3>(temReg, floatReg, mask);
                        Reg::Cast<FloatType_, RangeType_, castTrait6>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                      mask);
                        Reg::Muls(floatReg, floatReg, invStride3, mask);
                        Reg::Adds(floatReg, floatReg, epsilon, mask);
                        Reg::Cast<RangeType_, FloatType_, castTrait5>(temReg2, floatReg, mask);
                        Reg::Select(outIdx3Reg, (Reg::RegTensor<IdxType_>&)temReg2, (Reg::RegTensor<IdxType_>&)temReg,
                                    allMask);
                    }
                    Reg::Muls(tmpCalReg, outIdx3Reg, outStride3, mask);
                    Reg::Sub(tmpModReg, tmpModReg, tmpCalReg, mask);
                    if constexpr (sizeof(T) > sizeof(uint16_t)) {
                        Reg::Mull(tmpCalReg, tmpCompareReg, tmpModReg, outStride4Reg, mask);
                        Reg::Add(tmpCalReg, tmpModReg, tmpCompareReg, mask);
                        Reg::ShiftRights(outIdx4Reg, tmpCalReg, k4, mask);
                    } else {
                        Reg::Cast<FloatType_, RangeType_, castTrait4>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                      mask);
                        Reg::Muls(floatReg, floatReg, invStride4, mask);
                        Reg::Adds(floatReg, floatReg, epsilon, mask);
                        Reg::Cast<RangeType_, FloatType_, castTrait3>(temReg, floatReg, mask);
                        Reg::Cast<FloatType_, RangeType_, castTrait6>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                      mask);
                        Reg::Muls(floatReg, floatReg, invStride4, mask);
                        Reg::Adds(floatReg, floatReg, epsilon, mask);
                        Reg::Cast<RangeType_, FloatType_, castTrait5>(temReg2, floatReg, mask);
                        Reg::Select(outIdx4Reg, (Reg::RegTensor<IdxType_>&)temReg2, (Reg::RegTensor<IdxType_>&)temReg,
                                    allMask);
                    }

                    Reg::Muls(tmpCalReg, outIdx4Reg, outStride4, mask);
                    if constexpr (BLOCK_DIM_NUM < THIRD_DIMENSION) {
                        Reg::Sub(outIdx5Reg, tmpModReg, tmpCalReg, mask);
                    }
                }

                if constexpr (BLOCK_DIM_NUM == THIRD_DIMENSION) {
                    Reg::Sub(tmpModReg, tmpModReg, tmpCalReg, mask);
                    if constexpr (sizeof(T) > sizeof(uint16_t)) {
                        Reg::Mull(tmpCalReg, tmpCompareReg, tmpModReg, outStride5Reg, mask);
                        Reg::Add(tmpCalReg, tmpModReg, tmpCompareReg, mask);
                        Reg::ShiftRights(outIdx5Reg, tmpCalReg, k5, mask);
                    } else {
                        Reg::Cast<FloatType_, RangeType_, castTrait4>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                      mask);
                        Reg::Muls(floatReg, floatReg, invStride5, mask);
                        Reg::Adds(floatReg, floatReg, epsilon, mask);
                        Reg::Cast<RangeType_, FloatType_, castTrait3>(temReg, floatReg, mask);
                        Reg::Cast<FloatType_, RangeType_, castTrait6>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                      mask);
                        Reg::Muls(floatReg, floatReg, invStride5, mask);
                        Reg::Adds(floatReg, floatReg, epsilon, mask);
                        Reg::Cast<RangeType_, FloatType_, castTrait5>(temReg2, floatReg, mask);
                        Reg::Select(outIdx5Reg, (Reg::RegTensor<IdxType_>&)temReg2, (Reg::RegTensor<IdxType_>&)temReg,
                                    allMask);
                    }
                    Reg::Muls(tmpCalReg, outIdx5Reg, outStride5, mask);
                    Reg::Sub(tmpModReg, tmpModReg, tmpCalReg, mask);
                    if constexpr (sizeof(T) > sizeof(uint16_t)) {
                        Reg::Mull(tmpCalReg, tmpCompareReg, tmpModReg, outStride6Reg, mask);
                        Reg::Add(tmpCalReg, tmpModReg, tmpCompareReg, mask);
                        Reg::ShiftRights(outIdx6Reg, tmpCalReg, k6, mask);
                    } else {
                        Reg::Cast<FloatType_, RangeType_, castTrait4>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                      mask);
                        Reg::Muls(floatReg, floatReg, invStride6, mask);
                        Reg::Adds(floatReg, floatReg, epsilon, mask);
                        Reg::Cast<RangeType_, FloatType_, castTrait3>(temReg, floatReg, mask);
                        Reg::Cast<FloatType_, RangeType_, castTrait6>(floatReg, (Reg::RegTensor<RangeType_>&)tmpModReg,
                                                                      mask);
                        Reg::Muls(floatReg, floatReg, invStride6, mask);
                        Reg::Adds(floatReg, floatReg, epsilon, mask);
                        Reg::Cast<RangeType_, FloatType_, castTrait5>(temReg2, floatReg, mask);
                        Reg::Select(outIdx6Reg, (Reg::RegTensor<IdxType_>&)temReg2, (Reg::RegTensor<IdxType_>&)temReg,
                                    allMask);
                    }

                    Reg::Muls(tmpCalReg, outIdx6Reg, outStride6, mask);
                    Reg::Sub(outIdx7Reg, tmpModReg, tmpCalReg, mask);
                }

                for (uint16_t i = 0; i < bs0Axis; ++i) {
                    Reg::Compares(offsetMask, outIdx1Reg, 0, mask);
                    Reg::Adds(tmpModReg, outIdx2Reg, offset0, mask);
                    Reg::Duplicate(idxReg, shape2);
                    IdxType_ value = shape2;
                    Reg::Compares<IdxType_, AscendC::CMPMODE::GE>(vmask, tmpModReg, value, mask);
                    Reg::Sub(idxReg, tmpModReg, idxReg, vmask);
                    Reg::Select(tmpModReg, idxReg, tmpModReg, vmask);
                    Reg::Select(outIdx2Reg, tmpModReg, outIdx2Reg, offsetMask);
                    Reg::Adds(tmpCalReg, outIdx1Reg, 1, vmask);
                    Reg::Select(tmpCalReg, tmpCalReg, outIdx1Reg, vmask);
                    Reg::Select(outIdx1Reg, tmpCalReg, outIdx1Reg, offsetMask);
                }
                for (uint16_t i = 0; i < noBs0Axis; ++i) {
                    Reg::Adds(tmpModReg, outIdx2Reg, offset0, mask);
                    Reg::Duplicate(idxReg, shape2);
                    IdxType_ value = shape2;
                    Reg::Compares<IdxType_, AscendC::CMPMODE::GE>(vmask, tmpModReg, value, mask);
                    Reg::Sub(idxReg, tmpModReg, idxReg, vmask);
                    Reg::Select(outIdx2Reg, idxReg, tmpModReg, vmask);
                    Reg::Adds(tmpCalReg, outIdx1Reg, 1, vmask);
                    Reg::Select(outIdx1Reg, tmpCalReg, outIdx1Reg, vmask);
                }

                if constexpr (BLOCK_DIM_NUM >= TWO_DIMENSION) {
                    for (uint16_t i = 0; i < bs1Axis; ++i) {
                        Reg::Compares(offsetMask, outIdx3Reg, 0, mask);
                        Reg::Adds(tmpModReg, outIdx4Reg, offset1, mask);
                        Reg::Duplicate(idxReg, shape4);
                        IdxType_ value = shape4;
                        Reg::Compares<IdxType_, AscendC::CMPMODE::GE>(vmask, tmpModReg, value, mask);
                        Reg::Sub(idxReg, tmpModReg, idxReg, vmask);
                        Reg::Select(tmpModReg, idxReg, tmpModReg, vmask);
                        Reg::Select(outIdx4Reg, tmpModReg, outIdx4Reg, offsetMask);
                        Reg::Adds(tmpCalReg, outIdx3Reg, 1, vmask);
                        Reg::Select(tmpCalReg, tmpCalReg, outIdx3Reg, vmask);
                        Reg::Select(outIdx3Reg, tmpCalReg, outIdx3Reg, offsetMask);
                    }
                    for (uint16_t i = 0; i < noBs1Axis; ++i) {
                        Reg::Adds(tmpModReg, outIdx4Reg, offset1, mask);
                        Reg::Duplicate(idxReg, shape4);
                        IdxType_ value = shape4;
                        Reg::Compares<IdxType_, AscendC::CMPMODE::GE>(vmask, tmpModReg, value, mask);
                        Reg::Sub(idxReg, tmpModReg, idxReg, vmask);
                        Reg::Select(outIdx4Reg, idxReg, tmpModReg, vmask);
                        Reg::Adds(tmpCalReg, outIdx3Reg, 1, vmask);
                        Reg::Select(outIdx3Reg, tmpCalReg, outIdx3Reg, vmask);
                    }
                }

                if constexpr (BLOCK_DIM_NUM == THIRD_DIMENSION) {
                    for (uint16_t i = 0; i < bs2Axis; ++i) {
                        Reg::Compares(offsetMask, outIdx5Reg, 0, mask);
                        Reg::Adds(tmpModReg, outIdx6Reg, offset2, mask);
                        Reg::Duplicate(idxReg, shape6);
                        IdxType_ value = shape6;
                        Reg::Compares<IdxType_, AscendC::CMPMODE::GE>(vmask, tmpModReg, value, mask);
                        Reg::Sub(idxReg, tmpModReg, idxReg, vmask);
                        Reg::Select(tmpModReg, idxReg, tmpModReg, vmask);
                        Reg::Select(outIdx6Reg, tmpModReg, outIdx6Reg, offsetMask);
                        Reg::Adds(tmpCalReg, outIdx5Reg, 1, vmask);
                        Reg::Select(tmpCalReg, tmpCalReg, outIdx5Reg, vmask);
                        Reg::Select(outIdx5Reg, tmpCalReg, outIdx5Reg, offsetMask);
                    }
                    for (uint16_t i = 0; i < noBs2Axis; ++i) {
                        Reg::Adds(tmpModReg, outIdx6Reg, offset2, mask);
                        Reg::Duplicate(idxReg, shape6);
                        IdxType_ value = shape6;
                        Reg::Compares<IdxType_, AscendC::CMPMODE::GE>(vmask, tmpModReg, value, mask);
                        Reg::Sub(idxReg, tmpModReg, idxReg, vmask);
                        Reg::Select(outIdx6Reg, idxReg, tmpModReg, vmask);
                        Reg::Adds(tmpCalReg, outIdx5Reg, 1, vmask);
                        Reg::Select(outIdx5Reg, tmpCalReg, outIdx5Reg, vmask);
                    }
                }

                // outAddr = i.N * s.N + ... + n.C
                Reg::Duplicate(idxReg, 0);
                Reg::Muls(tmpCalReg, outIdx0Reg, tiledInStride0, mask);
                Reg::Add(idxReg, idxReg, tmpCalReg, mask);
                Reg::Muls(tmpCalReg, outIdx1Reg, tiledInStride1, mask);
                Reg::Add(idxReg, idxReg, tmpCalReg, mask);
                Reg::Muls(tmpCalReg, outIdx2Reg, tiledInStride2, mask);
                Reg::Add(idxReg, idxReg, tmpCalReg, mask);
                Reg::Muls(tmpCalReg, outIdx3Reg, tiledInStride3, mask);
                Reg::Add(idxReg, idxReg, tmpCalReg, mask);
                if constexpr (BLOCK_DIM_NUM >= TWO_DIMENSION) {
                    Reg::Muls(tmpCalReg, outIdx4Reg, tiledInStride4, mask);
                    Reg::Add(idxReg, idxReg, tmpCalReg, mask);
                    Reg::Muls(tmpCalReg, outIdx5Reg, tiledInStride5, mask);
                    Reg::Add(idxReg, idxReg, tmpCalReg, mask);
                }
                if constexpr (BLOCK_DIM_NUM == THIRD_DIMENSION) {
                    Reg::Muls(tmpCalReg, outIdx6Reg, tiledInStride6, mask);
                    Reg::Add(idxReg, idxReg, tmpCalReg, mask);
                    Reg::Muls(tmpCalReg, outIdx7Reg, tiledInStride7, mask);
                    Reg::Add(idxReg, idxReg, tmpCalReg, mask);
                }

                // gather
                Reg::Gather((Reg::RegTensor<CastType_>&)dstReg, inputAddr, idxReg, mask);
                outputAddrTmp = outputAddr + loopIdx * vlSize;
                if constexpr (sizeof(T) == 1) {
                    Reg::Pack(dstRegT, (Reg::RegTensor<CastType_>&)dstReg);
                    Reg::StoreUnAlign(outputAddrTmp, dstRegT, uReg, vlSize);
                } else {
                    Reg::StoreUnAlign(outputAddrTmp, dstReg, uReg, vlSize);
                }
            }
            Reg::StoreUnAlignPost(outputAddrTmp, uReg, 0);
        }
    }

    __aicore__ inline void CopyOut(uint64_t idx)
    {
        LocalTensor<T> dst = outQueue_.DeQue<T>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyExtParams copyOutParamsL;
        copyOutParamsL.blockCount = 1;
        copyOutParamsL.srcStride = 0;
        copyOutParamsL.dstStride = 0;
        DataCopyExtParams copyOutParamsF;
        copyOutParamsF.blockCount = 1;
        copyOutParamsF.srcStride = 0;
        copyOutParamsF.dstStride = 0;
        LoopModeParams loopParams = {1, 1, 0, 0, 0, 0};
        uint64_t outAddr = 0;
        uint64_t tempOffset = 0;
        int8_t axis1 = 0;
        int8_t axis2 = 0;
        for (int8_t i = MAX_CROP_DIM_NUM * BLOCK_DIM_NUM + 1; i >= 0; i--) {
            int8_t outI = indexMap_[i];
            if (outI < BLOCK_DIM_NUM) {
                tempOffset = (inIndex_[outI + BLOCK_DIM_NUM + 1] + cropIndex_[outI + BLOCK_DIM_NUM + 1]) *
                                 td_->oriInShape[outI] +
                             inIndex_[outI] + cropIndex_[outI];
                tempOffset = (tempOffset > td_->crops[outI][0]) ?
                                 tempOffset - td_->crops[outI][0] :
                             (tempOffset + tiledInShape_[outI] <= td_->crops[outI][0]) ?
                                 td_->croppedInShape[outI] - td_->crops[outI][0] % td_->croppedInShape[outI] +
                                     inIndex_[outI] :
                                 0;
                outAddr = outAddr + tempOffset * outStride_[i];
            } else if (outI == BLOCK_DIM_NUM) {
                outAddr = outAddr + inIndex_[outI] * outStride_[i];
            }
            if ((outI == outUbAxis_ || outI == inUbAxis_) && !noCut_) {
                if (axis1 == 0) {
                    axis1 = i;
                } else {
                    axis2 = i;
                }
            }
        }
        int8_t tempAxis = FindOuterIndex(axis2, axis1);
        if (indexMap_[axis1] >= BLOCK_DIM_NUM) {
            copyOutParams.blockLen = tiledInShape_[indexMap_[axis1]] * ubOutStride_[axis1] * sizeof(T);
            if (axis1 > 0) {
                copyOutParams.blockLen = (tiledInShape_[indexMap_[axis1]] * tiledInShape_[indexMap_[axis1 + 1]] -
                                          cropOffset_[indexMap_[axis1 + 1]][0] - cropOffset_[indexMap_[axis1 + 1]][1]) *
                                         ubOutStride_[axis1 + 1] * sizeof(T);
            }
            copyOutParams.srcStride = axis1 == 0 ?
                                          0 :
                                          (ubOutStride_[axis1 - 1] - copyOutParams.blockLen / sizeof(T)) / BLK_ELEMS;
            copyOutParams.dstStride = axis1 == 0 ?
                                          0 :
                                          (outStride_[axis1 - 1] - copyOutParams.blockLen / sizeof(T)) * sizeof(T);
            if (tempAxis > 0) {
                copyOutParams.dstStride = (outStride_[tempAxis - 1] - copyOutParams.blockLen / sizeof(T)) * sizeof(T);
            }
            if (axis2 == 0) {
                copyOutParams.blockCount = axis1 == 0 ?
                                               1 :
                                               ubOutStride_[0] * tiledInShape_[indexMap_[0]] / ubOutStride_[axis1 - 1];
                DataCopyPad(outputGM_[outAddr], dst, copyOutParams);
            } else {
                tempAxis = FindOuterIndex(0, axis2 - 1);
                int8_t hasFirst = (cropOffset_[indexMap_[axis2]][0] > 0 &&
                                   tiledInShape_[indexMap_[axis2]] > cropOffset_[indexMap_[axis2]][0]) ?
                                      1 :
                                      0;
                int8_t hasLast = (cropOffset_[indexMap_[axis2]][1] > 0 &&
                                  tiledInShape_[indexMap_[axis2]] > cropOffset_[indexMap_[axis2]][1]) ?
                                     1 :
                                     0;
                loopParams.loop1SrcStride = ubOutStride_[axis2 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                loopParams.loop1DstStride = tempAxis > 0 ? outStride_[tempAxis - 1] * sizeof(T) :
                                                           outStride_[axis2 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                uint64_t tempBlockCount = ubOutStride_[axis2] / ubOutStride_[axis1 - 1];
                uint64_t tempLoopSize = ubOutStride_[0] * tiledInShape_[indexMap_[0]] /
                                        ubOutStride_[axis2 - OUTER_STRIDE_AXIS_OFFSET];
                if (hasFirst) {
                    copyOutParams.blockCount = (tiledInShape_[indexMap_[axis2]] - cropOffset_[indexMap_[axis2]][0]) *
                                               tempBlockCount;
                    loopParams.loop1Size = tempLoopSize;
                    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                    DataCopyPad(outputGM_[outAddr], dst, copyOutParams);
                    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
                }
                uint64_t inOffset = hasFirst ? ubOutStride_[axis2 - 1] : 0;
                uint64_t outOffset = hasFirst ?
                                         (td_->croppedInShape[indexMap_[axis2]] - cropOffset_[indexMap_[axis2]][0]) *
                                             outStride_[axis2] :
                                         0;
                if (hasLast) {
                    uint64_t inOffsetL = ubOutStride_[axis2 - OUTER_STRIDE_AXIS_OFFSET] - ubOutStride_[axis2 - 1];
                    uint64_t outOffsetL = outOffset + loopParams.loop1Size / tempLoopSize * outStride_[axis2 - 1];
                    copyOutParams.blockCount = (tiledInShape_[indexMap_[axis2]] - cropOffset_[indexMap_[axis2]][1]) *
                                               tempBlockCount;
                    loopParams.loop1Size = tempLoopSize;
                    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                    DataCopyPad(outputGM_[outAddr + outOffsetL], dst[inOffsetL], copyOutParams);
                    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
                }
                copyOutParams.blockCount = tiledInShape_[indexMap_[axis2]] * tempBlockCount;
                loopParams.loop1SrcStride = ubOutStride_[axis2 - 1] * sizeof(T);
                loopParams.loop1DstStride = outStride_[axis2 - 1] * sizeof(T);
                loopParams.loop1Size = tiledInShape_[indexMap_[axis2 - 1]] -
                                       (cropOffset_[indexMap_[axis2]][0] > 0 ? 1 : 0) -
                                       (cropOffset_[indexMap_[axis2]][1] > 0 ? 1 : 0);
                loopParams.loop2SrcStride = ubOutStride_[axis2 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                loopParams.loop2DstStride = tempAxis > 0 ? outStride_[tempAxis - 1] * sizeof(T) :
                                                           outStride_[axis2 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                loopParams.loop2Size = tempLoopSize;
                SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                DataCopyPad(outputGM_[outAddr + outOffset], dst[inOffset], copyOutParams);
                ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
            }
        } else {
            int8_t hasFirst = cropOffset_[indexMap_[axis1]][0] > 0 ? 1 : 0;
            int8_t hasLast = cropOffset_[indexMap_[axis1]][1] > 0 ? 1 : 0;
            copyOutParams.blockLen = tiledInShape_[indexMap_[axis1]] * ubOutStride_[axis1] * sizeof(T);
            copyOutParams.srcStride = (ubOutStride_[axis1 - 1] - copyOutParams.blockLen / sizeof(T)) / BLK_ELEMS;
            copyOutParams.dstStride = (outStride_[axis1 - 1] - copyOutParams.blockLen / sizeof(T)) * sizeof(T);
            copyOutParams.blockCount = tiledInShape_[indexMap_[axis1 - 1]] - hasFirst - hasLast;
            copyOutParamsF.blockLen = ubOutStride_[axis1] *
                                      (tiledInShape_[indexMap_[axis1]] - cropOffset_[indexMap_[axis1]][0]) * sizeof(T);
            copyOutParamsF.srcStride = (ubOutStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] -
                                        copyOutParamsF.blockLen / sizeof(T)) /
                                       BLK_ELEMS;
            copyOutParamsF.dstStride = (outStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] -
                                        copyOutParamsF.blockLen / sizeof(T)) *
                                       sizeof(T);
            copyOutParamsL.blockLen = ubOutStride_[axis1] *
                                      (tiledInShape_[indexMap_[axis1]] - cropOffset_[indexMap_[axis1]][1]) * sizeof(T);
            copyOutParamsL.srcStride = (ubOutStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] -
                                        copyOutParamsL.blockLen / sizeof(T)) /
                                       BLK_ELEMS;
            copyOutParamsL.dstStride = (outStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] -
                                        copyOutParamsL.blockLen / sizeof(T)) *
                                       sizeof(T);
            tempAxis = FindOuterIndex(axis2, axis1 - 1);
            if (tempAxis > 0) {
                copyOutParamsF.dstStride = (outStride_[tempAxis - 1] - copyOutParamsF.blockLen / sizeof(T)) * sizeof(T);
                copyOutParamsL.dstStride = (outStride_[tempAxis - 1] - copyOutParamsL.blockLen / sizeof(T)) * sizeof(T);
            }
            hasFirst = (hasFirst > 0 && copyOutParamsF.blockLen > 0) ? 1 : 0;
            hasLast = (hasLast > 0 && copyOutParamsL.blockLen > 0) ? 1 : 0;
            uint64_t inOffset = hasFirst ? ubOutStride_[axis1 - 1] : 0;
            uint64_t outOffset = hasFirst ? (td_->croppedInShape[indexMap_[axis1]] - cropOffset_[indexMap_[axis1]][0]) *
                                                outStride_[axis1] :
                                            0;
            uint64_t inOffsetL = inOffset + copyOutParams.blockCount * ubOutStride_[axis1 - 1];
            uint64_t outOffsetL = outOffset + copyOutParams.blockCount * outStride_[axis1 - 1];
            const int8_t b0Index = 2;
            if (axis1 - axis2 > OUTLOOP1_IDX && tempAxis >= b0Index) {
                uint64_t tempFactor = ubOutStride_[b0Index - 1] / ubOutStride_[tempAxis - 1];
                loopParams.loop1SrcStride = ubOutStride_[b0Index - 1] * sizeof(T);
                loopParams.loop1DstStride = outStride_[b0Index - 1] * sizeof(T);
                loopParams.loop1Size = ubOutStride_[0] / ubOutStride_[b0Index - 1];
                loopParams.loop2SrcStride = ubOutStride_[0] * sizeof(T);
                loopParams.loop2DstStride = outStride_[0] * sizeof(T);
                loopParams.loop2Size = tiledInShape_[indexMap_[0]];
                if (hasFirst) {
                    copyOutParamsF.blockCount = copyOutParamsF.blockCount * tempFactor;
                    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                    DataCopyPad(outputGM_[outAddr], dst, copyOutParamsF);
                    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
                }
                if (hasLast) {
                    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                    copyOutParamsL.blockCount = copyOutParamsL.blockCount * tempFactor;
                    DataCopyPad(outputGM_[outAddr + outOffsetL], dst[inOffsetL], copyOutParamsL);
                    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
                }
                uint64_t outSize = loopParams.loop2Size;
                uint64_t outSrcStride = loopParams.loop2SrcStride / sizeof(T);
                uint64_t outDstStride = loopParams.loop2DstStride / sizeof(T);
                loopParams.loop2Size = loopParams.loop1Size;
                loopParams.loop2SrcStride = loopParams.loop1SrcStride;
                loopParams.loop2DstStride = loopParams.loop1DstStride;
                loopParams.loop1Size = tempFactor;
                loopParams.loop1SrcStride = ubOutStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                loopParams.loop1DstStride = tempAxis > 0 ? outStride_[tempAxis - 1] * sizeof(T) :
                                                           outStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                for (auto a = 0; a < outSize; a++) {
                    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                    DataCopyPad(outputGM_[outAddr + outOffset + a * outDstStride], dst[inOffset + a * outSrcStride],
                                copyOutParams);
                    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
                }
            } else if (axis2 == 0) {
                uint64_t tempFactor = ubOutStride_[0] * tiledInShape_[indexMap_[0]] /
                                      ubOutStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET];
                if (hasFirst) {
                    copyOutParamsF.blockCount = copyOutParamsF.blockCount * tempFactor;
                    DataCopyPad(outputGM_[outAddr], dst, copyOutParamsF);
                }
                loopParams.loop1SrcStride = ubOutStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                loopParams.loop1DstStride = tempAxis > 0 ? outStride_[tempAxis - 1] * sizeof(T) :
                                                           outStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                loopParams.loop1Size = tempFactor;
                SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                DataCopyPad(outputGM_[outAddr + outOffset], dst[inOffset], copyOutParams);
                ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
                if (hasLast) {
                    copyOutParamsL.blockCount = copyOutParamsL.blockCount * tempFactor;
                    DataCopyPad(outputGM_[outAddr + outOffsetL], dst[inOffsetL], copyOutParamsL);
                }
            } else {
                int8_t tempAxis2 = FindOuterIndex(0, axis2);
                uint64_t tempFactor = ubOutStride_[axis2 - 1] / ubOutStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET];
                loopParams.loop1SrcStride = ubOutStride_[axis2 - 1] * sizeof(T);
                loopParams.loop1DstStride = (tempAxis2 > 0) ? outStride_[tempAxis2 - 1] * sizeof(T) :
                                                              outStride_[axis2 - 1] * sizeof(T);
                loopParams.loop1Size = ubOutStride_[0] * tiledInShape_[indexMap_[0]] / ubOutStride_[axis2 - 1];
                SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                if (hasFirst) {
                    copyOutParamsF.blockCount = copyOutParamsF.blockCount * tempFactor;
                    DataCopyPad(outputGM_[outAddr], dst, copyOutParamsF);
                }
                if (hasLast) {
                    copyOutParamsL.blockCount = copyOutParamsL.blockCount * tempFactor;
                    DataCopyPad(outputGM_[outAddr + outOffsetL], dst[inOffsetL], copyOutParamsL);
                }
                ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
                loopParams.loop1SrcStride = ubOutStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                loopParams.loop1DstStride = tempAxis > 0 ? outStride_[tempAxis - 1] * sizeof(T) :
                                                           outStride_[axis1 - OUTER_STRIDE_AXIS_OFFSET] * sizeof(T);
                loopParams.loop1Size = tempFactor;
                loopParams.loop2SrcStride = ubOutStride_[axis2 - 1] * sizeof(T);
                loopParams.loop2DstStride = tempAxis2 > 0 ? outStride_[tempAxis2 - 1] * sizeof(T) :
                                                            outStride_[axis2 - 1] * sizeof(T);
                loopParams.loop2Size = ubOutStride_[0] * tiledInShape_[indexMap_[0]] / ubOutStride_[axis2 - 1];
                SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                DataCopyPad(outputGM_[outAddr + outOffset], dst[inOffset], copyOutParams);
                ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
            }
        }
        outQueue_.FreeTensor(dst);
    }

    // 计算公共外轴
    __aicore__ inline void GetOuterAxes(uint8_t inUbAxis, uint8_t outUbAxis, uint8_t& outerAxes)
    {
        outerAxes = 0;
        outerAxes |= (1 << inUbAxis);
        outerAxes |= (1 << outUbAxis);

        if (inUbAxis >= SHAPE_DIM_NUM || outUbAxis >= SHAPE_DIM_NUM) {
            return;
        }

        for (uint8_t axisIdx = 0; axisIdx < outdexMap_[outUbAxis]; ++axisIdx) {
            if (indexMap_[axisIdx] < inUbAxis) {
                outerAxes |= (1 << indexMap_[axisIdx]);
            }
        }
    }

    // 计算转置前张量索引和转后张量索引的映射
    __aicore__ inline void CalIndexMap()
    {
        uint8_t outputIdx = 0;
        indexMap_[outputIdx++] = BLOCK_DIM_NUM;
        for (uint8_t i = 0; i < BLOCK_DIM_NUM; ++i) {
            indexMap_[outputIdx++] = BLOCK_DIM_NUM + i + 1;
            indexMap_[outputIdx++] = i;
        }
        indexMap_[outputIdx++] = SHAPE_DIM_NUM - 1;
        for (uint8_t i = 0; i < outputIdx; ++i) {
            outdexMap_[indexMap_[i]] = i;
        }
    }

    __aicore__ inline void CalUbStride()
    {
        uint64_t curInStride = 1;
        uint64_t curOutStride = 1;
        uint64_t blockLen = 0;
        for (int8_t i = SHAPE_DIM_NUM - 1; i >= 0; i--) {
            ubInStride_[i] = curInStride;
            ubOutStride_[i] = curOutStride;
            if (i == 0) {
                break;
            }
            int8_t outI = indexMap_[i];
            curInStride = curInStride * tiledInShape_[i];
            if (tiledInShape_[i] != td_->oriInShape[i]) {
                copyInAxis |= (1 << i);
                if (blockLen > 0 && (tiledInShape_[i] > 1 || curInStride > blockLen)) {
                    curInStride = Ops::Base::CeilAlign(curInStride, static_cast<uint64_t>(BLK_ELEMS));
                }
                blockLen = curInStride;
            }
            curOutStride = curOutStride * tiledInShape_[outI];
            // W/H/D轴需处理crop
            if (outI > BLOCK_DIM_NUM && outI < SHAPE_DIM_NUM - 1) {
                int8_t bsI = outI - BLOCK_DIM_NUM - 1;
                bool flag = (bsI == inUbAxis_ || bsI == outUbAxis_) && !noCut_ && cropOffset_[bsI][1] > 0 &&
                            cropOffset_[bsI][1] < tiledInShape_[bsI];
                if (!copyMode_[bsI] && !flag) {
                    uint64_t cropOff = (cropOffset_[bsI][0] + cropOffset_[bsI][1]) * ubOutStride_[i] /
                                       tiledInShape_[indexMap_[i + 1]];
                    curOutStride = curOutStride > cropOff ? curOutStride - cropOff : 0;
                    ubOutStride_[i] = (ubOutStride_[i] > curOutStride) ? curOutStride : ubOutStride_[i];
                }
            }
            if ((outI == outUbAxis_ || outI == inUbAxis_) && !noCut_) {
                curOutStride = Ops::Base::CeilAlign(curOutStride, static_cast<uint64_t>(BLK_ELEMS));
            }
        }
    }

    __aicore__ inline uint32_t CalUbFactor(int8_t index)
    {
        uint32_t res = 1;
        if ((outIndexs_ & (1 << index)) != 0) {
            if (index == inUbAxis_) {
                res = inIndex_[index] + inUbFactor_ > td_->croppedInShape[index] ?
                          td_->croppedInShape[index] - inIndex_[index] :
                          inUbFactor_;
            } else if (index == outUbAxis_) {
                res = inIndex_[index] + outUbFactor_ > td_->croppedInShape[index] ?
                          td_->croppedInShape[index] - inIndex_[index] :
                          outUbFactor_;
            }
        } else {
            res = td_->croppedInShape[index];
        }
        return res;
    }

    // 计算输入shape截取数据后的偏移
    __aicore__ inline void CalcCropIndex(uint64_t result[MAX_DIMS_NUM])
    {
        for (uint8_t dim = 0; dim < BLOCK_DIM_NUM; ++dim) {
            int8_t pixelDim = BS_PIXEL_MAP[BLOCK_DIM_NUM][dim];
            result[pixelDim] = td_->crops[dim][0] / td_->oriInShape[dim];
            if (td_->crops[dim][0] / td_->oriInShape[dim] ==
                (td_->oriInShape[pixelDim] * td_->oriInShape[dim] - td_->crops[dim][1] - 1) / td_->oriInShape[dim]) {
                result[dim] = td_->crops[dim][0] % td_->oriInShape[dim];
            }
        }
    }

    __aicore__ inline int8_t FindOuterIndex(int8_t axis1, int8_t axis2)
    {
        int8_t res = -1;
        for (int8_t i = axis2 - 1; i > axis1; i--) {
            if (ubOutStride_[i - 1] > ubOutStride_[axis2 - 1]) {
                break;
            }
            if ((outIndexs_ & (1 << indexMap_[i])) != 0) {
                res = i;
            }
        }
        return res;
    }

    __aicore__ inline uint32_t CeilLog2(uint32_t input)
    {
        input--;
#pragma unroll
        for (uint16_t i = 0; i < LOG_OFFSET_NUM; ++i) {
            input |= input >> LOG_OFFSET[i];
        }
        input++;
        uint32_t res = 0;
        while (input >>= 1) {
            res++;
        }
        return res;
    }
};

} // namespace B2SND

#endif // _BATCH_TO_SPACE_N_D_SMALL_C_H_
