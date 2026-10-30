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
 * \file reduce_nansum_operator.h
 * \brief Custom ReduceNansumOp that integrates NaN->0 replacement inside the reduce pipeline.
 *
 * By inheriting ReduceSumOp and overriding PadValue, the NaN detection and replacement
 * (Compare(x, x, EQ) + Select(mask, x, 0)) is performed on each UB data chunk after
 * data copy and padding, right before the reduction computation.
 *
 * 注意: NaN->0 不能放在 CopyIn 中执行，因为 reduce 框架的 CopyInAux 在调用
 * reduceOp_->CopyIn() 之后，还会无条件调用 CopyInWithMoveAlign/CopyInWithNddma
 * 再次从 GM 搬数据到 UB，会覆盖 CopyIn 中对 UB 数据的修改。
 * PadValue 的调用时机在数据搬运和 padding 完成后、reduce 计算前，此时数据
 * 已在 UB 中且不会再被覆盖，是执行 NaN->0 替换的正确位置。
 *
 * This eliminates the need for separate Compare, Select, and Duplicate DAG nodes,
 * avoiding buffer conflict issues caused by multi-consumer patterns in the DAG pipeline.
 */

#ifndef REDUCE_NANSUM_OPERATOR_H
#define REDUCE_NANSUM_OPERATOR_H
#ifdef __CCE_AICORE__
#include "atvoss/reduce/reduce_operator.h"
#include "atvoss/reduce/reduce_util.h"
#endif

namespace AscendC {
namespace ReduceNansumVec {
using namespace Ops::Base::ReduceOpTmpl;
using namespace Ops::Base::Vec;
using namespace Ops::Base;

template <typename PromteT>
class ReduceNansumOp : public ReduceSumOp<PromteT> {
public:
    __aicore__ inline ReduceNansumOp() {}

#ifdef __CCE_AICORE__
private:
    template <typename T, class V>
    __aicore__ inline void SetCopyInAndLoopParams(DataCopyExtParams& copyInParams, LoopModeParams& loopParams,
                                                  const V& view, bool isAxisA)
    {
        copyInParams.blockCount = isAxisA ? 1 : view.axis[CONST1].repeat;
        copyInParams.blockLen = view.axis[CONST0].repeat * sizeof(T);
        copyInParams.srcStride = (view.axis[CONST1].srcStride - view.axis[CONST0].repeat) * sizeof(T);
        copyInParams.dstStride = (view.axis[CONST1].dstStride - view.axis[CONST0].repeat) * sizeof(T) /
                                 GetUbBlockSize();
        loopParams.loop1Size = view.axis[CONST2].repeat;
        loopParams.loop1SrcStride = view.axis[CONST2].srcStride * sizeof(T);
        loopParams.loop1DstStride = view.axis[CONST2].dstStride * sizeof(T);
        loopParams.loop2Size = view.axis[CONST3].repeat;
        loopParams.loop2SrcStride = view.axis[CONST3].srcStride * sizeof(T);
        loopParams.loop2DstStride = view.axis[CONST3].dstStride * sizeof(T);
    }

    template <typename T, class V>
    __aicore__ inline void CopyInLoop(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const V& view,
                                      const DataCopyExtParams& copyInParams, const DataCopyPadExtParams<T>& padParams)
    {
        if (view.axisSize <= CONST4) {
            AscendC::DataCopyPad(dst, src[view.addr], copyInParams, padParams);
        } else {
            for (uint64_t i = 0; i < view.axis[CONST7].repeat; ++i) {
                for (uint64_t j = 0; j < view.axis[CONST6].repeat; ++j) {
                    for (uint64_t k = 0; k < view.axis[CONST5].repeat; ++k) {
                        for (uint64_t l = 0; l < view.axis[CONST4].repeat; ++l) {
                            const int64_t dstStride = i * view.axis[CONST7].dstStride +
                                                      j * view.axis[CONST6].dstStride +
                                                      k * view.axis[CONST5].dstStride + l * view.axis[CONST4].dstStride;
                            const int64_t srcStride = i * view.axis[CONST7].srcStride +
                                                      j * view.axis[CONST6].srcStride +
                                                      k * view.axis[CONST5].srcStride + l * view.axis[CONST4].srcStride;
                            AscendC::DataCopyPad(dst[dstStride], src[view.addr + srcStride], copyInParams, padParams);
                        }
                    }
                }
            }
        }
    }

    template <class InnerPattern, typename T>
    __aicore__ inline void DoUbBroadCast(__ubuf__ T* dstAddr, uint64_t dimA, uint64_t dimR)
    {
        const uint16_t loopA = CeilDiv(dimA * sizeof(T), static_cast<uint64_t>(VL_LEN));
        const uint16_t loopR = CeilDiv(dimR * sizeof(T), static_cast<uint64_t>(VL_LEN));
        if constexpr (InnerPattern::TailA) {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<T, AscendC::Reg::RegTraitNumOne> vreg;
                AscendC::Reg::MaskReg mask;
                for (uint16_t i = 0; i < dimR - 1; ++i) {
                    uint32_t scalar = dimA;
                    for (uint16_t j = 0; j < loopA; ++j) {
                        AscendC::Reg::DataCopy(vreg, dstAddr + i * dimA + j * VL_LEN / sizeof(T));
                        auto mask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(scalar);
                        AscendC::Reg::DataCopy(dstAddr + (i + 1) * dimA + j * VL_LEN / sizeof(T), vreg, mask);
                    }
                }
            }
        } else {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<T, AscendC::Reg::RegTraitNumOne> vregSrc, vregDst;
                AscendC::Reg::MaskReg maskFull = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
                AscendC::Reg::MaskReg mask;
                for (uint16_t i = 0; i < dimA; ++i) {
                    AscendC::Reg::DataCopy(vregSrc, dstAddr + i * dimR);
                    uint32_t scalar = dimR;
                    for (uint16_t j = 0; j < loopR; ++j) {
                        AscendC::Reg::Duplicate(vregDst, vregSrc, maskFull);
                        auto mask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(scalar);
                        AscendC::Reg::DataCopy(dstAddr + i * dimR + j * VL_LEN / sizeof(T), vregDst, mask);
                    }
                }
            }
        }
    }

    /**
     * \brief Process NaN->0 replacement on UB data using Reg vector registers.
     *
     * Uses Compare(x, x, EQ) to detect NaN (IEEE 754: NaN != NaN -> mask=0, normal == normal -> mask=1).
     * Then Select(data, data, 0, mask): mask=1(normal)->keep data, mask=0(NaN)->replace with 0.
     * All operations use vector registers, no extra UB buffer allocation needed.
     */
    template <typename T>
    __aicore__ inline void NaNToZero(LocalTensor<T>& data, uint64_t elemCount)
    {
        if (elemCount == 0) {
            return;
        }

        constexpr uint16_t vLElems = VL_LEN / sizeof(T);
        const uint16_t loopNum = CeilDiv(elemCount, static_cast<uint64_t>(vLElems));
        __ubuf__ T* dataAddr = reinterpret_cast<__ubuf__ T*>(data.GetPhyAddr());

        __VEC_SCOPE__
        {
            Reg::RegTensor<T, Reg::RegTraitNumOne> dataReg;
            Reg::RegTensor<T, Reg::RegTraitNumOne> zeroReg;
            Reg::MaskReg cmpMask;
            Reg::MaskReg opMask;

            Reg::Duplicate(zeroReg, static_cast<T>(0));

            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t count = (i == loopNum - 1) ?
                                     static_cast<uint32_t>(elemCount - static_cast<uint64_t>(i) * vLElems) :
                                     vLElems;
                opMask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);

                // Load data from UB to VR
                Reg::DataCopy(dataReg, dataAddr + i * vLElems);

                // Compare(x, x, EQ): NaN != NaN -> mask=0, normal == normal -> mask=1
                Reg::Compare<T, CMPMODE::EQ>(cmpMask, dataReg, dataReg, opMask);

                // Select: mask=1(normal) -> data, mask=0(NaN) -> 0
                Reg::Select(dataReg, dataReg, zeroReg, cmpMask);

                // Store back to UB
                Reg::DataCopy(dataAddr + i * vLElems, dataReg, opMask);
            }
        }
    }

    /**
     * \brief Calculate total element count from view.
     */
    template <class V>
    __aicore__ inline uint64_t GetViewElemCount(const V& view)
    {
        uint64_t count = 1;
        for (uint8_t i = 0; i < view.axisSize; ++i) {
            count *= view.axis[i].repeat;
        }
        return count;
    }

public:
    /**
     * \brief Override CopyIn for inner-loop broadcast (pos!=0).
     *
     * For pos==0: no-op here. 数据搬运由 CopyInAux 的后续函数 (CopyInWithMoveAlign /
     *             CopyInWithNddma) 完成。不能在此处做 NaN->0，因为 CopyInAux 在调用
     *             reduceOp_->CopyIn() 之后，还会无条件调用 CopyInWithMoveAlign/
     *             CopyInWithNddma 再次从 GM 搬数据到 UB，会覆盖此处对 UB 数据的修改。
     *             NaN->0 替换改在 PadValue 中执行（数据搬运和 padding 完成后、reduce 计算前）。
     * For pos!=0 (inner loop / dichotomy): does standard broadcast same as ReduceSumOp.
     */
    template <int32_t pos, class InnerPattern, typename T, class V>
    __aicore__ inline void CopyIn(const LocalTensor<T>& dst, const GlobalTensor<T>& src, V& view)
    {
        DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
        DataCopyExtParams copyInParams;
        LoopModeParams loopParams;
        uint64_t addrOffset = 0;
        for (int32_t i = 0; i < view.axisSize; i++) {
            addrOffset += view.axis[i].start * view.axis[i].srcStride;
        }
        view.addr = addrOffset;

        if constexpr (pos == 0) {
            // pos==0: 数据搬运交给 CopyInAux 的后续函数完成，此处不做任何操作。
            // NaN->0 在 PadValue 中执行，避免被 CopyInWithMoveAlign/CopyInWithNddma 覆盖。
        } else {
            // For inner loop patterns, do broadcast (same as ReduceLogSumExpOp)
            V viewCopy = view;
            V orinView = view;
            const uint64_t blkSize = BLOCK_SIZE / sizeof(T);
            const uint64_t burstLenBlockAlign = CeilAlign(viewCopy.axis[CONST0].repeat, blkSize);
            uint64_t dimA = 1;
            uint64_t dimR = 1;
            for (uint8_t i = 1; i < viewCopy.axisSize; ++i) {
                viewCopy.axis[i].isAxisA ? (dimA *= viewCopy.axis[i].repeat) : (dimR *= viewCopy.axis[i].repeat);
            }

            for (int64_t i = 0; i < viewCopy.axisSize; i++) {
                if (i == 0) {
                    orinView.axis[0].repeat = view.axis[1].srcStride;
                } else {
                    orinView.axis[i].repeat = view.axis[i + 1].srcStride / view.axis[i].srcStride;
                }
            }
            int64_t continuousBlockLen = 1;
            int64_t continuousBlockLenDimA = 1;
            int64_t cutBlockLenDimA = 1;

            for (int64_t i = 0; i < viewCopy.axisSize; i++) {
                if (orinView.axis[i].repeat == viewCopy.axis[i].repeat) {
                    continuousBlockLen *= orinView.axis[i].repeat;
                    if (orinView.axis[i].isAxisA) {
                        continuousBlockLenDimA *= orinView.axis[i].repeat;
                    }
                } else {
                    if (orinView.axis[i].isAxisA) {
                        cutBlockLenDimA = orinView.axis[i].repeat;
                    }
                    break;
                }
            }
            viewCopy.addr = (viewCopy.addr / continuousBlockLen) % cutBlockLenDimA * continuousBlockLenDimA;

            if constexpr (InnerPattern::TailA) {
                int64_t rRepeatProd = 1;
                for (uint8_t i = 0; i < viewCopy.axisSize; ++i) {
                    viewCopy.axis[i].srcStride /= rRepeatProd;
                    if (!viewCopy.axis[i].isAxisA) {
                        rRepeatProd *= viewCopy.axis[i].repeat;
                        viewCopy.axis[i].dstStride = dimA;
                        if (i != 0) {
                            viewCopy.axis[i].repeat = 1;
                        }
                    }
                }

                SetCopyInAndLoopParams<T>(copyInParams, loopParams, viewCopy, true);
                SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
                CopyInLoop<T>(dst, src, viewCopy, copyInParams, padParams);
                ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            } else {
                copyInParams.blockCount = dimA;
                copyInParams.blockLen = sizeof(T);
                copyInParams.srcStride = 0;
                copyInParams.dstStride = (dimR - 1) * sizeof(T) / GetUbBlockSize();
                AscendC::DataCopyPad(dst, src[viewCopy.addr], copyInParams, padParams);
            }
            int32_t eventIdMte2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            DoUbBroadCast<InnerPattern, T>((__ubuf__ T*)dst.GetPhyAddr(), dimA, dimR);
        }
    }

    template <class Pattern, typename InputT, class S, class P>
    __aicore__ inline void PadValue(const LocalTensor<PromteT>& dst, S& shape, P& padding)
    {
        PromteT padValue = this->template GetPaddingValue<PromteT>();
        if constexpr (IsB64<PromteT>()) {
            DoPadding<AscendC::Reg::RegTraitNumTwo, true, Pattern, InputT, PromteT>((__ubuf__ PromteT*)dst.GetPhyAddr(),
                                                                                    padValue, shape, padding);
        } else {
            DoPadding<AscendC::Reg::RegTraitNumOne, true, Pattern, InputT, PromteT>((__ubuf__ PromteT*)dst.GetPhyAddr(),
                                                                                    padValue, shape, padding);
        }

        // PadValue 在数据搬运(CopyInWithMoveAlign)和 padding 完成后、reduce 计算前调用，
        // 此时数据已在 UB 中且不会再被覆盖，执行 NaN->0 替换。
        // padding 区域已被填充为 0，NaN->0 对其无影响（0 不是 NaN）。
        uint64_t elemCount = static_cast<uint64_t>(shape.value[0]) * static_cast<uint64_t>(shape.value[1]);
        NaNToZero<PromteT>(const_cast<LocalTensor<PromteT>&>(dst), elemCount);
    }
#endif
};
} // namespace ReduceNansumVec
} // namespace AscendC

#endif // REDUCE_NANSUM_OPERATOR_H
