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
 * \file pow_tensor_scalar_integer.h
 * \brief
 */
#ifndef ASCENDC_POW_POWS_INTEGER_H_
#define ASCENDC_POW_POWS_INTEGER_H_

namespace Pow {
using namespace AscendC;
template <typename T1, typename T2>
class PowTensorScalarInteger {
public:
    __aicore__ inline PowTensorScalarInteger(){};

    __aicore__ inline void Init(GM_ADDR base, GM_ADDR exponent, GM_ADDR pow,
                                const PowTensorScalarTilingData* __restrict ordTilingData, TPipe* pipeIn)
    {
        cBlockIdx = GetBlockIdx();
        tilingData = ordTilingData;

        // init pipe
        pipe = pipeIn;

        // init gm
        baseGm.SetGlobalBuffer((__gm__ T1*)base);
        expGm.SetGlobalBuffer((__gm__ T2*)exponent);
        powGm.SetGlobalBuffer((__gm__ T1*)pow);

        // init ub buffer
        constexpr static int64_t EXP_UB_SIZE = 256;
        constexpr static int64_t DB_BUFFER = 2;
        pipe->InitBuffer(vecBaseQue, DB_BUFFER, tilingData->ubSize);
        pipe->InitBuffer(vecExpQue, 1, EXP_UB_SIZE);
        pipe->InitBuffer(vecPowQue, 1, tilingData->ubSize);
        pipe->InitBuffer(tmp1Tensor, tilingData->ubSize);
        pipe->InitBuffer(tmp2Tensor, tilingData->ubSize);
        pipe->InitBuffer(tmpExpTensor, EXP_UB_SIZE);

        // get exp value
        expValue = static_cast<int64_t>(expGm(0));

        // add sync v wait for s
        TEventID eventID_S_V = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventID_S_V);
        WaitFlag<HardEvent::S_V>(eventID_S_V);
    }

    __aicore__ inline void Process()
    {
        int64_t beginIdx = cBlockIdx * tilingData->blockFactor;
        int64_t endIdx = (cBlockIdx + 1) * tilingData->blockFactor;
        endIdx = endIdx > tilingData->ubOuter ? tilingData->ubOuter : endIdx;
        int64_t realEndIdx = tilingData->ubOuter - 1;
        int64_t curDataLength = tilingData->ubFactor;
        int64_t gmOffset = beginIdx * curDataLength;

        if (expValue == 0) {
            LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
            ComputeZeros(dstLocal, tilingData->ubFactor);
            vecPowQue.EnQue(dstLocal);
            vecPowQue.DeQue<T1>();
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                CopyPow(dstLocal, gmOffset, curDataLength);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
            vecPowQue.FreeTensor(dstLocal);
        } else if (expValue == 1) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                ComputeCopy(dstLocal, srcLocal, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else if (expValue == EXP_SQUARED) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                ComputeSquare(dstLocal, srcLocal, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else if (expValue == EXP_CUBE) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                ComputeCube(dstLocal, srcLocal, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                Power<T1, false, powsConfig_>(dstLocal, srcLocal, static_cast<T1>(expValue),
                                              static_cast<uint32_t>(curDataLength));
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        }
    }

private:
    __aicore__ inline void CopyBase(LocalTensor<T1>& dstTensor, int64_t offset, int64_t gatherLength)
    {
        DataCopyPad(dstTensor, baseGm[offset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(gatherLength * sizeof(T1)), 0, 0, 0},
                    {false, 0, 0, 0});
    }

    __aicore__ inline void CopyPow(LocalTensor<T1>& srcTensor, int64_t offset, int64_t gatherLength)
    {
        DataCopyPad(powGm[offset], srcTensor,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(gatherLength * sizeof(T1)), 0, 0, 0});
    }

    __aicore__ inline void ComputeZeros(LocalTensor<T1>& dstTensor, int64_t dataLength)
    {
        T1 inputVal(1);
        Duplicate<T1>(dstTensor, inputVal, static_cast<int32_t>(dataLength));
    }

    __aicore__ inline void ComputeCopy(LocalTensor<T1>& dstTensor, LocalTensor<T1>& srcTensor, int64_t dataLength)
    {
        if constexpr (sizeof(T1) == sizeof(int32_t)) {
            uint32_t scalarValue = 0;
            ShiftLeft(dstTensor.template ReinterpretCast<uint32_t>(), srcTensor.template ReinterpretCast<uint32_t>(),
                      scalarValue, static_cast<int32_t>(dataLength));
        } else if constexpr (sizeof(T1) == sizeof(int16_t)) {
            uint16_t scalarValue = 0;
            ShiftLeft(dstTensor.template ReinterpretCast<uint16_t>(), srcTensor.template ReinterpretCast<uint16_t>(),
                      scalarValue, static_cast<int32_t>(dataLength));
        } else {
            uint16_t scalarValue = 0;
            ShiftLeft(dstTensor.template ReinterpretCast<uint16_t>(), srcTensor.template ReinterpretCast<uint16_t>(),
                      scalarValue, static_cast<int32_t>((dataLength + 1) / B8_TO_B16)); // 将数据长度B8转换为B16长度
        }
    }

    __aicore__ inline void ComputeSquare(LocalTensor<T1>& dstTensor, LocalTensor<T1>& srcTensor, int64_t dataLength)
    {
        // 设置控制寄存器的第60位为0，开启非饱和模式
        SetCtrlSpr<SAT_POS, SAT_POS>(0);
        __VEC_SCOPE__
        {
            if constexpr (IsSameType<T1, int32_t>::value || IsSameType<T1, int16_t>::value) {
                Reg::RegTensor<T1> vreg0;
                Reg::RegTensor<T1> vreg1;
                Reg::MaskReg preg0;
                uint32_t size = dataLength;
                uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(T1)) - 1) /
                                     (VECTOR_REG_WIDTH / sizeof(T1));
                __local_mem__ T1* bufferIn0Addr = (__local_mem__ T1*)srcTensor.GetPhyAddr();
                __local_mem__ T1* bufferOut0Addr = (__local_mem__ T1*)dstTensor.GetPhyAddr();
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    preg0 = Reg::UpdateMask<T1>(size);
                    Reg::DataCopy<T1, Reg::LoadDist::DIST_NORM>(vreg0,
                                                                bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(T1)));
                    Reg::Mul<T1, Reg::MaskMergeMode::ZEROING>(vreg1, vreg0, vreg0, preg0);
                    Reg::DataCopy<T1, Reg::StoreDist::DIST_NORM_B32>(
                        bufferOut0Addr + i * (VECTOR_REG_WIDTH / sizeof(T1)), vreg1, preg0);
                }
            } else if constexpr (IsSameType<T1, uint8_t>::value) {
                Reg::RegTensor<T1> vreg0;
                Reg::RegTensor<uint16_t> scalar0;
                Reg::RegTensor<uint16_t> vreg1;
                Reg::RegTensor<uint16_t> vreg2;
                Reg::RegTensor<uint16_t> vreg3;
                Reg::RegTensor<T1> vreg4;
                Reg::MaskReg preg0;
                uint32_t size = dataLength;
                uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(uint16_t)) - 1) /
                                     (VECTOR_REG_WIDTH / sizeof(uint16_t));
                __local_mem__ T1* bufferIn0Addr = (__local_mem__ T1*)srcTensor.GetPhyAddr();
                __local_mem__ T1* bufferOut0Addr = (__local_mem__ T1*)dstTensor.GetPhyAddr();
                Reg::Duplicate(scalar0, uint16_t(255));
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    preg0 = Reg::UpdateMask<uint16_t>(size);
                    Reg::DataCopy<T1, Reg::LoadDist::DIST_UNPACK_B8>(
                        vreg0, bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(uint16_t)));
                    Reg::Cast<uint16_t, T1, castTrait0>(vreg1, vreg0, preg0);
                    Reg::Mul<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);
                    Reg::And<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, scalar0, preg0);
                    Reg::Cast<T1, uint16_t, castTrait1>(vreg4, vreg3, preg0);
                    Reg::DataCopy<T1, Reg::StoreDist::DIST_PACK_B16>(
                        bufferOut0Addr + i * (VECTOR_REG_WIDTH / sizeof(uint16_t)), vreg4, preg0);
                }
            } else {
                Reg::RegTensor<T1> vreg0;
                Reg::RegTensor<int16_t> scalar0;
                Reg::RegTensor<int16_t> vreg1;
                Reg::RegTensor<int16_t> vreg2;
                Reg::RegTensor<int16_t> vreg3;
                Reg::RegTensor<half> vreg4;
                Reg::RegTensor<T1> vreg5;
                Reg::MaskReg preg0;
                uint32_t size = dataLength;
                uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(int16_t)) - 1) /
                                     (VECTOR_REG_WIDTH / sizeof(int16_t));
                __local_mem__ T1* bufferIn0Addr = (__local_mem__ T1*)srcTensor.GetPhyAddr();
                __local_mem__ T1* bufferOut0Addr = (__local_mem__ T1*)dstTensor.GetPhyAddr();
                Reg::Duplicate(scalar0, int16_t(255));
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    preg0 = Reg::UpdateMask<int16_t>(size);
                    Reg::DataCopy<T1, Reg::LoadDist::DIST_UNPACK_B8>(
                        vreg0, bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(int16_t)));
                    Reg::Cast<int16_t, T1, castTrait0>(vreg1, vreg0, preg0);
                    Reg::Mul<int16_t, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);
                    Reg::And<int16_t, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, scalar0, preg0);
                    Reg::Cast<half, int16_t, castTrait2>(vreg4, vreg3, preg0);
                    Reg::Cast<T1, half, castTrait3>(vreg5, vreg4, preg0);
                    Reg::DataCopy<T1, Reg::StoreDist::DIST_PACK_B16>(
                        bufferOut0Addr + i * (VECTOR_REG_WIDTH / sizeof(int16_t)), vreg5, preg0);
                }
            }
        }
        // 设置控制寄存器的第60位为1，回退到饱和模式
        SetCtrlSpr<SAT_POS, SAT_POS>(1);
    }

    __aicore__ inline void ComputeCube(LocalTensor<T1>& dstTensor, LocalTensor<T1>& srcTensor, int64_t dataLength)
    {
        // 设置控制寄存器的第60位为0，开启非饱和模式
        SetCtrlSpr<SAT_POS, SAT_POS>(0);
        __VEC_SCOPE__
        {
            if constexpr (IsSameType<T1, int32_t>::value || IsSameType<T1, int16_t>::value) {
                Reg::RegTensor<T1> vreg0;
                Reg::RegTensor<T1> vreg1;
                Reg::RegTensor<T1> vreg2;
                Reg::MaskReg preg0;
                uint32_t size = dataLength;
                uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(T1)) - 1) /
                                     (VECTOR_REG_WIDTH / sizeof(T1));
                __local_mem__ T1* bufferIn0Addr = (__local_mem__ T1*)srcTensor.GetPhyAddr();
                __local_mem__ T1* bufferOut0Addr = (__local_mem__ T1*)dstTensor.GetPhyAddr();
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    preg0 = Reg::UpdateMask<T1>(size);
                    Reg::DataCopy<T1, Reg::LoadDist::DIST_NORM>(vreg0,
                                                                bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(T1)));
                    Reg::Mul<T1, Reg::MaskMergeMode::ZEROING>(vreg1, vreg0, vreg0, preg0);
                    Reg::Mul<T1, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg0, preg0);
                    Reg::DataCopy<T1, Reg::StoreDist::DIST_NORM_B32>(
                        bufferOut0Addr + i * (VECTOR_REG_WIDTH / sizeof(T1)), vreg2, preg0);
                }
            } else if constexpr (IsSameType<T1, uint8_t>::value) {
                Reg::RegTensor<T1> vreg0;
                Reg::RegTensor<uint16_t> scalar0;
                Reg::RegTensor<uint16_t> vreg1;
                Reg::RegTensor<uint16_t> vreg2;
                Reg::RegTensor<uint16_t> vreg3;
                Reg::RegTensor<uint16_t> vreg4;
                Reg::RegTensor<T1> vreg5;
                Reg::MaskReg preg0;
                uint32_t size = dataLength;
                uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(uint16_t)) - 1) /
                                     (VECTOR_REG_WIDTH / sizeof(uint16_t));
                __local_mem__ T1* bufferIn0Addr = (__local_mem__ T1*)srcTensor.GetPhyAddr();
                __local_mem__ T1* bufferOut0Addr = (__local_mem__ T1*)dstTensor.GetPhyAddr();
                Reg::Duplicate(scalar0, uint16_t(255));
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    preg0 = Reg::UpdateMask<uint16_t>(size);
                    Reg::DataCopy<T1, Reg::LoadDist::DIST_UNPACK_B8>(
                        vreg0, bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(uint16_t)));
                    Reg::Cast<uint16_t, T1, castTrait0>(vreg1, vreg0, preg0);
                    Reg::Mul<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);
                    Reg::Mul<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg1, preg0);
                    Reg::And<uint16_t, Reg::MaskMergeMode::ZEROING>(vreg4, vreg3, scalar0, preg0);
                    Reg::Cast<T1, uint16_t, castTrait1>(vreg5, vreg4, preg0);
                    Reg::DataCopy<T1, Reg::StoreDist::DIST_PACK_B16>(
                        bufferOut0Addr + i * (VECTOR_REG_WIDTH / sizeof(uint16_t)), vreg5, preg0);
                }
            } else {
                Reg::RegTensor<T1> vreg0;
                Reg::RegTensor<int16_t> scalar0;
                Reg::RegTensor<int16_t> vreg1;
                Reg::RegTensor<int16_t> vreg2;
                Reg::RegTensor<int16_t> vreg3;
                Reg::RegTensor<int16_t> vreg4;
                Reg::RegTensor<half> vreg5;
                Reg::RegTensor<T1> vreg6;
                Reg::MaskReg preg0;
                uint32_t size = dataLength;
                uint16_t vfLoopNum = (dataLength + (VECTOR_REG_WIDTH / sizeof(int16_t)) - 1) /
                                     (VECTOR_REG_WIDTH / sizeof(int16_t));
                __local_mem__ T1* bufferIn0Addr = (__local_mem__ T1*)srcTensor.GetPhyAddr();
                __local_mem__ T1* bufferOut0Addr = (__local_mem__ T1*)dstTensor.GetPhyAddr();
                Reg::Duplicate(scalar0, int16_t(255));
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    preg0 = Reg::UpdateMask<int16_t>(size);
                    Reg::DataCopy<T1, Reg::LoadDist::DIST_UNPACK_B8>(
                        vreg0, bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(int16_t)));
                    Reg::Cast<int16_t, T1, castTrait0>(vreg1, vreg0, preg0);
                    Reg::Mul<int16_t, Reg::MaskMergeMode::ZEROING>(vreg2, vreg1, vreg1, preg0);
                    Reg::Mul<int16_t, Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg1, preg0);
                    Reg::And<int16_t, Reg::MaskMergeMode::ZEROING>(vreg4, vreg3, scalar0, preg0);
                    Reg::Cast<half, int16_t, castTrait2>(vreg5, vreg4, preg0);
                    Reg::Cast<T1, half, castTrait3>(vreg6, vreg5, preg0);
                    Reg::DataCopy<T1, Reg::StoreDist::DIST_PACK_B16>(
                        bufferOut0Addr + i * (VECTOR_REG_WIDTH / sizeof(int16_t)), vreg6, preg0);
                }
            }
        }
        // 设置控制寄存器的第60位为1，回退到饱和模式
        SetCtrlSpr<SAT_POS, SAT_POS>(1);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> vecBaseQue;
    TQue<QuePosition::VECIN, 1> vecExpQue;
    TQue<QuePosition::VECOUT, 1> vecPowQue;
    TBuf<> tmp1Tensor;
    TBuf<> tmp2Tensor;
    TBuf<> tmpExpTensor;

    int64_t cBlockIdx;
    const PowTensorScalarTilingData* __restrict tilingData;

    // input
    GlobalTensor<T1> baseGm;
    GlobalTensor<T2> expGm;

    // output
    GlobalTensor<T1> powGm;

    // exp value
    int64_t expValue;

    // cast info
    constexpr static Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr static Reg::CastTrait castTrait1 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr static Reg::CastTrait castTrait2 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait castTrait3 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static PowerConfig powsConfig_ = {PowerAlgo::INTRINSIC};

    constexpr static int64_t EXP_SQUARED = 2;
    constexpr static int64_t EXP_CUBE = 3;
    constexpr static int64_t B8_TO_B16 = 2;
    constexpr static int8_t SAT_POS = 60;
};
} // namespace Pow
#endif // ASCENDC_POW_POWS_INTEGER_H_
