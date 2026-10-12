/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"
#include "sign_bits_pack_tiling_data.h"

constexpr int kPhysNodes = 3;
constexpr int kDB = 2;
constexpr int kSlotElems = 256;
constexpr int kOutSlotBytes = 32;

__simd_vf__ inline void SignBitsPackBitReverseVF(__ubuf__ uint16_t* addr, uint32_t count, uint32_t oneRepeatSize,
                                                 uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<uint16_t> regV, regT, regMask, regNotMask;
    AscendC::Reg::MaskReg predMask;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<uint16_t>(i, oneRepeatSize);
        predMask = AscendC::Reg::UpdateMask<uint16_t>(count);
        AscendC::Reg::LoadAlign(regV, addr, aReg);

        AscendC::Reg::Duplicate(regMask, static_cast<uint16_t>(0x5555u));
        AscendC::Reg::Duplicate(regNotMask, static_cast<uint16_t>(0xAAAAu));
        AscendC::Reg::And(regT, regV, regMask, predMask);
        AscendC::Reg::ShiftLefts(regT, regT, static_cast<int16_t>(1), predMask);
        AscendC::Reg::And(regV, regV, regNotMask, predMask);
        AscendC::Reg::ShiftRights(regV, regV, static_cast<int16_t>(1), predMask);
        AscendC::Reg::Or(regV, regV, regT, predMask);

        AscendC::Reg::Duplicate(regMask, static_cast<uint16_t>(0x3333u));
        AscendC::Reg::Duplicate(regNotMask, static_cast<uint16_t>(0xCCCCu));
        AscendC::Reg::And(regT, regV, regMask, predMask);
        AscendC::Reg::ShiftLefts(regT, regT, static_cast<int16_t>(2), predMask);
        AscendC::Reg::And(regV, regV, regNotMask, predMask);
        AscendC::Reg::ShiftRights(regV, regV, static_cast<int16_t>(2), predMask);
        AscendC::Reg::Or(regV, regV, regT, predMask);

        AscendC::Reg::Duplicate(regMask, static_cast<uint16_t>(0x0F0Fu));
        AscendC::Reg::Duplicate(regNotMask, static_cast<uint16_t>(0xF0F0u));
        AscendC::Reg::And(regT, regV, regMask, predMask);
        AscendC::Reg::ShiftLefts(regT, regT, static_cast<int16_t>(4), predMask);
        AscendC::Reg::And(regV, regV, regNotMask, predMask);
        AscendC::Reg::ShiftRights(regV, regV, static_cast<int16_t>(4), predMask);
        AscendC::Reg::Or(regV, regV, regT, predMask);

        AscendC::Reg::StoreAlign(addr, regV, aReg, predMask);
    }
}

template <typename T, int UB_AXES_IN_BLOCK>
class SignBitsPackKernel {
    AscendC::TPipe pipe_;
    const SignBitsPackTilingData* td_ = nullptr;
    AscendC::GlobalTensor<T> gmIn_;
    AscendC::GlobalTensor<uint8_t> gmOut_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> inputTBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroCmpTBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outTBuf_;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SignBitsPackTilingData* td)
    {
        td_ = td;
        gmIn_.SetGlobalBuffer((__gm__ T*)x, td->n);
        gmOut_.SetGlobalBuffer((__gm__ uint8_t*)y, td->packedLen);

        const uint32_t inSlotBytes = static_cast<uint32_t>(kSlotElems * sizeof(T));
        const uint32_t outSlotBytes = static_cast<uint32_t>(kOutSlotBytes);
        pipe_.InitBuffer(inputTBuf_, kDB * inSlotBytes);
        pipe_.InitBuffer(zeroCmpTBuf_, kDB * inSlotBytes);
        pipe_.InitBuffer(outTBuf_, kDB * outSlotBytes);

        for (int s = 0; s < kDB; s++) {
            auto zeroLocal = zeroCmpTBuf_.template GetWithOffset<T>(kSlotElems, static_cast<uint32_t>(s) * inSlotBytes);
            AscendC::Duplicate<T>(zeroLocal, static_cast<T>(0), kSlotElems);
        }
    }

    __aicore__ inline void Process()
    {
        const uint64_t coreId = AscendC::GetBlockIdx();
        if (coreId >= td_->realCoreNum) {
            return;
        }

        const uint64_t coreStart = coreId * td_->perCoreCount;
        uint64_t coreEnd = coreStart + td_->perCoreCount;
        if (coreEnd > td_->totalCount) {
            coreEnd = td_->totalCount;
        }
        const uint64_t localCount = coreEnd - coreStart;

        for (uint64_t bi = coreStart; bi < coreEnd; ++bi) {
            const uint32_t ubSlot = static_cast<uint32_t>(bi % kDB);
            const uint64_t localIdx = bi - coreStart;
            const bool isTail = (bi == td_->totalCount - 1) && (td_->tailElemCount > 0);

            if (localIdx >= 2) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ubSlot);
            }

            CopyIn(ubSlot, bi, isTail);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubSlot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubSlot);

            Compute(ubSlot);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSlot);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSlot);

            CopyOut(ubSlot, bi, isTail);

            if (localIdx + 2 < localCount) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ubSlot);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline AscendC::LocalTensor<T> GetInputSlot(uint32_t slot)
    {
        const uint32_t slotBytes = static_cast<uint32_t>(kSlotElems * sizeof(T));
        return inputTBuf_.template GetWithOffset<T>(kSlotElems, slot * slotBytes);
    }
    __aicore__ inline AscendC::LocalTensor<T> GetZeroCmpSlot(uint32_t slot)
    {
        const uint32_t slotBytes = static_cast<uint32_t>(kSlotElems * sizeof(T));
        return zeroCmpTBuf_.template GetWithOffset<T>(kSlotElems, slot * slotBytes);
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> GetOutSlot(uint32_t slot)
    {
        return outTBuf_.template GetWithOffset<uint8_t>(kOutSlotBytes, slot * kOutSlotBytes);
    }

    __aicore__ inline void CopyIn(uint32_t slot, uint64_t blockIdx, bool isTail)
    {
        const uint64_t inElemBase = blockIdx * kSlotElems;
        const uint32_t realElemCount = isTail ? static_cast<uint32_t>(td_->tailElemCount) :
                                                static_cast<uint32_t>(kSlotElems);
        const uint32_t blockLen = realElemCount * static_cast<uint32_t>(sizeof(T));

        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = blockLen;
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams(true, 0, 0, static_cast<T>(-1.0));

        auto inLocal = GetInputSlot(slot);
        AscendC::DataCopyPad(inLocal, gmIn_[inElemBase], extParams, padParams);
    }

    static constexpr bool kEnableBitReverse = true;
    static constexpr uint32_t kVecLenU16 = AscendC::GetVecLen() / sizeof(uint16_t);

    // CMPMODE::LT: bit = (x < 0). +0/-0 均视为非负（IEEE 754 -0 == +0），
    // nan 视为非负（nan < 0 == false）。符号位语义为"是否为负数"而非 IEEE 754 sign bit。
    __aicore__ inline void Compute(uint32_t slot)
    {
        auto outLocal = GetOutSlot(slot);
        auto xLocal = GetInputSlot(slot);
        auto zeroLocal = GetZeroCmpSlot(slot);

        AscendC::Compare<T, uint8_t>(outLocal, xLocal, zeroLocal, AscendC::CMPMODE::LT,
                                     static_cast<uint32_t>(kSlotElems));

        if constexpr (kEnableBitReverse) {
            __ubuf__ uint16_t* outAddr = (__ubuf__ uint16_t*)outLocal.GetPhyAddr();
            const uint32_t count = static_cast<uint32_t>(kOutSlotBytes / sizeof(uint16_t));
            const uint16_t repeatTimes = static_cast<uint16_t>((count + kVecLenU16 - 1) / kVecLenU16);
            asc_vf_call<SignBitsPackBitReverseVF>(outAddr, count, kVecLenU16, repeatTimes);
        }
    }

    __aicore__ inline void CopyOut(uint32_t slot, uint64_t blockIdx, bool isTail)
    {
        const uint64_t outByteBase = blockIdx * kOutSlotBytes;
        const uint32_t byteCount = isTail ? static_cast<uint32_t>(td_->tailByteCount) :
                                            static_cast<uint32_t>(kOutSlotBytes);

        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = byteCount;
        extParams.srcStride = 0;
        extParams.dstStride = 0;

        auto outLocal = GetOutSlot(slot);
        AscendC::DataCopyPad(gmOut_[outByteBase], outLocal, extParams);
    }
};
