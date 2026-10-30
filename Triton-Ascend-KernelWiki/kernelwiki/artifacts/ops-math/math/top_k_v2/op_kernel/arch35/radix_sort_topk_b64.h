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
 * \file radix_sort_topk_b64.h
 * \brief radix_sort_topk_b64 impl
 */
#ifndef RADIX_SORT_TOPK_B64_H
#define RADIX_SORT_TOPK_B64_H
#include "kernel_operator.h"
#include "top_k_util_type_simd.h"

using namespace topkV2;
template <typename T, typename UNSIGNED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
struct RadixSortTopKB64 {
    __aicore__ inline RadixSortTopKB64() {}
    __aicore__ inline void GetCumSum(LocalTensor<UNSIGNED_TYPE> inputX, LocalTensor<int32_t> blockCumSum,
                                     UNSIGNED_TYPE andDataMask, UNSIGNED_TYPE involvedDataMask, uint16_t round,
                                     uint32_t numTileData);
};

template <typename T, typename UNSIGNED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
__aicore__ inline void RadixSortTopKB64<T, UNSIGNED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX>::GetCumSum(
    LocalTensor<UNSIGNED_TYPE> inputX, LocalTensor<int32_t> blockCumSum, UNSIGNED_TYPE andDataMask,
    UNSIGNED_TYPE involvedDataMask, uint16_t round, uint32_t numTileData)
{
    __local_mem__ UNSIGNED_TYPE* inputXValuePtr = (__ubuf__ UNSIGNED_TYPE*)inputX.GetPhyAddr();
    __local_mem__ int32_t* blockCumSumPtr = (__ubuf__ int32_t*)blockCumSum.GetPhyAddr();
    int16_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repateTime = (numTileData + ONE_TIMES_B64_NUM - 1) / ONE_TIMES_B64_NUM;
    UNSIGNED_TYPE roundValue = NUM_PASS - 1;
    UNSIGNED_TYPE newRound = static_cast<UNSIGNED_TYPE>(round);
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint64_t> inputVectorOne;
        Reg::RegTensor<uint16_t> chistVectorZero, chistVectorOne;
        Reg::RegTensor<uint16_t> zeroVectorU16;
        Reg::RegTensor<uint32_t> zeroVectorU32;
        Reg::RegTensor<uint64_t> zeroVectorU64;
        Reg::RegTensor<uint64_t> andMaskVector;
        Reg::RegTensor<uint64_t> firstRoundVector;
        Reg::MaskReg predicateDefaultB64 = Reg::CreateMask<uint64_t>();
        Reg::MaskReg predicateDefaultB32 = Reg::CreateMask<uint32_t>();
        Reg::MaskReg predicateDefaultB16 = Reg::CreateMask<uint16_t>();
        Reg::Duplicate(chistVectorZero, 0, predicateDefaultB16);
        Reg::Duplicate(chistVectorOne, 0, predicateDefaultB16);
        Reg::Duplicate(zeroVectorU16, 0, predicateDefaultB16);
        Reg::Duplicate(zeroVectorU32, 0, predicateDefaultB32);
        Reg::Duplicate(zeroVectorU64, 0, predicateDefaultB64);
        Reg::Duplicate(andMaskVector, andDataMask, predicateDefaultB64);
        Reg::Duplicate(firstRoundVector, roundValue, predicateDefaultB64);
        uint32_t inputElementNum = numTileData;
        for (uint16_t i = 0; i < repateTime; i++) {
            Reg::MaskReg histMask = Reg::UpdateMask<uint64_t>(inputElementNum);
            // load input
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr,
                                                                        ONE_TIMES_B64_NUM);
            // get high bit value
            Reg::RegTensor<uint64_t> andValueZero;
            Reg::And(andValueZero, inputVectorOne, andMaskVector, histMask);
            // get same value
            Reg::MaskReg cmpValueMask;
            Reg::CompareScalar<uint64_t, CMPMODE::EQ>(cmpValueMask, andValueZero, involvedDataMask, histMask);
            // update mask for next step calc
            Reg::MaskReg newHistMask;
            Reg::MaskAnd(newHistMask, cmpValueMask, histMask, predicateDefaultB64);
            // is first round
            Reg::MaskReg finalMask, firstRoundMask;
            Reg::CompareScalar<uint64_t, CMPMODE::EQ>(firstRoundMask, firstRoundVector, newRound, histMask);
            // mask sel
            Reg::MaskSel(finalMask, histMask, newHistMask, firstRoundMask);
            // vshr
            Reg::RegTensor<uint64_t> shiftVecOne;
            Reg::ShiftRights<uint64_t, int16_t>(shiftVecOne, inputVectorOne, bitOffset, finalMask);
            // get 256 8bit
            // get u64 low bit
            Reg::RegTensor<uint32_t> shiftVecU64LowBitOne, shiftVecU64HighBitOne;
            Reg::DeInterleave(shiftVecU64LowBitOne, shiftVecU64HighBitOne, (Reg::RegTensor<uint32_t>&)shiftVecOne,
                              (Reg::RegTensor<uint32_t>&)zeroVectorU64);
            // get u32 low bit
            Reg::RegTensor<uint16_t> shiftVecU16LowBitOne, shiftVecU16HighBitOne;
            Reg::DeInterleave(shiftVecU16LowBitOne, shiftVecU16HighBitOne,
                              (Reg::RegTensor<uint16_t>&)shiftVecU64LowBitOne,
                              (Reg::RegTensor<uint16_t>&)zeroVectorU32);
            // get u16 low bit
            Reg::RegTensor<uint8_t> shiftVecU8LowBit, shiftVecU8HighBit;
            Reg::DeInterleave(shiftVecU8LowBit, shiftVecU8HighBit, (Reg::RegTensor<uint8_t>&)shiftVecU16LowBitOne,
                              (Reg::RegTensor<uint8_t>&)zeroVectorU16);
            // copy u16 mask
            Reg::MaskReg maskU8, maskU16, maskU32;
            Reg::MaskPack(maskU32, finalMask);
            Reg::MaskPack(maskU16, maskU32);
            Reg::MaskPack(maskU8, maskU16);
            // get hist
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
                chistVectorZero, shiftVecU8LowBit, maskU8);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
                chistVectorOne, shiftVecU8LowBit, maskU8);
        }
        StoreCumsumToUB<int32_t, uint16_t>(blockCumSumPtr, chistVectorZero, chistVectorOne, zeroVectorU16);
    }
}
#endif
