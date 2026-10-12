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
 * \file reduce_var_welford.h
 * \brief reduce var welford
 */

#ifndef _REDUCE_VAR_WELFORD_H_
#define _REDUCE_VAR_WELFORD_H_

#include "reduce_var_vf_common.h"

namespace ReduceOpTmpl
{
using namespace AscendC;

template <typename T>
__aicore__ inline void VFWelfordParallelUpdateWithInit(__local_mem__ T *x1Local, __local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, uint32_t calLen, float scale)
{
    uint16_t loopCount = Ops::Base::CeilDiv(calLen, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> tmpVar;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t sreg0 = calLen;
        AscendC::MicroAPI::Duplicate(tmpVar, 0.0, pregMain);
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadOneTensorForDtypeT(x1Local, x1, pregLoop, i * VL_FP32);
            DataCopy(tmpMeanLocal + i * VL_FP32, x1, pregLoop);
            DataCopy(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
        }
    }
}

template <typename T>
__aicore__ inline void VFWelfordParallelUpdate(__local_mem__ T *x1Local, __local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, uint32_t calLen, float scale)
{
    uint16_t loopCount = Ops::Base::CeilDiv(calLen, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x1Scale;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpMeanScale;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delta3;
        RegTensor<float> delta4;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregLoop;
        uint32_t sreg0 = calLen;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadOneTensorForDtypeT(x1Local, x1, pregLoop, i * VL_FP32);

            DataCopy(tmpMean, tmpMeanLocal + i * VL_FP32);
            // delta1 = x1 - mean
            AscendC::MicroAPI::Sub(delta1, x1, tmpMean, pregLoop);
            // delta2 = delta1 * scale = x1 * scale - mean * scale // 防止越界溢出
            AscendC::MicroAPI::Muls(x1Scale, x1, scale, pregLoop);
            AscendC::MicroAPI::Muls(tmpMeanScale, tmpMean, scale, pregLoop);
            AscendC::MicroAPI::Sub(delta2, x1Scale, tmpMeanScale, pregLoop);
            // mean = mean + delta2
            AscendC::MicroAPI::Add(tmpMean, tmpMean, delta2, pregLoop);
            DataCopy(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

            DataCopy(tmpVar, tmpVarLocal + i * VL_FP32);
            // delta3 = x1 - mean
            AscendC::MicroAPI::Sub(delta3, x1, tmpMean, pregLoop);
            // delta4 = delta1 * delta3
            AscendC::MicroAPI::Mul(delta4, delta1, delta3, pregLoop);
            // var = var + delta4
            AscendC::MicroAPI::Add(tmpVar, tmpVar, delta4, pregLoop);
            DataCopy(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
        }
    }
}

template <typename T>
__aicore__ inline void VFWelfordParallelUpdateARWithTail(__local_mem__ T *x1Local, __local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, uint32_t ANum, uint32_t RStride, uint32_t realRLen, float scale)
{
    uint16_t aloopCount = static_cast<uint16_t>(ANum);
    uint16_t loopCount = Ops::Base::CeilDiv(realRLen, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x1Scale;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpMeanScale;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delta3;
        RegTensor<float> delta4;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregLoop;
        for (uint16_t j = 0; j < aloopCount; j++) {
            uint32_t sreg0 = realRLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadOneTensorForDtypeT(x1Local, x1, pregLoop, j * RStride + i * VL_FP32);

                DataCopy(tmpMean, tmpMeanLocal + j * RStride + i * VL_FP32);
                // delata1 = x1 - mean
                AscendC::MicroAPI::Sub(delta1, x1, tmpMean, pregLoop);
                // delta2 = delta1 * scale = x1 * scale - mean * scale // 防止越界溢出
                AscendC::MicroAPI::Muls(x1Scale, x1, scale, pregLoop);
                AscendC::MicroAPI::Muls(tmpMeanScale, tmpMean, scale, pregLoop);
                AscendC::MicroAPI::Sub(delta2, x1Scale, tmpMeanScale, pregLoop);
                // mean = mean + delta2
                AscendC::MicroAPI::Add(tmpMean, tmpMean, delta2, pregLoop);
                DataCopy(tmpMeanLocal + j * RStride + i * VL_FP32, tmpMean, pregLoop);

                DataCopy(tmpVar, tmpVarLocal + j * RStride + i * VL_FP32);
                // delta3 = x1 - mean
                AscendC::MicroAPI::Sub(delta3, x1, tmpMean, pregLoop);
                // delta4 = delta1 * delta3
                AscendC::MicroAPI::Mul(delta4, delta1, delta3, pregLoop);
                // var = var + delta4
                AscendC::MicroAPI::Add(tmpVar, tmpVar, delta4, pregLoop);
                DataCopy(tmpVarLocal + j * RStride + i * VL_FP32, tmpVar, pregLoop);
            }
        }
    }
}

/*
    Welford Finalize对齐场景计算公式如下:
    finalize_mean = sum_fun(mean) / parallel_N
    finalize_delta = mean - finalize_mean
    finalize_delta_square = finalize_delta * finalize_delta
    M2_fixed = M2 + float(count) * finalize_delta_square
    finalize_std = sum_fun(M2_fixed) / float(parallel_N * count)
*/
template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeAlignLessVL(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint32_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt)
{
    uint16_t aLoopNum = static_cast<uint16_t>(ANum);
    __VEC_SCOPE__
    {
        RegTensor<float> tmpMeanReg;
        RegTensor<float> tmpVarReg;
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> delta;

        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < aLoopNum; j++) {
            uint32_t sreg0 = RNum;
            // 计算mean
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(tmpMeanReg, tmpMeanLocal + j * RStride);
            Muls(tmpMeanReg, tmpMeanReg, meanScale, pregLoop);
            ReduceSum(mean, tmpMeanReg, pregLoop);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);

            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            DataCopy(tmpMeanReg, tmpMeanLocal + j * RStride);
            Sub(delta, tmpMeanReg, mean, pregLoop);
            Mul(delta, delta, delta, pregLoop);
            Muls(delta, delta, cnt, pregLoop);

            DataCopy(tmpVarReg, tmpVarLocal + j * RStride);
            Add(tmpVarReg, tmpVarReg, delta, pregLoop);
            if constexpr (isM2Out == false) {
                Muls(tmpVarReg, tmpVarReg, varScale, pregLoop);
            }
            ReduceSum(var, tmpVarReg, pregLoop);

            if constexpr (isStd == true && isM2Out == false) {
                Sqrt(var, var, pregMerge);
            }

            // var out
            StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
        }
    }
}

template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeARAlign(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ float *dichotomyAddLocal,
    __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint32_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt)
{
    if (RNum <= VL_FP32) {
        VFWelfordParallelFinalizeAlignLessVL<T, isStd, isM2Out>(tmpMeanLocal, tmpVarLocal, outMeanLocal, outVarLocal,
                                                    ANum, RNum, RStride, varScale, meanScale, cnt);
        return;
    }

    uint32_t dichotomyAddPower = Ops::Base::ReduceOpTmpl::FindNearestPower2(RNum);
    uint32_t dichotomyAddReminder = RNum - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = Ops::Base::CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;
    uint16_t dichtomyPowerReminder =  (dichotomyAddPowerLoopCount > dichotomyAddReminderLoopCount) ?
        (dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount) : 0;
    uint16_t outerLoop = 0;
    uint16_t dichotomyAddLastNum = 0;
    if (dichotomyAddPowerLoopCount <= VL_FP32) {
        outerLoop = 0;
        dichotomyAddLastNum = dichotomyAddPowerLoopCount;
    } else {
        int64_t tmpBinaryAddNum = 1;
        while (tmpBinaryAddNum < innerLoopCountOrigin) {
            outerLoop = outerLoop + 1;
            tmpBinaryAddNum = tmpBinaryAddNum * Ops::Base::ReduceOpTmpl::CONST2;
        }
        dichotomyAddLastNum = VL_FP32;
    }

    uint16_t aLoopNum = static_cast<uint16_t>(ANum);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
        RegTensor<float> std;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < aLoopNum; j++) {
            uint32_t sreg0 = dichotomyAddReminder;
            // 计算mean, 如果有尾块可以带着pad(0)一起计算
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);

            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

            sreg0 = dichotomyAddReminder;
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }

                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                DataCopy(dichotomyAddVarR, tmpVarLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarR, dichotomyAddVarR, varScale, pregLoop);
                }

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);

            // var out
            if constexpr (isStd == true && isM2Out == false) {
                Sqrt(std, var, pregMerge);
                StoreOneElementForDtypeT<T>(outVarLocal, std, pregMerge, j);
            } else {
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
            }

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>(); // dichotomyAddLocal 是复用的
        }
    }
}

/*
无尾块的计算公式：
    mean = sum(meant) / Ab
        其中 meant 是update后的结果，Ab是welford的并行度，即Ub内R的大小
    var = sum(vart + Rn * (meant - mean)^2) / (Rn*Ab)
        其中 vart 是update后的结果，Rn是R轴的for循环次数, Rn*Ab为实际的R轴大小
*/
template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeAlignLessVLPad(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint16_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt,
    uint32_t lastRAxisLen,       // lastRAxisLen: 尾轴R的实际大小
    uint32_t lastRAxisLenAlign,  // lastRAxisLen block对齐后的大小
    uint16_t loopLastRNum)       // 有多少个lastR
{
    uint16_t lastRAxisLoopNum = Ops::Base::CeilDiv(lastRAxisLen, VL_FP32);

    __VEC_SCOPE__
    {
        RegTensor<float> tmpMeanReg;
        RegTensor<float> tmpVarReg;
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> deltaMean;
        RegTensor<float> delta;
        RegTensor<float> padded;
        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < ANum; j++) {
            uint32_t sreg0 = RNum;
            // 计算mean
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(tmpMeanReg, tmpMeanLocal + j * RStride);
            Muls(deltaMean, tmpMeanReg, meanScale, pregLoop);
            ReduceSum(mean, deltaMean, pregLoop);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregLoop);
            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
            
            // 计算var时，pad的位置需要清零
            for (uint16_t ro = 0; ro < loopLastRNum; ro++) {
                uint32_t sreg1 = lastRAxisLen;
                for (uint16_t ri = 0; ri < lastRAxisLoopNum; ri++) {
                    pregLoop1 = UpdateMask<float>(sreg1);
                    DataCopy(tmpMeanReg, tmpMeanLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32);
                    Sub(delta, tmpMeanReg, mean, pregLoop1);
                    Mul(delta, delta, delta, pregLoop1);
                    Muls(delta, delta, cnt, pregLoop1);
                    DataCopy(tmpVarReg, tmpVarLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32);
                    Add(tmpVarReg, tmpVarReg, delta, pregLoop1);
                    if constexpr (isM2Out == false) {
                        Muls(tmpVarReg, tmpVarReg, varScale, pregLoop1);
                    }
                    DataCopy(((__local_mem__ float *)tmpVarLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32),
                            tmpVarReg, pregLoop1);
                }
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            DataCopy(tmpVarReg, tmpVarLocal + j * RStride);
            Muls(tmpVarReg, tmpVarReg, float(1.0), pregLoop);
            ReduceSum(var, tmpVarReg, pregLoop);

            if constexpr (isStd == true && isM2Out == false) {
                Sqrt(var, var, pregMerge);
            }

            StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
        }
    }
}

template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeARAlignPad(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ float *dichotomyAddLocal,
    __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint16_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt,
    uint32_t lastRAxisLen,       // lastRAxisLen: 尾轴R的实际大小
    uint32_t lastRAxisLenAlign,  // lastRAxisLen block对齐后的大小
    uint16_t loopLastRNum)       // 有多少个lastR
{
    if (RNum <= VL_FP32) {
        VFWelfordParallelFinalizeAlignLessVLPad<T, isStd, isM2Out>(tmpMeanLocal, tmpVarLocal,
            outMeanLocal, outVarLocal, ANum, RNum, RStride, varScale, meanScale, cnt, lastRAxisLen,
            lastRAxisLenAlign, loopLastRNum);
        return;
    }

    uint32_t dichotomyAddPower = Ops::Base::ReduceOpTmpl::FindNearestPower2(RNum);
    uint32_t dichotomyAddReminder = RNum - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = Ops::Base::CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;
    uint16_t dichtomyPowerReminder = (dichotomyAddPowerLoopCount > dichotomyAddReminderLoopCount) ?
        (dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount) : 0;
    uint16_t outerLoop = 0;
    uint16_t dichotomyAddLastNum = 0;
    if (dichotomyAddPowerLoopCount <= VL_FP32) {
        outerLoop = 0;
        dichotomyAddLastNum = dichotomyAddPowerLoopCount;
    } else {
        int64_t tmpBinaryAddNum = 1;
        while (tmpBinaryAddNum < innerLoopCountOrigin) {
            outerLoop = outerLoop + 1;
            tmpBinaryAddNum = tmpBinaryAddNum * Ops::Base::ReduceOpTmpl::CONST2;
        }
        dichotomyAddLastNum = VL_FP32;
    }

    uint16_t lastRAxisLoopNum = Ops::Base::CeilDiv(lastRAxisLen, VL_FP32);

    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
        RegTensor<float> std;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        RegTensor<float> padded;

        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < ANum; j++) {
            uint32_t sreg0 = dichotomyAddReminder;
            // 计算mean, 可以带着pad(0)一起计算
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);

            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

            // 计算var时，pad的位置需要清零
            for (uint16_t ro = 0; ro < loopLastRNum; ro++) {
                sreg0 = lastRAxisLen;
                for (uint16_t ri = 0; ri < lastRAxisLoopNum; ri++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32);
                    Sub(deltaL, dichotomyAddMeanL, mean, pregLoop);
                    Mul(deltaL, deltaL, deltaL, pregLoop);
                    Muls(deltaL, deltaL, cnt, pregLoop);
                    DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32);
                    Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregLoop);
                    if constexpr (isM2Out == false) {
                        Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregLoop);
                    }
                    DataCopy(((__local_mem__ float *)tmpVarLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32),
                            dichotomyAddVarL, pregLoop);
                }
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            // 二分累加
            sreg0 = dichotomyAddReminder;
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + i * VL_FP32);
                DataCopy(dichotomyAddVarR, tmpVarLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                // dichotomyAddVarR 可能非VL对齐，有脏数据
                Muls(deltaR, dichotomyAddVarR, float(1.0), pregLoop);
                Add(sumVar, dichotomyAddVarL, deltaR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }
            DichotomyAdd(var, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);

            if constexpr (isStd == true && isM2Out == false) {
                Sqrt(std, var, pregMerge);
                StoreOneElementForDtypeT<T>(outVarLocal, std, pregMerge, j);
            } else {
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
            }

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>(); // dichotomyAddLocal 是复用的
        }
    }
}

/*
    // Welford Finalize非对齐场景计算公式如下:
    counts = np.ones(len(mean), dtype=np.float32)*count
    for i in range(tail_size):
        counts[i] += 1
    finalize_mean = np.sum(mean*counts) / np.sum(counts)
    finalize_delta = mean - finalize_mean
    finalize_delta_square = finalize_delta * finalize_delta
    M2_fixed = M2 + counts * finalize_delta_square
    finalize_std = np.sum(M2_fixed) / np.sum(counts)

    // Welford Finalize非对齐场景下，二分累加存在如下几种场景:
    首先,非对齐场景下存在两种尾块类型
    1. welford自身的整块和尾块，需要注意的是，welford自身的整块也可能非对齐，整块+尾块=并行度
    2. 二分累加的整块和尾块
    3.
    3.1 welford整块小于二分累加整块，这种场景下，可以进一步细分为两种场景
    3.1.1 welford整块小于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
    3.1.2 welford整块大于等于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现剩余整块回刷UB逻辑中
    3.2 welford整块大于等于二分累加整块，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
*/
// welford整块大于等于二分累加整块
template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeARNonAlign1(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ float *dichotomyAddLocal,
    __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint32_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt, float tailCnt,
    uint32_t tailSize, uint32_t dichotomyAddPower)
{
    // cnt  是不带尾块的累加次数
    // tailCnt 是带尾块的累加次数， tailCnt >= cnt
    float coeff = tailCnt / cnt;
    float tailMeanCountScale = tailCnt * meanScale;
    float meanCountScale = cnt * meanScale;

    uint32_t dichotomyAddReminder = RNum - dichotomyAddPower;
    uint16_t dichotomyAddReminderRealLoopCount = Ops::Base::CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;
    uint16_t dichtomyPowerReminder = (dichotomyAddPowerLoopCount > dichotomyAddReminderRealLoopCount) ?
        (dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount) : 0;

    uint32_t welfordDiff = tailSize - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplit =
        (dichotomyAddReminder > (welfordDiffLoopCount + welfordReminderLoopCount) * VL_FP32) ?
        (dichotomyAddReminder - (welfordDiffLoopCount + welfordReminderLoopCount) * VL_FP32) : 0;
    uint16_t dichotomyAddReminderLoopCount = Ops::Base::CeilDiv(dichotomyAddReminderAfterSplit, VL_FP32);

    uint32_t dichotomyAddReminderRem = (dichotomyAddReminder > welfordDiffLoopCount * VL_FP32) ? 
        (dichotomyAddReminder - welfordDiffLoopCount * VL_FP32) : 0;

    uint16_t outerLoop = 0;
    uint16_t dichotomyAddLastNum = 0;
    if (dichotomyAddPowerLoopCount <= VL_FP32) {
        outerLoop = 0;
        dichotomyAddLastNum = dichotomyAddPowerLoopCount;
    } else {
        int64_t tmpBinaryAddNum = 1;
        while (tmpBinaryAddNum < innerLoopCountOrigin) {
            outerLoop = outerLoop + 1;
            tmpBinaryAddNum = tmpBinaryAddNum * Ops::Base::ReduceOpTmpl::CONST2;
        }
        dichotomyAddLastNum = VL_FP32;
    }

    uint16_t aLoopNum = static_cast<uint16_t>(ANum);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        RegTensor<float> std;
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < aLoopNum; j++) {
            uint32_t sreg0;

            // 整块使用tailCountScale,尾块使用tailCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailMeanCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, tailMeanCountScale, pregMain);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 处理welford第一次非对齐点, 整块使用tailCountScale,尾块部分使用tailCountScale, 部分使用countScale
            sreg0 = dichotomyAddReminderRem;
            uint32_t sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                pregLoop1 = UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailMeanCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanCountScale, pregLoop);
                Muls(tmp, dichotomyAddMeanR, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanR, tmp, pregLoop1);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount, mean,
                    pregMerge);
            }

            // 整块使用tailCountScale,尾块使用countScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                DataCopy(dichotomyAddMeanR,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailMeanCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanCountScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
            }
            // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailMeanCountScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
            }
            DichotomyAdd(mean, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);

            // 计算var
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
                Mul(deltaR, deltaR, deltaR, pregMain);
                Muls(deltaR, deltaR, tailCnt, pregMain);

                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                DataCopy(dichotomyAddVarR, tmpVarLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarR, dichotomyAddVarR, varScale, pregMain);
                }

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }
            sreg0 = dichotomyAddReminderRem;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                pregLoop1 = UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                Muls(tmp, deltaR, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(deltaR, tmp, pregLoop1);

                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                DataCopy(dichotomyAddVarR, tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarR, dichotomyAddVarR, varScale, pregLoop);
                }

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount, var,
                    pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);

                DataCopy(dichotomyAddVarL,
                    tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                DataCopy(dichotomyAddVarR,
                    tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarR, dichotomyAddVarR, varScale, pregLoop);
                }
                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
            }
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);

            // var out
            if constexpr (isStd == true && isM2Out == false) {
                Sqrt(std, var, pregMerge);
                StoreOneElementForDtypeT<T>(outVarLocal, std, pregMerge, j);
            } else {
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
            }

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>(); // dichotomyAddLocal 是复用的
        }
    }
}

// welford整块小于二分累加整块，并且小于二分累加尾块向上对齐
template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeARNonAlign2(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ float *dichotomyAddLocal,
    __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint32_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt, float tailCnt,
    uint32_t tailSize, uint32_t dichotomyAddPower)
{
    float coeff = tailCnt / cnt;
    float tailMeanCountScale = tailCnt * meanScale;
    float meanCountScale = cnt * meanScale;

    uint32_t dichotomyAddReminder = RNum - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = tailSize / VL_FP32;
    uint32_t welfordDiffReminder = tailSize - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint16_t dichotomyAddReminderRealLoopCount = Ops::Base::CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;
    uint16_t dichtomyPowerReminder = (dichotomyAddPowerLoopCount > dichotomyAddReminderRealLoopCount) ?
        (dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount) : 0;

    uint32_t dichotomyAddReminderAfterSplit =
        (dichotomyAddReminder > welfordDiffLoopCount * VL_FP32 + welfordDiffReminderAlign) ?
        (dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 - welfordDiffReminderAlign) : 0;
    uint16_t dichotomyAddReminderLoopCount = Ops::Base::CeilDiv(dichotomyAddReminderAfterSplit, VL_FP32);

    uint32_t dichotomyAddReminderRem = (dichotomyAddReminder > welfordDiffLoopCount * VL_FP32) ? 
        (dichotomyAddReminder - welfordDiffLoopCount * VL_FP32) : 0;

    uint16_t outerLoop = 0;
    uint16_t dichotomyAddLastNum = 0;
    if (dichotomyAddPowerLoopCount <= VL_FP32) {
        outerLoop = 0;
        dichotomyAddLastNum = dichotomyAddPowerLoopCount;
    } else {
        int64_t tmpBinaryAddNum = 1;
        while (tmpBinaryAddNum < innerLoopCountOrigin) {
            outerLoop = outerLoop + 1;
            tmpBinaryAddNum = tmpBinaryAddNum * Ops::Base::ReduceOpTmpl::CONST2;
        }
        dichotomyAddLastNum = VL_FP32;
    }

    uint16_t aLoopNum = static_cast<uint16_t>(ANum);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        RegTensor<float> std;
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < aLoopNum; j++) {
            uint32_t sreg0;
            uint32_t sreg1;

            // 整块使用tailMeanCountScale,尾块使用meanCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailMeanCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanCountScale, pregMain);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 处理welford第一次非对齐点, 尾块使用meanCountScale,整块部分使用tailMeanCountScale, 部分使用meanCountScale
            sreg0 = dichotomyAddReminderRem;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                pregLoop1 = UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanCountScale, pregLoop);
                Muls(tmp, dichotomyAddMeanL, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop1);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount, mean,
                    pregMerge);
            }

            // 整块使用meanCountScale,尾块使用meanCountScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                DataCopy(dichotomyAddMeanR,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanCountScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
            }
            // PART2: 整块剩余部分vcadd回刷UB,使用meanCountScale
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanCountScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);

            // 计算var
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
                Mul(deltaR, deltaR, deltaR, pregMain);
                Muls(deltaR, deltaR, cnt, pregMain);

                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                DataCopy(dichotomyAddVarR, tmpVarLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarR, dichotomyAddVarR, varScale, pregMain);
                }

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }
            sreg0 = dichotomyAddReminderRem;
            sreg1 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                pregLoop1 = UpdateMask<float>(sreg1);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                Muls(tmp, deltaL, coeff, pregLoop1);
                Copy<float, MaskMergeMode::MERGING>(deltaL, tmp, pregLoop1);

                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                DataCopy(dichotomyAddVarR, tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarR, dichotomyAddVarR, varScale, pregLoop);
                }

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount, var,
                    pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddMeanR,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);

                DataCopy(dichotomyAddVarL,
                    tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                DataCopy(dichotomyAddVarR,
                    tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarR, dichotomyAddVarR, varScale, pregLoop);
                }
                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            // var out
            if constexpr (isStd == true && isM2Out == false) {
                Sqrt(std, var, pregMerge);
                StoreOneElementForDtypeT<T>(outVarLocal, std, pregMerge, j);
            } else {
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
            }

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>(); // dichotomyAddLocal 是复用的
        }
    }
}

// 场景3：welford整块小于二分累加整块，并且大于等于二分累加尾块向上对齐
template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeARNonAlign3(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ float *dichotomyAddLocal,
    __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint32_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt, float tailCnt,
    uint32_t tailSize, uint32_t dichotomyAddPower)
{
    float coeff = tailCnt / cnt;
    float tailMeanCountScale = tailCnt * meanScale;
    float meanCountScale = cnt * meanScale;

    // 二分累加
    uint32_t dichotomyAddReminder = RNum - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = Ops::Base::CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;
    uint32_t dichotomyAddReminderRoundUp = dichotomyAddReminderLoopCount * VL_FP32;

    uint32_t welfordDiff = tailSize - dichotomyAddReminderRoundUp;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;
    uint16_t dichotomyAddPowerRemainLoopCount = dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount -
        welfordDiffLoopCount - welfordReminderLoopCount;
    uint32_t dichotomyAddPowerOffset =
        dichotomyAddReminderRoundUp + welfordDiffLoopCount * VL_FP32 + welfordDiffReminderAlign;

    uint16_t outerLoop = 0;
    uint16_t dichotomyAddLastNum = 0;
    if (dichotomyAddPowerLoopCount <= VL_FP32) {
        outerLoop = 0;
        dichotomyAddLastNum = dichotomyAddPowerLoopCount;
    } else {
        int64_t tmpBinaryAddNum = 1;
        while (tmpBinaryAddNum < innerLoopCountOrigin) {
            outerLoop = outerLoop + 1;
            tmpBinaryAddNum = tmpBinaryAddNum * Ops::Base::ReduceOpTmpl::CONST2;
        }
        dichotomyAddLastNum = VL_FP32;
    }

    uint16_t aLoopNum = static_cast<uint16_t>(ANum);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        RegTensor<float> std;
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < aLoopNum; j++) {
            uint32_t sreg0 = dichotomyAddReminder;
            // 整块使用tailMeanCountScale, 尾块使用meanCountScale
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailMeanCountScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanCountScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // 剩余整块需要拆分成多部分
            // 整块剩余部分回刷UB，整块使用tailMeanCountScale
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailMeanCountScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            sreg0 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanCountScale, pregMain);
                Muls(tmp, dichotomyAddMeanL, coeff, pregLoop);
                Copy<float, MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, mean, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPowerOffset);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanCountScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + dichotomyAddReminderLoopCount +
                    welfordDiffLoopCount + welfordReminderLoopCount + i,
                    mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);

            // 计算var
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

            sreg0 = dichotomyAddReminder;
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + i * VL_FP32);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }

                DataCopy(dichotomyAddMeanR, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
                Mul(deltaR, deltaR, deltaR, pregLoop);
                Muls(deltaR, deltaR, cnt, pregLoop);
                DataCopy(dichotomyAddVarR, tmpVarLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarR, dichotomyAddVarR, varScale, pregLoop);
                }

                Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }

            // 整块剩余部分回刷UB
            for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, tailCnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + i * VL_FP32 + dichotomyAddReminderRoundUp);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }

            sreg0 = welfordDiffReminder;
            for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddMeanL,
                    tmpMeanLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                Muls(tmp, deltaL, coeff, pregLoop);
                Copy<float, MaskMergeMode::MERGING>(deltaL, tmp, pregLoop);
                DataCopy(dichotomyAddVarL,
                    tmpVarLocal + j * RStride + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, var, pregMerge);
            }

            for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
                DataCopy(dichotomyAddMeanL, tmpMeanLocal + j * RStride + i * VL_FP32 + dichotomyAddPowerOffset);
                Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
                Mul(deltaL, deltaL, deltaL, pregMain);
                Muls(deltaL, deltaL, cnt, pregMain);
                DataCopy(dichotomyAddVarL, tmpVarLocal + j * RStride + i * VL_FP32 + dichotomyAddPowerOffset);
                Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
                if constexpr (isM2Out == false) {
                    Muls(dichotomyAddVarL, dichotomyAddVarL, varScale, pregMain);
                }
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + dichotomyAddReminderLoopCount +
                    welfordDiffLoopCount + welfordReminderLoopCount + i,
                    var, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            // var out
            if constexpr (isStd == true && isM2Out == false) {
                Sqrt(std, var, pregMerge);
                StoreOneElementForDtypeT<T>(outVarLocal, std, pregMerge, j);
            } else {
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
            }

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>(); // dichotomyAddLocal 是复用的
        }
    }
}

template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeARNonAlignLessVL(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint32_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt, float tailCnt,
    uint32_t tailSize)
{
    uint16_t aLoopNum = static_cast<uint16_t>(ANum);
    // cnt  是不带尾块的累加次数
    // tailCnt 是带尾块的累加次数， tailCnt >= cnt
    float coeff = tailCnt / cnt;
    float meanCountScale = cnt * meanScale;

    __VEC_SCOPE__
    {
        RegTensor<float> tmpMeanReg;
        RegTensor<float> tmpVarReg;
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> delta;
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregLoopTail;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();

        for (uint16_t j = 0; j < aLoopNum; j++) {
            uint32_t sreg0 = RNum;
            uint32_t sreg1 = tailSize;
            // 计算mean
            pregLoop = UpdateMask<float>(sreg0);
            pregLoopTail = UpdateMask<float>(sreg1);

            DataCopy(tmpMeanReg, tmpMeanLocal + j * RStride);
            Muls(tmpMeanReg, tmpMeanReg, meanCountScale, pregLoop);
            Muls(tmp, tmpMeanReg, coeff, pregLoopTail);
            Copy<float, MaskMergeMode::MERGING>(tmpMeanReg, tmp, pregLoopTail);
            ReduceSum(mean, tmpMeanReg, pregLoop);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);

            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            DataCopy(tmpMeanReg, tmpMeanLocal + j * RStride);
            Sub(delta, tmpMeanReg, mean, pregLoop);
            Mul(delta, delta, delta, pregLoop);
            Muls(delta, delta, cnt, pregLoop);
            Muls(tmp, delta, coeff, pregLoopTail);
            Copy<float, MaskMergeMode::MERGING>(delta, tmp, pregLoopTail);
            DataCopy(tmpVarReg, tmpVarLocal + j * RStride);
            Add(tmpVarReg, tmpVarReg, delta, pregLoop);
            if constexpr (isM2Out == false) {
                Muls(tmpVarReg, tmpVarReg, varScale, pregLoop);
            }
            ReduceSum(var, tmpVarReg, pregLoop);

            if constexpr (isStd == true && isM2Out == false) {
                Sqrt(var, var, pregMerge);
            }
            // var out
            StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
        }
    }
}

// cnt: 只有整块的累加次数
// tailCnt: 包含尾块的累加次数, tailCnt >= cnt
// RNum: welford整块的长度
// tailSize: welford尾块的长度
template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeARNonAlign(__local_mem__ float *tmpMeanLocal,
    __local_mem__ float *tmpVarLocal, __local_mem__ float *dichotomyAddLocal,
    __local_mem__ T *outMeanLocal, __local_mem__ T *outVarLocal,
    uint32_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt, float tailCnt,
    uint32_t tailSize)
{
    if (RNum <= VL_FP32) {
        VFWelfordParallelFinalizeARNonAlignLessVL<T, isStd, isM2Out>(tmpMeanLocal, tmpVarLocal,
            outMeanLocal, outVarLocal, ANum, RNum, RStride, varScale, meanScale, cnt, tailCnt, tailSize);
        return;
    }

    uint32_t dichotomyAddPower = Ops::Base::ReduceOpTmpl::FindNearestPower2(RNum); // 4096  8192   32768

    // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
    uint32_t dichotomyAddReminder = RNum - dichotomyAddPower;
    uint32_t dichotomyAddReminderRoundUp = Ops::Base::CeilAlign(dichotomyAddReminder, VL_FP32);
    if (tailSize >= dichotomyAddPower) {
        VFWelfordParallelFinalizeARNonAlign1<T, isStd, isM2Out>(tmpMeanLocal, tmpVarLocal, dichotomyAddLocal,
            outMeanLocal, outVarLocal, ANum, RNum, RStride, varScale, meanScale, cnt, tailCnt,
            tailSize, dichotomyAddPower);
        return;
    }
    if (tailSize < dichotomyAddReminderRoundUp) {
        VFWelfordParallelFinalizeARNonAlign2<T, isStd, isM2Out>(tmpMeanLocal, tmpVarLocal, dichotomyAddLocal,
            outMeanLocal, outVarLocal, ANum, RNum, RStride, varScale, meanScale, cnt, tailCnt,
            tailSize, dichotomyAddPower);
        return;
    }
    VFWelfordParallelFinalizeARNonAlign3<T, isStd, isM2Out>(tmpMeanLocal, tmpVarLocal, dichotomyAddLocal,
            outMeanLocal, outVarLocal, ANum, RNum, RStride, varScale, meanScale, cnt, tailCnt,
            tailSize, dichotomyAddPower);
}

template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordParallelFinalizeARNonAlignPad(LocalTensor<float> &tMeanTensor,
    LocalTensor<float> &tVarTensor, LocalTensor<float> &tDichTensor,
    LocalTensor<T> &outMeanTensor, LocalTensor<T> &outVarTensor,
    uint32_t ANum, uint32_t RNum, uint32_t RStride, float varScale, float meanScale, float cnt, float tailCnt,
    uint32_t tailSize,
    uint32_t loopWelfTailRCnt,   // 单次ub内，welford尾块包含多少个lastR
    uint32_t lastRAxisLen,       // lastRAxisLen: 尾轴R的实际大小
    uint32_t lastRAxisLenAlign,  // lastRAxisLen block对齐后的大小
    uint32_t loopLastRNum)       // 有多少个lastR
{
    __local_mem__ float *tmpMeanLocal = (__local_mem__ float *)tMeanTensor.GetPhyAddr();
    __local_mem__ float *tmpVarLocal = (__local_mem__ float *)tVarTensor.GetPhyAddr();
    __local_mem__ float *dichotomyAddLocal = (__local_mem__ float *)tDichTensor.GetPhyAddr();

    uint16_t aLoopNum = static_cast<uint16_t>(ANum);
    uint16_t lastRAxisLoopNum = Ops::Base::CeilDiv(lastRAxisLen, VL_FP32);
    uint16_t reminderLoopLastRNum = (loopLastRNum > loopWelfTailRCnt) ? (loopLastRNum - loopWelfTailRCnt) : 0;
    // cnt  是不带尾块的累加次数
    // tailCnt 是带尾块的累加次数， tailCnt >= cnt
    float coeff = tailCnt / cnt;
    float tailMeanCountScale = tailCnt * meanScale;
    float meanCountScale = cnt * meanScale;

    float tailMeanCountScaleVrt = static_cast<float>(1.0) / tailMeanCountScale;
    float meanCountScaleVrt = static_cast<float>(1.0) / meanCountScale;

    uint32_t srcShape[DICHOTOMY_ADD_COEFF] = {ANum, RNum};

    uint16_t loopWelfTailRCntLoops = static_cast<uint16_t>(loopWelfTailRCnt);
    __VEC_SCOPE__
    {
        RegTensor<float> regTmpMean;
        RegTensor<float> regTmpMeanTail;
        MaskReg pregLoop;
        MaskReg pregLoopTail;

        for (uint16_t j = 0; j < aLoopNum; j++) {
            // 计算mean时，pad的位置需要清零
            for (uint16_t ro = 0; ro < loopWelfTailRCntLoops; ro++) {
                uint32_t sreg0 = lastRAxisLen;
                for (uint16_t ri = 0; ri < lastRAxisLoopNum; ri++) {
                    pregLoopTail = UpdateMask<float>(sreg0);
                    DataCopy(regTmpMeanTail, tmpMeanLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32);
                    Muls(regTmpMeanTail, regTmpMeanTail, tailMeanCountScale, pregLoopTail);
                    // 写回 tmpMeanLocal, 保持pad位置还是0
                    DataCopy((tmpMeanLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32),
                            regTmpMeanTail, pregLoopTail);
                }
            }

            for (uint16_t ro = 0; ro < reminderLoopLastRNum; ro++) {
                uint32_t sreg0 = lastRAxisLen;
                for (uint16_t ri = 0; ri < lastRAxisLoopNum; ri++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    DataCopy(regTmpMean, tmpMeanLocal + j * RStride + (ro + loopWelfTailRCntLoops) * lastRAxisLenAlign + ri * VL_FP32);
                    Muls(regTmpMean, regTmpMean, meanCountScale, pregLoop);
                    // 写回 tmpMeanLocal, 保持pad位置还是0
                    DataCopy((tmpMeanLocal + j * RStride + (ro + loopWelfTailRCntLoops) * lastRAxisLenAlign + ri * VL_FP32),
                            regTmpMean, pregLoop);
                }
            }
        }
    }

    auto tmpBuf = tDichTensor.template ReinterpretCast<uint8_t>();
    if constexpr (IsSameType<T, float>::value) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(outMeanTensor, tMeanTensor, tmpBuf, srcShape, false);
    } else {
        LocalTensor<float> outMeanFp32 = outMeanTensor.template ReinterpretCast<float>();
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(outMeanFp32, tMeanTensor, tmpBuf, srcShape, false);
    }

    __local_mem__ float *outMeanFp32Addr = (__local_mem__ float*)outMeanTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> mean;
        RegTensor<float> regTmpMean;
        RegTensor<float> regTmpVar;
        RegTensor<float> delta;
        MaskReg pregLoop;
        for (uint16_t j = 0; j < aLoopNum; j++) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(mean, outMeanFp32Addr + j);
            for (uint16_t ro = 0; ro < loopWelfTailRCntLoops; ro++) {
                uint32_t sreg0 = lastRAxisLen;
                for (uint16_t ri = 0; ri < lastRAxisLoopNum; ri++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    DataCopy(regTmpMean, tmpMeanLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32);
                    Muls(regTmpMean, regTmpMean, tailMeanCountScaleVrt, pregLoop);
                    Sub(delta, regTmpMean, mean, pregLoop);
                    Mul(delta, delta, delta, pregLoop);
                    Muls(delta, delta, tailCnt, pregLoop);
                    DataCopy(regTmpVar, tmpVarLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32);
                    Add(regTmpVar, regTmpVar, delta, pregLoop);
                    if constexpr (isM2Out == false) {
                        Muls(regTmpVar, regTmpVar, varScale, pregLoop);
                    }
                    // 写回 tmpMeanLocal, 保持pad位置还是0
                    DataCopy((tmpVarLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32),
                            regTmpVar, pregLoop);
                }
            }

            for (uint16_t ro = 0; ro < reminderLoopLastRNum; ro++) {
                uint32_t sreg0 = lastRAxisLen;
                for (uint16_t ri = 0; ri < lastRAxisLoopNum; ri++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    DataCopy(regTmpMean, tmpMeanLocal + j * RStride + (ro + loopWelfTailRCntLoops) * lastRAxisLenAlign + ri * VL_FP32);
                    Muls(regTmpMean, regTmpMean, meanCountScaleVrt, pregLoop);
                    Sub(delta, regTmpMean, mean, pregLoop);
                    Mul(delta, delta, delta, pregLoop);
                    Muls(delta, delta, cnt, pregLoop);
                    DataCopy(regTmpVar, tmpVarLocal + j * RStride + (ro + loopWelfTailRCntLoops) * lastRAxisLenAlign + ri * VL_FP32);
                    Add(regTmpVar, regTmpVar, delta, pregLoop);
                    if constexpr (isM2Out == false) {
                        Muls(regTmpVar, regTmpVar, varScale, pregLoop);
                    }
                    // 写回 tmpMeanLocal, 保持pad位置还是0
                    DataCopy((tmpVarLocal + j * RStride + (ro + loopWelfTailRCntLoops) * lastRAxisLenAlign + ri * VL_FP32),
                            regTmpVar, pregLoop);
                }
            }
        }
    }

    if constexpr (IsSameType<T, float>::value) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(outVarTensor, tVarTensor, srcShape, false);
        if constexpr (isStd == true && isM2Out == false) {
            AscendC::Sqrt(outVarTensor, outVarTensor, ANum);
        }
    } else {
        LocalTensor<float> outVarFp32 = outVarTensor.template ReinterpretCast<float>();

        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(outVarFp32, tVarTensor, srcShape, false);

        if constexpr (isStd == true && isM2Out == false) {
            AscendC::Sqrt(outVarFp32, outVarFp32, ANum);
        }
        Cast(outVarTensor, outVarFp32, RoundMode::CAST_RINT, ANum);

        LocalTensor<float> outMeanFp32V1 = outMeanTensor.template ReinterpretCast<float>();
        Cast(outMeanTensor, outMeanFp32V1, RoundMode::CAST_RINT, ANum);
    }
}

// welford 累加次数, addTailCnt >= addCnt
__aicore__ inline void CaculateCountBuf(__local_mem__ float *tmpCountLocal, uint32_t RNum, uint32_t addTailRNum,
    uint32_t addCnt, uint32_t addTailCnt)
{
    uint16_t addCntLoopCnt = Ops::Base::CeilDiv(RNum, VL_FP32);
    uint16_t addTailCntLoopCnt = (addTailRNum == 0) ? 0 : Ops::Base::CeilDiv(addTailRNum, VL_FP32);
    float addCntF = static_cast<float>(addCnt);
    float addTailCntF = static_cast<float>(addTailCnt);

    __VEC_SCOPE__
    {
        RegTensor<float> tmpCount;

        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregLoop;

        uint32_t sreg1 = RNum;
        Duplicate(tmpCount, addCntF, pregMain);
        for (uint16_t i = 0; i < addCntLoopCnt; i++) {
            pregLoop = UpdateMask<float>(sreg1);
            DataCopy(((__local_mem__ float *)tmpCountLocal + i * VL_FP32), tmpCount, pregLoop);
        }
        uint32_t sreg2 = addTailRNum;
        Duplicate(tmpCount, addTailCntF, pregMain);
        for (uint16_t i = 0; i < addTailCntLoopCnt; i++) {
            pregLoop = UpdateMask<float>(sreg2);
            DataCopy(((__local_mem__ float *)tmpCountLocal + i * VL_FP32), tmpCount, pregLoop);
        }
    }
}

template <typename T>
__aicore__ inline void CalcFinalizeMean(uint32_t RNum, uint32_t ANum, LocalTensor<float> &updateMeanTensor,
    __local_mem__ float *tmpCountLocal, LocalTensor<T> &outMeanTensor, __local_mem__ float *dichotomyAddAddr,
    float meanFactor)
{
    __local_mem__ float *groupMeanBufAddr = (__local_mem__ float *)updateMeanTensor.GetPhyAddr();
    uint16_t aLoopCount = Ops::Base::CeilDiv(ANum, VL_FP32); // 1
    uint16_t rLoopCount = static_cast<uint16_t>(RNum);

    uint32_t srcShape[DICHOTOMY_ADD_COEFF] = {RNum, ANum};

    __VEC_SCOPE__
    {
        RegTensor<float> regTmpMean;
        RegTensor<float> regCnt;
        MaskReg pregLoop;
        uint32_t sreg0 = ANum;
        for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
            pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t rIndx = 0; rIndx < rLoopCount; rIndx++) {
                DataCopy(regTmpMean, groupMeanBufAddr + aIndex * VL_FP32 + rIndx * ANum);
                DataCopy(dichotomyAddAddr + aIndex * VL_FP32 + rIndx * ANum, regTmpMean, pregLoop);
                DataCopy<float, LoadDist::DIST_BRC_B32>(regCnt, (__local_mem__ float *)(tmpCountLocal) + rIndx);
                Muls(regTmpMean, regTmpMean, meanFactor, pregLoop);
                Mul(regTmpMean, regTmpMean, regCnt, pregLoop);
                DataCopy(groupMeanBufAddr + aIndex * VL_FP32 + rIndx * ANum, regTmpMean, pregLoop);
            }
        }
    }

    if constexpr (IsSameType<T, float>::value) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(outMeanTensor, updateMeanTensor, srcShape, false);
    } else {
        LocalTensor<float> outMeanFp32 = outMeanTensor.template ReinterpretCast<float>();
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(outMeanFp32, updateMeanTensor, srcShape, false);
    }
}

template <typename T, bool isM2Out = false>
__aicore__ inline void CalcFinalizeVar(uint32_t RNum, uint32_t ANum,
    LocalTensor<float> &updateVarTensor, __local_mem__ float *tmpCountLocal, __local_mem__ float *dichotomyAddAddr,
    LocalTensor<T> &outMeanTensor, LocalTensor<T> &outVarTensor, float varFactor)
{
    uint32_t srcShape[DICHOTOMY_ADD_COEFF] = {RNum, ANum};

    uint16_t aLoopCount = Ops::Base::CeilDiv(ANum, VL_FP32); // 1
    uint16_t rLoopCount = static_cast<uint16_t>(RNum);

    __local_mem__ float *groupVarBufAddr = (__local_mem__ float *)updateVarTensor.GetPhyAddr();

    __local_mem__ float *outMeanFp32Addr = (__local_mem__ float*)outMeanTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> mean;
        RegTensor<float> regCnt;
        RegTensor<float> regTmpMean;
        RegTensor<float> regTmpVar;
        RegTensor<float> delta;
        MaskReg pregLoop;

        uint32_t sreg0 = ANum;
        for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
            pregLoop = UpdateMask<float>(sreg0);
            
            for (uint16_t rIndx = 0; rIndx < rLoopCount; rIndx++) {
                DataCopy(regTmpMean, dichotomyAddAddr + aIndex * VL_FP32 + rIndx * ANum);
                DataCopy<float, LoadDist::DIST_BRC_B32>(regCnt, (__local_mem__ float *)(tmpCountLocal) + rIndx);
                DataCopy(mean, outMeanFp32Addr + aIndex * VL_FP32);
                
                Sub(delta, regTmpMean, mean, pregLoop);
                Mul(delta, delta, delta, pregLoop);
                Mul(delta, delta, regCnt, pregLoop);

                DataCopy(regTmpVar, groupVarBufAddr + aIndex * VL_FP32 + rIndx * ANum);
                Add(regTmpVar, regTmpVar, delta, pregLoop);
                if constexpr (isM2Out == false) {
                    Muls(regTmpVar, regTmpVar, varFactor, pregLoop);
                }
                DataCopy(groupVarBufAddr + aIndex * VL_FP32 + rIndx * ANum, regTmpVar, pregLoop);
            }
        }
    }

    if constexpr (IsSameType<T, float>::value) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(outVarTensor, updateVarTensor, srcShape, false);
    } else {
        LocalTensor<float> outVarFp32 = outVarTensor.template ReinterpretCast<float>();
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(outVarFp32, updateVarTensor, srcShape, false);
    }
}

template <typename T, bool isStd = false, bool isM2Out = false>
__aicore__ inline void VFWelfordFinalizeRA(uint32_t RNum, uint32_t ANum,
    LocalTensor<float> &updateMeanTensor,
    LocalTensor<float> &updateVarTensor,
    __local_mem__ float *tmpCountLocal,
    LocalTensor<T> &outMeanTensor, 
    LocalTensor<T> &outVarTensor,
    __local_mem__ float *dichotomyAddAddr,
    float meanFactor,
    float varFactor)
{
    CalcFinalizeMean<T>(RNum, ANum, updateMeanTensor, tmpCountLocal, outMeanTensor, dichotomyAddAddr, meanFactor);
    CalcFinalizeVar<T, isM2Out>(RNum, ANum, updateVarTensor, tmpCountLocal, dichotomyAddAddr, outMeanTensor,
        outVarTensor, varFactor);

    if constexpr (IsSameType<T, float>::value) {
        if constexpr (isStd == true && isM2Out == false) {
            AscendC::Sqrt(outVarTensor, outVarTensor, ANum);
        }
    } else {
        LocalTensor<float> outVarFp32 = outVarTensor.template ReinterpretCast<float>();
        if constexpr (isStd == true && isM2Out == false) {
            AscendC::Sqrt(outVarFp32, outVarFp32, ANum);
        }
        Cast(outVarTensor, outVarFp32, RoundMode::CAST_RINT, ANum);

        LocalTensor<float> outMeanFp32V1 = outMeanTensor.template ReinterpretCast<float>();
        Cast(outMeanTensor, outMeanFp32V1, RoundMode::CAST_RINT, ANum);
    }
}

}  // namespace ReduceOpTmpl
#endif // _REDUCE_VAR_WELFORD_H_