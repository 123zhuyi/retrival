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
 * \file reduce_var_twopass.h
 * \brief reduce var twopass
 */

#ifndef _REDUCE_VAR_TWOPASS_H_
#define _REDUCE_VAR_TWOPASS_H_

#include "reduce_var_vf_common.h"

namespace ReduceOpTmpl
{
using namespace AscendC;

template <typename T, bool isStd = false>
__aicore__ inline void VFMeanVarTwoPassARLessVL(__local_mem__ T* xInUb, __local_mem__ float* dichotomyAddLocal,
                                          __local_mem__ T* outMeanLocal, __local_mem__ T* outVarLocal, uint32_t ANum,
                                          uint32_t RStride, uint32_t RNum, float varScale)
{
    uint32_t nextR = FindNextPower2(RNum);
    float meanScale = float(1.0) / float(nextR);
    float meanCorrection = float(nextR) / float(RNum);

    uint16_t aLoopNum = static_cast<uint16_t>(ANum);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> meanSum;
        RegTensor<float> mean;

        // RegTensor<float> x1;
        RegTensor<float> y1;
        RegTensor<float> y1Scale;
        RegTensor<float> y1Pow;
        RegTensor<float> var;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        uint32_t sreg0 = RNum;
        MaskReg pregLoop = UpdateMask<float>(sreg0);
        for (uint16_t k = 0; k < aLoopNum; k++) {
            LoadOneTensorForDtypeT(xInUb, x, pregLoop, (k * RStride));
            Muls(meanSum, x, meanScale, pregLoop);
            ReduceSum(mean, meanSum, pregLoop);
            Muls(mean, mean, meanCorrection, pregMerge);

            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, k);
            Duplicate(mean, mean, pregMain);

            Sub(y1, x, mean, pregLoop);
            // 先乘以系数，防止平方溢出
            Muls(y1Scale, y1, varScale, pregLoop);
            Mul(y1Pow, y1Scale, y1, pregLoop);
            ReduceSum(var, y1Pow, pregLoop);
            // var out
            if constexpr (isStd == true) {
                Sqrt(var, var, pregMerge);
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, k);
            } else {
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, k);
            }
        }
    }
}

template <typename T, bool isStd = false>
__aicore__ inline void VFMeanVarTwoPassAR(__local_mem__ T* xInUb, __local_mem__ float* dichotomyAddLocal,
                                          __local_mem__ T* outMeanLocal, __local_mem__ T* outVarLocal, uint32_t ANum,
                                          uint32_t RStride, uint32_t RNum, float varScale)
{
    if (RNum <= VL_FP32) {
        VFMeanVarTwoPassARLessVL<T, isStd>(xInUb, dichotomyAddLocal, outMeanLocal, outVarLocal, ANum, RStride, RNum,
                                           varScale);
        return;
    }

    uint32_t nextR = FindNextPower2(RNum);
    float meanScale = float(1.0) / float(nextR);
    float meanCorrection = float(nextR) / float(RNum);

    uint32_t dichotomyAddPower = Ops::Base::ReduceOpTmpl::FindNearestPower2(RNum);
    uint32_t dichotomyAddReminder = RNum - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = Ops::Base::CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = dichotomyAddPowerLoopCount / VL_FP32;
    uint16_t dichtomyPowerReminder = (dichotomyAddPowerLoopCount > dichotomyAddReminderLoopCount)
                                         ? (dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount)
                                         : 0;
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
        RegTensor<float> x;
        RegTensor<float> meanSum;
        RegTensor<float> mean;

        RegTensor<float> x1;
        RegTensor<float> y1;
        RegTensor<float> y1Scale;
        RegTensor<float> y1Pow;
        RegTensor<float> varSum;
        RegTensor<float> var;

        RegTensor<float> binaryAddQ;
        RegTensor<float> binaryAddR;
        RegTensor<float> vlMean;

        RegTensor<float> binaryAddQScale;
        RegTensor<float> binaryAddRScale;
        RegTensor<float> binaryAddQPow;
        RegTensor<float> binaryAddRPow;
        RegTensor<float> vlVar;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t k = 0; k < aLoopNum; k++) {
            uint32_t sreg0 = dichotomyAddReminder;
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadTwoTensorForDtypeT(xInUb, xInUb, binaryAddQ, binaryAddR, pregMain, pregLoop,
                                       (i * VL_FP32 + k * RStride),
                                       (i * VL_FP32 + k * RStride + dichotomyAddPower));
                Muls(binaryAddQ, binaryAddQ, meanScale, pregMain);
                Muls(binaryAddR, binaryAddR, meanScale, pregLoop);
                Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                ReduceSum(vlMean, binaryAddQ, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)dichotomyAddLocal + i),
                                                                   vlMean, pregMerge);
            }
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                LoadOneTensorForDtypeT(xInUb, x, pregMain, ((i + dichotomyAddReminderLoopCount) * VL_FP32 + k * RStride));
                Muls(x, x, meanScale, pregMain);
                ReduceSum(vlMean, x, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)dichotomyAddLocal + dichotomyAddReminderLoopCount + i), vlMean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(mean, mean, meanCorrection, pregMerge);

            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, k);

            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);
            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

            uint32_t sreg1 = dichotomyAddReminder;
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg1);
                LoadTwoTensorForDtypeT(xInUb, xInUb, binaryAddQ, binaryAddR, pregMain, pregLoop,
                                       (i * VL_FP32 + k * RStride),
                                       (i * VL_FP32 + k * RStride + dichotomyAddPower));
                Sub(binaryAddQ, binaryAddQ, mean, pregMain);
                Sub(binaryAddR, binaryAddR, mean, pregLoop);
                // 先乘以系数，防止平方溢出
                Muls(binaryAddQScale, binaryAddQ, varScale, pregMain);
                Muls(binaryAddRScale, binaryAddR, varScale, pregLoop);
                Mul(binaryAddQPow, binaryAddQScale, binaryAddQ, pregMain);
                Mul(binaryAddRPow, binaryAddRScale, binaryAddR, pregLoop);
                Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregMain);
                ReduceSum(vlVar, binaryAddQPow, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)dichotomyAddLocal + i),
                                                                   vlVar, pregMerge);
            }
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                LoadOneTensorForDtypeT(xInUb, x1, pregMain, ((i + dichotomyAddReminderLoopCount) * VL_FP32 + k * RStride));
                Sub(y1, x1, mean, pregMain);
                Muls(y1Scale, y1, varScale, pregMain);
                Mul(y1Pow, y1Scale, y1, pregMain);
                ReduceSum(vlVar, y1Pow, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)dichotomyAddLocal + dichotomyAddReminderLoopCount + i), vlVar, pregMerge);
            }

            DichotomyAdd(var, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            // var out
            if constexpr (isStd == true) {
                Sqrt(var, var, pregMerge);
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, k);
            } else {
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, k);
            }
            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
        }
    }
}

template <typename T, bool isStd = false>
__aicore__ inline void VFMeanVarTwoPassARPadLessVL(__local_mem__ T* xInUb, __local_mem__ float* dichotomyAddLocal,
                                          __local_mem__ T* outMeanLocal, __local_mem__ T* outVarLocal, uint16_t ANum,
                                          uint16_t RStride, uint32_t RealRNum, float varScale, uint32_t lastRAxisLen,
                                          uint32_t lastRAxisLenAlign,
                                          uint16_t loopLastRNum)
{
    uint32_t nextR = FindNextPower2(RealRNum);
    float meanScale = float(1.0) / float(nextR);
    float meanCorrection = float(nextR) / float(RealRNum);

    uint16_t lastRAxisLoopNum = Ops::Base::CeilDiv(lastRAxisLen, VL_FP32);

    __VEC_SCOPE__
    {
        RegTensor<float> tmpMeanReg;
        RegTensor<float> tmpVarReg;
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> deltaMean;
        RegTensor<float> delta;
        RegTensor<float> deltaScale;
        RegTensor<float> deltaPow;
        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregLoop2;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < ANum; j++) {
            uint32_t sreg0 = RStride;
            // 计算mean
            pregLoop = UpdateMask<float>(sreg0);
            LoadOneTensorForDtypeT(xInUb, tmpMeanReg, pregLoop, j * RStride);
            Muls(deltaMean, tmpMeanReg, meanScale, pregLoop);
            ReduceSum(mean, deltaMean, pregLoop);
            Muls(mean, mean, meanCorrection, pregMerge);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);
            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregLoop);
            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
            
            // 计算var时，pad的位置需要清零
            uint32_t sreg1 = lastRAxisLenAlign;
            uint32_t sreg2 = lastRAxisLen;
            pregLoop1 = UpdateMask<float>(sreg1);
            pregLoop2 = UpdateMask<float>(sreg2);
            for (uint16_t ro = 0; ro < loopLastRNum; ro++) {
                // 用 lastRAxisLenAlign, 将 lastRAxisLen -> lastRAxisLenAlign 之间的数据清零
                LoadOneTensorForDtypeT(xInUb, tmpMeanReg, pregLoop2, j * RStride + ro * lastRAxisLenAlign);
                Sub(delta, tmpMeanReg, mean, pregLoop2);
                Muls(deltaScale, delta, varScale, pregLoop2);
                Mul(deltaPow, deltaScale, delta, pregLoop2);
                DataCopy(((__local_mem__ float *)dichotomyAddLocal + j * RStride + ro * lastRAxisLenAlign),
                            deltaPow, pregLoop1);
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            DataCopy(tmpVarReg, dichotomyAddLocal + j * RStride);
            Muls(tmpVarReg, tmpVarReg, float(1.0), pregLoop);
            ReduceSum(var, tmpVarReg, pregLoop);

            if constexpr (isStd == true) {
                Sqrt(var, var, pregMerge);
            }

            StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
        }
    }
}

template <typename T, bool isStd = false>
__aicore__ inline void VFMeanVarTwoPassARPad(__local_mem__ T* xInUb, __local_mem__ float* dichotomyAddLocal,
                                          __local_mem__ T* outMeanLocal, __local_mem__ T* outVarLocal, uint16_t ANum,
                                          uint16_t RStride, uint32_t RealRNum, float varScale, uint32_t lastRAxisLen,
                                          uint32_t lastRAxisLenAlign,
                                          uint16_t loopLastRNum)
{
    if (RStride <= VL_FP32) {
        VFMeanVarTwoPassARPadLessVL<T, isStd>(xInUb, dichotomyAddLocal, outMeanLocal, outVarLocal, ANum,
            RStride, RealRNum, varScale, lastRAxisLen, lastRAxisLenAlign, loopLastRNum);
        return;
    }

    uint32_t nextR = FindNextPower2(RealRNum);
    float meanScale = float(1.0) / float(nextR);
    float meanCorrection = float(nextR) / float(RealRNum);

    uint32_t dichotomyAddPower = Ops::Base::ReduceOpTmpl::FindNearestPower2(RStride);
    uint32_t dichotomyAddReminder = RStride - dichotomyAddPower;
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

    uint16_t lastRAxisLoopNumT = Ops::Base::CeilDiv(lastRAxisLen, VL_FP32);
    if (lastRAxisLoopNumT > 0) {
        lastRAxisLoopNumT = lastRAxisLoopNumT - 1;
    }

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

        RegTensor<float> x1;
        RegTensor<float> x1Scale;
        RegTensor<float> x1Pow;

        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t j = 0; j < ANum; j++) {
            uint32_t sreg0 = dichotomyAddReminder;
            // 计算mean, 可以带着pad(0)一起计算
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadTwoTensorForDtypeT(xInUb, xInUb, dichotomyAddMeanL, dichotomyAddMeanR, pregMain, pregLoop,
                                       (i * VL_FP32 + j * RStride),
                                       (i * VL_FP32 + j * RStride + dichotomyAddPower));
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanScale, pregMain);
                Muls(dichotomyAddMeanR, dichotomyAddMeanR, meanScale, pregLoop);
                Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                LoadOneTensorForDtypeT(xInUb, dichotomyAddMeanL, pregMain, j * RStride + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddMeanL, dichotomyAddMeanL, meanScale, pregMain);
                ReduceSum(mean, dichotomyAddMeanL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(mean, mean, meanCorrection, pregMerge);
            // mean out
            StoreOneElementForDtypeT<T>(outMeanLocal, mean, pregMerge, j);

            Duplicate<float, AscendC::MicroAPI::HighLowPart::LOWEST, MaskMergeMode::ZEROING>(mean, mean, pregMain);

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

            // 计算var时，pad的位置需要清零
            uint32_t sreg1 = lastRAxisLenAlign - lastRAxisLoopNumT * VL_FP32;
            pregLoop1 = UpdateMask<float>(sreg1);
            for (uint16_t ro = 0; ro < loopLastRNum; ro++) {
                sreg0 = lastRAxisLen;
                for (uint16_t ri = 0; ri < lastRAxisLoopNumT; ri++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadOneTensorForDtypeT(xInUb, x1, pregLoop, j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32);
                    Sub(x1, x1, mean, pregLoop);
                    // 先乘以系数，防止平方溢出
                    Muls(x1Scale, x1, varScale, pregLoop);
                    Mul(x1Pow, x1Scale, x1, pregLoop);
                    DataCopy(((__local_mem__ float *)dichotomyAddLocal + j * RStride + ro * lastRAxisLenAlign + ri * VL_FP32),
                            x1Pow, pregLoop);
                }
                {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadOneTensorForDtypeT(xInUb, x1, pregLoop, j * RStride + ro * lastRAxisLenAlign + lastRAxisLoopNumT * VL_FP32);
                    Sub(x1, x1, mean, pregLoop);
                    Muls(x1Scale, x1, varScale, pregLoop);
                    Mul(x1Pow, x1Scale, x1, pregLoop);
                    DataCopy(((__local_mem__ float *)dichotomyAddLocal + j * RStride + ro * lastRAxisLenAlign + lastRAxisLoopNumT * VL_FP32),
                            x1Pow, pregLoop1);
                }
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            // 二分累加
            sreg0 = dichotomyAddReminder;
            // PART1: 整尾块合并
            for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(dichotomyAddVarL, dichotomyAddLocal + j * RStride + i * VL_FP32);
                DataCopy(dichotomyAddVarR, dichotomyAddLocal + j * RStride + i * VL_FP32 + dichotomyAddPower);
                // dichotomyAddVarR 可能非VL对齐，有脏数据
                Muls(deltaR, dichotomyAddVarR, float(1.0), pregLoop);
                Add(sumVar, dichotomyAddVarL, deltaR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
            }

            // PART2: 整块剩余部分vcadd回刷UB
            for (uint16_t i = 0; i < dichtomyPowerReminder; i++) {
                DataCopy(dichotomyAddVarL, dichotomyAddLocal + j * RStride + (i + dichotomyAddReminderLoopCount) * VL_FP32);
                ReduceSum(var, dichotomyAddVarL, pregMain);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
            }
            DichotomyAdd(var, dichotomyAddLocal, outerLoop, innerLoopCountOrigin, dichotomyAddLastNum);

            if constexpr (isStd == true) {
                Sqrt(std, var, pregMerge);
                StoreOneElementForDtypeT<T>(outVarLocal, std, pregMerge, j);
            } else {
                StoreOneElementForDtypeT<T>(outVarLocal, var, pregMerge, j);
            }

            LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>(); // dichotomyAddLocal 是复用的
        }
    }
}

template <typename T, bool isStd = false>
__aicore__ inline void VFMeanVarTwoPassRAFp32(__local_mem__ T* xInUb, LocalTensor<float>& dichotomyAddLocal,
                                          LocalTensor<float>& tmpMeanTensor, LocalTensor<float>& tmpVarTensor,
                                          LocalTensor<T>& outMeanTensor, LocalTensor<T>& outVarTensor, uint32_t ANum,
                                          uint32_t RNum, float varScale)
{
    // xIn -> tmpMean ---reducesum---> out
    uint16_t aLoopCount = Ops::Base::CeilDiv(ANum, VL_FP32); // 1
    uint16_t rLoopCount = static_cast<uint16_t>(RNum);

    uint32_t nextR = FindNextPower2(RNum);
    float meanScale = float(1.0) / float(nextR);
    float meanCorrection = float(nextR) / float(RNum);

    uint32_t srcShape[DICHOTOMY_ADD_COEFF] = {RNum, ANum};

    __local_mem__ float* tmpMeanAddr = (__local_mem__ float*)tmpMeanTensor.GetPhyAddr();
    __local_mem__ float* tmpVarAddr = (__local_mem__ float*)tmpVarTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        MaskReg pregLoop;
        uint32_t sreg0 = ANum;
        for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
            pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t rIndx = 0; rIndx < rLoopCount; rIndx++) {
                LoadOneTensorForDtypeT(xInUb, x1, pregLoop, aIndex * VL_FP32 + rIndx * ANum);
                Muls(x1, x1, meanScale, pregLoop);
                DataCopy(tmpMeanAddr + aIndex * VL_FP32 + rIndx * ANum, x1, pregLoop);
            }
        }
    }

    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(outMeanTensor, tmpMeanTensor, srcShape, false);

    __local_mem__ T* outMeanAddr = (__local_mem__ T*)outMeanTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> mean;
        RegTensor<float> regCnt;
        RegTensor<float> regTmpMean;
        RegTensor<float> regTmpVar;
        RegTensor<float> delta;
        RegTensor<float> deltaScale;
        RegTensor<float> deltaPow;
        MaskReg pregLoop;

        uint32_t sreg0 = ANum;
        for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(mean, outMeanAddr + aIndex * VL_FP32);
            Muls(mean, mean, meanCorrection, pregLoop);
            DataCopy(outMeanAddr + aIndex * VL_FP32, mean, pregLoop);
            for (uint16_t rIndx = 0; rIndx < rLoopCount; rIndx++) {
                LoadOneTensorForDtypeT(xInUb, x1, pregLoop, aIndex * VL_FP32 + rIndx * ANum);

                Sub(delta, x1, mean, pregLoop);
                Muls(deltaScale, delta, varScale, pregLoop);
                Mul(deltaPow, deltaScale, delta, pregLoop);
                DataCopy(tmpVarAddr + aIndex * VL_FP32 + rIndx * ANum, deltaPow, pregLoop);
            }
        }
    }
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(outVarTensor, tmpVarTensor, srcShape, false);
    if constexpr (isStd == true) {
        AscendC::Sqrt(outVarTensor, outVarTensor, ANum);
    }
}

template <typename T, bool isStd = false>
__aicore__ inline void VFMeanVarTwoPassRAB16(__local_mem__ T* xInUb, LocalTensor<float>& dichotomyAddLocal,
                                          LocalTensor<float>& tmpMeanTensor, LocalTensor<float>& tmpVarTensor,
                                          LocalTensor<T>& outMeanTensor, LocalTensor<T>& outVarTensor, uint32_t ANum,
                                          uint32_t RNum, float varScale)
{
    // xIn -> cast -> tmpMean ---reducesum---> cast -> out
    uint16_t aLoopCount = Ops::Base::CeilDiv(ANum, VL_FP32);
    uint16_t rLoopCount = static_cast<uint16_t>(RNum);

    uint32_t nextR = FindNextPower2(RNum);
    float meanScale = float(1.0) / float(nextR);
    float meanCorrection = float(nextR) / float(RNum);

    uint32_t srcShape[DICHOTOMY_ADD_COEFF] = {RNum, ANum};

    __local_mem__ float* dichotomyAddAddr = (__local_mem__ float*)dichotomyAddLocal.GetPhyAddr();
    __local_mem__ float* tmpMeanAddr = (__local_mem__ float*)tmpMeanTensor.GetPhyAddr();
    __local_mem__ float* tmpVarAddr = (__local_mem__ float*)tmpVarTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        // RegTensor<float> regTmpMean;
        // RegTensor<float> regCnt;
        MaskReg pregLoop;
        uint32_t sreg0 = ANum;
        for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
            pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t rIndx = 0; rIndx < rLoopCount; rIndx++) {
                LoadOneTensorForDtypeT(xInUb, x1, pregLoop, aIndex * VL_FP32 + rIndx * ANum);
                Muls(x1, x1, meanScale, pregLoop);
                DataCopy(dichotomyAddAddr + aIndex * VL_FP32 + rIndx * ANum, x1, pregLoop);
            }
        }
    }

    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tmpMeanTensor, dichotomyAddLocal, srcShape, false);

    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> mean;
        RegTensor<float> regCnt;
        RegTensor<float> regTmpMean;
        RegTensor<float> regTmpVar;
        RegTensor<float> delta;
        RegTensor<float> deltaScale;
        RegTensor<float> deltaPow;
        MaskReg pregLoop;

        uint32_t sreg0 = ANum;
        for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(mean, tmpMeanAddr + aIndex * VL_FP32);
            Muls(mean, mean, meanCorrection, pregLoop);
            DataCopy(tmpMeanAddr + aIndex * VL_FP32, mean, pregLoop);
            for (uint16_t rIndx = 0; rIndx < rLoopCount; rIndx++) {
                LoadOneTensorForDtypeT(xInUb, x1, pregLoop, aIndex * VL_FP32 + rIndx * ANum);

                Sub(delta, x1, mean, pregLoop);
                Muls(deltaScale, delta, varScale, pregLoop);
                Mul(deltaPow, deltaScale, delta, pregLoop);
                DataCopy(dichotomyAddAddr + aIndex * VL_FP32 + rIndx * ANum, deltaPow, pregLoop);
            }
        }
    }
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tmpVarTensor, dichotomyAddLocal, srcShape, false);

    // 优化： 合并成一个VF
    if constexpr (isStd == true) {
        AscendC::Sqrt(tmpVarTensor, tmpVarTensor, ANum);
    }
    Cast(outVarTensor, tmpVarTensor, RoundMode::CAST_RINT, ANum);
    Cast(outMeanTensor, tmpMeanTensor, RoundMode::CAST_RINT, ANum);
}

template <typename T, bool isStd = false>
__aicore__ inline void VFMeanVarTwoPassRA(__local_mem__ T* xInUb, LocalTensor<float>& dichotomyAddLocal,
                                          LocalTensor<float>& tmpMeanTensor, LocalTensor<float>& tmpVarTensor,
                                          LocalTensor<T>& outMeanTensor, LocalTensor<T>& outVarTensor, uint32_t ANum,
                                          uint32_t RNum, float varScale)
{
    if constexpr (IsSameType<T, float>::value) {
        // 直接输出到outbuf上
        VFMeanVarTwoPassRAFp32<T, isStd>(xInUb, dichotomyAddLocal, tmpMeanTensor, tmpVarTensor, outMeanTensor, outVarTensor,
            ANum, RNum, varScale);
    } else {
        VFMeanVarTwoPassRAB16<T, isStd>(xInUb, dichotomyAddLocal, tmpMeanTensor, tmpVarTensor, outMeanTensor, outVarTensor,
            ANum, RNum, varScale);
    }
}

}  // namespace ReduceOpTmpl
#endif  // _REDUCE_VAR_TWOPASS_H_