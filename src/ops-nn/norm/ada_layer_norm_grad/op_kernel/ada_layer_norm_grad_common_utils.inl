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
 * \file ada_layer_norm_grad_common_utils.inl
 * \brief
 */


#ifndef ADA_LAYER_NORM_GRAD_COMMON_UTILS_INL
#define ADA_LAYER_NORM_GRAD_COMMON_UTILS_INL

namespace AdaLayerNormGrad {

template <typename T, typename U, bool isDeterministic>
template <typename dType>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::BlockBroadcast(
    const LocalTensor<dType>& dst, const LocalTensor<dType>& src, const int64_t curRowsNum)
{
    Brcb(dst, src, (curRowsNum + COMMON_CONSTANT_EIGHT - 1) / COMMON_CONSTANT_EIGHT, {1, COMMON_CONSTANT_EIGHT});
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::LastBrcFP32LargeStride(
    const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
    const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tiling,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
        BinaryRepeatParams paramsOne;
        paramsOne.dstBlkStride = 1;
        paramsOne.src0BlkStride = 1;
        paramsOne.src1BlkStride = 0;
        paramsOne.dstRepStride = 0;
        paramsOne.src0RepStride = 0;
        paramsOne.src1RepStride = 0;

        int64_t colMain = tiling->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t colRemain = tiling->colAlignV % COMMON_B32_REPEAT_SIZE;

        for (int64_t i = 0; i < curRowsNum; i++) {
            int64_t rowOffset = i * tiling->colAlignV;
            int64_t src1Offset = i * 8;

            for (int64_t j = 0; j < colMain; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            func(
                output[rowOffset + colOffset], input0[rowOffset + colOffset], input1[src1Offset],
                COMMON_B32_REPEAT_SIZE, 1, paramsOne);
            }
            if (colRemain != 0) {
            int64_t colOffset = colMain * COMMON_B32_REPEAT_SIZE;
            func(
                output[rowOffset + colOffset], input0[rowOffset + colOffset], input1[src1Offset], colRemain, 1,
                paramsOne);
            }
        }
    
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::BinElemWithInlinedLastBrcFP32(
    const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
    const int64_t rows, const AdaLayerNormGradTilingDataCommon* tiling,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
        int64_t colMain = tiling->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t colRemain = tiling->colAlignV - colMain * COMMON_B32_REPEAT_SIZE;
        int64_t rowMain = rows / COMMON_MAX_REPEAT;
        int64_t rowRemain = rows - rowMain * COMMON_MAX_REPEAT;

        if (tiling->colAlignV / COMMON_CONSTANT_EIGHT > 255) {
            LastBrcFP32LargeStride(output, input0, input1, rows, tiling, func);
        } else {
            BinaryRepeatParams params;
            params.dstBlkStride = 1;
            params.src0BlkStride = 1;
            params.src1BlkStride = 0;
            params.dstRepStride = tiling->colAlignV / COMMON_CONSTANT_EIGHT;
            params.src0RepStride = tiling->colAlignV / COMMON_CONSTANT_EIGHT;
            params.src1RepStride = 1;

            for (int64_t i = 0; i < rowMain; i++) {
            int64_t rowOffset = i * COMMON_MAX_REPEAT * tiling->colAlignV;
            for (int64_t j = 0; j < colMain; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(
                    output[rowOffset + colOffset], input0[rowOffset + colOffset],
                    input1[i * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT,
                    params);
            }
            if (likely(colRemain != 0)) {
                int64_t colOffset = colMain * COMMON_B32_REPEAT_SIZE;
                func(
                    output[rowOffset + colOffset], input0[rowOffset + colOffset],
                    input1[i * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], colRemain, COMMON_MAX_REPEAT, params);
            }
            }

            if (likely(rowRemain != 0)) {
            int64_t rowOffset = rowMain * COMMON_MAX_REPEAT * tiling->colAlignV;
            for (int64_t j = 0; j < colMain; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(
                    output[rowOffset + colOffset], input0[rowOffset + colOffset],
                    input1[rowMain * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], COMMON_B32_REPEAT_SIZE, rowRemain,
                    params);
            }
            if (likely(colRemain != 0)) {
                int64_t colOffset = colMain * COMMON_B32_REPEAT_SIZE;
                func(
                    output[rowOffset + colOffset], input0[rowOffset + colOffset],
                    input1[rowMain * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], colRemain, rowRemain, params);
            }
            }
        }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::BinElemNLastBrcLargeStride(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t, const BinaryRepeatParams&))
{
    BinaryRepeatParams paramsOne;
    paramsOne.dstRepStride = 0;
    paramsOne.src0RepStride = 0;
    paramsOne.src1RepStride = 0; 
    paramsOne.dstBlkStride = 1;
    paramsOne.src0BlkStride = 1;
    paramsOne.src1BlkStride = 1; 

    int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = tilingData->colAlignV % COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
            int64_t rowOffset = i * tilingData->colAlignV;

            for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            func(
                dst[rowOffset + colOffset], src0[rowOffset + colOffset], src1[colOffset], COMMON_B32_REPEAT_SIZE, 1,
                paramsOne);
            }

            if (remainderCol != 0) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            func(dst[rowOffset + colOffset], src0[rowOffset + colOffset], src1[colOffset], remainderCol, 1, paramsOne);
            }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::BinElemWithInlinedNLastBrcFP32(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
    if (unlikely(tilingData->colAlignV / COMMON_CONSTANT_EIGHT > 255)) {
            BinElemNLastBrcLargeStride(dst, src0, src1, curRowsNum, tilingData, func);
    } else {
            int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
            int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
            int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
            int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;

            BinaryRepeatParams intriParams;
            intriParams.dstBlkStride = 1;
            intriParams.src0BlkStride = 1;
            intriParams.src1BlkStride = 1;
            intriParams.dstRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
            intriParams.src0RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
            intriParams.src1RepStride = 0;

            for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(
                    dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset],
                    COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                func(
                    dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset], remainderCol,
                    COMMON_MAX_REPEAT, intriParams);
            }
            }

            if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(
                    dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset],
                    COMMON_B32_REPEAT_SIZE, remainderRepeat, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                func(
                    dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset], remainderCol,
                    remainderRepeat, intriParams);
            }
            }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::NlastBatchReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum,
    const int64_t startRow, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    int64_t computeStartBatch = startRow / tilingData->seq;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t curBatch = (startRow + i) / tilingData->seq;
        int64_t batchOffset = curBatch - computeStartBatch;
        int64_t dstOffset = batchOffset * tilingData->colAlignV;
        int64_t srcOffset = i * tilingData->colAlignV;
        
        Add(dst[dstOffset], src[srcOffset], dst[dstOffset], tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::NlastBatchMul(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum, const int64_t startRow, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    int64_t computeStartBatch = startRow / tilingData->seq;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t curBatch = (startRow + i) / tilingData->seq;
        int64_t batchOffset = curBatch - computeStartBatch;
        int64_t srcOffset = batchOffset * tilingData->colAlignV;
        int64_t dstOffset = i * tilingData->colAlignV;
        Mul(dst[dstOffset], src0[dstOffset], src1[srcOffset], tilingData->colAlignV);
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::NlastReduceSumLargeStride(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum,
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    BinaryRepeatParams paramsOne;
    paramsOne.dstBlkStride = 1;
    paramsOne.src0BlkStride = 1;
    paramsOne.src1BlkStride = 1;
    paramsOne.dstRepStride = 0;
    paramsOne.src0RepStride = 0;
    paramsOne.src1RepStride = 0;

    int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = tilingData->colAlignV % COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t srcRowOffset = i * tilingData->colAlignV;

        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            Add(dst[colOffset], src[srcRowOffset + colOffset], dst[colOffset],
                COMMON_B32_REPEAT_SIZE, 1, paramsOne);
        }

        if (remainderCol != 0) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            Add(dst[colOffset], src[srcRowOffset + colOffset], dst[colOffset],
                remainderCol, 1, paramsOne);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::NlastReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum,
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    if (unlikely(tilingData->colAlignV / COMMON_CONSTANT_EIGHT > 255)) {
        NlastReduceSumLargeStride(dst, src, curRowsNum, tilingData);
    } else {
        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;

        BinaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.src0BlkStride = 1;
        intriParams.src1BlkStride = 1;
        intriParams.dstRepStride = 0;
        intriParams.src0RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        intriParams.src1RepStride = 0;

        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], COMMON_B32_REPEAT_SIZE,
                    COMMON_MAX_REPEAT, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], remainderCol, COMMON_MAX_REPEAT,
                    intriParams);
            }
            PipeBarrier<PIPE_V>();
        }

        if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], COMMON_B32_REPEAT_SIZE,
                    remainderRepeat, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], remainderCol, remainderRepeat,
                    intriParams);
            }
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::LastReduceSumLargeStride(
    const LocalTensor<float>& tmp, const LocalTensor<float>& src, 
    const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    BinaryRepeatParams intriParamsOne;
    intriParamsOne.dstBlkStride = 1;
    intriParamsOne.src0BlkStride = 1;
    intriParamsOne.src1BlkStride = 1;
    intriParamsOne.dstRepStride = 0;
    intriParamsOne.src0RepStride = 0;
    intriParamsOne.src1RepStride = 0;

    int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = tilingData->colAlignV % COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t tmpRowOffset = i * COMMON_CONSTANT_EIGHT * 8; 
        int64_t srcRowOffset = i * tilingData->colAlignV;

        for (int64_t j = 1; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            Add(tmp[tmpRowOffset], tmp[tmpRowOffset], src[srcRowOffset + colOffset], 
                COMMON_B32_REPEAT_SIZE, 1, intriParamsOne);
        }
        
        if (likely(remainderCol != 0)) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            Add(tmp[tmpRowOffset], tmp[tmpRowOffset], src[srcRowOffset + colOffset], 
                remainderCol, 1, intriParamsOne);
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::LastReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmp,
    const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    if (tilingData->colAlignV <= COMMON_B32_REPEAT_SIZE) {
        int64_t repeatLoops = curRowsNum / COMMON_VC_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_VC_MAX_REPEAT;

        for (int64_t i = 0; i < repeatLoops; i++) {
            WholeReduceSum(
                dst[i * COMMON_VC_MAX_REPEAT], src[i * COMMON_VC_MAX_REPEAT * tilingData->colAlignV],
                tilingData->colAlignV, COMMON_VC_MAX_REPEAT, 1, 1, tilingData->colAlignV / COMMON_CONSTANT_EIGHT);
        }

        if (likely(remainderRepeat != 0)) {
            WholeReduceSum(
                dst[repeatLoops * COMMON_VC_MAX_REPEAT],
                src[repeatLoops * COMMON_VC_MAX_REPEAT * tilingData->colAlignV], tilingData->colAlignV, remainderRepeat,
                1, 1, tilingData->colAlignV / COMMON_CONSTANT_EIGHT);
        }
    } else {
        DataCopyParams copyIntriParams;
        copyIntriParams.blockCount = curRowsNum;
        copyIntriParams.blockLen = COMMON_CONSTANT_EIGHT;
        copyIntriParams.srcStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT - COMMON_CONSTANT_EIGHT;
        copyIntriParams.dstStride = 0;

        DataCopy(tmp, src, copyIntriParams);
        PipeBarrier<PIPE_V>();

        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
        if (tilingData->colAlignV / COMMON_CONSTANT_EIGHT > 255) {
            LastReduceSumLargeStride(tmp, src, curRowsNum, tilingData);
        } else {
            BinaryRepeatParams intriParams;
            intriParams.dstBlkStride = 1;
            intriParams.src0BlkStride = 1;
            intriParams.src1BlkStride = 1;
            intriParams.dstRepStride = COMMON_CONSTANT_EIGHT;
            intriParams.src0RepStride = COMMON_CONSTANT_EIGHT;
            intriParams.src1RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;

            for (int64_t i = 1; i < formerColLoops; i++) {
                Add(tmp, tmp, src[i * COMMON_B32_REPEAT_SIZE], COMMON_B32_REPEAT_SIZE, curRowsNum, intriParams);
                PipeBarrier<PIPE_V>();
            }
            if (likely(remainderCol != 0)) {
                Add(tmp, tmp, src[formerColLoops * COMMON_B32_REPEAT_SIZE], remainderCol, curRowsNum, intriParams);
            }
        }
        PipeBarrier<PIPE_V>();

        WholeReduceSum(dst, tmp, COMMON_B32_REPEAT_SIZE, curRowsNum, 1, 1, COMMON_CONSTANT_EIGHT);
    }
}

} // namespace AdaLayerNormGrad

#endif