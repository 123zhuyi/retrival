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
 * \file bincount_simt_base.h
 * \brief The base class of bincount. The main function of this file is:
 */

#ifndef BINCOUNT_SIMT_BASE_H
#define BINCOUNT_SIMT_BASE_H

#include "bincount_tiling_data.h"

namespace BincountSimt {
using namespace AscendC;

// The depth of TQue when transfer out.
constexpr uint32_t OUT_QUE_DEPTH = 1;

// The number of double buffer.
constexpr uint32_t NUM_DOUBLE_BUFFER = 1;

// blcokcount of datacopypad.
constexpr uint16_t DATA_COPY_PAD_BLOCK_COUNT = 1;

// weight is 1.
constexpr int32_t WEIGHT_ONE = 1;

constexpr uint32_t THREAD_NUM = 1024;

template <typename WEIGHT_TYPE>
class BincountSimtBase {
public:
    // Constructor fucntion
    __aicore__ inline BincountSimtBase(){};
    // Init common global tensor and tiling data
    __aicore__ inline void BaseInit(
        GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
        TPipe* tPipe, bool isWeightEmpty);

protected:
    // Parse the tiling data.
    __aicore__ inline void ParseTilingData(const BincountTilingData* __restrict tilingData);

protected:
    // input array in global memory.
    GlobalTensor<int32_t> arrayGm_;
    // input weights in global memory.
    GlobalTensor<WEIGHT_TYPE> weightsGm_;
    // output in global memory
    GlobalTensor<WEIGHT_TYPE> binsGm_;

    // current aicore index.
    int32_t blockIdx_ = 0;

    // weights is float, isWeightEmpty_ equal false. weights is one, isWeightEmpty_ is true.
    bool isWeightEmpty_ = false;
    // the size of out, example: size = 5 the out is [0,0,0,0,0].
    int64_t size_ = 0;
    // the number of aicore for partcipate in calculations.
    int64_t needCoreNum_ = 0;
    // the length of ahead aicore.
    int64_t formerLength_ = 0;
    // the length of last aicore.
    int64_t tailLength_ = 0;
    // the number of aicore for reset bins.
    int64_t resetBinsCoreNum_ = 0;
    // length of each aicore for reset bins.
    int64_t resetBinsLength_ = 0;
    // last aicore length for reset bins.
    int64_t resetBinsTailLength_ = 0;
};

/**
 * @param array array of bincount
 * @param weights weight of array, if all weight of x is one, we don't need get weights by H2D.
 * @param size size of out, size can transfer to device by tiling data, so we don't need get size by H2D (host to
 * device).
 * @param bins output
 * @param tilingData tiling data
 * @param tPipe memory of management
 */
template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtBase<WEIGHT_TYPE>::BaseInit(
    GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
    TPipe* tPipe, bool isWeightEmpty)
{
    // parse tiling data and assign class variable.
    ParseTilingData(tilingData);
    isWeightEmpty_ = isWeightEmpty;
    arrayGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(array));
    if (!isWeightEmpty_) {
        weightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ WEIGHT_TYPE*>(weights));
    }
    binsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ WEIGHT_TYPE*>(bins));

    blockIdx_ = static_cast<int32_t>(GetBlockIdx());
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtBase<WEIGHT_TYPE>::ParseTilingData(const BincountTilingData* __restrict tilingData)
{
    size_ = tilingData->size;
    needCoreNum_ = tilingData->needXCoreNum;
    formerLength_ = tilingData->formerLength;
    tailLength_ = tilingData->tailLength;
    resetBinsCoreNum_ = tilingData->clearYCoreNum;
    resetBinsLength_ = tilingData->clearYFactor;
    resetBinsTailLength_ = tilingData->clearYTail;
}

} // namespace BincountSimt

#endif // BINCOUNT_SIMT_BASE_H