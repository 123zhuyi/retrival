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
 * \file ones_like.h
 * \brief
 */
#ifndef ONES_LIKE_H
#define ONES_LIKE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ones_like_tiling_data.h"
#include "ones_like_tiling_key.h"

#define Sync(PIPE_SRC, PIPE_DST)                                                                               \
    {                                                                                                          \
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::PIPE_SRC##_##PIPE_DST)); \
        TQueSync<PIPE_##PIPE_SRC, PIPE_##PIPE_DST> queSync;                                                    \
        queSync.SetFlag(eventID);                                                                              \
        queSync.WaitFlag(eventID);                                                                             \
    }

namespace NsOnesLike {

using namespace AscendC;

constexpr int32_t ALIGN_BYTE = 32;

template <typename T>
class OnesLike {
public:
    __aicore__ inline OnesLike(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const OnesLikeTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut(AscendC::LocalTensor<T> yLocal, uint32_t offset, uint32_t dataLength);
    __aicore__ inline void Compute(AscendC::LocalTensor<T> yLocal, uint32_t dataLength);

private:
    TBuf<TPosition::VECOUT> yBuf;

    GlobalTensor<T> outputGMY;

    int64_t tileNum_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t lastTileLength_ = 0;
};

template <typename T>
__aicore__ inline void OnesLike<T>::Init(GM_ADDR x, GM_ADDR y, const OnesLikeTilingData* tilingData, TPipe* pipe)
{
    if (GetBlockIdx() < tilingData->formerNum) {
        this->tileNum_ = tilingData->formerTileNum;
        this->tileLength_ = tilingData->formerTileLength;
        this->lastTileLength_ = tilingData->formerLastTileLength;

        outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->formerLength * GetBlockIdx(), tilingData->formerLength);
    } else {
        this->tileNum_ = tilingData->tailTileNum;
        this->tileLength_ = tilingData->tailTileLength;
        this->lastTileLength_ = tilingData->tailLastTileLength;

        outputGMY.SetGlobalBuffer(
            (__gm__ T*)y + tilingData->formerLength * tilingData->formerNum +
                tilingData->tailLength * (GetBlockIdx() - tilingData->formerNum),
            tilingData->tailLength);
    }

    pipe->InitBuffer(yBuf, tileLength_ * sizeof(T));
}
template <typename T>
__aicore__ inline void OnesLike<T>::CopyOut(AscendC::LocalTensor<T> yLocal, uint32_t offset, uint32_t dataLength)
{
    if (dataLength * sizeof(T) % ALIGN_BYTE == 0) {
        AscendC::DataCopy(outputGMY[offset], yLocal, dataLength);
    } else {
        DataCopyExtParams copyParams = {1, (uint32_t)(dataLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
}

template <typename T>
__aicore__ inline void OnesLike<T>::Compute(AscendC::LocalTensor<T> yLocal, uint32_t dataLength)
{
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, bool>) {
        AscendC::Duplicate(yLocal.template ReinterpretCast<uint16_t>(), (uint16_t)0x0101, (dataLength + 1) / 2);
    } else if constexpr (std::is_same_v<T, half>) {
        AscendC::Duplicate(yLocal, (T)1, dataLength);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        AscendC::Duplicate(yLocal, (T)1, dataLength);
    } else if constexpr (std::is_same_v<T, float>) {
        AscendC::Duplicate(yLocal, (T)1, dataLength);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        AscendC::Duplicate(yLocal.template ReinterpretCast<uint16_t>(), (uint16_t)0x3F80, dataLength);
    }
}

template <typename T>
__aicore__ inline void OnesLike<T>::Process()
{
    LocalTensor<T> yLocal = yBuf.Get<T>();

    Compute(yLocal, tileLength_);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < tileNum_ - 1; i++) {
        Sync(V, MTE3) CopyOut(yLocal, offset, tileLength_);
        offset += tileLength_;
    }
    if (lastTileLength_ > 0) {
        Sync(V, MTE3) CopyOut(yLocal, offset, lastTileLength_);
    }
}

} // namespace NsOnesLike
#endif // ONES_LIKE_H