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
 * \file chunk_cat_common.h
 * \brief
 */

#ifndef _CHUNK_CAT_COMMON_DATA_H_
#define _CHUNK_CAT_COMMON_DATA_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "chunk_cat_tiling_data.h"

constexpr uint32_t UB_BLOCK_SIZE = 32; // UB块大小
constexpr uint32_t TRANS_BLOCK = 16;   // 转置行数
constexpr uint32_t HALF = 2;           // 半对齐/UB对半切分
constexpr int64_t NUM_THIRTY_ONE = 31;
constexpr int64_t NUM_TWO = 2;

struct TensorInfo {
    bool isSplit{false};
    bool isZero{false};
    int64_t chunkDimSize{0};
    int64_t chunkCol{0};
    int64_t chunkRow{0};
    int64_t chunkRowAlign{0};
    int64_t originCol{1};
    int64_t tensorCol{0};
    int64_t splitCol{0};
    int64_t splitColAlign{0};
    int64_t startOffset{0};
};

struct UbLoopInfo {
    bool isAllZero{true};
    int64_t count{0};
    int64_t currentUbRowFactor{0};
    int64_t currentUbColFactor{0};
    int64_t ubRowGroup{0};
    int64_t ubColGroup{0};
    int64_t totalUbCol{0};
    int64_t totalUbColAlign{0};
    int64_t colStart{0};
    int64_t rowStart{0};
    int64_t preCatCol{0};
    int64_t* inputCol;
};

using namespace AscendC;
template <typename T1, typename T2>
class ChunkCatCommon {
public:
    __aicore__ inline ChunkCatCommon(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void InitCommon(GM_ADDR x, GM_ADDR y, const ChunkCatTilingData& tilingData)
    {
        blockIdx_ = GetBlockIdx();
        // 获取tiling信息
        dim_ = tilingData.dim;
        numChunk_ = tilingData.numChunk;
        outputRow_ = tilingData.outputRow;
        outputCol_ = tilingData.outputCol;
        blockRowFactor_ = tilingData.blockRowFactor;
        blockColFactor_ = tilingData.blockColFactor;
        tailBlockRowFactor_ = tilingData.tailBlockRowFactor;
        tailBlockColFactor_ = tilingData.tailBlockColFactor;
        ubRowFactor_ = tilingData.ubRowFactor;
        ubColFactor_ = tilingData.ubColFactor;
        inputNum_ = tilingData.inputNum;
        srcEleUbBlock_ = UB_BLOCK_SIZE / sizeof(T1);
        dstEleUbBlock_ = UB_BLOCK_SIZE / sizeof(T2);

        blockRowGroup_ = blockIdx_ / tilingData.blockColNum;
        blockColGroup_ = blockIdx_ % tilingData.blockColNum;
        currentBlockRowFactor_ = blockRowGroup_ == tilingData.blockRowNum - 1 ? tailBlockRowFactor_ : blockRowFactor_;
        currentBlockColFactor_ = blockColGroup_ == tilingData.blockColNum - 1 ? tailBlockColFactor_ : blockColFactor_;
        int64_t dstGmOffset = blockRowGroup_ * blockRowFactor_ * outputCol_ + blockColGroup_ * blockColFactor_;
        dstGlobal_.SetGlobalBuffer((__gm__ T2*)y + dstGmOffset);
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));

        pipe_->InitBuffer(srcBuf_, tilingData.inUbSize);
        pipe_->InitBuffer(dstBuf_, tilingData.outUbSize);
        srcLocal_ = srcBuf_.Get<T1>();
        dstLocal_ = dstBuf_.Get<T2>();
    }

    __aicore__ inline int64_t GetAlign(int64_t value, int64_t align)
    {
        return align == 0 ? value : (value + align - 1) / align * align;
    }

    __aicore__ inline void GetChunkInfo(int32_t idx, TensorInfo& tensorInfo)
    {
        inputList_.GetDesc(desc_, idx); // scalar很大(将buf改为局部变量有改善)
        // 获取chunk相关信息
        tensorInfo.chunkDimSize = desc_.GetShape(dim_);
        tensorInfo.chunkCol = (tensorInfo.chunkDimSize + numChunk_ - 1) / numChunk_;
        // 获取concat阶段输入的col
        for (uint32_t j = 1; j < desc_.GetDim(); j++) {
            tensorInfo.originCol *= desc_.GetShape(j);
        }
        tensorInfo.tensorCol = tensorInfo.chunkCol * tensorInfo.originCol;
    }

    __aicore__ inline bool IsTensorInRange(int64_t totalCol, const UbLoopInfo& ubLoopInfo, const TensorInfo& tensorInfo)
    {
        return (totalCol < ubLoopInfo.colStart + ubLoopInfo.currentUbColFactor) &&
               (totalCol + tensorInfo.tensorCol > ubLoopInfo.colStart);
    }

    __aicore__ inline void SplitTensorDim0(int64_t& totalCol, const UbLoopInfo& ubLoopInfo, TensorInfo& tensorInfo)
    {
        // tensor是否被切分
        tensorInfo.splitCol = tensorInfo.tensorCol;
        int64_t colEnd = ubLoopInfo.colStart + ubLoopInfo.currentUbColFactor;
        if (totalCol < ubLoopInfo.colStart && (totalCol + tensorInfo.tensorCol) > colEnd) {
            // 中间部分
            tensorInfo.isSplit = true;
            tensorInfo.splitCol = ubLoopInfo.currentUbColFactor;
            tensorInfo.startOffset = ubLoopInfo.colStart - totalCol;
        } else if (totalCol < ubLoopInfo.colStart) {
            // 被切分的后半部分
            tensorInfo.isSplit = true;
            tensorInfo.splitCol = totalCol + tensorInfo.tensorCol - ubLoopInfo.colStart;
            tensorInfo.startOffset = ubLoopInfo.colStart - totalCol;
        } else if ((totalCol + tensorInfo.tensorCol) > colEnd) {
            // 被切分的前半部分
            tensorInfo.isSplit = true;
            tensorInfo.splitCol = colEnd - totalCol;
        }
        tensorInfo.splitColAlign = GetAlign(tensorInfo.splitCol, srcEleUbBlock_);
    }

    __aicore__ inline void ExecuteDataCopy(int64_t localOffset, int64_t gmOffset, uint16_t blockCount,
                                           uint32_t blockLen, uint32_t srcStride)
    {
        AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, 0, 0};
        uint8_t rightPadValue = (GetAlign(blockLen, UB_BLOCK_SIZE) - blockLen) / sizeof(T1);
        AscendC::DataCopyPadExtParams<T1> padParams{true, 0, rightPadValue, 0};
#if __CCE_AICORE__ == 310
        AscendC::DataCopyPad<T1, PaddingMode::Compact>(srcLocal_[localOffset], srcGlobal_[gmOffset], copyParams,
                                                       padParams);
#else
        AscendC::DataCopyPad(srcLocal_[localOffset], srcGlobal_[gmOffset], copyParams, padParams);
#endif
    }

    __aicore__ inline void DoRowsCopy(int64_t localOffset, const UbLoopInfo& ubLoopInfo, const TensorInfo& tensorInfo)
    {
        uint16_t blockCount = tensorInfo.isSplit ? static_cast<uint16_t>(ubLoopInfo.currentUbRowFactor) : 1;
        uint32_t blockLen = tensorInfo.isSplit ?
                                static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1)) :
                                static_cast<uint32_t>(ubLoopInfo.currentUbRowFactor * tensorInfo.splitCol * sizeof(T1));
        uint32_t srcStride = (tensorInfo.tensorCol - tensorInfo.splitCol) * sizeof(T1);
        int64_t gmOffset = ubLoopInfo.rowStart * tensorInfo.tensorCol + tensorInfo.startOffset;
        ExecuteDataCopy(localOffset, gmOffset, blockCount, blockLen, srcStride);
    }

    __aicore__ inline void DoLastRowsCopy(int64_t localOffset, const UbLoopInfo& ubLoopInfo,
                                          const TensorInfo& tensorInfo)
    {
        // 0 无切分
        int64_t srcGmOffset = ubLoopInfo.rowStart * tensorInfo.tensorCol;
        uint32_t srcStride = (tensorInfo.tensorCol - tensorInfo.splitCol) * sizeof(T1);
        if (!tensorInfo.isSplit) {
            uint32_t blockLen = static_cast<uint32_t>(
                (tensorInfo.chunkDimSize * tensorInfo.originCol - ubLoopInfo.rowStart * tensorInfo.tensorCol) *
                sizeof(T1));
            ExecuteDataCopy(localOffset, srcGmOffset + tensorInfo.startOffset, 1, blockLen, srcStride);
            return;
        }

        uint16_t blockCount = 0;
        uint32_t blockLen = 0;
        int64_t remainderCol = (tensorInfo.chunkDimSize % tensorInfo.chunkCol) * tensorInfo.originCol;
        // 1 有切分
        // 1.0 remainder等于0
        if (remainderCol == 0) {
            blockCount = static_cast<uint16_t>(tensorInfo.chunkRow - ubLoopInfo.rowStart);
            blockLen = static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1));
        }
        // 1.1 切分+偏移值小于等于remainder
        else if (tensorInfo.startOffset + tensorInfo.splitCol <= remainderCol) {
            blockCount = static_cast<uint16_t>(tensorInfo.chunkRowAlign - ubLoopInfo.rowStart);
            blockLen = static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1));
        }
        // 1.2 偏移值大于等于remainder
        else if (tensorInfo.startOffset >= remainderCol) {
            blockCount = static_cast<uint16_t>(tensorInfo.chunkRow - ubLoopInfo.rowStart);
            blockLen = static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1));
        }
        // 1.3  偏移值小于remainder，且切分+偏移值大于remainder
        else {
            // 1.3.1
            blockLen = (remainderCol - tensorInfo.startOffset) * sizeof(T1);
            int64_t localOffsetPart = 0;
#if __CCE_AICORE__ == 310
            localOffsetPart = localOffset + GetAlign((tensorInfo.chunkRow - ubLoopInfo.rowStart) * tensorInfo.splitCol,
                                                     srcEleUbBlock_);
#else
            localOffsetPart = localOffset + (tensorInfo.chunkRow - ubLoopInfo.rowStart) * tensorInfo.splitColAlign;
#endif

            int64_t gmOffsetPart = srcGmOffset + tensorInfo.startOffset +
                                   (tensorInfo.chunkRow - ubLoopInfo.rowStart) * tensorInfo.tensorCol;
            ExecuteDataCopy(localOffsetPart, gmOffsetPart, 1, blockLen, srcStride);
            // 1.3.2
            blockCount = static_cast<uint16_t>(tensorInfo.chunkRow - ubLoopInfo.rowStart);
            blockLen = static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1));
        }
        ExecuteDataCopy(localOffset, srcGmOffset + tensorInfo.startOffset, blockCount, blockLen, srcStride);
    }

    __aicore__ inline void CopyInChunk(int64_t& totalCol, int64_t& localOffset, UbLoopInfo& ubLoopInfo,
                                       TensorInfo& tensorInfo)
    {
        ubLoopInfo.isAllZero = false;
        int64_t rowEnd = ubLoopInfo.rowStart + ubLoopInfo.currentUbRowFactor;
        if (rowEnd < tensorInfo.chunkRowAlign) {
            DoRowsCopy(localOffset, ubLoopInfo, tensorInfo);
        } else {
            DoLastRowsCopy(localOffset, ubLoopInfo, tensorInfo);
        }
    }

protected:
    int64_t blockIdx_{0};
    int64_t inputNum_{0};
    int64_t ubRowFactor_{0};
    int64_t ubColFactor_{0};
    int64_t srcEleUbBlock_{0};
    int64_t dstEleUbBlock_{0};
    int64_t dim_{0};
    int64_t numChunk_{0};
    int64_t outputRow_{0};
    int64_t outputCol_{0};
    int64_t blockRowFactor_{0};
    int64_t blockColFactor_{0};
    int64_t tailBlockRowFactor_{0};
    int64_t tailBlockColFactor_{0};
    int64_t blockRowGroup_{0};
    int64_t blockColGroup_{0};
    int64_t currentBlockRowFactor_{0};
    int64_t currentBlockColFactor_{0};

    TPipe* pipe_;
    TEventID event_{0};
    TensorDesc<T1> desc_;
    ListTensorDesc inputList_;
    GlobalTensor<T2> dstGlobal_;
    GlobalTensor<T1> srcGlobal_;
    TBuf<AscendC::TPosition::VECCALC> srcBuf_;
    TBuf<AscendC::TPosition::VECCALC> dstBuf_;
    LocalTensor<T1> srcLocal_;
    LocalTensor<T2> dstLocal_;
};

#endif // _CHUNK_CAT_COMMON_DATA_H_
