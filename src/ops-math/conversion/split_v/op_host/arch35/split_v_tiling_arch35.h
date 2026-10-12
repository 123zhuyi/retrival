/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_TILING_H
#define SPLIT_V_TILING_H
#include <cstdint>
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr int64_t BASE_2 = 2;
constexpr int64_t MAX_CORE_COUNT = 72;
constexpr int32_t MAX_COL_OFFSET_COUNT = 128;

// SIMT 全局 chunk 均衡模板(204)相关常量
// constexpr int32_t LOGIC_BLOCK_NUM = 1024;        // 64 物理核 * 每核 16 个逻辑 block
// constexpr int32_t LOGIC_BLOCK_NUM = 896;     // 56 物理核 * 每核 16 个逻辑 block
constexpr int32_t LOGIC_BLOCK_PER_CORE = 16; // 每核逻辑 block 数
constexpr int32_t SIMT_THREAD_DIM = 2048;    // asc_vf_call 单核拉起线程数(16*128)

BEGIN_TILING_DATA_DEF(SplitVTilingData)
TILING_DATA_FIELD_DEF(int64_t, ubSize);            // asc 接口获取到的ub大小，Bytes
TILING_DATA_FIELD_DEF(int64_t, splitDim);          // attr里带的原始的split的轴, 负数转正数
TILING_DATA_FIELD_DEF(int64_t, sizeAfterSplitDim); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int64_t, mBlockFactor);      // m轴block切分factor, 个数  --对应Split: mUBFactor
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorTail);  // m轴切分factor尾快, 个数   --对应Split: mUBFactorTail
TILING_DATA_FIELD_DEF(int64_t, mBlockFactorNum);   // m轴主块的个数             --对应Split: mSize
TILING_DATA_FIELD_DEF(int64_t, mBlockCount);       // m轴切分块的总个数         --对应Split: mUBCount
TILING_DATA_FIELD_DEF(int64_t, gUBFactor);         // Split: Gi
TILING_DATA_FIELD_DEF(int64_t, gUBFactorTail);     // Split: GiTail
TILING_DATA_FIELD_DEF(int64_t, gSize);             // Split: gSize
TILING_DATA_FIELD_DEF(int64_t, gUBCount);          // Split: Go
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);      // n轴block切分factor, 个数  --对应Split: nUBFactor
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorAlign); // n轴block切分factor对齐到32字节, 个数  --对应Split: nUBFactorAlign
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorTail); // n轴切分factor尾快, 个数     --对应Split: nUBFactorTail
TILING_DATA_FIELD_DEF(int64_t,
                      nBlockFactorTailAlign); // n轴切分factor尾快对齐到32字节, 个数   --对应Split: nUBFactorTailAlign
TILING_DATA_FIELD_DEF(int64_t, nBlockFactorNum); // n轴主块的个数               --对应Split: nSize
TILING_DATA_FIELD_DEF(int64_t, nBlockCount);     // n轴切分块的总个数           --对应Split: nUBCount
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);     // 真实使用的核数
TILING_DATA_FIELD_DEF(int64_t, blockFactor);     // Split: blockFactor
TILING_DATA_FIELD_DEF(int64_t, blockFactorTail); // Split: blockFactorTail
TILING_DATA_FIELD_DEF(int64_t, nSize);           // 整个n轴的大小, 个数
TILING_DATA_FIELD_DEF(int64_t, nSizeAlign);      // 整个n轴的大小, 对齐到32字节, 个数
TILING_DATA_FIELD_DEF(int64_t, pureOutIdx);      // split个数为1的纯搬运场景，真实的输出的idx
TILING_DATA_FIELD_DEF(int64_t, negIdx);          // sizeSplits为负数的索引
TILING_DATA_FIELD_DEF(int64_t, negValue);        // sizeSplits为负数的实际split轴大小
TILING_DATA_FIELD_DEF(int64_t, isInt32);         // sizeSplits是否为int32类型
TILING_DATA_FIELD_DEF(int64_t, mSize);           // 整个m轴的大小, 个数
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT,
                          nBlockSplitOffset); // 所有核在N轴上每个block起始位置在第几个Split块上, 从0开始
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, nBlockSplitOffsetEnd); // end - start = 处理的split块数, 至少为1
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT,
                          nBlockSplitPrefixStart); // 所有核在N轴上每个block起始位置在第几个Split块上, 从0开始
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, nBlockSplitPrefixEnd); // end - start = 处理的split块数, 至少为1
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV, SplitVTilingData)
REGISTER_TILING_DATA_CLASS(Split, SplitVTilingData)

BEGIN_TILING_DATA_DEF(SplitVSIMTTilingData)
TILING_DATA_FIELD_DEF(int32_t, mSize); // 整个m轴的大小, 元素个数
TILING_DATA_FIELD_DEF(int32_t, nSize); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int32_t, sizeAfterSplit);
TILING_DATA_FIELD_DEF(int32_t, splitNum);
TILING_DATA_FIELD_DEF(int32_t, realCoreNum); // 真实使用的核数
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_COL_OFFSET_COUNT, colOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV_200, SplitVSIMTTilingData)
REGISTER_TILING_DATA_CLASS(SplitV_202, SplitVSIMTTilingData)

// SIMT 全局 chunk 均衡模板 per-output 预存版: host 预存 q/r/magic/shift/strideDiff, kernel 零派生(无 div/magic 计算)
// 与 SplitVSIMTChunkTilingData 区别: 移除 splitWidth, 新增 qPerBlk/rPerBlk/divMagic/divShift/strideDiff
BEGIN_TILING_DATA_DEF(SplitVSIMTChunkPreTilingData)
TILING_DATA_FIELD_DEF(int32_t, mSize);       // 整个m轴的大小, 元素个数
TILING_DATA_FIELD_DEF(int32_t, nSize);       // split轴之后所有shape的乘积, 个数(输入行宽 nLen)
TILING_DATA_FIELD_DEF(int32_t, splitNum);    // output 个数
TILING_DATA_FIELD_DEF(int32_t, realCoreNum); // 真实使用的核数(64)
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_COL_OFFSET_COUNT, blockPrefix); // 各 output 逻辑 block 数前缀, Σ=1024
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_COL_OFFSET_COUNT, colOffset);   // 各 output 输入行内列起点 Σ_{j<i} wj
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_COL_OFFSET_COUNT, qPerBlk);      // = (M*wi)/cnt_i, cnt_i = blkForTensor[i]
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_COL_OFFSET_COUNT, rPerBlk);      // = (M*wi)%cnt_i
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_COL_OFFSET_COUNT, divMagic);    // 对 wi 的 fast-div magic
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_COL_OFFSET_COUNT, divShift);     // fast-div shift
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_COL_OFFSET_COUNT, strideDiff);   // = nSize - wi
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV_204, SplitVSIMTChunkPreTilingData)
REGISTER_TILING_DATA_CLASS(Split_204, SplitVSIMTChunkPreTilingData)

BEGIN_TILING_DATA_DEF(SplitSIMTTilingData)
TILING_DATA_FIELD_DEF(int32_t, mSize); // 整个m轴的大小, 元素个数
TILING_DATA_FIELD_DEF(int32_t, nSize); // split轴之后所有shape的乘积, 个数
TILING_DATA_FIELD_DEF(int32_t, splitSize);
TILING_DATA_FIELD_DEF(int32_t, splitNum);
TILING_DATA_FIELD_DEF(int32_t, realCoreNum);         // 真实使用的核数
TILING_DATA_FIELD_DEF(int32_t, tensorSplitNum);      // tensor切分的块数
TILING_DATA_FIELD_DEF(int32_t, mainTensorSplitSize); // tensor切分主块大小
TILING_DATA_FIELD_DEF(int32_t, tailTensorSplitSize); // tensor切分尾块大小
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitV_201, SplitSIMTTilingData)
REGISTER_TILING_DATA_CLASS(Split_201, SplitSIMTTilingData)
REGISTER_TILING_DATA_CLASS(SplitV_203, SplitSIMTTilingData)
REGISTER_TILING_DATA_CLASS(Split_203, SplitSIMTTilingData)

ge::graphStatus SplitVTilingAscendC(gert::TilingContext* context, int32_t maxCoreNum, uint32_t ubSize,
                                    int32_t isSameLen);

struct DualSplitNode {
    int64_t m = 0;     // M轴切分块数
    int64_t n = 0;     // N轴切分块数
    int64_t t = 0;     // m*n, 实际使用核数
    int64_t delta = 0; // 最大块和最小块面积的差值, 仅作观测和最后兜底

    // 以下字段由 host 侧成本估算填充, 用于以 kernel 真实 UB 循环次数为核心的候选选择
    int64_t mTimes = 0;       // 最重核在 M 方向的 UB 循环次数
    int64_t nTimes = 0;       // 最重核在 N 方向的 UB 循环次数
    int64_t maxSplitCnt = 0;  // 单个 N block 覆盖的最大 split 块数
    int64_t computeCost = 0;  // 综合计算成本 = mTimes * (maxSplitCnt + nTimes), 主排序键
    int64_t maxDataBytes = 0; // 最重核的数据量, 仅作极端失衡保护

    DualSplitNode() {};
    DualSplitNode(int64_t mV, int64_t nV, int64_t tV, int64_t deltaV) : m(mV), n(nV), t(tV), delta(deltaV) {}

    // 计算瓶颈是每核 UB 循环放大后的 split 处理量, 因此以 computeCost 为主排序键。
    // 数据量随硬件带宽提升不再是主瓶颈, 只在 ChooseBestSplitInfo 中作为极端失衡过滤, 不参与主评分。
    // 排序顺序: computeCost 小 -> 用满核(t 大) -> maxSplitCnt 小 -> maxDataBytes 小 -> n 小(保守少切 N)。
    bool operator<(const DualSplitNode& x) const
    {
        if (computeCost != x.computeCost) {
            return computeCost < x.computeCost;
        }
        if (t != x.t) {
            return t > x.t;
        }
        if (maxSplitCnt != x.maxSplitCnt) {
            return maxSplitCnt < x.maxSplitCnt;
        }
        if (maxDataBytes != x.maxDataBytes) {
            return maxDataBytes < x.maxDataBytes;
        }
        return n < x.n;
    }
};

struct SplitVCompileInfo {
    int64_t core_num;
    uint32_t maxCoreNum{0};
    uint32_t ubSizePlatform{0};
};

class SplitVTiling {
public:
    explicit SplitVTiling(gert::TilingContext* tilingContext) : context_(tilingContext) {};
    ge::graphStatus DoSplitVTiling(int32_t maxCoreNum, uint32_t ubSize);
    ge::graphStatus DoSplitTiling(int32_t maxCoreNum, uint32_t ubSize);
    ge::graphStatus DoSplitVSIMTTiling(int32_t maxCoreNum);

protected:
    virtual ge::graphStatus InitParamsSameLen(int32_t maxCoreNum, uint32_t ubSize);
    virtual ge::graphStatus GetInputParamsSameLen();

    int64_t splitDim_ = 0;
    int64_t numSplit_ = 0;
    int32_t coreNum_ = 0;
    uint32_t ubSize_ = 0;
    gert::TilingContext* context_{nullptr};
    int64_t xDtypeSize_ = 0;
    gert::Shape inputShape_;

private:
    template <typename T>
    std::string ArrayToString(const T* vec, size_t num) const;
    template <typename T>
    std::string VectorToString(const std::vector<T>& vec) const;
    ge::graphStatus ModifySizeSplitList();
    template <typename T>
    bool GetData(const gert::Tensor* tensor, std::vector<int64_t>& values) const;
    bool GetSizeSplitsList();
    ge::graphStatus GetInputParams();
    ge::graphStatus InitParams(int32_t maxCoreNum, uint32_t ubSize);
    void FuseAllShape();
    void SetPureMoveTilingMode();
    void SetBlockSplitInfo(int64_t mBlockCnt, int64_t nBlockCnt);
    void FuseInputShape();
    void CalEmptyInputTiling();
    void FindAllPossibleCutCnt(std::set<int64_t>& cutCountSet, int64_t cores) const;
    void GetAllCutInfoNode(int64_t M, int64_t N, int64_t cores, const std::set<int64_t>& cutCountSet,
                           std::vector<DualSplitNode>& splitNodeList) const;
    void EstimateSplitNodeCost(int64_t dataBytes, DualSplitNode& node) const;
    int64_t CalcMaxSplitCntPerNBlock(int64_t nBlockCnt) const;
    void ChooseBestSplitInfo(std::vector<DualSplitNode>& splitNodeList, DualSplitNode& splitInfo) const;
    void CalBlockSplitTwoAxis(int64_t rowM, int64_t colN, int64_t dataBytes, int64_t coreNum,
                              DualSplitNode& splitInfo) const;
    void CalBlockTilingParams();
    void CalcSplitLenCond(int64_t curSplitNumN, int64_t& condNum, int64_t& total) const;
    void CalcSplitUBSizeCond(int64_t curSplitNumN, int64_t& condNum, int64_t& total);
    int64_t UpdataNextBlockFactor(int64_t sizeSplitIdx, int64_t handedSplitSize, int64_t leftSplitSize,
                                  int64_t& handedBlockNumN);
    void UpdateMAxisSplitOffset();
    void CalcTilingKey(double condN, double condM, double totalCompareN, double totalCompareM);
    void SetTilingMode();
    void FillTilingData();
    void SetBlockDimAndTilingKey() const;
    void SetWorkspaceSize() const;
    // Used for split
    int64_t CeilDiv(int64_t value, int64_t factor) const;
    int64_t FloorAlign(int64_t value, int64_t factor) const;
    void FuseInputShapeSameLen();
    int64_t AdjustUbFactor(int64_t factor, int64_t alignFactor) const;
    void CalcSameLenCopyTilingInfo(int64_t halfUbEleNum, int64_t blockEleNum);
    void CalcSameLenSplitTilingInfo(int64_t halfUbEleNum, int64_t blockEleNum);
    void CalcSameLenTilingInfo();
    void FillSIMTSplitVTilingData();
    int64_t CalInputDataSize();
    void CountSplitPrefix();
    int64_t SplitPrefix(int64_t i);
    bool IsDoSplitVSIMT(int32_t maxCoreNum);
    void DoDiffLenModeSplitVSIMTTiling(int32_t maxCoreNum);
    void CalSplitSizeDiff(int32_t maxCoreNum, int64_t& maxSizeSplit, int64_t& minSizeSplit, int32_t countPerCore);
    bool IsDoSplitVSIMTChunk(int32_t maxCoreNum);
    void DoSplitVSIMTChunkTiling(int32_t maxCoreNum);
    void FillSIMTChunkTilingData();
    bool IsFallbackToChunk();

    // 204 per-output 预存版: host 算好 q/r/magic/shift/strideDiff, kernel 零派生; 准入复用 IsDoSplitVSIMTChunk
    void DoSplitVSIMTChunkPreTiling(int32_t maxCoreNum);
    void FillSIMTChunkPreTilingData();

private:
    int64_t nonZeroSplitCnt_ = 0;
    int64_t pureOutIdx_ = -1;
    bool isSameLenMode_ = true;

    int64_t splitStride_ = 0; // 切分轴右侧的所有轴的乘积 --对应Split: splitDimFactor_
    int64_t fusedShape_[BASE_2] = {0};

    int64_t mSimtSize_ = 1;
    int64_t nSimtSize_ = 1;
    int64_t simtSizeAfterSplit_ = 0;
    int64_t maxFuseSizeAfterSplit_ = 0;
    int64_t minFuseSizeAfterSplit_ = 0;
    int64_t simtRelativeSkew_ = 0;  // IsDoSplitVSIMT 计算的相对偏差比
    int64_t simtAbsoluteSkew_ = 0;  // IsDoSplitVSIMT 计算的绝对偏差值
    bool simtSkewRejected_ = false; // IsDoSplitVSIMT 是否因偏差被拒
    int64_t realCoreNum_ = 0;       // 处理空tensor需要两个变量，realCoreNum_ 和 blockDim_分开赋值
    int64_t blockDim_ = 0;
    int32_t logicBlockNum_ = 0; // = realCoreNum * LOGIC_BLOCK_PER_CORE(这里把一个core分成16个block), 动态计算
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;

    int64_t mBlockCount_ = 0;  // M轴方向核切分块数
    int64_t mBlockFactor_ = 0; // M轴核切分的 main_factor -- tail_factor=main_factor-1
    int64_t mBlockFactorTail_ = 0;
    int64_t mBlockFactorCount_ = 0; // M轴核切分的主块个数 -- 尾块个数=mBlockCount_ - mBlockFactorCount_
    int64_t gUBCount_ = 0;
    int64_t gUBFactor_ = 0;
    int64_t gUBFactorTail_ = 0;
    int64_t nBlockCount_ = 0;
    int64_t nBlockFactor_ = 0;
    int64_t nBlockFactorTail_ = 0;
    int64_t nBlockFactorCount_ = 0;
    int64_t tilingKey_ = 0;
    int64_t negIdx_ = -1;
    int64_t negValue_ = 0;
    int64_t isInt32_ = 0;

    int64_t nBlockSplitOffsetStart_[MAX_CORE_COUNT] = {
        0}; // 所有核在N轴上每个blockfactor起始位置在第几个Split块上, 从idx 0开始
    int64_t nBlockSplitOffsetEnd_[MAX_CORE_COUNT] = {0}; // end - start = 处理的split块数
    int64_t nBlockSplitPrefixStart_[MAX_CORE_COUNT] = {
        0}; // 所有核在N轴上每个blockfactor起始位置所在split块的split轴偏移
    int64_t nBlockSplitPrefixEnd_[MAX_CORE_COUNT] = {0}; // 所有核在N轴上每个blockfactor结束位置所在split块的split轴偏移

    SplitVTilingData tilingData_;
    SplitVSIMTTilingData simtSplitVTilingData_;
    std::vector<int64_t> oriSizeSplits_;
    std::vector<int64_t> sizeSplits_; // 合轴后的
    uint32_t colOffset_[MAX_COL_OFFSET_COUNT] = {0};
    // 204 SIMT chunk 均衡模板 per-output 缓冲(均 MAX_COL_OFFSET_COUNT 维)
    uint32_t blockPrefixBuf_[MAX_COL_OFFSET_COUNT] = {0};
    int32_t splitWidthBuf_[MAX_COL_OFFSET_COUNT] = {0};
    uint32_t colOffsetChunkBuf_[MAX_COL_OFFSET_COUNT] = {0};
    // 204 SIMT chunk 均衡 per-output 预存版缓冲(均 MAX_COL_OFFSET_COUNT 维)
    SplitVSIMTChunkPreTilingData simtChunkPreTilingData_;
    int64_t qPerBlkBuf_[MAX_COL_OFFSET_COUNT] = {0};
    int64_t rPerBlkBuf_[MAX_COL_OFFSET_COUNT] = {0};
    uint64_t divMagicBuf_[MAX_COL_OFFSET_COUNT] = {0};
    int32_t divShiftBuf_[MAX_COL_OFFSET_COUNT] = {0};
    int64_t strideDiffBuf_[MAX_COL_OFFSET_COUNT] = {0};
};

} // namespace optiling
#endif
