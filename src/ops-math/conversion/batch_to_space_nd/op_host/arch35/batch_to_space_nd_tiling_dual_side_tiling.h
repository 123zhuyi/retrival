/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_BATCH_TO_SPACE_ND_TILING_DUAL_SIDE_TILING_H
#define OP_HOST_BATCH_TO_SPACE_ND_TILING_DUAL_SIDE_TILING_H

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <set>
#include <utility>
#include "op_host/util/math_util.h"

namespace optiling {
constexpr int16_t INVALID_ALIGN_AXIS = -1;

namespace {
struct CutInfo {                  // 切分信息
    const size_t* axisPerm;       // 轴的排列
    const int16_t alignAxis;      // 需要对齐的轴
    std::set<size_t> axisSet{};   // 轴集合
    size_t cutAxis{0};            // 切分轴
    uint32_t cutFactor{1};        // 切分轴维度数量
    uint32_t innerProd{1};        // 内轴积
    uint32_t alignedInnerProd{1}; // 对齐的内轴积
    uint32_t outterProd{1};       // 外轴积

    CutInfo(const size_t* perm, const int16_t align, uint32_t initInnerProd)
        : axisPerm(perm), alignAxis(align), innerProd(initInnerProd), alignedInnerProd(initInnerProd) {};
    ~CutInfo() {};

    inline size_t Idx2Axis(int16_t idx) const { return axisPerm == nullptr ? idx : axisPerm[idx]; }

    inline bool IsNeedAlign(int16_t idx) const { return alignAxis == idx; }
};
} // namespace

class DualSideTiling {
private:
    // tiling context
    gert::TilingContext* context_;

    // 输入
    const uint32_t ubBlockElements_; // UB block 对应元素个数
    const uint64_t* axisSizeList_;   // 每根轴大小
    const size_t* outAxisPerm_;      // 输出轴的排列
    const int16_t needAlignAxis_;    // 需要对齐的轴的序号
    const size_t rank_;              // 轴的数量
    uint32_t maxBufElements_{0};     // 单侧最大缓存元素个数

    // 中间变量
    int startIdx_{0};                  // 起始轴
    uint32_t commAxisProd_{1};         // 公共轴的积
    uint64_t expectMaxInnerProd_{0};   // 每一侧内轴积期望最大值
    std::set<size_t> commAxisSet_{};   // 公共轴集合
    std::set<size_t> kernelAxisSet_{}; // 所有核间轴集合
    bool isCommAxisUpdated_{false};    // 公共轴是否有更新
public:
    // 输出
    size_t inAxis{0};       // 输入切分轴
    size_t outAxis{0};      // 输出切分轴（按输入索引）
    uint32_t inFactor{1};   // 输入切分轴维度数量
    uint32_t outFactor{1};  // 输出切分轴维度数量
    uint64_t totalCount{1}; // 块数
public:
    DualSideTiling(gert::TilingContext* context, const uint32_t ubBlockElements, const uint64_t* axisSizeList,
                   const size_t* outAxisPerm, const size_t rank)
        : context_(context),
          ubBlockElements_(ubBlockElements),
          axisSizeList_(axisSizeList),
          outAxisPerm_(outAxisPerm),
          needAlignAxis_(INVALID_ALIGN_AXIS),
          rank_(rank) {};
    DualSideTiling(gert::TilingContext* context, const uint32_t ubBlockElements, const uint64_t* axisSizeList,
                   const size_t* outAxisPerm, const int16_t needAlign, const size_t rank)
        : context_(context),
          ubBlockElements_(ubBlockElements),
          axisSizeList_(axisSizeList),
          outAxisPerm_(outAxisPerm),
          needAlignAxis_(needAlign),
          rank_(rank) {};
    ~DualSideTiling() {};

    void DoTiling(uint32_t maxBufElements);

private:
    // block 对齐
    template <typename T>
    inline T CeilAlignBlockElement(T elementCount) const;
    template <typename T>
    inline T FloorAlignBlockElement(T elementCount) const;

    // 初始化
    void Init();
    // 计算最大内轴积
    inline uint32_t ComputeActualMaxInnerProd(uint32_t outterProd) const;
    inline uint32_t ComputeActualMaxInnerProd(const CutInfo& currInfo) const;
    // 新增公共轴
    inline void AddCommonAxis(size_t axis);
    // 计算新增轴后的内轴积
    inline uint64_t ComputeAlignedInnerProdAfterAddAxis(CutInfo& currInfo, size_t axis) const;
    // 新增内轴
    inline void AddInnerAxis(CutInfo& currInfo, size_t axis);
    // 计算应该切哪根轴
    inline size_t ComputeCutAxis(int16_t& currIdx, CutInfo& currInfo, CutInfo& otherInfo);
    // 计算切分轴上的最大维度数量
    inline uint64_t ComputeMaxFactor(const CutInfo& currInfo, uint64_t maxInnerProd) const;
    // 计算切分轴上的维度数量
    inline uint32_t ComputeAxisFactor(const CutInfo& currInfo) const;
    // 填满切分轴维度数量
    inline void FillAxisFactor(CutInfo& currInfo, const CutInfo& otherInfo);
    // 调整每侧的切分轴维度数量
    inline void AdjustAxisFactor(CutInfo& inputInfo, CutInfo& outputInfo);
    // 切轴
    void CutAxis();
    // 计算总块数
    void ComputeTotalCount();

    // 全载
    bool TryFullLoad();
    // 非全载
    void DoNonFullLoad();
};

template <typename T>
inline T DualSideTiling::CeilAlignBlockElement(T elementCount) const
{
    return Ops::Base::CeilAlign(elementCount, static_cast<T>(ubBlockElements_));
}

template <typename T>
inline T DualSideTiling::FloorAlignBlockElement(T elementCount) const
{
    return Ops::Base::FloorAlign(elementCount, static_cast<T>(ubBlockElements_));
}

void DualSideTiling::Init()
{
    startIdx_ = rank_ - 1;
    // 核间轴初始为 0 ~ -2 轴，必不包含 -1 轴
    for (size_t i = 0; i < rank_ - 1; ++i) {
        kernelAxisSet_.emplace_hint(kernelAxisSet_.end(), i);
    }
    // 本模板尾轴小，如果存在公共尾轴则从-2轴开始切
    if (static_cast<size_t>(startIdx_) == outAxisPerm_[startIdx_]) {
        AddCommonAxis(startIdx_--);
    } else {
        expectMaxInnerProd_ = FloorAlignBlockElement(static_cast<uint64_t>(std::sqrt(maxBufElements_)));
    }
}

bool DualSideTiling::TryFullLoad()
{
    uint64_t needSize = 1;
    for (int16_t i = static_cast<int16_t>(rank_) - 1; i >= 0; --i) {
        needSize *= axisSizeList_[i];
        if (i == needAlignAxis_) {
            needSize = CeilAlignBlockElement(needSize);
        }
    }
    // 能全载
    if (needSize > maxBufElements_) {
        return false;
    }
    inAxis = 0;
    inFactor = axisSizeList_[0];
    outAxis = outAxisPerm_[0];
    outFactor = axisSizeList_[outAxis];
    totalCount = 1;
    return true;
}

inline uint32_t DualSideTiling::ComputeActualMaxInnerProd(uint32_t outterProd) const
{
    // 每侧元素数 = 内轴向上block对齐 * 外轴
    // 则每侧的最大值为：最大元素数/当前侧的UB外轴，然后向下block对齐
    return FloorAlignBlockElement(Ops::Base::FloorDiv(maxBufElements_, outterProd));
}

inline uint32_t DualSideTiling::ComputeActualMaxInnerProd(const CutInfo& currInfo) const
{
    return ComputeActualMaxInnerProd(currInfo.outterProd);
}

inline void DualSideTiling::AddCommonAxis(size_t axis)
{
    commAxisProd_ *= axisSizeList_[axis];
    commAxisSet_.insert(axis);
    expectMaxInnerProd_ = FloorAlignBlockElement(static_cast<uint64_t>(std::sqrt(maxBufElements_ * commAxisProd_)));
    isCommAxisUpdated_ = true;
}

inline uint64_t DualSideTiling::ComputeAlignedInnerProdAfterAddAxis(CutInfo& currInfo, size_t axis) const
{
    uint64_t tmpInner = currInfo.alignedInnerProd * axisSizeList_[axis];
    if (currInfo.IsNeedAlign(axis)) {
        tmpInner = CeilAlignBlockElement(tmpInner);
    }
    return tmpInner;
}

inline void DualSideTiling::AddInnerAxis(CutInfo& currInfo, size_t axis)
{
    currInfo.innerProd *= axisSizeList_[axis];
    currInfo.alignedInnerProd = ComputeAlignedInnerProdAfterAddAxis(currInfo, axis);
}

inline size_t DualSideTiling::ComputeCutAxis(int16_t& currIdx, CutInfo& currInfo, CutInfo& otherInfo)
{
    uint32_t maxInnerProd = ComputeActualMaxInnerProd(currInfo);
    size_t currAxis;
    for (; currIdx >= 0; --currIdx) {
        currAxis = currInfo.Idx2Axis(currIdx);
        uint64_t axisSize = axisSizeList_[currAxis];
        // 如果另一边已经放下了这跟轴，则表示为公共轴，必然能放下，跳过
        if (otherInfo.axisSet.find(currAxis) != otherInfo.axisSet.end()) {
            // 加入公共轴
            AddCommonAxis(currAxis);
            otherInfo.axisSet.erase(currAxis);
            // 加入内轴
            AddInnerAxis(currInfo, currAxis);
            // 从外轴中去除
            currInfo.outterProd /= std::max(axisSize, 1UL);
            maxInnerProd = ComputeActualMaxInnerProd(currInfo);
            continue;
        }
        uint64_t tmpInner = ComputeAlignedInnerProdAfterAddAxis(currInfo, currAxis);
        // 放不下，当前轴即为切分轴
        if (tmpInner > expectMaxInnerProd_ || tmpInner > maxInnerProd) {
            return currAxis;
        }
        currInfo.alignedInnerProd = tmpInner;
        currInfo.innerProd *= axisSize;
        currInfo.axisSet.emplace(currAxis);
        otherInfo.outterProd *= axisSize; // 加入另一侧外轴
        kernelAxisSet_.erase(currAxis);
    }
    return currInfo.Idx2Axis(0);
}

inline uint64_t DualSideTiling::ComputeMaxFactor(const CutInfo& currInfo, uint64_t maxInnerProd) const
{
    return Ops::Base::FloorDiv(maxInnerProd, static_cast<uint64_t>(currInfo.alignedInnerProd));
}

inline uint32_t DualSideTiling::ComputeAxisFactor(const CutInfo& currInfo) const
{
    // 实际最大内轴积
    uint32_t actualMaxInnerProd = ComputeActualMaxInnerProd(currInfo);
    uint64_t maxInnerProd = std::min(expectMaxInnerProd_, static_cast<uint64_t>(actualMaxInnerProd));
    uint64_t axisFactor = ComputeMaxFactor(currInfo, maxInnerProd);
    return static_cast<uint32_t>(std::clamp(axisFactor, 1UL, axisSizeList_[currInfo.cutAxis]));
}

inline void DualSideTiling::FillAxisFactor(CutInfo& currInfo, const CutInfo& otherInfo)
{
    // 使用UB内最终外轴计算，最终外轴不会超过UB大小，不会超过uint32
    uint32_t maxInnerProd = ComputeActualMaxInnerProd(currInfo.outterProd * otherInfo.cutFactor);
    uint64_t axisFactor = ComputeMaxFactor(currInfo, maxInnerProd);
    currInfo.cutFactor = static_cast<uint32_t>(std::clamp(axisFactor, 1UL, axisSizeList_[currInfo.cutAxis]));
}

inline void DualSideTiling::AdjustAxisFactor(CutInfo& inputInfo, CutInfo& outputInfo)
{
    // 把剩余部分调给小轴
    uint32_t inputInner = inputInfo.cutFactor * inputInfo.innerProd;
    uint32_t outputInner = outputInfo.cutFactor * outputInfo.innerProd;
    if (inputInner <= outputInner) {
        FillAxisFactor(inputInfo, outputInfo);
    } else {
        FillAxisFactor(outputInfo, inputInfo);
    }
}

void DualSideTiling::CutAxis()
{
    // 起始索引
    int16_t xIdx = startIdx_;
    int16_t yIdx = startIdx_;
    // 初始内积为公共内积
    CutInfo inputInfo{nullptr, needAlignAxis_, commAxisProd_};           // 输入切分信息
    CutInfo outputInfo{outAxisPerm_, INVALID_ALIGN_AXIS, commAxisProd_}; // 输出切分信息

    while (true) {
        isCommAxisUpdated_ = false;
        // 遍历输入切分轴
        inputInfo.cutAxis = ComputeCutAxis(xIdx, inputInfo, outputInfo);
        // 遍历输出切分轴
        outputInfo.cutAxis = ComputeCutAxis(yIdx, outputInfo, inputInfo);
        if (isCommAxisUpdated_) {
            continue;
        }
        // 切同一根轴，即当前切分轴为公共轴
        if (inputInfo.cutAxis == outputInfo.cutAxis) {
            uint64_t axisSize = axisSizeList_[inputInfo.cutAxis];
            // 输入侧当前轴最大切分值
            uint64_t maxInFactor = ComputeMaxFactor(inputInfo, ComputeActualMaxInnerProd(inputInfo));
            // 输出侧当前轴最大切分值
            uint64_t maxOutFactor = ComputeMaxFactor(outputInfo, ComputeActualMaxInnerProd(outputInfo));
            // 能放下
            if (axisSize <= maxInFactor && axisSize <= maxOutFactor) {
                AddCommonAxis(inputInfo.cutAxis);
                kernelAxisSet_.erase(inputInfo.cutAxis);
                // 加入内轴，不需要更新外轴
                AddInnerAxis(inputInfo, inputInfo.cutAxis);
                AddInnerAxis(outputInfo, inputInfo.cutAxis);
                xIdx--;
                yIdx--;
                continue;
            }
            // 放不下，就切这根轴
            uint64_t factor = std::min(maxInFactor, maxOutFactor);
            factor = std::clamp(factor, 1UL, axisSize);
            inputInfo.cutFactor = factor;
            outputInfo.cutFactor = factor;
            break;
        }
        // 计算切分大小
        inputInfo.cutFactor = ComputeAxisFactor(inputInfo);
        outputInfo.cutFactor = ComputeAxisFactor(outputInfo);
        AdjustAxisFactor(inputInfo, outputInfo);
        break;
    }
    // 剔除当前切分轴
    kernelAxisSet_.erase(inputInfo.cutAxis);
    kernelAxisSet_.erase(outputInfo.cutAxis);
    // 赋值
    inAxis = inputInfo.cutAxis;
    inFactor = inputInfo.cutFactor;
    outAxis = outputInfo.cutAxis;
    outFactor = outputInfo.cutFactor;
}

void DualSideTiling::ComputeTotalCount()
{
    // 核间轴乘积即为总块数
    totalCount = 1;
    // 被切的轴的外轴
    if (inAxis == outAxis) {
        totalCount *= Ops::Base::CeilDiv(axisSizeList_[inAxis], static_cast<uint64_t>(inFactor));
    } else {
        totalCount *= Ops::Base::CeilDiv(axisSizeList_[inAxis], static_cast<uint64_t>(inFactor));
        totalCount *= Ops::Base::CeilDiv(axisSizeList_[outAxis], static_cast<uint64_t>(outFactor));
    }
    // 剩余核间轴
    for (auto i : kernelAxisSet_) {
        totalCount *= axisSizeList_[i];
    }
}

void DualSideTiling::DoNonFullLoad()
{
    Init();

    CutAxis();

    // 计算核间轴
    ComputeTotalCount();
}

void DualSideTiling::DoTiling(uint32_t maxBufElements)
{
    maxBufElements_ = maxBufElements;
    // 全载
    if (TryFullLoad()) {
        return;
    }

    DoNonFullLoad();
}
} // namespace optiling

#endif
