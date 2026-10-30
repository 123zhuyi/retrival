/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include <vector>
#include <algorithm>
#include "register/op_impl_registry.h"
#include "split_v_tiling_arch35.h"
#include "util/const_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "log/log.h"

using namespace gert;
using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr int64_t HUGE_SPLIT_NUM = 4L * 1024;
constexpr int64_t HUGE_M = 64;
constexpr int64_t SIZE_SPLIT_INDEX = 1;
constexpr int64_t SPLIT_DIM_INDEX = 2;
constexpr int64_t SPLIT_DIM_INDEX_SAME_LEN = 0;
constexpr int64_t BASE_256 = 256;
constexpr int64_t BASE_512 = 512;
constexpr int64_t PURE_MOVE_BASE_LEN = 128;              // 128B, cacheline
constexpr int64_t PURE_MOVE_BASE_UB_SIZE = 48 * 1024;    // 48KB
constexpr int64_t PURE_MOVE_BASE_UB_SIZE_M1 = 24 * 1024; // M == 1 24KB
constexpr int64_t USED_MIN_UB_SIZE = 8 * 1024;           // 8KB
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t SPLIT_KEY_PURE_MOVE = 100;                // 等长纯搬运模板
constexpr int64_t SPLIT_KEY_UB_SPLIT = 101;                 // 等长UB内split模板
constexpr int64_t SPLIT_KEY_UB_SPLIT_SMALL_G = 111;         // 等长UB内split模板 g轴小的场景
constexpr int64_t SPLIT_KEY_UB_SPLIT_DEINTERLEAVE = 112;    // 等长UB内split模板 deinterleave场景
constexpr int64_t SPLITV_KEY_PURE_MOVE = 102;               // 纯搬运模板
constexpr int64_t SPLITV_KEY_UB_SPLIT_WITH_PURE_MOVE = 103; // UB内split与纯搬运混合模板
constexpr int64_t SPLITV_KEY_PURE_MOVE_1 = 104;             // split个数为1的纯搬运
constexpr int64_t SIMT_KEY = 200;
constexpr int64_t SIMT_SAME_LEN_KEY = 201;
constexpr int64_t SIMT_KEY_SPLIT_TENSOR = 202;
constexpr int64_t SIMT_SAME_LEN_KEY_SPLIT_TENSOR = 203;
constexpr int64_t SIMT_CHUNK_KEY = 204; // SIMT 全局 chunk 均衡模板
constexpr int64_t BLOCK_SIZE = 32;
constexpr double PURE_MOVE_RATIO = 0.9;    // 90%
constexpr double PURE_MOVE_RATIO_M1 = 0.8; // M == 1 80%
constexpr int64_t INT16_MAX_VALUE = 32767;
constexpr int64_t PREF_BRANCH_GSIZE = 15;
constexpr int64_t MAX_SIMT_PER_CORE_SIZE = 131072; // 128KB
constexpr int32_t SIMT_THREAD_NUM = 1024;
constexpr int64_t MAX_NUM = 9223372036854775807;
constexpr int64_t SMALL_G_TILE_RATIO_LIMIT = 4;
constexpr int64_t PERCENT_BASE = 100;
constexpr int32_t BASE_TWO = 2;
constexpr int32_t BASE_FOUR = 4;
constexpr int32_t SIMT_MIN_CORE_NUM = 8;
// 数据量只做极端失衡保护, 不进入主评分。只有当某候选最重核数据量同时超过最优候选的
// MAX_DATA_IMBALANCE_RATIO 倍且绝对差超过 MAX_DATA_IMBALANCE_BYTES 时才过滤。
constexpr int64_t MAX_DATA_IMBALANCE_RATIO = 2;
constexpr int64_t MAX_DATA_IMBALANCE_BYTES = USED_MIN_UB_SIZE * 4; // 32KB, 数倍 USED_MIN_UB_SIZE

// 204 SIMT chunk 均衡模板准入与分配相关常量
// constexpr int32_t SIMT_VECTOR_CORE_NUM = 56;                // 1024 逻辑 block = 64 核 * 16
constexpr int32_t THREADS_PER_LOGIC_BLOCK = 128;            // 每个逻辑 block 128 线程
constexpr int64_t SIMT_CHUNK_SKEW_RATIO_THRESHOLD = 2;      // max(M*wi)/min(M*wi) >= 2 视为偏斜
constexpr int64_t SIMT_CHUNK_TOTAL_ELEM_THRESHOLD = 131072; // 1024 block * 128 thread, 总元素下限
constexpr int64_t SIMT_REJECT_RELATIVE_SKEW_UPPER = 16;     // SIMT拒绝后回退Chunk的相对偏差上界
constexpr int64_t SIMT_REJECT_ABSOLUTE_SKEW_UPPER = 2048;   // SIMT拒绝后回退Chunk的绝对偏差上界

template <typename T>
std::string SplitVTiling::ArrayToString(const T* vec, size_t num) const
{
    std::stringstream ss;
    for (size_t i = 0; i < num; i++) {
        ss << vec[i] << " ";
    }
    return ss.str();
}

template <typename T>
std::string SplitVTiling::VectorToString(const std::vector<T>& vec) const
{
    std::stringstream ss;
    for (size_t i = 0; i < vec.size(); i++) {
        ss << vec[i] << " ";
    }
    return ss.str();
}

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

ge::graphStatus SplitVTiling::ModifySizeSplitList()
{
    // sizeSplits_ 中最多只能有一个负数，需要根据轴的大小来推导实际值
    int32_t negNum = 0;
    int32_t negIdx = 0;
    int64_t sizeSplitsTotal = 0;
    int64_t splitDimShape = inputShape_.GetDim(splitDim_);
    for (int64_t i = 0; i < numSplit_; i++) {
        if (oriSizeSplits_[i] < 0) {
            negNum++;
            negIdx = i;
            continue;
        }
        sizeSplitsTotal += oriSizeSplits_[i];
        nonZeroSplitCnt_ = (oriSizeSplits_[i] == 0) ? nonZeroSplitCnt_ : (nonZeroSplitCnt_ + 1);
        pureOutIdx_ = (oriSizeSplits_[i] == 0) ? pureOutIdx_ : i;
    }

    // 没有负数，sizeSplitsTotal 需要和切分的dim轴大小相等
    OP_CHECK_IF((negNum == 0 && sizeSplitsTotal != splitDimShape),
                OP_LOGE(context_->GetNodeName(), "sizeSplitsTotal:%ld not equal splitDimShape:%ld", sizeSplitsTotal,
                        splitDimShape),
                return ge::GRAPH_FAILED);

    // 最多只能有一个负数
    OP_CHECK_IF(negNum > 1,
                OP_LOGE(context_->GetNodeName(), "sizeSplits negtive num:%d cannot be bigger than 1", negNum),
                return ge::GRAPH_FAILED);

    if (negNum > 0) {
        // 保证splitSize不能为负数
        OP_CHECK_IF(sizeSplitsTotal > splitDimShape,
                    OP_LOGE(context_->GetNodeName(), "sizeSplitsTotal:%ld cannot be bigger than splitDimShape:%ld",
                            sizeSplitsTotal, splitDimShape),
                    return ge::GRAPH_FAILED);
        oriSizeSplits_[negIdx] = splitDimShape - sizeSplitsTotal;
        nonZeroSplitCnt_ = (oriSizeSplits_[negIdx] == 0) ? nonZeroSplitCnt_ : (nonZeroSplitCnt_ + 1);
        pureOutIdx_ = (oriSizeSplits_[negIdx] == 0) ? pureOutIdx_ : negIdx;
        negIdx_ = negIdx;
        negValue_ = oriSizeSplits_[negIdx];
    }

    // Add SameLenMode branch
    int64_t firstElement = oriSizeSplits_[0];
    for (size_t i = 1; i < oriSizeSplits_.size(); i++) {
        if (oriSizeSplits_[i] != firstElement) {
            isSameLenMode_ = false;
            break;
        }
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
bool SplitVTiling::GetData(const gert::Tensor* tensor, std::vector<int64_t>& values) const
{
    size_t shape_size = tensor->GetShapeSize();
    values.resize(shape_size);
    auto* tensor_data = tensor->GetData<T>();
    if (tensor_data == nullptr) {
        return false;
    }
    for (size_t i = 0; i < shape_size; i++) {
        values[i] = static_cast<int64_t>(*(tensor_data + i));
    }
    return true;
}

bool SplitVTiling::GetSizeSplitsList()
{
    const gert::Tensor* sizeSplitsT = context_->GetInputTensor(SIZE_SPLIT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sizeSplitsT);
    if (sizeSplitsT->GetDataType() == ge::DT_INT32) {
        isInt32_ = 1L;
        return GetData<int32_t>(sizeSplitsT, oriSizeSplits_);
    } else if (sizeSplitsT->GetDataType() == ge::DT_INT64) {
        return GetData<int64_t>(sizeSplitsT, oriSizeSplits_);
    } else if (sizeSplitsT->GetDataType() == ge::DT_UINT32) {
        isInt32_ = 1L;
        return GetData<uint32_t>(sizeSplitsT, oriSizeSplits_);
    } else if (sizeSplitsT->GetDataType() == ge::DT_UINT64) {
        return GetData<uint64_t>(sizeSplitsT, oriSizeSplits_);
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
        context_->GetNodeType(), "size_splits", Ops::Base::ToString(sizeSplitsT->GetDataType()).c_str(),
        "The dtype of size_splits must be within the range [DT_INT32, DT_INT64, DT_UINT32, DT_UINT64].");
    return false;
}

ge::graphStatus SplitVTiling::GetInputParams()
{
    auto xInput = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    const gert::Shape& xInputShape = Ops::Base::EnsureNotScalar(xInput->GetStorageShape());
    inputShape_ = xInputShape;

    // 不能使用 GetConstIntToShape, gert::Shape最多只能支持25个数，超过25就获取不到了
    OP_CHECK_IF(!GetSizeSplitsList(), OP_LOGE(context_->GetNodeName(), "SplitV tiling get size_splits failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!Ops::Base::GetConstInt(context_, SPLIT_DIM_INDEX, splitDim_),
                OP_LOGE(context_->GetNodeName(), "SplitV tiling get split_dim failed"), return ge::GRAPH_FAILED);

    int64_t inputXDimNum = static_cast<int64_t>(inputShape_.GetDimNum());
    OP_CHECK_IF((splitDim_ < -inputXDimNum || splitDim_ >= inputXDimNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "split_dim", std::to_string(splitDim_).c_str(),
                    ("The value of split_dim must be within the range [-" + std::to_string(inputXDimNum) + ", " +
                     std::to_string(inputXDimNum - 1) + "].")
                        .c_str()),
                return ge::GRAPH_FAILED);
    if (splitDim_ < 0) {
        splitDim_ = splitDim_ + inputXDimNum;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto attr0 = attrs->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attr0);
    numSplit_ = *attr0;
    OP_CHECK_IF(
        numSplit_ < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_split", std::to_string(numSplit_).c_str(),
                                              "The value of num_split must be greater than or equal to 0."),
        return ge::GRAPH_FAILED);

    int64_t sizeSplitsNum = static_cast<int64_t>(oriSizeSplits_.size());
    OP_CHECK_IF(
        numSplit_ != sizeSplitsNum,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_split", std::to_string(numSplit_).c_str(),
                                              "The value of num_split must be equal to the size of size_splits."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitVTiling::InitParams(int32_t maxCoreNum, uint32_t ubSize)
{
    coreNum_ = std::min(maxCoreNum, static_cast<int32_t>(MAX_CORE_COUNT));
    ubSize_ = ubSize;

    auto xInputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    ge::DataType xDtype = xInputDesc->GetDataType();
    xDtypeSize_ = ge::GetSizeByDataType(xDtype);

    if (GetInputParams() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ModifySizeSplitList() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context_->GetNodeName(),
            "coreNum_:%d ubSize_:%d xDtypeSize_:%ld inputShape_:%s "
            "splitDim_:%ld numSplit_:%ld oriSizeSplits_:%s nonZeroSplitCnt_:%ld pureOutIdx_:%ld",
            coreNum_, ubSize_, xDtypeSize_, Shape2String(inputShape_).c_str(), splitDim_, numSplit_,
            VectorToString(oriSizeSplits_).c_str(), nonZeroSplitCnt_, pureOutIdx_);
    return ge::GRAPH_SUCCESS;
}

void SplitVTiling::FuseAllShape()
{
    fusedShape_[0] = 1;
    fusedShape_[1] = inputShape_.GetShapeSize();
    OP_LOGI(context_->GetNodeName(), "After fusedShape_:%ld %ld, pureOutIdx_:%ld", fusedShape_[0], fusedShape_[1],
            pureOutIdx_);
}

void SplitVTiling::SetPureMoveTilingMode()
{
    int64_t cores = std::min(static_cast<int64_t>(coreNum_),
                             Ops::Base::CeilDiv(fusedShape_[0] * fusedShape_[1] * xDtypeSize_, BASE_512));
    SetBlockSplitInfo(1, cores);
    tilingKey_ = SPLITV_KEY_PURE_MOVE_1;
}

void SplitVTiling::FuseInputShape()
{
    int64_t outerSize = 1;
    int64_t innerSize = 1;
    for (size_t i = 0; i < static_cast<size_t>(splitDim_); i++) {
        outerSize *= inputShape_.GetDim(i);
    }

    for (size_t i = splitDim_ + 1; i < inputShape_.GetDimNum(); i++) {
        innerSize *= inputShape_.GetDim(i);
    }

    splitStride_ = innerSize;

    fusedShape_[0] = outerSize;
    fusedShape_[1] = inputShape_.GetDim(splitDim_) * innerSize;

    // 合轴后, sizeSplits也需要同步合并
    sizeSplits_.resize(numSplit_);
    for (int64_t i = 0; i < numSplit_; i++) {
        sizeSplits_[i] = oriSizeSplits_[i] * innerSize;
    }

    OP_LOGI(context_->GetNodeName(), "After fused shape, splitStride_:%ld fusedShape_:%ld %ld, sizeSplits_:%s",
            splitStride_, fusedShape_[0], fusedShape_[1], VectorToString(sizeSplits_).c_str());
}

void SplitVTiling::CalEmptyInputTiling()
{
    realCoreNum_ = 0;
    blockDim_ = 1; // blockDim不能为0，最少为1
    tilingKey_ = SPLITV_KEY_PURE_MOVE;
}

void SplitVTiling::FindAllPossibleCutCnt(std::set<int64_t>& cutCountSet, int64_t cores) const
{
    // M切分成m块，N切分成n块，找到 m*n<=coreNum 且 m*n尽可能接近coreNum的所有m和n的可能
    int64_t upBound = static_cast<int64_t>(std::ceil(std::sqrt(cores) + 1.0));
    int64_t n = 1;
    for (int64_t m = 1; m < upBound; m++) {
        n = cores / m;
        cutCountSet.insert(m);
        cutCountSet.insert(n);
    }
}

void SplitVTiling::GetAllCutInfoNode(int64_t M, int64_t N, int64_t cores, const std::set<int64_t>& cutCountSet,
                                     std::vector<DualSplitNode>& splitNodeList) const
{
    // 给定一个shape[M,N],和coreNum
    // M切分成m块，N切分成n块, 找到尽可能均匀且尽量用满核的切分方式
    // 采用尽量均匀的方式，如 14, 切分为4块，则切分为 4 4 3 3，有多个尾块，而不是 4 4 4 2这种.
    int64_t n = 0;
    int64_t mFactor = 0;
    int64_t nFactor = 0;
    int64_t delta = 0; // 获取完整切分块的大小和最小的尾块之间的差值
    int64_t blockNum = BLOCK_SIZE / xDtypeSize_;
    for (int64_t m : cutCountSet) {
        if (m <= 0) {
            continue;
        }
        n = cores / m;
        if (m > M || n > N || n <= 0) {
            continue;
        }
        mFactor = Ops::Base::CeilDiv(M, m);
        nFactor = Ops::Base::CeilDiv(N, n);
        delta = mFactor * nFactor;
        if (m * n == cores) {
            if (M % m == 0 && N % n == 0) {
                delta = 0;
            } else if (M % m == 0) {
                delta = delta - mFactor * (N / n); // N/n为尾块的切分因子
            } else if (N % n == 0) {
                delta = delta - nFactor * (M / m);
            } else {
                delta = delta - (M / m) * (N / n);
            }
        }
        if ((M / m) < blockNum && M > blockNum && splitNodeList.size() > 0) {
            continue;
        }
        DualSplitNode node(m, n, m * n, delta);
        EstimateSplitNodeCost(xDtypeSize_, node);
        splitNodeList.push_back(node);
        OP_LOGD(context_->GetNodeName(), "m:%ld n:%ld delta:%ld mTimes:%ld nTimes:%ld maxSplitCnt:%ld computeCost:%ld",
                m, n, delta, node.mTimes, node.nTimes, node.maxSplitCnt, node.computeCost);
    }
}

int64_t SplitVTiling::CalcMaxSplitCntPerNBlock(int64_t nBlockCnt) const
{
    // 给定 N 方向核切分块数, 统计每个 N block 覆盖的非零 split 块数, 返回最大值。
    // 边界切在 split 块内部时, 相邻两个 N block 都会处理该 split, 因此两侧都计入。
    if (nBlockCnt <= 1) {
        int64_t nonZero = 0;
        for (int64_t i = 0; i < numSplit_ && i < static_cast<int64_t>(sizeSplits_.size()); i++) {
            if (sizeSplits_[i] > 0) {
                nonZero++;
            }
        }
        return nonZero;
    }

    int64_t N = fusedShape_[1];
    int64_t nBlockFactor = Ops::Base::CeilDiv(N, nBlockCnt);
    int64_t nBlockFactorCount = (N % nBlockCnt == 0) ? nBlockCnt : (N % nBlockCnt);
    int64_t nBlockFactorTail = N / nBlockCnt;

    int64_t maxCnt = 0;
    int64_t prefixStart = 0; // 当前 split 块起始的 N 轴偏移
    int64_t splitIdx = 0;
    int64_t nStart = 0;
    for (int64_t b = 0; b < nBlockCnt; b++) {
        int64_t factor = (b < nBlockFactorCount) ? nBlockFactor : nBlockFactorTail;
        int64_t nEnd = nStart + factor;
        // 统计与 [nStart, nEnd) 重叠且长度大于 0 的 split 块个数
        int64_t cnt = 0;
        int64_t scanStart = prefixStart;
        int64_t scanIdx = splitIdx;
        while (scanIdx < numSplit_ && scanIdx < static_cast<int64_t>(sizeSplits_.size())) {
            int64_t splitLen = sizeSplits_[scanIdx];
            int64_t splitEnd = scanStart + splitLen;
            if (scanStart >= nEnd) {
                break;
            }
            if (splitLen > 0 && splitEnd > nStart && scanStart < nEnd) {
                cnt++;
            }
            // split 块完全落在当前 N block 左侧才推进全局游标, 保证跨界 split 在下个 block 仍被统计
            if (splitEnd <= nEnd) {
                splitIdx = scanIdx + 1;
                prefixStart = splitEnd;
            }
            scanStart = splitEnd;
            scanIdx++;
        }
        maxCnt = std::max(maxCnt, cnt);
        nStart = nEnd;
    }
    return maxCnt;
}

void SplitVTiling::EstimateSplitNodeCost(int64_t dataBytes, DualSplitNode& node) const
{
    // 以 kernel split_v_ub_split.h 的 UB 切分口径估算最重核的 M/N 方向 UB 循环次数。
    // host 估算无需与 kernel 逐字节一致, 只要与 kernel 循环次数同阶且单调一致即可正确排序候选。
    // M/N 主排序统一走成员 fusedShape_ 口径, 候选枚举侧已用 512 对齐 N, 此处不再重复入参。
    constexpr int64_t SPLIT_UB_NUM = 4; // 输入输出均开 double buffer, 与 kernel 一致
    constexpr int64_t INT_NUM_TWO = 2;

    int64_t blockEleNum = (dataBytes > 0) ? (BLOCK_SIZE / dataBytes) : 1; // kernel oneUbNum_
    blockEleNum = std::max<int64_t>(blockEleNum, 1);

    int64_t mBlockFactor = Ops::Base::CeilDiv(fusedShape_[0], node.m);
    int64_t nBlockFactor = Ops::Base::CeilDiv(fusedShape_[1], node.n);

    // 复现 kernel Init 中的 UB 可用空间计算
    int64_t splitBufSize = Ops::Base::CeilAlign(numSplit_ * SPLIT_UB_NUM * INT_NUM_TWO, BLOCK_SIZE);
    int64_t splitBufSizeInt32 = Ops::Base::CeilAlign(numSplit_ * SPLIT_UB_NUM, BLOCK_SIZE);
    int64_t ubUsable = static_cast<int64_t>(ubSize_) - splitBufSize - splitBufSizeInt32;
    ubUsable = (ubUsable > 0) ? (ubUsable / SPLIT_UB_NUM / BLOCK_SIZE * BLOCK_SIZE) : 0;
    int64_t ubSizeNum = (dataBytes > 0) ? (ubUsable / dataBytes) : 0;
    ubSizeNum = std::max<int64_t>(ubSizeNum, blockEleNum);

    // 复现 kernel Process 中 mUbFactor/nUbFactor 的推导 (含一次 lastUbNum 回补)
    int64_t mUbFactor = std::min(blockEleNum, mBlockFactor);
    mUbFactor = std::max<int64_t>(mUbFactor, 1);
    int64_t nUbBudget = ubSizeNum / mUbFactor / blockEleNum * blockEleNum;
    int64_t nUbFactor = std::max<int64_t>(nUbBudget, blockEleNum);
    int64_t nBlockFactorAlign = Ops::Base::CeilAlign(nBlockFactor, blockEleNum);
    if (nBlockFactorAlign <= nUbFactor) {
        nUbFactor = nBlockFactor;
    }
    nUbFactor = std::max<int64_t>(nUbFactor, 1);

    int64_t mUbFactorAlign = Ops::Base::CeilAlign(nUbFactor, blockEleNum) + INT_NUM_TWO * blockEleNum;
    if (mUbFactorAlign > 0) {
        int64_t lastUbNum = (ubSizeNum - mUbFactorAlign * mUbFactor) / mUbFactorAlign;
        if (mBlockFactor <= lastUbNum + mUbFactor) {
            mUbFactor = mBlockFactor;
        } else if (lastUbNum > 0) {
            mUbFactor = mUbFactor + (lastUbNum / blockEleNum) * blockEleNum;
        }
    }
    mUbFactor = std::max<int64_t>(mUbFactor, 1);

    node.mTimes = Ops::Base::CeilDiv(mBlockFactor, mUbFactor);
    node.nTimes = Ops::Base::CeilDiv(nBlockFactor, nUbFactor);
    node.maxSplitCnt = CalcMaxSplitCntPerNBlock(node.n);
    // 计算瓶颈: split 处理嵌在 M 两层 UB 循环最内层, 被 mTimes 放大;
    // mTimes*nTimes 覆盖 UB tile 调度开销。取最重核视角即用 CeilDiv 后的主块 factor。
    node.computeCost = node.mTimes * (node.maxSplitCnt + node.nTimes);
    node.maxDataBytes = mBlockFactor * nBlockFactor * dataBytes;

    // MTE burst 碎片化惩罚: 成本模型只建模 UB 循环与 split 遍历, 未覆盖"搬运主体被 N 切核切碎"。
    // 当存在占比过半的主导 split 时, N 切越细, 主导 split 的每段连续搬运字节越小, strided burst
    // 固定开销占比越高。此处对"每段低于 HBM 高效 burst 阈值(512B)"的候选按碎化倍数线性抬升 computeCost,
    // 使切核在 MTE-bound 场景偏向更粗的 N 切分。占比不足或每段已达 512B 时不触发, 对小/均匀 split 零副作用。
    int64_t alignBytes = BASE_512;
    int64_t avgSplitBytes = (numSplit_ > 0) ? (fusedShape_[1] * dataBytes / numSplit_) : (fusedShape_[1] * dataBytes);
    if (avgSplitBytes < PURE_MOVE_BASE_LEN) {
        alignBytes = PURE_MOVE_BASE_LEN;
    }
    int64_t maxSplitEle = 0;
    for (int64_t i = 0; i < numSplit_ && i < static_cast<int64_t>(sizeSplits_.size()); i++) {
        maxSplitEle = std::max(maxSplitEle, sizeSplits_[i]);
    }
    int64_t maxSplitBytes = maxSplitEle * dataBytes;
    int64_t totalBytes = fusedShape_[1] * dataBytes;
    if (maxSplitBytes * INT_NUM_TWO >= totalBytes && maxSplitBytes > 0) {
        int64_t dominantAlignBlocks = Ops::Base::CeilDiv(maxSplitBytes, alignBytes);
        int64_t dominantSpan = std::min(node.n, dominantAlignBlocks);
        int64_t perSpanBytes = maxSplitBytes / std::max<int64_t>(dominantSpan, 1);
        if (perSpanBytes < BASE_512) {
            int64_t fragFactor = Ops::Base::CeilDiv(BASE_512, std::max<int64_t>(perSpanBytes, 1));
            node.computeCost += node.mTimes * dominantSpan * (fragFactor - 1);
        }
    }
}

void SplitVTiling::ChooseBestSplitInfo(std::vector<DualSplitNode>& splitNodeList, DualSplitNode& splitInfo) const
{
    if (splitNodeList.empty()) {
        return;
    }

    // 数据量随硬件带宽提升不再是主瓶颈, 仅做极端失衡保护: 过滤掉会把某核明显拖成尾核的候选。
    int64_t bestDataBytes = splitNodeList.front().maxDataBytes;
    for (const auto& node : splitNodeList) {
        bestDataBytes = std::min(bestDataBytes, node.maxDataBytes);
    }

    std::vector<DualSplitNode> filtered;
    for (const auto& node : splitNodeList) {
        bool tooSkewed = (node.maxDataBytes > bestDataBytes * MAX_DATA_IMBALANCE_RATIO) &&
                         (node.maxDataBytes - bestDataBytes > MAX_DATA_IMBALANCE_BYTES);
        if (!tooSkewed) {
            filtered.push_back(node);
        }
    }
    if (filtered.empty()) {
        filtered = splitNodeList;
    }

    // filtered 内以 computeCost 为主排序键 (见 DualSplitNode::operator<), huge split 场景会因
    // maxSplitCnt 高而抬升 computeCost, 自然偏向切 N; 小 split 场景切 N 会抬高 mTimes, 自然少切 N。
    std::sort(filtered.begin(), filtered.end());
    splitInfo.m = filtered.front().m;
    splitInfo.n = filtered.front().n;
    splitInfo.t = filtered.front().t;
    splitInfo.delta = filtered.front().delta;

    // huge split + 大M兜底: computeCost=mTimes*(maxSplitCnt+nTimes)为乘法结构, 大M时切M能直接压低外层
    // 乘数mTimes, 收益盖过maxSplitCnt上升, 模型会误偏向切M导致N轴分核骤减性能回退。此处强制选
    // m==SPLIT_DIM_INDEX(=2)的候选, M只切2份其余核全给N, 保证huge split场景N轴充分分核。
    if (numSplit_ > HUGE_SPLIT_NUM && fusedShape_[0] >= HUGE_M) {
        for (const auto& node : filtered) {
            if (node.m == SPLIT_DIM_INDEX) {
                splitInfo.m = node.m;
                splitInfo.n = node.n;
                splitInfo.t = node.t;
                splitInfo.delta = node.delta;
                break;
            }
        }
    }

    OP_LOGD(context_->GetNodeName(),
            "choose best split m:%ld n:%ld t:%ld delta:%ld mTimes:%ld nTimes:%ld maxSplitCnt:%ld computeCost:%ld",
            splitInfo.m, splitInfo.n, splitInfo.t, splitInfo.delta, filtered.front().mTimes, filtered.front().nTimes,
            filtered.front().maxSplitCnt, filtered.front().computeCost);
}

void SplitVTiling::CalBlockSplitTwoAxis(int64_t rowM, int64_t colN, int64_t dataBytes, int64_t coreNum,
                                        DualSplitNode& splitInfo) const
{
    // 给定一个shape[M,N],和coreNum, 找到尽可能均匀且尽量用满核的切分方式
    // N按512Bytes对齐的原因是在n_real较小的时候，如果切了N轴会导致搬运性能差，设置512对齐可以减少切N轴的概率
    int64_t M = rowM;
    // split块平均n轴字节数<cacheline(128B)时，512B对齐会把N切核上界压得过低，且此时单块搬运粒度本就碎，
    // 512对齐无搬运收益，放宽到cacheline以释放更多N切核候选。平均split较大时仍保持512对齐降低切N概率。
    int64_t alignBytes = BASE_512;
    int64_t avgSplitBytes = (numSplit_ > 0) ? (colN * dataBytes / numSplit_) : (colN * dataBytes);
    if (avgSplitBytes < PURE_MOVE_BASE_LEN) {
        alignBytes = PURE_MOVE_BASE_LEN;
    }
    int64_t N = Ops::Base::CeilDiv(colN, alignBytes / dataBytes); // 平均小split时降到cacheline对齐
    int64_t cores = std::min(coreNum, Ops::Base::CeilDiv(rowM * colN * dataBytes, USED_MIN_UB_SIZE));

    OP_LOGD(context_->GetNodeName(), "get split info rowM:%ld colN:%ld M:%ld N:%ld cores:%ld", rowM, colN, M, N, cores);

    std::set<int64_t> cutCountSet;
    std::vector<DualSplitNode> splitNodeList;
    FindAllPossibleCutCnt(cutCountSet, cores);
    GetAllCutInfoNode(M, N, cores, cutCountSet, splitNodeList);
    ChooseBestSplitInfo(splitNodeList, splitInfo);

    // 仅做异常保护，不太可能发生
    if (splitInfo.m > rowM || splitInfo.n > colN) {
        splitInfo.m = std::min(splitInfo.m, rowM);
        splitInfo.n = std::min(splitInfo.n, colN);
    }
    if (splitInfo.m == 0 || splitInfo.n == 0) {
        splitInfo.m = std::min(cores, rowM);
        splitInfo.n = 1;
    }
}

void SplitVTiling::SetBlockSplitInfo(int64_t mBlockCnt, int64_t nBlockCnt)
{
    // 采用尽量均匀的方式，如 14, 切分为4块，则切分为 4 4 3 3，有多个尾块，而不是 4 4 4 2这种.
    mBlockCount_ = mBlockCnt;
    mBlockFactor_ = Ops::Base::CeilDiv(fusedShape_[0], mBlockCount_);
    mBlockFactorCount_ = (fusedShape_[0] % mBlockCount_ == 0) ? mBlockCount_ : (fusedShape_[0] % mBlockCount_);
    mBlockFactorTail_ = fusedShape_[0] / mBlockCount_;

    nBlockCount_ = nBlockCnt;
    nBlockFactor_ = Ops::Base::CeilDiv(fusedShape_[1], nBlockCount_);
    nBlockFactorCount_ = (fusedShape_[1] % nBlockCount_ == 0) ? nBlockCount_ : (fusedShape_[1] % nBlockCount_);
    nBlockFactorTail_ = fusedShape_[1] / nBlockCount_;

    realCoreNum_ = mBlockCount_ * nBlockCount_;
    blockDim_ = realCoreNum_;

    OP_LOGI(context_->GetNodeName(),
            "Get block split TotalNum-BlockCnt-MainFactor-MainCnt-TailFactor, "
            "M:%ld %ld %ld %ld %ld, N:%ld %ld %ld %ld %ld",
            fusedShape_[0], mBlockCount_, mBlockFactor_, mBlockFactorCount_, mBlockFactorTail_, fusedShape_[1],
            nBlockCount_, nBlockFactor_, nBlockFactorCount_, nBlockFactorTail_);
}

void SplitVTiling::CalBlockTilingParams()
{
    DualSplitNode splitInfo;
    CalBlockSplitTwoAxis(fusedShape_[0], fusedShape_[1], xDtypeSize_, coreNum_, splitInfo);

    SetBlockSplitInfo(splitInfo.m, splitInfo.n);
}

void SplitVTiling::CalcSplitLenCond(int64_t curSplitNumN, int64_t& condNum, int64_t& total) const
{
    if (curSplitNumN * xDtypeSize_ > PURE_MOVE_BASE_LEN) {
        condNum++;
    }
    total++;
}

void SplitVTiling::CalcSplitUBSizeCond(int64_t curSplitNumN, int64_t& condNum, int64_t& total)
{
    int64_t limitUb = fusedShape_[0] == 1 ? PURE_MOVE_BASE_UB_SIZE_M1 : PURE_MOVE_BASE_UB_SIZE;
    if (mBlockFactor_ * curSplitNumN * xDtypeSize_ > limitUb) {
        condNum += mBlockFactorCount_;
    }

    if (mBlockFactorTail_ * curSplitNumN * xDtypeSize_ > limitUb) {
        condNum += (mBlockCount_ - mBlockFactorCount_);
    }

    total += mBlockCount_;
}

int64_t SplitVTiling::UpdataNextBlockFactor(int64_t sizeSplitIdx, int64_t handedSplitSize, int64_t leftSplitSize,
                                            int64_t& handedBlockNumN)
{
    int64_t leftBlockFactorN = (handedSplitSize >= nBlockFactor_ * nBlockFactorCount_) ? nBlockFactorTail_ :
                                                                                         nBlockFactor_;

    // 更新N方向当前核的endOffset，和下一个核的startOffset
    nBlockSplitOffsetEnd_[handedBlockNumN] = sizeSplitIdx + 1;
    handedBlockNumN++;
    if (handedBlockNumN < MAX_CORE_COUNT && handedBlockNumN < nBlockCount_) {
        nBlockSplitOffsetStart_[handedBlockNumN] = (leftSplitSize == 0) ? (sizeSplitIdx + 1) : sizeSplitIdx;
    }

    return leftBlockFactorN;
}

void SplitVTiling::UpdateMAxisSplitOffset()
{
    // 要计算所有核，在N方向的idx
    for (int64_t i = 1; i < mBlockCount_; i++) {
        if (memcpy_s(nBlockSplitOffsetStart_ + i * nBlockCount_, (MAX_CORE_COUNT - i * nBlockCount_) * sizeof(int64_t),
                     nBlockSplitOffsetStart_, nBlockCount_ * sizeof(int64_t)) != EOK) {
            OP_LOGE(context_->GetNodeType(), "memcpy_s offset start failed, i:%ld mBlockCount_:%ld nBlockCount_:%ld", i,
                    mBlockCount_, nBlockCount_);
        }

        if (memcpy_s(nBlockSplitOffsetEnd_ + i * nBlockCount_, (MAX_CORE_COUNT - i * nBlockCount_) * sizeof(int64_t),
                     nBlockSplitOffsetEnd_, nBlockCount_ * sizeof(int64_t)) != EOK) {
            OP_LOGE(context_->GetNodeType(), "memcpy_s offset end failed, i:%ld mBlockCount_:%ld nBlockCount_:%ld", i,
                    mBlockCount_, nBlockCount_);
        }
    }
}

void SplitVTiling::CalcTilingKey(double condN, double condM, double totalCompareN, double totalCompareM)
{
    // 不需要做切分的场景，也走纯搬运模板
    double nowRatio = (fusedShape_[0] == 1) ? PURE_MOVE_RATIO_M1 : PURE_MOVE_RATIO;
    if ((condN > totalCompareN * nowRatio && condM > totalCompareM * nowRatio) || numSplit_ == 1) {
        tilingKey_ = SPLITV_KEY_PURE_MOVE;
    } else {
        tilingKey_ = SPLITV_KEY_UB_SPLIT_WITH_PURE_MOVE;
    }
    OP_LOGD(context_->GetNodeName(), "condN:%f condM:%f totalCompareN:%f totalCompareM:%f tilingKey_:%ld numSplit_:%ld",
            condN, condM, totalCompareN, totalCompareM, tilingKey_, numSplit_);
}

int64_t SplitVTiling::SplitPrefix(int64_t i)
{
    int64_t shape = 0;
    for (int64_t index = 0; index < i + 1; index++) {
        if (index < numSplit_) {
            shape += oriSizeSplits_[index] * splitStride_;
        }
    }
    return shape;
}

void SplitVTiling::CountSplitPrefix()
{
    for (int64_t i = 0; i < MAX_CORE_COUNT; i++) {
        nBlockSplitPrefixStart_[i] = SplitPrefix(nBlockSplitOffsetStart_[i]);
        nBlockSplitPrefixEnd_[i] = SplitPrefix(nBlockSplitOffsetEnd_[i]);
    }
}

void SplitVTiling::FillSIMTSplitVTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering FillSIMTSplitVTilingData.");
    simtSplitVTilingData_.set_mSize(static_cast<int32_t>(mSimtSize_));
    simtSplitVTilingData_.set_nSize(static_cast<int32_t>(nSimtSize_));
    simtSplitVTilingData_.set_sizeAfterSplit(static_cast<int32_t>(simtSizeAfterSplit_));
    simtSplitVTilingData_.set_splitNum(static_cast<int32_t>(numSplit_));
    simtSplitVTilingData_.set_realCoreNum(static_cast<int32_t>(realCoreNum_));
    simtSplitVTilingData_.set_colOffset(colOffset_);

    OP_LOGI(context_->GetNodeName(), "TilingData mSize:%d nSize:%d sizeAfterSplit:%d splitNum:%d realCoreNum:%d ",
            simtSplitVTilingData_.get_mSize(), simtSplitVTilingData_.get_nSize(),
            simtSplitVTilingData_.get_sizeAfterSplit(), simtSplitVTilingData_.get_splitNum(),
            simtSplitVTilingData_.get_realCoreNum());
    OP_LOGI(context_->GetNodeName(), "TilingData colOffset:   %s",
            ArrayToString(simtSplitVTilingData_.get_colOffset(), realCoreNum_).c_str());

    simtSplitVTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                       context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(simtSplitVTilingData_.GetDataSize());
}

void SplitVTiling::SetTilingMode()
{
    // 纯搬运模板的准入条件：
    // 1.
    // 多核切分后，核内统计每个Split块的N轴大小，N轴大于cacheLine(128B)的比例超过90%。(此条件是为了保证UB内单次搬运的效率)
    // 2. 多核切分后，核内统计每个Split块的size，size大于48K(经验值)的比例超过90%。(此条件是为了保证充分的利用UB空间)
    // 采用滑窗方式计算
    int64_t leftSplitSize = 0;                // 当前split_size中余留待处理的split个数
    int64_t handedSplitSize = 0;              // 已处理过的split_size个数的总和
    int64_t leftBlockFactorN = nBlockFactor_; // 余留待处理的blockFactor
    int64_t handedBlockNumN = 0;              // N轴上已处理过的block个数
    int64_t condNum[BASE_2] = {0, 0};
    int64_t totalCompareN = 0;
    int64_t totalCompareM = 0;
    for (int64_t i = 0; i < numSplit_; i++) {
        leftSplitSize = sizeSplits_[i];
        while (leftSplitSize > leftBlockFactorN) {
            // 处理掉 leftBlockFactorN 长度, 并更新到下一个block切分块. 此处 leftBlockFactorN 不可能为 0
            CalcSplitLenCond(leftBlockFactorN, condNum[0], totalCompareN);
            CalcSplitUBSizeCond(leftBlockFactorN, condNum[1], totalCompareM);
            leftSplitSize -= leftBlockFactorN;
            handedSplitSize += leftBlockFactorN;
            leftBlockFactorN = UpdataNextBlockFactor(i, handedSplitSize, leftSplitSize, handedBlockNumN);
        }

        // 如果存在splitSize长度为0的块, 或上面的while循环刚好处理完 leftSplitSize
        // 则不计入总数，不计算cacheline和ubsize是否满足准入条件
        if (leftSplitSize <= 0) {
            continue;
        }
        CalcSplitLenCond(leftSplitSize, condNum[0], totalCompareN);
        CalcSplitUBSizeCond(leftSplitSize, condNum[1], totalCompareM);
        leftBlockFactorN -= leftSplitSize;
        handedSplitSize += leftSplitSize;
        leftSplitSize = 0;
        if (leftBlockFactorN == 0) {
            leftBlockFactorN = UpdataNextBlockFactor(i, handedSplitSize, leftSplitSize, handedBlockNumN);
        }
    }

    OP_LOGD(context_->GetNodeName(), "handedSplitSize:%ld handedBlockNumN:%ld nBlockCount_:%ld", handedSplitSize,
            handedBlockNumN, nBlockCount_);

    UpdateMAxisSplitOffset();
    CountSplitPrefix();
    CalcTilingKey(condNum[0], condNum[1], totalCompareN, totalCompareM);
}

int64_t SplitVTiling::CalInputDataSize()
{
    // 计算输入的data大小
    for (size_t i = 0; i < static_cast<size_t>(splitDim_); i++) {
        mSimtSize_ *= inputShape_.GetDim(i);
    }
    for (size_t i = splitDim_; i < inputShape_.GetDimNum(); i++) {
        nSimtSize_ *= inputShape_.GetDim(i);
    }
    if (inputShape_.GetDim(splitDim_) <= 0) {
        return false;
    }
    simtSizeAfterSplit_ = nSimtSize_ / inputShape_.GetDim(splitDim_);
    return xDtypeSize_ * mSimtSize_ * nSimtSize_;
}

void SplitVTiling::DoDiffLenModeSplitVSIMTTiling(int32_t maxCoreNum)
{
    if (numSplit_ <= static_cast<int64_t>(maxCoreNum / BASE_TWO) &&
        static_cast<int64_t>(mSimtSize_ * maxFuseSizeAfterSplit_) > SIMT_THREAD_NUM) {
        int32_t perBlkSize = static_cast<int32_t>(maxCoreNum / numSplit_);
        int32_t preSum = 0;
        int32_t curSplitLen = 0;
        for (int32_t i = 0; i < numSplit_; i++) {
            int32_t curNLen = oriSizeSplits_[i] * simtSizeAfterSplit_;
            int32_t curBlkFactor = Ops::Base::CeilDiv(curNLen, perBlkSize);
            // 累加，直到curNLen耗尽
            for (int32_t j = 0; j < perBlkSize; j++) {
                if (curNLen >= curBlkFactor) {
                    curSplitLen = curBlkFactor;
                    curNLen -= curSplitLen;
                } else {
                    curSplitLen = curNLen;
                    curNLen -= curSplitLen;
                }
                preSum += curSplitLen;
                colOffset_[i * perBlkSize + j] = static_cast<uint32_t>(preSum);
            }
        }
        realCoreNum_ = perBlkSize * numSplit_;
        tilingKey_ = SIMT_KEY_SPLIT_TENSOR;
    } else {
        int32_t preSum = 0;
        for (int32_t i = 0; i < numSplit_; i++) {
            preSum += oriSizeSplits_[i] * simtSizeAfterSplit_;
            colOffset_[i] = static_cast<uint32_t>(preSum);
        }
        realCoreNum_ = numSplit_ <= maxCoreNum ? numSplit_ : maxCoreNum;
        tilingKey_ = SIMT_KEY;
    }

    FillSIMTSplitVTilingData();
}

ge::graphStatus SplitVTiling::DoSplitVSIMTTiling(int32_t maxCoreNum)
{
    OP_LOGD("SplitTilingForAscendC", "DoSplitVSIMTTiling start");
    DoDiffLenModeSplitVSIMTTiling(maxCoreNum);
    blockDim_ = realCoreNum_;
    SetBlockDimAndTilingKey();
    SetWorkspaceSize();
    OP_LOGD("SplitTilingForAscendC", "DoSplitVSIMTTiling end");
    return ge::GRAPH_SUCCESS;
}

void SplitVTiling::CalSplitSizeDiff(int32_t maxCoreNum, int64_t& maxSizeSplit, int64_t& minSizeSplit,
                                    int32_t countPerCore)
{
    if (numSplit_ <= maxCoreNum) {
        for (size_t i = 0; i < oriSizeSplits_.size(); i++) {
            if (oriSizeSplits_[i] == 0) {
                continue;
            }
            maxSizeSplit = maxSizeSplit > oriSizeSplits_[i] ? maxSizeSplit : oriSizeSplits_[i];
            minSizeSplit = minSizeSplit < oriSizeSplits_[i] ? minSizeSplit : oriSizeSplits_[i];
        }
    } else {
        for (int32_t i = 0; i < maxCoreNum; i++) {
            int32_t curSizeSplits = 0;
            for (int32_t j = 0; j < countPerCore; j++) {
                if (static_cast<int64_t>(j * maxCoreNum + i) >= numSplit_) {
                    break;
                }
                curSizeSplits += oriSizeSplits_[j * maxCoreNum + i];
            }
            if (curSizeSplits == 0) {
                continue;
            }
            maxSizeSplit = maxSizeSplit > curSizeSplits ? maxSizeSplit : curSizeSplits;
            minSizeSplit = minSizeSplit < curSizeSplits ? minSizeSplit : curSizeSplits;
        }
    }
    maxFuseSizeAfterSplit_ = maxSizeSplit * simtSizeAfterSplit_;
    minFuseSizeAfterSplit_ = minSizeSplit * simtSizeAfterSplit_;
}

// 1) 数据量过大，SIMT的性能不行
// 2) 单核处理数据量偏差大， 占总数据量的比值过高，且差值> 1024, SIMT的性能也不行
// 3) 分块少的， SIMD没有什么劣势 暂定numSplit_ <= 16
// 4) 偏差大，但是分块多的，SIMD性能不好 暂定 numSplit_ > 64
// 5) numSplit_ > 128的限制能否往上提高
// 6) 等长模式下，simd的性能更好
bool SplitVTiling::IsDoSplitVSIMT(int32_t maxCoreNum)
{
    if (isSameLenMode_) {
        return false;
    }
    if (maxCoreNum <= SIMT_MIN_CORE_NUM) {
        return false;
    }
    // (3) (5) 分块限制
    if (numSplit_ > static_cast<int64_t>(MAX_COL_OFFSET_COUNT) ||
        numSplit_ <= static_cast<int64_t>(maxCoreNum / BASE_FOUR)) {
        return false;
    }

    // 1) 数据量过大
    if (CalInputDataSize() > maxCoreNum * MAX_SIMT_PER_CORE_SIZE) {
        return false;
    }

    int32_t countPerCore = 1;
    if (numSplit_ > static_cast<int64_t>(maxCoreNum / BASE_TWO) && numSplit_ <= static_cast<int64_t>(maxCoreNum)) {
        countPerCore = 1;
    } else if (numSplit_ > maxCoreNum) {
        countPerCore = Ops::Base::CeilDiv(static_cast<int32_t>(numSplit_), maxCoreNum);
    }

    int64_t maxSizeSplit = 0;
    int64_t minSizeSplit = MAX_NUM;
    CalSplitSizeDiff(maxCoreNum, maxSizeSplit, minSizeSplit, countPerCore);

    // (2) (4)
    // 根据实际kernel侧计算的数量来判断, 单核处理的数据量不能差距不能过大
    // 考虑输出tensor被切分的场景，每个核处理的数据偏差会减小
    int32_t divPara = 1;
    if (numSplit_ <= static_cast<int64_t>(maxCoreNum / BASE_TWO)) {
        divPara = static_cast<int32_t>(maxCoreNum / numSplit_);
    }
    int64_t relativeSkew = maxSizeSplit * maxCoreNum / inputShape_.GetDim(splitDim_) / divPara;
    int64_t absoluteSkew = mSimtSize_ * (maxFuseSizeAfterSplit_ - minFuseSizeAfterSplit_) / divPara;
    if (relativeSkew >= BASE_TWO && absoluteSkew >= SIMT_THREAD_NUM) {
        simtRelativeSkew_ = relativeSkew;
        simtAbsoluteSkew_ = absoluteSkew;
        simtSkewRejected_ = true;
        return false;
    }

    return true;
}

bool SplitVTiling::IsFallbackToChunk()
{
    if (!simtSkewRejected_) {
        return false;
    }
    if (simtRelativeSkew_ >= SIMT_REJECT_RELATIVE_SKEW_UPPER || simtAbsoluteSkew_ >= SIMT_REJECT_ABSOLUTE_SKEW_UPPER) {
        return false;
    }
    OP_LOGI(context_->GetNodeName(), "SIMT rejected by skew, fallback to chunk 204. relativeSkew:%ld absoluteSkew:%ld",
            simtRelativeSkew_, simtAbsoluteSkew_);
    return true;
}

void SplitVTiling::FillTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering FillTilingData.");
    tilingData_.set_ubSize(static_cast<int64_t>(ubSize_));
    tilingData_.set_splitDim(splitDim_);
    tilingData_.set_sizeAfterSplitDim(splitStride_);
    tilingData_.set_mBlockFactor(mBlockFactor_);
    tilingData_.set_mBlockFactorTail(mBlockFactorTail_);
    tilingData_.set_mBlockFactorNum(mBlockFactorCount_);
    tilingData_.set_mBlockCount(mBlockCount_);
    tilingData_.set_gUBFactor(gUBFactor_);
    tilingData_.set_gUBFactorTail(gUBFactorTail_);
    tilingData_.set_gSize(numSplit_);
    tilingData_.set_gUBCount(gUBCount_);
    tilingData_.set_nBlockFactor(nBlockFactor_);
    tilingData_.set_nBlockFactorAlign(Ops::Base::CeilAlign(nBlockFactor_, BLOCK_SIZE / xDtypeSize_));
    tilingData_.set_nBlockFactorTail(nBlockFactorTail_);
    tilingData_.set_nBlockFactorTailAlign(Ops::Base::CeilAlign(nBlockFactorTail_, BLOCK_SIZE / xDtypeSize_));
    tilingData_.set_nBlockFactorNum(nBlockFactorCount_);
    tilingData_.set_nBlockCount(nBlockCount_);
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockFactorTail(blockFactorTail_);
    tilingData_.set_nSize(fusedShape_[1]);
    tilingData_.set_nSizeAlign(Ops::Base::CeilAlign(fusedShape_[1], BLOCK_SIZE / xDtypeSize_));
    tilingData_.set_pureOutIdx(pureOutIdx_);
    tilingData_.set_negIdx(negIdx_);
    tilingData_.set_negValue(negValue_);
    tilingData_.set_isInt32(isInt32_);
    tilingData_.set_mSize(fusedShape_[0]);

    tilingData_.set_nBlockSplitOffset(nBlockSplitOffsetStart_);
    tilingData_.set_nBlockSplitOffsetEnd(nBlockSplitOffsetEnd_);
    tilingData_.set_nBlockSplitPrefixStart(nBlockSplitPrefixStart_);
    tilingData_.set_nBlockSplitPrefixEnd(nBlockSplitPrefixEnd_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    OP_LOGI(context_->GetNodeName(),
            "TilingData ubSize:%u splitDim:%ld sizeAfterSplitDim:%ld mBlockFactor:%ld "
            "mBlockFactorTail:%ld mBlockFactorNum:%ld mBlockCount:%ld gUBFactor:%ld gUBFactorTail:%ld gSize:%ld "
            "gUBCount:%ld "
            "nBlockFactor:%ld nBlockFactorAlign:%ld nBlockFactorTail:%ld nBlockFactorTailAlign:%ld nBlockFactorNum:%ld "
            "nBlockCount:%ld realCoreNum:%ld blockFactor:%ld blockFactorTail:%ld nSize:%ld nSizeAlign:%ld "
            "pureOutIdx_:%ld",
            ubSize_, splitDim_, splitStride_, mBlockFactor_, mBlockFactorTail_, mBlockFactorCount_, mBlockCount_,
            gUBFactor_, gUBFactorTail_, numSplit_, gUBCount_, nBlockFactor_, tilingData_.get_nBlockFactorAlign(),
            nBlockFactorTail_, tilingData_.get_nBlockFactorTailAlign(), nBlockFactorCount_, nBlockCount_, realCoreNum_,
            blockFactor_, blockFactorTail_, fusedShape_[1], tilingData_.get_nSizeAlign(), pureOutIdx_);
    OP_LOGI(context_->GetNodeName(), "TilingData nBlockSplitOffset:   %s",
            ArrayToString(tilingData_.get_nBlockSplitOffset(), realCoreNum_).c_str());
    OP_LOGI(context_->GetNodeName(), "TilingData nBlockSplitOffsetEnd:%s",
            ArrayToString(tilingData_.get_nBlockSplitOffsetEnd(), realCoreNum_).c_str());
}

void SplitVTiling::SetBlockDimAndTilingKey() const
{
    context_->SetBlockDim(blockDim_);
    context_->SetTilingKey(tilingKey_);
    OP_LOGI(context_->GetNodeName(), "Tiling blockDim_:%ld tilingKey_:%ld", blockDim_, tilingKey_);
}

void SplitVTiling::SetWorkspaceSize() const
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;
}

// host 侧 uint64 fast-div magic/shift 计算, 算法与 kernel 侧 GetUintDivMagicAndShiftImpl(uint64_t) + GetUintDivMagic
// 一致, 供 kernel Simt::UintDiv<uint64_t> 使用; 用 __int128 实现 128 位除法避免溢出
static inline void HostGetUintDivMagicAndShift64(uint64_t& magic, uint32_t& shift, uint64_t divisor)
{
    if (divisor == 0) {
        magic = 0;
        shift = 0;
        return;
    }
    // pos = 64 - clz(divisor) = floor(log2(divisor)) + 1
    int64_t pos = 64 - static_cast<int64_t>(__builtin_clzll(divisor));
    int64_t cnt1 = static_cast<int64_t>(__builtin_popcountll(divisor));
    shift = (cnt1 == 1) ? static_cast<uint32_t>(pos - 1) : static_cast<uint32_t>(pos);
    uint64_t dividend = 0;
    if (shift < 64) {
        dividend = (1ULL << shift) - divisor;
    } else {
        // shift == 64, 2^64 - divisor
        dividend = 0xFFFFFFFFFFFFFFFFULL - divisor + 1;
    }
    // magic = floor(2^64 * dividend / divisor) + 1
    __uint128_t num = (static_cast<__uint128_t>(dividend) << 64);
    magic = static_cast<uint64_t>(num / divisor) + 1;
}

// 204 per-output 预存版: 与 DoSplitVSIMTChunkTiling 同样的 blockPrefix/colOffset 均衡, 额外在 host 预存
// qPerBlk/rPerBlk/divMagic/divShift/strideDiff, kernel 不再做 div/magic/strideDiff 派生
void SplitVTiling::DoSplitVSIMTChunkPreTiling(int32_t maxCoreNum)
{
    OP_LOGD(context_->GetNodeName(), "DoSplitVSIMTChunkPreTiling start");
    int32_t M = static_cast<int32_t>(mSimtSize_);
    int32_t nLen = static_cast<int32_t>(nSimtSize_);
    realCoreNum_ = maxCoreNum;
    // logicBlockNum_ = static_cast<int32_t>(realCoreNum_) * LOGIC_BLOCK_PER_CORE;

    // 1) wi / colOffset / totalElem / strideDiff
    int64_t totalElem = 0;
    uint32_t colStart = 0;
    for (int32_t i = 0; i < numSplit_; i++) {
        int32_t wi = static_cast<int32_t>(oriSizeSplits_[i] * simtSizeAfterSplit_);
        colOffsetChunkBuf_[i] = colStart;
        colStart += static_cast<uint32_t>(wi);
        strideDiffBuf_[i] = static_cast<int64_t>(nLen) - static_cast<int64_t>(wi);
        totalElem += static_cast<int64_t>(M) * static_cast<int64_t>(wi);
    }

    // 限制 logicBlockNum_: 每个 block 至少 128 个元素(每线程1个), 避免数据量少时大量线程空跑
    int32_t maxLogicBlockByData = static_cast<int32_t>((totalElem + THREADS_PER_LOGIC_BLOCK - 1) /
                                                       THREADS_PER_LOGIC_BLOCK);
    int32_t maxLogicBlockByCore = static_cast<int32_t>(realCoreNum_) * LOGIC_BLOCK_PER_CORE;
    logicBlockNum_ = std::min(maxLogicBlockByCore, maxLogicBlockByData);
    logicBlockNum_ = std::max(logicBlockNum_, static_cast<int32_t>(numSplit_)); // 至少每个 output 1 个 block

    // 2) 按 M*wi 比例分配整数逻辑 block, 空输出(wi==0)分配 0; 暂存到 blockPrefixBuf_
    int32_t blkSum = 0;
    int32_t maxCopyIdx = -1;
    int64_t maxCopy = -1;
    for (int32_t i = 0; i < numSplit_; i++) {
        int32_t wi = static_cast<int32_t>(oriSizeSplits_[i] * simtSizeAfterSplit_);
        if (wi <= 0) {
            blockPrefixBuf_[i] = 0;
            continue;
        }
        int64_t copy = static_cast<int64_t>(M) * static_cast<int64_t>(wi);
        int32_t blk = static_cast<int32_t>((static_cast<int64_t>(logicBlockNum_) * copy + totalElem / 2) / totalElem);
        blk = std::max(1, blk);
        blockPrefixBuf_[i] = static_cast<uint32_t>(blk);
        blkSum += blk;
        if (copy > maxCopy) {
            maxCopy = copy;
            maxCopyIdx = i;
        }
    }

    // 3) 校正使 Σ blkForTensor == logicBlockNum_
    //    diff >= 0 (欠分配): 差值全部补到拷贝量最大的 output
    //    diff <  0 (超分配): 按拷贝量降序逐个扣减, 每个非空 output 保留至少 1 块, 避免下游 q=total/blkCnt 除零
    if (maxCopyIdx >= 0) {
        // int32_t diff = logicBlockNum_ - blkSum;
        int32_t diff = maxLogicBlockByCore - blkSum;
        if (diff >= 0) {
            // blockPrefixBuf_[maxCopyIdx] = static_cast<uint32_t>(static_cast<int32_t>(blockPrefixBuf_[maxCopyIdx]) +
            //                                                     diff);
            realCoreNum_ = CeilDiv(blkSum, 16);
        } else {
            int32_t need = -diff;
            // 收集可扣减的输出(blk > 1), 按拷贝量 M*wi 降序排序后贪心扣减
            std::vector<int32_t> order;
            order.reserve(static_cast<size_t>(numSplit_));
            for (int32_t i = 0; i < numSplit_; i++) {
                if (static_cast<int32_t>(blockPrefixBuf_[i]) > 1) {
                    order.push_back(i);
                }
            }
            std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
                int32_t wa = static_cast<int32_t>(oriSizeSplits_[a] * simtSizeAfterSplit_);
                int32_t wb = static_cast<int32_t>(oriSizeSplits_[b] * simtSizeAfterSplit_);
                return static_cast<int64_t>(M) * wa > static_cast<int64_t>(M) * wb;
            });
            for (int32_t idx : order) {
                if (need <= 0) {
                    break;
                }
                int32_t spare = static_cast<int32_t>(blockPrefixBuf_[idx]) - 1;
                int32_t take = std::min(spare, need);
                blockPrefixBuf_[idx] = static_cast<uint32_t>(static_cast<int32_t>(blockPrefixBuf_[idx]) - take);
                need -= take;
            }
            // need > 0 表示非空输出数 > logicBlockNum_, 无法在保证每输出 >= 1 块的前提下归零
            if (need > 0) {
                OP_LOGW(context_->GetNodeName(),
                        "SIMT chunk-pre blk overflow cannot be fully corrected: "
                        "numSplit=%ld remain=%d, total blocks exceed logicBlockNum_=%d.",
                        numSplit_, need, logicBlockNum_);
            }
        }
    }

    // 4) 前缀求和填 blockPrefix, 同时算 qPerBlk/rPerBlk/divMagic/divShift
    uint32_t pre = 0;
    for (int32_t i = 0; i < numSplit_; i++) {
        int32_t wi = static_cast<int32_t>(oriSizeSplits_[i] * simtSizeAfterSplit_);
        if (wi <= 0) {
            // 空 output: block 数为 0, 前缀保持不变, kernel 自然跳过
            blockPrefixBuf_[i] = pre;
            qPerBlkBuf_[i] = 0;
            rPerBlkBuf_[i] = 0;
            divMagicBuf_[i] = 0;
            divShiftBuf_[i] = 0;
            continue;
        }
        uint32_t blkCnt = blockPrefixBuf_[i]; // 该 output 的 block 数(前缀覆盖前)
        int64_t total = static_cast<int64_t>(M) * static_cast<int64_t>(wi);
        // q/r: 前 r 个 block 每个多分 1 个元素, 与 split_v_simt_chunk_ub.h 一致
        int64_t q = total / static_cast<int64_t>(blkCnt);
        int64_t r = total % static_cast<int64_t>(blkCnt);
        qPerBlkBuf_[i] = q;
        rPerBlkBuf_[i] = r;
        // fast-div magic/shift for wi (uint64), 与 kernel Simt::UintDiv<uint64_t> 配套
        uint64_t magic = 0;
        uint32_t shift = 0;
        HostGetUintDivMagicAndShift64(magic, shift, static_cast<uint64_t>(wi));
        divMagicBuf_[i] = magic;
        divShiftBuf_[i] = static_cast<int32_t>(shift);

        pre += blockPrefixBuf_[i];
        blockPrefixBuf_[i] = pre;
    }

    // 5) 1024 逻辑 block = 64 核 * 16
    // realCoreNum_ = std::min(maxCoreNum, SIMT_VECTOR_CORE_NUM);
    blockDim_ = realCoreNum_;
    tilingKey_ = SIMT_CHUNK_KEY;

    OP_LOGI(context_->GetNodeName(), "SIMT chunk-pre 204 M:%d nLen:%d splitNum:%ld totalElem:%ld blockDim:%ld", M, nLen,
            numSplit_, totalElem, blockDim_);
    OP_LOGI(context_->GetNodeName(), "SIMT chunk-pre 204 blockPrefix:  %s",
            ArrayToString(blockPrefixBuf_, static_cast<size_t>(numSplit_)).c_str());
    OP_LOGI(context_->GetNodeName(), "SIMT chunk-pre 204 colOffset:    %s",
            ArrayToString(colOffsetChunkBuf_, static_cast<size_t>(numSplit_)).c_str());
    OP_LOGI(context_->GetNodeName(), "SIMT chunk-pre 204 strideDiff:   %s",
            ArrayToString(strideDiffBuf_, static_cast<size_t>(numSplit_)).c_str());

    FillSIMTChunkPreTilingData();
    SetBlockDimAndTilingKey();
    SetWorkspaceSize();
    OP_LOGD(context_->GetNodeName(), "DoSplitVSIMTChunkPreTiling end");
}

void SplitVTiling::FillSIMTChunkPreTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering FillSIMTChunkPreTilingData.");
    simtChunkPreTilingData_.set_mSize(static_cast<int32_t>(mSimtSize_));
    simtChunkPreTilingData_.set_nSize(static_cast<int32_t>(nSimtSize_));
    simtChunkPreTilingData_.set_splitNum(static_cast<int32_t>(numSplit_));
    simtChunkPreTilingData_.set_realCoreNum(static_cast<int32_t>(realCoreNum_));
    simtChunkPreTilingData_.set_blockPrefix(blockPrefixBuf_);
    simtChunkPreTilingData_.set_colOffset(colOffsetChunkBuf_);
    simtChunkPreTilingData_.set_qPerBlk(qPerBlkBuf_);
    simtChunkPreTilingData_.set_rPerBlk(rPerBlkBuf_);
    simtChunkPreTilingData_.set_divMagic(divMagicBuf_);
    simtChunkPreTilingData_.set_divShift(divShiftBuf_);
    simtChunkPreTilingData_.set_strideDiff(strideDiffBuf_);

    OP_LOGI(context_->GetNodeName(), "ChunkPreTilingData mSize:%d nSize:%d splitNum:%d realCoreNum:%d",
            simtChunkPreTilingData_.get_mSize(), simtChunkPreTilingData_.get_nSize(),
            simtChunkPreTilingData_.get_splitNum(), simtChunkPreTilingData_.get_realCoreNum());

    simtChunkPreTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                         context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(simtChunkPreTilingData_.GetDataSize());
}

ge::graphStatus SplitVTiling::DoSplitVTiling(int32_t maxCoreNum, uint32_t ubSize)
{
    OP_CHECK_IF((InitParams(maxCoreNum, ubSize) != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "SplitV Tiling InitParams Failed."), return ge::GRAPH_FAILED);

    if (IsDoSplitVSIMT(maxCoreNum)) {
        return DoSplitVSIMTTiling(maxCoreNum);
    }
    // if (IsFallbackToChunk()) {
    //     DoSplitVSIMTChunkPreTiling(maxCoreNum);
    //     // DoSplitVSIMTChunkTiling(maxCoreNum);
    //     return ge::GRAPH_SUCCESS;
    // }
    if (numSplit_ != 0 && numSplit_ != 1 && isSameLenMode_) {
        // SamLenMode fuse to [M G N]
        FuseInputShapeSameLen();
    } else if (nonZeroSplitCnt_ == 1) {
        // 完全的纯搬运场景
        // input (12800, 2)  split_dim:1  sizeSplit:[0, 2, -1]
        FuseAllShape();
    } else {
        FuseInputShape();
    }

    if (numSplit_ != 0 && numSplit_ != 1 && isSameLenMode_) {
        tilingKey_ = (numSplit_ == 1 || fusedShape_[1] >= (PURE_MOVE_BASE_LEN / xDtypeSize_)) ? SPLIT_KEY_PURE_MOVE :
                                                                                                SPLIT_KEY_UB_SPLIT;
        CalcSameLenTilingInfo();
    } else if (fusedShape_[0] == 0 || fusedShape_[1] == 0) {
        CalEmptyInputTiling();
    } else if (nonZeroSplitCnt_ == 1) {
        SetPureMoveTilingMode();
    } else {
        CalBlockTilingParams();
        SetTilingMode();
    }

    FillTilingData();
    SetBlockDimAndTilingKey();
    SetWorkspaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitVTilingAscendC(gert::TilingContext* context, int32_t maxCoreNum, uint32_t ubSize,
                                    int32_t isSameLen)
{
    SplitVTiling tiling(context);
    // SameLenSplit goto new branch
    return isSameLen == 1 ? tiling.DoSplitTiling(maxCoreNum, ubSize) : tiling.DoSplitVTiling(maxCoreNum, ubSize);
}

int64_t SplitVTiling::CeilDiv(int64_t value, int64_t factor) const
{
    int64_t valueNum = 0;
    if (factor == 0) {
        OP_LOGE(context_->GetNodeName(), "SplitTiling CeilDiv divideNum is 0!");
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

int64_t SplitVTiling::FloorAlign(int64_t value, int64_t factor) const
{
    int64_t valueNum = 0;
    if (factor == 0) {
        OP_LOGE(context_->GetNodeName(), "SplitTiling FloorAlign divideNum is 0!");
        return value;
    }
    if (value % factor == 0 || value < factor) {
        valueNum = value;
    } else {
        valueNum = value / factor * factor;
    }
    return valueNum;
}

ge::graphStatus SplitVTiling::GetInputParamsSameLen()
{
    auto xInput = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    const gert::Shape& xInputShape = Ops::Base::EnsureNotScalar(xInput->GetStorageShape());
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto attr0 = attrs->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attr0);
    inputShape_ = xInputShape;
    numSplit_ = *attr0;
    int64_t inputXDimNum = static_cast<int64_t>(inputShape_.GetDimNum());

    if (numSplit_ == 1) {
        // 如果不切分, 则默认splitDim为0
        splitDim_ = 0;
    } else {
        OP_CHECK_IF(!Ops::Base::GetConstInt(context_, SPLIT_DIM_INDEX_SAME_LEN, splitDim_),
                    OP_LOGE(context_->GetNodeName(), "SplitTiling get split_dim failed"), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((splitDim_ < -inputXDimNum || splitDim_ >= inputXDimNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "split_dim", std::to_string(splitDim_).c_str(),
                    ("The value of split_dim must be within the range [-" + std::to_string(inputXDimNum) + ", " +
                     std::to_string(inputXDimNum - 1) + "].")
                        .c_str()),
                return ge::GRAPH_FAILED);
    if (splitDim_ < 0) {
        splitDim_ = splitDim_ + inputXDimNum;
    }

    OP_CHECK_IF(
        numSplit_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_split", std::to_string(numSplit_).c_str(),
                                              "The value of num_split must be greater than 0."),
        return ge::GRAPH_FAILED);

    const int64_t dim = inputShape_.GetDim(splitDim_);
    OP_CHECK_IF(dim % numSplit_ != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x", Ops::Base::ToString(inputShape_).c_str(),
                    ("Shape [" + std::to_string(splitDim_) + "] of x must be exactly divided by num_split (" +
                     std::to_string(numSplit_) + ").")
                        .c_str()),
                return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitVTiling::InitParamsSameLen(int32_t maxCoreNum, uint32_t ubSize)
{
    coreNum_ = std::min(maxCoreNum, static_cast<int32_t>(MAX_CORE_COUNT));
    ubSize_ = ubSize;

    auto xInputDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    ge::DataType xDtype = xInputDesc->GetDataType();
    xDtypeSize_ = ge::GetSizeByDataType(xDtype);
    OP_CHECK_IF(xDtypeSize_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dtype_size",
                                                      std::to_string(xDtypeSize_).c_str(),
                                                      "The value of dtype_size must be greater than 0."),
                return ge::GRAPH_FAILED);

    if (GetInputParamsSameLen() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void SplitVTiling::FuseInputShapeSameLen()
{
    int64_t mSize = 1;
    int64_t nSize = inputShape_.GetDim(splitDim_) / numSplit_;
    for (size_t i = 0; i < static_cast<size_t>(splitDim_); i++) {
        mSize *= inputShape_.GetDim(i);
    }

    for (size_t i = splitDim_ + 1; i < inputShape_.GetDimNum(); i++) {
        nSize *= inputShape_.GetDim(i);
    }

    // G is numSplit_
    fusedShape_[0] = mSize;
    fusedShape_[1] = nSize;

    // Set mSzie and nSize
    mBlockFactorCount_ = fusedShape_[0];
    nBlockFactorCount_ = fusedShape_[1];

    OP_LOGI(context_->GetNodeName(), "After SameLen fused shape, fusedShape_:%ld %ld", fusedShape_[0], fusedShape_[1]);
}

void SplitVTiling::CalcSameLenTilingInfo()
{
    OP_LOGD(context_->GetNodeName(), "Entering SplitTiling CalcSameLenTilingInfo.");
    // First cut UB
    int64_t blockEleNum = BLOCK_SIZE / xDtypeSize_;
    int64_t halfUbEleNum = static_cast<int64_t>(ubSize_ / BASE_2) / static_cast<int64_t>(xDtypeSize_);
    int64_t quarterUbEleNum = ((static_cast<int64_t>(ubSize_)) / BASE_2 - BASE_256) / BASE_2 / xDtypeSize_;
    int64_t ubOutSize = 0;
    if (fusedShape_[1] > halfUbEleNum) {
        // UB only cut axisN
        mBlockFactor_ = 1;
        mBlockCount_ = fusedShape_[0];
        mBlockFactorTail_ = 1;
        gUBFactor_ = 1;
        gUBCount_ = numSplit_;
        gUBFactorTail_ = 1;
        nBlockFactor_ = FloorAlign(halfUbEleNum, blockEleNum); // Align to blockSize
        nBlockCount_ = CeilDiv(fusedShape_[1], halfUbEleNum);
        nBlockFactorTail_ = fusedShape_[1] % nBlockFactor_ == 0 ? nBlockFactor_ : fusedShape_[1] % nBlockFactor_;
    } else {
        // UB cut axisM/G
        nBlockFactor_ = fusedShape_[1];
        nBlockCount_ = 1;
        nBlockFactorTail_ = fusedShape_[1];
        if (tilingKey_ == SPLIT_KEY_PURE_MOVE) {
            CalcSameLenCopyTilingInfo(halfUbEleNum, blockEleNum);
        } else if (tilingKey_ == SPLIT_KEY_UB_SPLIT) {
            CalcSameLenSplitTilingInfo(quarterUbEleNum, blockEleNum);
        }
    }

    // Then cut Block
    ubOutSize = mBlockCount_ * gUBCount_ * nBlockCount_;
    realCoreNum_ = ubOutSize < coreNum_ ? ubOutSize : coreNum_;
    blockDim_ = realCoreNum_;
    blockFactor_ = ubOutSize / realCoreNum_;
    blockFactorTail_ = ubOutSize % realCoreNum_;
}

void SplitVTiling::CalcSameLenCopyTilingInfo(int64_t halfUbEleNum, int64_t blockEleNum)
{
    int64_t spaceWithoutN = 0;
    int64_t factorTmp = 0;
    spaceWithoutN = halfUbEleNum / (CeilDiv(fusedShape_[1], blockEleNum) * blockEleNum);
    factorTmp = std::floor(std::sqrt(spaceWithoutN));
    mBlockFactor_ = FloorAlign(factorTmp, blockEleNum) >= fusedShape_[0] ? fusedShape_[0] :
                                                                           FloorAlign(factorTmp, blockEleNum);
    mBlockCount_ = CeilDiv(fusedShape_[0], mBlockFactor_);
    mBlockFactorTail_ = fusedShape_[0] % mBlockFactor_ == 0 ? mBlockFactor_ : fusedShape_[0] % mBlockFactor_;
    gUBFactor_ = factorTmp >= numSplit_ ? numSplit_ : factorTmp;
    gUBCount_ = CeilDiv(numSplit_, gUBFactor_);
    gUBFactorTail_ = numSplit_ % gUBFactor_ == 0 ? gUBFactor_ : numSplit_ % gUBFactor_;
}

void SplitVTiling::CalcSameLenSplitTilingInfo(int64_t halfUbEleNum, int64_t blockEleNum)
{
    // SPLIT_KEY_UB_SPLIT need extra reserve output space
    int64_t factorTmp = 0;
    // UB layout: [Mi, GiN_align] and [Mi, N] ==> Mi * GiN_align + Mi * N
    int64_t nAlignNum = CeilDiv(fusedShape_[1], blockEleNum) * blockEleNum;

    factorTmp = std::floor(std::sqrt(halfUbEleNum / fusedShape_[1]));

    // ===== Pre-compute original branch (先算m再算g) =====
    int64_t origMBF = FloorAlign(factorTmp, blockEleNum) >= fusedShape_[0] ? fusedShape_[0] :
                                                                             FloorAlign(factorTmp, blockEleNum);
    int64_t origMBC = CeilDiv(fusedShape_[0], origMBF);
    int64_t origMBFT = fusedShape_[0] % origMBF == 0 ? origMBF : fusedShape_[0] % origMBF;

    int64_t origGBF = factorTmp - 1;
    int64_t origMNAlign = origMBF * nAlignNum;
    origGBF = AdjustUbFactor(origGBF, origMNAlign) > numSplit_ ? numSplit_ : AdjustUbFactor(origGBF, origMNAlign);
    int64_t origGBCount = CeilDiv(numSplit_, origGBF);
    int64_t origGBFT = numSplit_ % origGBF == 0 ? origGBF : numSplit_ % origGBF;
    int64_t origTiles = origMBC * origGBCount; // 优化前总tile数
    origTiles = origTiles < coreNum_ ? origTiles : coreNum_;

    // ===== Pre-compute small-G branch (g轴全载，剩余空间全给m轴) =====
    int64_t newMBF = 0, newMBC = 0, newMBFT = 0;
    int64_t newGBF = 0, newGBCount = 0, newGBFT = 0;
    int64_t newTiles = 0;
    bool smallGAvail = FloorAlign(factorTmp, blockEleNum) >= numSplit_;

    if (smallGAvail) {
        newGBF = numSplit_;
        int64_t tmp = FloorAlign(halfUbEleNum / fusedShape_[1] / newGBF, blockEleNum);
        int64_t newAlignNum = CeilDiv(newGBF * fusedShape_[1], blockEleNum) * blockEleNum;
        newMBF = AdjustUbFactor(tmp, newAlignNum) >= fusedShape_[0] ? fusedShape_[0] : AdjustUbFactor(tmp, newAlignNum);
        newMBC = CeilDiv(fusedShape_[0], newMBF);
        newMBFT = fusedShape_[0] % newMBF == 0 ? newMBF : fusedShape_[0] % newMBF;
        newGBCount = CeilDiv(numSplit_, newGBF);
        newGBFT = numSplit_ % newGBF == 0 ? newGBF : numSplit_ % newGBF;
        newTiles = newMBC * newGBCount; // 优化后总tile数
        newTiles = newTiles < coreNum_ ? newTiles : coreNum_;
    }

    // ===== Guard: avoid severe core under-utilization =====
    bool notUseSmallG = true;
    if (smallGAvail && newTiles > 0) {
        notUseSmallG = (origTiles / newTiles >= SMALL_G_TILE_RATIO_LIMIT) || (origGBF < newGBF);
    }
    if (!notUseSmallG && newGBF > BASE_FOUR) {
        notUseSmallG = true;
    }
    if (notUseSmallG) {
        mBlockFactor_ = origMBF;
        mBlockCount_ = origMBC;
        mBlockFactorTail_ = origMBFT;
        gUBFactor_ = origGBF;
        gUBCount_ = origGBCount;
        gUBFactorTail_ = origGBFT;
    } else {
        mBlockFactor_ = newMBF;
        mBlockCount_ = newMBC;
        mBlockFactorTail_ = newMBFT;
        gUBFactor_ = newGBF;
        gUBCount_ = newGBCount;
        gUBFactorTail_ = newGBFT;
        if (numSplit_ == BASE_2 && fusedShape_[1] == 1) {
            tilingKey_ = SPLIT_KEY_UB_SPLIT_DEINTERLEAVE;
        } else {
            tilingKey_ = SPLIT_KEY_UB_SPLIT_SMALL_G;
        }
    }
}

int64_t SplitVTiling::AdjustUbFactor(int64_t factor, int64_t alignFactor) const
{
    int64_t adjustFactor = factor;
    // B8 and B16 need judge whether exceed the range of gather-index
    if (xDtypeSize_ == BASE_2 && factor * alignFactor > INT16_MAX_VALUE) {
        adjustFactor = INT16_MAX_VALUE / alignFactor;
    } else if (xDtypeSize_ == 1 && factor * alignFactor > INT16_MAX_VALUE * BASE_2) {
        adjustFactor = INT16_MAX_VALUE * BASE_2 / alignFactor;
    }
    return adjustFactor;
}

ge::graphStatus SplitVTiling::DoSplitTiling(int32_t maxCoreNum, uint32_t ubSize)
{
    OP_LOGD("SplitTilingForAscendC", "DoSplitTiling start");
    OP_CHECK_IF((InitParamsSameLen(maxCoreNum, ubSize) != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "SplitTiling InitParams Failed."), return ge::GRAPH_FAILED);

    // Reshape to [M G N]
    FuseInputShapeSameLen();

    int64_t nSwitchSize = PURE_MOVE_BASE_LEN / xDtypeSize_;
    if (fusedShape_[0] == 0 || fusedShape_[1] == 0) {
        // Empty branch
        tilingKey_ = SPLIT_KEY_PURE_MOVE;
        realCoreNum_ = 0;
        blockDim_ = 1;
    } else {
        tilingKey_ = (numSplit_ == 1 || fusedShape_[1] >= nSwitchSize) ? SPLIT_KEY_PURE_MOVE : SPLIT_KEY_UB_SPLIT;
        CalcSameLenTilingInfo();
    }

    FillTilingData();
    SetBlockDimAndTilingKey();
    SetWorkspaceSize();
    OP_LOGD("SplitTilingForAscendC", "DoSplitTiling end");
    return ge::GRAPH_SUCCESS;
}

graphStatus Tiling4SplitV(TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do Tiling4SplitV");
    auto compile_info = static_cast<const SplitVCompileInfo*>(context->GetCompileInfo());

    OP_LOGD(context->GetNodeName(), "AscendC splitV tiling start");
    int32_t maxCoreNum = static_cast<int32_t>(compile_info->core_num);
    uint32_t ubSizePlatform = compile_info->ubSizePlatform;
    OP_CHECK_IF(
        ubSizePlatform <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ub_size", std::to_string(ubSizePlatform).c_str(),
                                              "The value of ub_size must be greater than 0."),
        return GRAPH_FAILED);
    int32_t isSameLen = 0;
    OP_CHECK_IF(SplitVTilingAscendC(context, maxCoreNum, ubSizePlatform, isSameLen) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "AscendC splitV tiling function call failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

graphStatus TilingPrepare4SplitV(TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<SplitVCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    OP_LOGD(context->GetNodeName(), "AscendC splitV tiling prepare");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compile_info->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compile_info->core_num <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core_num",
                                                      std::to_string(compile_info->core_num).c_str(),
                                                      "The value of core_num must be greater than 0."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compile_info->ubSizePlatform = static_cast<uint32_t>(ubSize);
    OP_CHECK_IF((compile_info->ubSizePlatform <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ub_size",
                                                      std::to_string(compile_info->ubSizePlatform).c_str(),
                                                      "The value of ub_size must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SplitV).Tiling(Tiling4SplitV).TilingParse<SplitVCompileInfo>(TilingPrepare4SplitV);
} // namespace optiling
