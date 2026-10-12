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
 * \file transpose_tiling_with_gather_arch35.cpp
 * \brief
 */
#include "transpose_tiling_with_gather_arch35.h"

namespace optiling {
namespace TransWithGather {
static constexpr int8_t NUM_TWO = 2;
static constexpr int8_t NUM_THREE = 3;
static constexpr int64_t NUM_FOUR = 4;
static constexpr int64_t NUM_EIGHT = 8;
static constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;

/**
 * @brief 计算 Gather 路径 UB 预算（dataTensorSize / indexTensorSize）
 *
 * Gather 路径 UB 中同时存在：输入 queue（双缓冲）×2 + 输出 queue（双缓冲）×2
 * + Gather 索引 buffer。按"输入2份 + 输出2份 + 索引1份"比例切分 UB 预算，
 * 并对 ubBlockSize 块对齐。
 *
 * 分档依据（元素字节数，决定索引占用比例）：
 *   - 1B（8bit）：data 占 ub/(2*2+2)=ub/6；索引需 16bit → index = data*2
 *   - 8B（64bit）：data 占 ub/(2*2*8+4)*8；索引 32bit → index = data/2
 *   - 其他（2/4B）：data 占 ub/(2*2+1)=ub/5；索引同宽 → index = data
 */
void TransposeGatherTiling::CalcTensorSize()
{
    // ping pong, input and output
    if (shapeInfo_.eleLenInBytes == 1L) {
        dataTensorSize_ = static_cast<uint32_t>(platInfo_.ubSize / (NUM_TWO * NUM_TWO + NUM_TWO) /
                                                platInfo_.ubBlockSize * platInfo_.ubBlockSize);
        indexTensorSize_ = dataTensorSize_ * static_cast<uint32_t>(NUM_TWO);
    } else if (shapeInfo_.eleLenInBytes == NUM_EIGHT) {
        dataTensorSize_ = static_cast<uint32_t>((platInfo_.ubSize / (NUM_TWO * NUM_TWO * NUM_EIGHT + NUM_FOUR) *
                                                 NUM_EIGHT / platInfo_.ubBlockSize * platInfo_.ubBlockSize));
        indexTensorSize_ = dataTensorSize_ / static_cast<uint32_t>(NUM_TWO);
    } else {
        dataTensorSize_ = static_cast<uint32_t>(platInfo_.ubSize / (NUM_TWO * NUM_TWO + 1) / platInfo_.ubBlockSize *
                                                platInfo_.ubBlockSize);
        indexTensorSize_ = dataTensorSize_;
    }
}

/**
 * @brief 计算 shape[beg, end) 区间的元素数乘积
 *
 * @param shape 输入 shape（此处传 reducedInShape 或 reducedOutShape）
 * @param beg   起始索引（含）
 * @param end   结束索引（不含）
 * @return 区间内各轴长度乘积
 */
int64_t TransposeGatherTiling::CalcShapeSize(const std::vector<int64_t>& shape, int64_t beg, int64_t end)
{
    int64_t res = 1;
    for (auto idx = beg; idx < end; ++idx) {
        res *= shape[idx];
    }
    return res;
}

/**
 * @brief 选择进入 UB 的输入轴（inUbPerm）
 *
 * 从输入最末轴（最连续）向前收集轴索引，最多 UB_MAX_BRW_NUM=3 根，
 * 直到已收集轴的元素容量乘积超过 sqrtedTensor（正方形 block 边长预算）：
 *   inUbPerm_[0..cnt-1] = [最末轴, 次末轴, ...]
 *
 * @param sqrtedTensor block 边长预算（元素数）
 */
void TransposeGatherTiling::CalcInUbPerm(int64_t sqrtedTensor)
{
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; --i) {
        inUbPerm_.perm[inUbPerm_.cnt++] = i;
        inUbPermSet_.insert(i);
        allUbPerm_.insert(i);
        if (inUbPerm_.cnt >= UB_MAX_BRW_NUM ||
            CalcShapeSize(shapeInfo_.reducedInShape, i, shapeInfo_.dim) > sqrtedTensor) {
            break;
        }
    }
}

/**
 * @brief 选择进入 UB 的输出轴（outUbPerm）
 *
 * 从输出最末轴向前收集输出轴对应的输入轴索引（reducedPerm[i]），
 * 最多 UB_MAX_BRW_NUM=3 根，直到乘积超过 sqrtedTensor。
 * 与 CalcInUbPerm 的结果合并为 allUbPerm（真正进入 UB 循环的轴集合）。
 *
 * @param sqrtedTensor block 边长预算（元素数）
 */
void TransposeGatherTiling::CalcOutUbPerm(int64_t sqrtedTensor)
{
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; --i) {
        outUbPerm_.perm[outUbPerm_.cnt++] = shapeInfo_.reducedPerm[i];
        allUbPerm_.insert(shapeInfo_.reducedPerm[i]);
        if (outUbPerm_.cnt >= UB_MAX_BRW_NUM ||
            CalcShapeSize(shapeInfo_.reducedOutShape, i, shapeInfo_.dim) > sqrtedTensor) {
            break;
        }
    }
}

/**
 * @brief 根据 UB 越界情况回调切分因子（AdjustUbCutAxisFactor）
 *
 * 对拟定的 cut factor（axisFactor）做输入/输出两侧的 UB 越界校验：
 * 若"当前 block 数据量 > elemInTensor"，按 elemInTensor/对侧尺寸/block 回退因子。
 *
 * 三种 axisFlag 场景（与 CalcUbAxisCutFactor 调用对应）：
 *   - 0：in/out 切同一条轴 —— 同时校验 dst（输出 UB）与 src（输入 UB）两侧；
 *   - 1 / 3：切"输出"轴 —— 校验 dst 溢出（输入侧已计入 inUbCutAxisFactor），
 *        src 侧累计整块是否超 elemInTensor；
 *   - 2：切"输入"轴 —— 校验 src 溢出（输出侧已计入 outUbCutAxisFactor）。
 *
 * 回退公式（以 axisFlag==0 的 dst 侧为例）：
 *   dstFactor = elemInTensor / dstInUbAxesSize / elemPerBlock * elemPerBlock / dstOutUbAxesSize
 *   即用剩余容量除以对侧轴乘积、再向 block 对齐后取整，确保回退后不越界。
 *
 * @param axisFactor  [in,out] 待回调的切分因子
 * @param axisFlag   0=同轴; 1=输出轴; 2=输入轴; 3=输出轴(计入输入因子后)
 * @param elemInTensor 数据 tensor 的元素容量上限
 */
void TransposeGatherTiling::AdjustUbCutAxisFactor(int32_t& axisFactor, int8_t axisFlag, int64_t elemInTensor)
{
    int64_t srcInUbAxesSize = 1;
    int64_t srcOutUbAxesSize = 1;
    int64_t dstInUbAxesSize = 1;
    int64_t dstOutUbAxesSize = 1;

    // dst 侧（输出视角）：outUbPerm 前 cnt-1 根轴（非切分轴）的乘积
    // dst 侧输入贡献：inUbPerm 中还未被 outUbPerm 占用的轴
    CalcUbAxesSizes(outUbPerm_, inUbPerm_, dstOutUbAxesSize, dstInUbAxesSize);
    // src 侧（输入视角）：inUbPerm 前 cnt-1 根非切分轴
    // src 侧输出贡献：outUbPerm 中还未被 inUbPerm 占用的轴
    CalcUbAxesSizes(inUbPerm_, outUbPerm_, srcInUbAxesSize, srcOutUbAxesSize);

    int64_t elemPerBlock = platInfo_.ubBlockSize / shapeInfo_.eleLenInBytes;
    // in and out ub cut same axis
    if (axisFlag == 0) {
        AdjustUbCutFactorSameAxis(axisFactor, dstInUbAxesSize, dstOutUbAxesSize, srcInUbAxesSize, srcOutUbAxesSize,
                                  elemInTensor, elemPerBlock);
        // out ub cut axis
    } else if (axisFlag == 1 || axisFlag == NUM_THREE) {
        AdjustUbCutFactorOutAxis(axisFactor, dstInUbAxesSize, dstOutUbAxesSize, srcInUbAxesSize, srcOutUbAxesSize,
                                 elemInTensor, elemPerBlock, axisFlag);
        // in ub cut axis
    } else if (axisFlag == NUM_TWO) {
        AdjustUbCutFactorInAxis(axisFactor, dstInUbAxesSize, dstOutUbAxesSize, srcInUbAxesSize, srcOutUbAxesSize,
                                elemInTensor, elemPerBlock);
    }
}

/**
 * @brief 计算 UB 内主/辅视角的轴尺寸乘积对（dst/src 对称段参数化）
 *
 * main 侧：mainUbPerm 前 cnt-1 根轴（非切分轴）的乘积，并从 allUbPerm 剔除；
 * vice 侧：viceUbPerm 中还未被 mainUbPerm 占用的轴的乘积。
 * dst 侧调用 (outUbPerm_, inUbPerm_, dstOutUbAxesSize, dstInUbAxesSize)，
 * src 侧调用 (inUbPerm_, outUbPerm_, srcInUbAxesSize, srcOutUbAxesSize)。
 *
 * @param mainUbPerm   主视角轴表（dst 侧为 outUbPerm，src 侧为 inUbPerm）
 * @param viceUbPerm   辅视角轴表（dst 侧为 inUbPerm，src 侧为 outUbPerm）
 * @param mainUbAxesSize [in,out] main 侧轴乘积（传入初始值 1）
 * @param viceUbAxesSize [in,out] vice 侧轴乘积（传入初始值 1）
 */
void TransposeGatherTiling::CalcUbAxesSizes(const UbPermInfo& mainUbPerm, const UbPermInfo& viceUbPerm,
                                            int64_t& mainUbAxesSize, int64_t& viceUbAxesSize)
{
    std::set<int8_t> viceUbPermLeft(allUbPerm_);
    for (int8_t i = 0; i < mainUbPerm.cnt - 1; ++i) {
        mainUbAxesSize *= shapeInfo_.reducedInShape[mainUbPerm.perm[i]];
        viceUbPermLeft.erase(mainUbPerm.perm[i]);
    }
    for (int8_t i = 0; i < viceUbPerm.cnt - 1; ++i) {
        if (viceUbPermLeft.find(viceUbPerm.perm[i]) != viceUbPermLeft.end()) {
            viceUbAxesSize *= shapeInfo_.reducedInShape[viceUbPerm.perm[i]];
        }
    }
}

/**
 * @brief AdjustUbCutAxisFactor 的 axisFlag==0 场景：in/out 切同一条轴
 * @param axisFactor [in,out] 待回调的切分因子
 */
void TransposeGatherTiling::AdjustUbCutFactorSameAxis(int32_t& axisFactor, int64_t dstInUbAxesSize,
                                                      int64_t dstOutUbAxesSize, int64_t srcInUbAxesSize,
                                                      int64_t srcOutUbAxesSize, int64_t elemInTensor,
                                                      int64_t elemPerBlock)
{
    int64_t dstFactor = 0;
    int64_t srcFactor = 0;
    bool isDstUbOverflow = (dstInUbAxesSize * Ops::Base::CeilAlign(dstOutUbAxesSize * axisFactor, elemPerBlock) >
                            elemInTensor);
    bool isSrcUbOverflow = (srcOutUbAxesSize * Ops::Base::CeilAlign(srcInUbAxesSize * axisFactor, elemPerBlock) >
                            elemInTensor);
    if (isDstUbOverflow || isSrcUbOverflow) {
        dstFactor = elemInTensor / dstInUbAxesSize / elemPerBlock * elemPerBlock / dstOutUbAxesSize;
        srcFactor = elemInTensor / srcOutUbAxesSize / elemPerBlock * elemPerBlock / srcInUbAxesSize;
        axisFactor = static_cast<int32_t>(std::min(dstFactor, srcFactor));
    }
}

/**
 * @brief AdjustUbCutAxisFactor 的 axisFlag==1/NUM_THREE 场景：切输出轴
 *        （1 与 3 仅差预调整两行，故传 axisFlag 区分）
 * @param axisFactor [in,out] 待回调的切分因子
 */
void TransposeGatherTiling::AdjustUbCutFactorOutAxis(int32_t& axisFactor, int64_t dstInUbAxesSize,
                                                     int64_t dstOutUbAxesSize, int64_t srcInUbAxesSize,
                                                     int64_t srcOutUbAxesSize, int64_t elemInTensor,
                                                     int64_t elemPerBlock, int8_t axisFlag)
{
    int64_t dstFactor = 0;
    int64_t srcFactor = 0;
    if (axisFlag == NUM_THREE) {
        dstInUbAxesSize *= ubSplitInfo_.inUbCutAxisFactor;
    }
    if (axisFlag == 1) {
        srcOutUbAxesSize /= ubSplitInfo_.inUbCutAxisFactor;
    }
    bool isDstUbOverflow = (dstInUbAxesSize * Ops::Base::CeilAlign(dstOutUbAxesSize * axisFactor, elemPerBlock) >
                            elemInTensor);
    bool isSrcUbOverflow = (axisFactor * srcOutUbAxesSize *
                                Ops::Base::CeilAlign(srcInUbAxesSize * ubSplitInfo_.inUbCutAxisFactor, elemPerBlock) >
                            elemInTensor);
    if (isDstUbOverflow || isSrcUbOverflow) {
        dstFactor = elemInTensor / dstInUbAxesSize / elemPerBlock * elemPerBlock / dstOutUbAxesSize;
        srcFactor = Ops::Base::FloorDiv(
            elemInTensor / srcOutUbAxesSize,
            Ops::Base::CeilAlign(srcInUbAxesSize * ubSplitInfo_.inUbCutAxisFactor, elemPerBlock));
        axisFactor = static_cast<int32_t>(std::min(dstFactor, srcFactor));
    }
}

/**
 * @brief AdjustUbCutAxisFactor 的 axisFlag==NUM_TWO 场景：切输入轴
 * @param axisFactor [in,out] 待回调的切分因子
 */
void TransposeGatherTiling::AdjustUbCutFactorInAxis(int32_t& axisFactor, int64_t dstInUbAxesSize,
                                                    int64_t dstOutUbAxesSize, int64_t srcInUbAxesSize,
                                                    int64_t srcOutUbAxesSize, int64_t elemInTensor,
                                                    int64_t elemPerBlock)
{
    int64_t dstFactor = 0;
    int64_t srcFactor = 0;
    dstInUbAxesSize /= ubSplitInfo_.outUbCutAxisFactor;
    bool isDstUbOverflow = (axisFactor * dstInUbAxesSize *
                                Ops::Base::CeilAlign(dstOutUbAxesSize * ubSplitInfo_.outUbCutAxisFactor, elemPerBlock) >
                            elemInTensor);
    bool isSrcUbOverflow = (srcOutUbAxesSize * Ops::Base::CeilAlign(srcInUbAxesSize * axisFactor, elemPerBlock) >
                            elemInTensor);
    if (isDstUbOverflow || isSrcUbOverflow) {
        dstFactor = Ops::Base::FloorDiv(
            elemInTensor / dstInUbAxesSize,
            Ops::Base::CeilAlign(dstOutUbAxesSize * ubSplitInfo_.outUbCutAxisFactor, elemPerBlock));
        srcFactor = elemInTensor / srcOutUbAxesSize / elemPerBlock * elemPerBlock / srcInUbAxesSize;
        axisFactor = static_cast<int32_t>(std::min(dstFactor, srcFactor));
    }
}

/**
 * @brief 确定 Gather 的 UB 内切分因子（inUbCutAxisFactor / outUbCutAxisFactor）
 *
 * 切分轴 = inUbPerm/outUbPerm 各自最末一根轴（被"切"以适应 UB 容量）。
 * 根据"输入 last 轴 / 输出 last 轴是否还剩余（isLastInPermLeft / isLastOutPermLeft）"
 * 分 4 种 case 求因子：
 *
 *   1) 两 last 都剩余（∈viceAllUbPerm）：转 CalcUbCutFactorBothLeft
 *      - last 轴不同：转 CalcUbCutFactorDiffLastAxis（无重叠取 sqrtedTensor/savedElems；
 *        有重叠按共用轴算 newSqrtedTensor 后回调 flag NUM_THREE）
 *      - last 轴相同：取 min(inUbCutAxisSize, maxCutAxisSize, maxOutCutAxisSize) 并
 *        让 outUbCutAxisFactor = inUbCutAxisFactor（同轴同切，flag 0）
 *   2) 都不剩余：取完整 cutAxisSize（该轴全量进 UB，无需切）
 *   3) 只剩输入 last：转 CalcUbCutFactorInLeftOnly（outCutAxisFactor 取完整，
 *      inCutAxisFactor 取 min(inUbCutAxisSize, maxCutAxisSize) 并回调 flag NUM_TWO）
 *   4) 只剩输出 last：转 CalcUbCutFactorOutLeftOnly（对称处理，回调 flag 1）
 *
 * maxOutCutAxisSize 额外限制：为 gather 索引预留 1/4 UB（NUM_FOUR 分母）。
 *
 * @param elemInTensor    数据 tensor 元素容量上限
 * @param sqrtedTensor    block 边长预算
 * @param isLastInPermLeft  输入 last 轴是否∈viceAllUbPerm（剩余未分配）
 * @param isLastOutPermLeft 输出 last 轴是否∈viceAllUbPerm
 * @param viceAllUbPerm    已分配轴集合（in/out 非切分轴已从 allUbPerm 移除）
 */
void TransposeGatherTiling::CalcUbAxisCutFactor(int64_t elemInTensor, int64_t sqrtedTensor, bool isLastInPermLeft,
                                                bool isLastOutPermLeft, const std::set<int8_t>& viceAllUbPerm)
{
    int64_t allSavedElems = 1;
    for (int8_t idx : allUbPerm_) {
        if (viceAllUbPerm.find(idx) == viceAllUbPerm.end()) {
            allSavedElems *= shapeInfo_.reducedInShape[idx];
        }
    }
    auto dim = shapeInfo_.dim;
    int64_t outSavedElems = CalcShapeSize(shapeInfo_.reducedOutShape, dim - outUbPerm_.cnt + 1, dim);
    int64_t inSavedElems = CalcShapeSize(shapeInfo_.reducedInShape, dim - inUbPerm_.cnt + 1, dim);
    // to save ub for gather index
    int64_t elemPerBlock = platInfo_.ubBlockSize / shapeInfo_.eleLenInBytes;
    int64_t maxOutCutAxisSize = elemInTensor / NUM_FOUR / elemPerBlock * elemPerBlock /
                                Ops::Base::CeilAlign(outSavedElems, elemPerBlock);
    int64_t maxCutAxisSize = elemInTensor / allSavedElems;

    if (isLastInPermLeft && isLastOutPermLeft) {
        CalcUbCutFactorBothLeft(elemInTensor, sqrtedTensor, outSavedElems, inSavedElems, elemPerBlock, maxCutAxisSize,
                                maxOutCutAxisSize);
    } else if (!isLastInPermLeft && !isLastOutPermLeft) {
        ubSplitInfo_.inUbCutAxisFactor = ubSplitInfo_.inUbCutAxisSize;
        ubSplitInfo_.outUbCutAxisFactor = ubSplitInfo_.outUbCutAxisSize;
    } else {
        if (!isLastInPermLeft) {
            CalcUbCutFactorOutLeftOnly(elemInTensor, maxCutAxisSize, maxOutCutAxisSize);
        } else {
            CalcUbCutFactorInLeftOnly(elemInTensor, maxCutAxisSize);
        }
    }
}

/**
 * @brief CalcUbAxisCutFactor 的 isLastInPermLeft && isLastOutPermLeft 场景：两 last 轴都剩余
 *
 * 原函数 both-left 分支逐行搬移：last 轴不同转 CalcUbCutFactorDiffLastAxis，
 * last 轴相同则按 maxCutAxisSize/maxOutCutAxisSize 取因子并同轴同切（flag 0）。
 *
 * @param elemInTensor    数据 tensor 元素容量上限
 * @param sqrtedTensor    block 边长预算
 * @param outSavedElems   输出侧已分配元素数
 * @param inSavedElems    输入侧已分配元素数
 * @param elemPerBlock    每个 block 的元素数
 * @param maxCutAxisSize  最大切分尺寸（受 allSavedElems 约束）
 * @param maxOutCutAxisSize 最大输出切分尺寸（为 gather 索引预留 1/4 UB）
 */
void TransposeGatherTiling::CalcUbCutFactorBothLeft(int64_t elemInTensor, int64_t sqrtedTensor, int64_t outSavedElems,
                                                    int64_t inSavedElems, int64_t elemPerBlock, int64_t maxCutAxisSize,
                                                    int64_t maxOutCutAxisSize)
{
    if (inUbPerm_.perm[inUbPerm_.cnt - 1] != outUbPerm_.perm[outUbPerm_.cnt - 1]) {
        CalcUbCutFactorDiffLastAxis(elemInTensor, sqrtedTensor, outSavedElems, inSavedElems, elemPerBlock);
    } else {
        ubSplitInfo_.inUbCutAxisFactor = std::min(std::min(ubSplitInfo_.inUbCutAxisSize, maxCutAxisSize),
                                                  maxOutCutAxisSize);
        AdjustUbCutAxisFactor(ubSplitInfo_.inUbCutAxisFactor, 0, elemInTensor);
        ubSplitInfo_.outUbCutAxisFactor = ubSplitInfo_.inUbCutAxisFactor;
    }
}

/**
 * @brief CalcUbAxisCutFactor 的 both-left 且 last 轴不同场景
 *
 * 原函数 both-left/diff-last 分支逐行搬移：
 *   - 无重叠（cnt 之和 == ubAxesCnt）：两因子各取 sqrtedTensor / savedElems
 *   - 有重叠：先按共用轴算 newSqrtedTensor 再分别除以非共用轴乘积，回调 flag NUM_THREE
 *
 * @param elemInTensor    数据 tensor 元素容量上限
 * @param sqrtedTensor    block 边长预算
 * @param outSavedElems   输出侧已分配元素数
 * @param inSavedElems    输入侧已分配元素数
 * @param elemPerBlock    每个 block 的元素数
 */
void TransposeGatherTiling::CalcUbCutFactorDiffLastAxis(int64_t elemInTensor, int64_t sqrtedTensor,
                                                        int64_t outSavedElems, int64_t inSavedElems,
                                                        int64_t elemPerBlock)
{
    if (outUbPerm_.cnt + inUbPerm_.cnt == ubSplitInfo_.ubAxesCnt) {
        ubSplitInfo_.inUbCutAxisFactor = std::min(ubSplitInfo_.inUbCutAxisSize, sqrtedTensor / inSavedElems);
        ubSplitInfo_.outUbCutAxisFactor = std::min(ubSplitInfo_.outUbCutAxisSize, sqrtedTensor / outSavedElems);
    } else {
        int64_t comSavedElems = 1;
        for (int8_t idx = 0; idx < outUbPerm_.cnt - 1; ++idx) {
            if (inUbPermSet_.find(outUbPerm_.perm[idx]) != inUbPermSet_.end()) {
                comSavedElems *= shapeInfo_.reducedInShape[outUbPerm_.perm[idx]];
            }
        }
        int64_t newSqrtedTensor = static_cast<int64_t>(
            std::sqrt(elemInTensor / comSavedElems / elemPerBlock * elemPerBlock));
        int64_t inLeft = inSavedElems / comSavedElems;
        int64_t outLeft = outSavedElems / comSavedElems;
        ubSplitInfo_.inUbCutAxisFactor = std::min(ubSplitInfo_.inUbCutAxisSize, newSqrtedTensor / inLeft);
        ubSplitInfo_.outUbCutAxisFactor = std::min(ubSplitInfo_.outUbCutAxisSize, newSqrtedTensor / outLeft);
        AdjustUbCutAxisFactor(ubSplitInfo_.outUbCutAxisFactor, NUM_THREE, elemInTensor);
    }
}

/**
 * @brief CalcUbAxisCutFactor 的只剩输出 last 场景（isLastOutPermLeft && !isLastInPermLeft）
 *
 * 原函数 single-left/out 分支逐行搬移：输入因子取完整尺寸，输出因子受
 * maxCutAxisSize/maxOutCutAxisSize 约束并回调 flag 1。
 *
 * @param elemInTensor     数据 tensor 元素容量上限
 * @param maxCutAxisSize   最大切分尺寸
 * @param maxOutCutAxisSize 最大输出切分尺寸
 */
void TransposeGatherTiling::CalcUbCutFactorOutLeftOnly(int64_t elemInTensor, int64_t maxCutAxisSize,
                                                       int64_t maxOutCutAxisSize)
{
    ubSplitInfo_.inUbCutAxisFactor = ubSplitInfo_.inUbCutAxisSize;
    ubSplitInfo_.outUbCutAxisFactor = std::min(std::min(ubSplitInfo_.outUbCutAxisSize, maxCutAxisSize),
                                               maxOutCutAxisSize);
    AdjustUbCutAxisFactor(ubSplitInfo_.outUbCutAxisFactor, 1, elemInTensor);
}

/**
 * @brief CalcUbAxisCutFactor 的只剩输入 last 场景（isLastInPermLeft && !isLastOutPermLeft）
 *
 * 原函数 single-left/in 分支逐行搬移：输出因子取完整尺寸，输入因子受
 * maxCutAxisSize 约束并回调 flag NUM_TWO。
 *
 * @param elemInTensor   数据 tensor 元素容量上限
 * @param maxCutAxisSize 最大切分尺寸
 */
void TransposeGatherTiling::CalcUbCutFactorInLeftOnly(int64_t elemInTensor, int64_t maxCutAxisSize)
{
    ubSplitInfo_.outUbCutAxisFactor = ubSplitInfo_.outUbCutAxisSize;
    ubSplitInfo_.inUbCutAxisFactor = std::min(ubSplitInfo_.inUbCutAxisSize, maxCutAxisSize);
    AdjustUbCutAxisFactor(ubSplitInfo_.inUbCutAxisFactor, NUM_TWO, elemInTensor);
}

/**
 * @brief 汇总 UB 内轴尺寸（inUbAxes / outUbAxes / ubPerm）并做硬件约束校验
 *
 * tmpInAxes/tmpOutAxes/tmpOutPerm 是"十字布局"的中间表示：
 *   对每个输出位置 j：outAxes[j] 记录该输出轴的尺寸，inAxes[perm[j]] 记录
 *   对应输入轴尺寸，ubPerm[j] 记录输入轴在 allUbPerm 中的序号。
 * 本函数把它们压实为 kernel 使用的紧凑数组（去掉 0 项），并计算：
 *   - totalSizeInUb = eleBytes × Π inUbAxes：UT 单次搬运容量
 *   - indexStep：gather 索引的步进
 *
 * 硬件约束（不满足则返回失败回退 NDDMA）：
 *   - totalSizeInUb ≥ MTE_GATE(0x8000)：MTE 搬运效率门槛
 *   - CheckBC(indexStep) 为 false：避免 gather 索引在 UB bank 上的冲突
 *
 * 同时把输入侧切分轴位置 inUbInCutPos/inUbOutCutPos 映射到输出侧
 * （outUbInCutPos / outUbOutCutPos），供 kernel GetOutLoopAxes 使用。
 *
 * @return GRAPH_SUCCESS 校验通过；GRAPH_FAILED MTE 效率不足或 bank 冲突
 */
ge::graphStatus TransposeGatherTiling::CalcUbAxesInfo(const int64_t (&tmpInAxes)[MAX_TRANS_AXIS_NUM],
                                                      const int64_t (&tmpOutAxes)[MAX_TRANS_AXIS_NUM],
                                                      const int8_t (&tmpOutPerm)[MAX_TRANS_AXIS_NUM])
{
    CompactUbAxes(tmpInAxes, tmpOutAxes, tmpOutPerm);
    if (CheckUbHwConstraint() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    MapUbCutPosToOut();
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief CalcUbAxesInfo 的压实段：把 tmpInAxes/tmpOutAxes/tmpOutPerm 压实为 UB 轴信息
 *
 * 原函数首个 for 循环逐行搬移：遍历 MAX_TRANS_AXIS_NUM，非零 tmpOutAxes[j] 填入
 * outUbAxes[outIdx] 并按 allUbPerm 序号填 ubPerm[outIdx]；非零 tmpInAxes[j] 填入
 * inUbAxes[inIdx]。inIdx/outIdx 为本段局部变量（原函数对应分支逐行同名同形）。
 */
void TransposeGatherTiling::CompactUbAxes(const int64_t (&tmpInAxes)[MAX_TRANS_AXIS_NUM],
                                          const int64_t (&tmpOutAxes)[MAX_TRANS_AXIS_NUM],
                                          const int8_t (&tmpOutPerm)[MAX_TRANS_AXIS_NUM])
{
    int8_t inIdx = 0;
    int8_t outIdx = 0;
    for (int8_t j = 0; j < MAX_TRANS_AXIS_NUM; ++j) {
        if (tmpOutAxes[j] != 0) {
            ubSplitInfo_.outUbAxes[outIdx] = static_cast<int32_t>(tmpOutAxes[j]);
            // do like [5,2,3,0] -> [3,1,2,0]
            ubSplitInfo_.ubPerm[outIdx] = static_cast<int8_t>(
                std::distance(allUbPerm_.begin(), allUbPerm_.find(tmpOutPerm[j])));
            ++outIdx;
        }
        if (tmpInAxes[j] != 0) {
            ubSplitInfo_.inUbAxes[inIdx++] = static_cast<int32_t>(tmpInAxes[j]);
        }
    }
}

/**
 * @brief CalcUbAxesInfo 的硬件校验段：MTE 效率门槛 + gather 索引 bank 冲突
 *
 * 原函数中段逐行搬移：totalSizeInUb = eleLenInBytes × inUbAxes 乘积；
 * indexStep = ubPerm 末轴右侧 inUbAxes 乘积。totalSizeInUb < MTE_GATE 或
 * CheckBC(indexStep) 为真则返回 GRAPH_FAILED。totalSizeInUb/indexStep 为本段局部变量。
 *
 * @return GRAPH_SUCCESS 校验通过；GRAPH_FAILED MTE 效率不足或 bank 冲突
 */
ge::graphStatus TransposeGatherTiling::CheckUbHwConstraint()
{
    int32_t totalSizeInUb = static_cast<int32_t>(shapeInfo_.eleLenInBytes);
    for (int8_t i = 0; i < ubSplitInfo_.ubAxesCnt; ++i) {
        totalSizeInUb *= ubSplitInfo_.inUbAxes[i];
    }
    int32_t indexStep = 1;
    for (int8_t i = ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] + 1; i < ubSplitInfo_.ubAxesCnt; ++i) {
        indexStep *= ubSplitInfo_.inUbAxes[i];
    }
    // MTE size must be >= gate
    if (totalSizeInUb < MTE_GATE) {
        OP_LOGD(context_, "total size too small, totalSizeInUb=%ld", totalSizeInUb);
        return ge::GRAPH_FAILED;
    }
    if (CheckBC(indexStep)) {
        OP_LOGD(context_, "may bank conflict, indexStep=%ld", indexStep);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief CalcUbAxesInfo 的映射段：输入侧切分轴位置映射到输出侧
 *
 * 原函数末段 for 循环逐行搬移：扫描 ubPerm[0..allUbPerm.size()-1]，
 * ubPerm[k]==inUbInCutPos 记 outUbInCutPos=k，==inUbOutCutPos 记 outUbOutCutPos=k。
 */
void TransposeGatherTiling::MapUbCutPosToOut()
{
    for (int8_t k = 0; k < static_cast<int8_t>(allUbPerm_.size()); ++k) {
        if (ubSplitInfo_.ubPerm[k] == ubSplitInfo_.inUbInCutPos) {
            ubSplitInfo_.outUbInCutPos = k;
        }
        if (ubSplitInfo_.ubPerm[k] == ubSplitInfo_.inUbOutCutPos) {
            ubSplitInfo_.outUbOutCutPos = k;
        }
    }
}

/**
 * @brief 构造 Gather 的 UB 轴信息（CalcUbSplitInfo4Gather）
 *
 * 把已选出的 inUbPerm/outUbPerm 轴整理为"十字布局"中间数组 tmpInAxes/
 * tmpOutAxes/tmpOutPerm，并调用 CalcUbAxisCutFactor 确定切分因子。
 *
 * 具体流程：
 *   1. ubAxesCnt = allUbPerm 大小（UB 内轴总数）
 *   2. inUbCutAxisSize = inUbPerm 最末轴长度（输入切分轴完整尺寸），
 *      inUbInCutPos = 该轴在 allUbPerm 中的序号
 *   3. 输入非切分轴：填入其在输出 perm 位置 idx 的 tmpOutPerm/tmpOutAxes，
 *      以及 tmpInAxes[perm[i]]（十字布局的第一条边）
 *   4. 输出非切分轴（且未在输入中出现）：填入对应位置（十字布局第二条边）
 *   5. 计算 isLastInPermLeft/isLastOutPermLeft（两 last 轴是否仍未分配）
 *   6. CalcUbAxisCutFactor 确定 inUbCutAxisFactor / outUbCutAxisFactor
 *   7. 若 last 轴剩余：把切分因子填入 tmpOutAxes[idx]（尺寸 = 因子）
 *   8. CalcUbAxesInfo 压实 + 硬件校验
 *
 * @param elemInTensor 数据 tensor 元素容量上限
 * @param sqrtedTensor block 边长预算
 * @return GRAPH_SUCCESS 成功；GRAPH_FAILED 校验失败回退 NDDMA
 */
ge::graphStatus TransposeGatherTiling::CalcUbSplitInfo4Gather(int64_t elemInTensor, int64_t sqrtedTensor)
{
    int8_t tmpOutPerm[MAX_TRANS_AXIS_NUM] = {0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf};
    int64_t tmpInAxes[MAX_TRANS_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t tmpOutAxes[MAX_TRANS_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::set<int8_t> viceAllUbPerm(allUbPerm_);

    ubSplitInfo_.ubAxesCnt = static_cast<int8_t>(allUbPerm_.size());
    ubSplitInfo_.inUbCutAxisSize = shapeInfo_.reducedInShape[inUbPerm_.perm[inUbPerm_.cnt - 1]];
    ubSplitInfo_.inUbInCutPos = static_cast<int8_t>(
        std::distance(allUbPerm_.begin(), allUbPerm_.find(inUbPerm_.perm[inUbPerm_.cnt - 1])));
    // all axes can be move in ub except last
    for (int8_t i = 0; i < inUbPerm_.cnt - 1; ++i) {
        auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(), inUbPerm_.perm[i]);
        auto idx = std::distance(shapeInfo_.reducedPerm.begin(), iter);
        tmpOutPerm[idx] = inUbPerm_.perm[i];
        tmpOutAxes[idx] = shapeInfo_.reducedInShape[inUbPerm_.perm[i]];
        tmpInAxes[inUbPerm_.perm[i]] = shapeInfo_.reducedInShape[inUbPerm_.perm[i]];
        viceAllUbPerm.erase(inUbPerm_.perm[i]);
    }
    ubSplitInfo_.outUbCutAxisSize = shapeInfo_.reducedInShape[outUbPerm_.perm[outUbPerm_.cnt - 1]];
    ubSplitInfo_.inUbOutCutPos = static_cast<int8_t>(
        std::distance(allUbPerm_.begin(), allUbPerm_.find(outUbPerm_.perm[outUbPerm_.cnt - 1])));
    for (int8_t i = 0; i < outUbPerm_.cnt - 1; ++i) {
        if (viceAllUbPerm.find(outUbPerm_.perm[i]) != viceAllUbPerm.end()) {
            auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(), outUbPerm_.perm[i]);
            auto idx = std::distance(shapeInfo_.reducedPerm.begin(), iter);
            // pattern like: [6, 5, 4, 0xf, 0, 2, 1, 0xf]
            tmpOutPerm[idx] = outUbPerm_.perm[i];
            // pattern like: [10, 20, 50, 0, 5, 7, 2, 0]
            tmpOutAxes[idx] = shapeInfo_.reducedInShape[outUbPerm_.perm[i]];
            // pattern like: [0, 10, 50, 20, 5, 0, 7, 2]
            tmpInAxes[outUbPerm_.perm[i]] = shapeInfo_.reducedInShape[outUbPerm_.perm[i]];
            viceAllUbPerm.erase(outUbPerm_.perm[i]);
        }
    }

    bool isLastInPermLeft = viceAllUbPerm.find(inUbPerm_.perm[inUbPerm_.cnt - 1]) != viceAllUbPerm.end();
    bool isLastOutPermLeft = viceAllUbPerm.find(outUbPerm_.perm[outUbPerm_.cnt - 1]) != viceAllUbPerm.end();
    CalcUbAxisCutFactor(elemInTensor, sqrtedTensor, isLastInPermLeft, isLastOutPermLeft, viceAllUbPerm);

    if (isLastInPermLeft) {
        auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(),
                              inUbPerm_.perm[inUbPerm_.cnt - 1]);
        auto idx = std::distance(shapeInfo_.reducedPerm.begin(), iter);
        tmpOutPerm[idx] = inUbPerm_.perm[inUbPerm_.cnt - 1];
        tmpOutAxes[idx] = ubSplitInfo_.inUbCutAxisFactor;
        tmpInAxes[inUbPerm_.perm[inUbPerm_.cnt - 1]] = ubSplitInfo_.inUbCutAxisFactor;
        viceAllUbPerm.erase(inUbPerm_.perm[inUbPerm_.cnt - 1]);
    }
    if (isLastOutPermLeft) {
        auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(),
                              outUbPerm_.perm[outUbPerm_.cnt - 1]);
        auto idx = std::distance(shapeInfo_.reducedPerm.begin(), iter);
        tmpOutPerm[idx] = outUbPerm_.perm[outUbPerm_.cnt - 1];
        tmpOutAxes[idx] = ubSplitInfo_.outUbCutAxisFactor;
        tmpInAxes[outUbPerm_.perm[outUbPerm_.cnt - 1]] = ubSplitInfo_.outUbCutAxisFactor;
    }

    return CalcUbAxesInfo(tmpInAxes, tmpOutAxes, tmpOutPerm);
}

/**
 * @brief 计算 Gather 搬入/搬出（MTE）的各循环轴跨步
 *
 * axis0/1/2InSrcStride：从 GM 搬入 UB 时，UB 内各循环轴对应的源跨步
 *   - 源跨步 = 输入视角下该轴右侧（更高维）所有轴的乘积
 *   - 优先保证"输出最末维连续搬入"（outUbPerm_[0] 的轴放在 axis0 首循环）
 * axis0/1/2OutDstStride：从 UB 搬出到 GM 时，各循环轴的目标跨步
 *   - 目标跨步 = 输出视角下该轴右侧乘积
 *
 * kernel 端 SetCopyInParams/SetCopyOutParams 用这些 stride 构建
 * DataCopyPad 的 blockCount/blockLen/stride 与 LoopMode 参数。
 */
void TransposeGatherTiling::CalcUbSplitInfo4MTE()
{
    auto dim = shapeInfo_.dim;
    int8_t axisIdx = 0;

    if (outUbPerm_.perm[0] < inUbPerm_.perm[inUbPerm_.cnt - 1]) {
        CalcInSrcStrideConsecutive(dim, axisIdx);
    } else {
        CalcInSrcStrideDefault(dim, axisIdx);
    }

    axisIdx = 0;
    CalcOutDstStride(dim, axisIdx);
}

/**
 * @brief CalcUbSplitInfo4MTE 的 if 分支：输出末维 < 输入切轴（连续搬入）
 *
 * 原函数 if 分支逐行搬移：预置 axis0InSrcStride 为输出末维右侧乘积，
 * 再从输入切轴向前扫描 UB 内轴（跳过输出末维），填 axis1/axis2InSrcStride。
 *
 * @param dim     简化后维度数
 * @param axisIdx [in,out] 循环内轴序号计数器
 */
void TransposeGatherTiling::CalcInSrcStrideConsecutive(int64_t dim, int8_t& axisIdx)
{
    // to make sure output last dim and move in cube are consecutive
    ubSplitInfo_.axis0InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, outUbPerm_.perm[0] + 1, dim);
    for (int8_t i = inUbPerm_.perm[inUbPerm_.cnt - 1] - 1; i >= 0; --i) {
        if (allUbPerm_.find(i) != allUbPerm_.end() && i != outUbPerm_.perm[0] && axisIdx == 0) {
            ubSplitInfo_.axis1InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim);
            ++axisIdx;
        } else if (allUbPerm_.find(i) != allUbPerm_.end() && i != outUbPerm_.perm[0] && axisIdx == 1) {
            ubSplitInfo_.axis2InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim);
        }
    }
}

/**
 * @brief CalcUbSplitInfo4MTE 的 else 分支：默认顺序搬入
 *
 * 原函数 else 分支逐行搬移：从输入切轴向前扫描 UB 内轴，填 axis0/1InSrcStride。
 *
 * @param dim     简化后维度数
 * @param axisIdx [in,out] 循环内轴序号计数器
 */
void TransposeGatherTiling::CalcInSrcStrideDefault(int64_t dim, int8_t& axisIdx)
{
    for (int8_t i = inUbPerm_.perm[inUbPerm_.cnt - 1] - 1; i >= 0; --i) {
        if (allUbPerm_.find(i) != allUbPerm_.end() && axisIdx == 0) {
            ubSplitInfo_.axis0InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim);
            ++axisIdx;
        } else if (allUbPerm_.find(i) != allUbPerm_.end() && axisIdx == 1) {
            ubSplitInfo_.axis1InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim);
        }
    }
}

/**
 * @brief CalcUbSplitInfo4MTE 的搬出跨步段：计算 OutDstStride
 *
 * 原函数 OutDstStride 循环逐行搬移：从输出侧向前扫描 UB 内轴，
 * 填 axis0/1/2OutDstStride。
 *
 * @param dim     简化后维度数
 * @param axisIdx [in,out] 循环内轴序号计数器
 */
void TransposeGatherTiling::CalcOutDstStride(int64_t dim, int8_t& axisIdx)
{
    for (int8_t j = dim - outUbPerm_.cnt - 1; j >= 0; --j) {
        if (allUbPerm_.find(shapeInfo_.reducedPerm[j]) != allUbPerm_.end() && axisIdx == 0) {
            ubSplitInfo_.axis0OutDstStride = CalcShapeSize(shapeInfo_.reducedOutShape, j + 1, dim);
            ++axisIdx;
        } else if (allUbPerm_.find(shapeInfo_.reducedPerm[j]) != allUbPerm_.end() && axisIdx == 1) {
            ubSplitInfo_.axis1OutDstStride = CalcShapeSize(shapeInfo_.reducedOutShape, j + 1, dim);
            ++axisIdx;
        } else if (allUbPerm_.find(shapeInfo_.reducedPerm[j]) != allUbPerm_.end() && axisIdx > 1) {
            ubSplitInfo_.axis2OutDstStride = CalcShapeSize(shapeInfo_.reducedOutShape, j + 1, dim);
        }
    }
}

/**
 * @brief 借轴调整：让"借入的轴"搬入 UB 时位于 axis0（输出末维连续）
 *
 * 当输出末维（outUbPerm 最后一轴的输入序号）在 UB 内的位置不是 axis0，
 * 且中间有空位（axis0Gap = inUbInCutPos - 1 - outLastDimInPos > 0）时，
 * 把 inUbAxes 中元素向左移位，把输出末维那根轴搬到 axis0 位置，
 * 保证搬入 GM→UB 时输出末维连续（搬出时无需跨步）。
 *
 * 同时修正 ubPerm 顺序，使 gather 索引生成阶段（GenIndex4OneDim/TwoDim/ThreeDim）
 * 按新的轴顺序枚举输出维度，并适配 outUbPerm.cnt==2/3 时的轴排列。
 */
void TransposeGatherTiling::AdjustInUbAxesPosition()
{
    int8_t outLastDimInPos = ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1];
    int8_t axis0Gap = ubSplitInfo_.inUbInCutPos - 1 - outLastDimInPos;
    // only for brorrow axis case, to make output last dim to be the axis0 when move data in
    if (axis0Gap > 0) {
        ShiftInUbAxes4Borrow(outLastDimInPos, axis0Gap);

        if (outUbPerm_.cnt == NUM_TWO) {
            AdjustUbPerm4BorrowCntTwo();
        } else if (outUbPerm_.cnt == NUM_THREE) {
            AdjustUbPerm4BorrowCntThree();
        }
    }
}

/**
 * @brief AdjustInUbAxesPosition 的 axis0Gap>0 共享段：借轴位移 inUbAxes
 *
 * 原函数 shift 段逐行搬移：把 inUbAxes 中元素向左移位 axis0Gap 格，
 * 把输出末维那根轴（outUbAxes 末项）搬到 axis0 位置，并修正 inUbOutCutPos。
 *
 * @param outLastDimInPos 输出末维在 UB 轴序列中的位置
 * @param axis0Gap        需左移的格数（inUbInCutPos - 1 - outLastDimInPos）
 */
void TransposeGatherTiling::ShiftInUbAxes4Borrow(int8_t outLastDimInPos, int8_t axis0Gap)
{
    for (int8_t i = 0; i < axis0Gap; ++i) {
        ubSplitInfo_.inUbAxes[outLastDimInPos + i] = ubSplitInfo_.inUbAxes[outLastDimInPos + i + 1];
    }
    ubSplitInfo_.inUbAxes[outLastDimInPos + axis0Gap] = ubSplitInfo_.outUbAxes[ubSplitInfo_.ubAxesCnt - 1];
    if (outLastDimInPos < ubSplitInfo_.inUbOutCutPos && ubSplitInfo_.inUbOutCutPos < ubSplitInfo_.inUbInCutPos) {
        ubSplitInfo_.inUbOutCutPos -= 1;
    }
}

/**
 * @brief AdjustInUbAxesPosition 的 outUbPerm_.cnt==NUM_TWO 场景：ubPerm 重排
 *
 * 原函数 cnt==2 分支体逐行搬移：末两位置为 [0, 1]。
 */
void TransposeGatherTiling::AdjustUbPerm4BorrowCntTwo()
{
    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] = 0;
    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = 1;
}

/**
 * @brief AdjustInUbAxesPosition 的 outUbPerm_.cnt==NUM_THREE 场景：ubPerm 重排分发
 *
 * 原函数 cnt==3 分支逐行搬移：按 in/out 轴是否无重叠（cnt 之和 == ubAxesCnt）
 * 分发到 NoOverlap / Overlap 两个子函数。
 */
void TransposeGatherTiling::AdjustUbPerm4BorrowCntThree()
{
    /*           no overlap: 2 1 0 -> 1 0 2
     *                       2 0 1 -> 1 0 2
     *                       0 2 1 -> 0 1 2
     *                       1 2 0 -> 0 1 2
     *              overlap: x 1 0 -> x 0 1
     *                       1 x 0 -> 0 x 1
     */
    if (inUbPerm_.cnt + outUbPerm_.cnt == ubSplitInfo_.ubAxesCnt) {
        AdjustUbPermCntThreeNoOverlap();
    } else {
        AdjustUbPermCntThreeOverlap();
    }
}

/**
 * @brief cnt==3 无重叠场景（inUbPerm_.cnt + outUbPerm_.cnt == ubAxesCnt）
 *
 * 原函数 cnt==3 / no-overlap 分支体逐行搬移：末三轴 perm 中含 2 的位置决定重排。
 */
void TransposeGatherTiling::AdjustUbPermCntThreeNoOverlap()
{
    if (ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] == NUM_TWO) {
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] = 1;
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] = 0;
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = NUM_TWO;
    } else if (ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] == NUM_TWO) {
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] = 0;
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] = 1;
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = NUM_TWO;
    }
}

/**
 * @brief cnt==3 有重叠场景（inUbPerm_.cnt + outUbPerm_.cnt != ubAxesCnt）
 *
 * 原函数 cnt==3 / overlap 分支体逐行搬移：末三轴 perm 中 >= 2 的位置决定重排。
 */
void TransposeGatherTiling::AdjustUbPermCntThreeOverlap()
{
    if (ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] >= NUM_TWO) {
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] = 0;
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = 1;
    } else if (ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] >= NUM_TWO) {
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] = 0;
        ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = 1;
    }
}

/**
 * @brief 检查 gather 索引的 UB Bank Conflict
 *
 * 索引步进（steps，元素数）× eleLenInBytes 得到字节跨步，按 8B 子 bank 对齐后，
 * 若在 128B bank 内所占的子 bank 序号为偶数（MOD 8 后 /8%2==0），说明 gather
 * 读取时多个并行访问落在同一子 bank → 硬件冲突，返回 true。
 *
 * @param steps 索引步进（元素数）
 * @return true 存在 bank 冲突（应回退 NDDMA）；false 安全
 */
bool TransposeGatherTiling::CheckBC(int32_t steps)
{
    int32_t bytesPerSubBank = 8;
    int32_t bytesPerBank = 128;
    int32_t stepBytes = steps * shapeInfo_.eleLenInBytes;
    int32_t stepBytesAlign = Ops::Base::CeilAlign(stepBytes, bytesPerSubBank);
    return (stepBytesAlign % bytesPerBank / bytesPerSubBank % NUM_TWO == 0);
}

/**
 * @brief 计算 Gather block 的边长预算（sqrt 后按 block/cacheline 对齐）
 *
 * sqrtedTensor = sqrt(elemInTensor)，先向 elemPerBlock 向下对齐，
 * 再限制到 cacheLineSize 以内（除以 cacheLine 后取整）。
 * 若输入最末轴比预算大且预算会引起 bank 冲突，则再减去 8B 对应元素数。
 *
 * @param elemInTensor 数据 tensor 元素容量
 * @return block 边长预算（元素数）
 */
int64_t TransposeGatherTiling::CalcSqrtedTensor(int64_t elemInTensor)
{
    int64_t elemPerBlock = platInfo_.ubBlockSize / shapeInfo_.eleLenInBytes;
    int64_t sqrtedTensor = static_cast<int64_t>(std::sqrt(elemInTensor)) / elemPerBlock * elemPerBlock;
    if (sqrtedTensor * shapeInfo_.eleLenInBytes > platInfo_.cacheLineSize) {
        sqrtedTensor = (sqrtedTensor * shapeInfo_.eleLenInBytes / platInfo_.cacheLineSize * platInfo_.cacheLineSize /
                        shapeInfo_.eleLenInBytes);
    }
    int32_t bytesPerSubBank = 8;
    int64_t lastInDim = shapeInfo_.reducedInShape[shapeInfo_.dim - 1];
    if (lastInDim > sqrtedTensor && CheckBC(static_cast<int32_t>(sqrtedTensor))) {
        sqrtedTensor -= (bytesPerSubBank / shapeInfo_.eleLenInBytes);
    }
    return sqrtedTensor;
}

/**
 * @brief Gather 切轴逻辑入口（组装 UB 内轴 + 切分因子 + MTE 跨步）
 *
 * 整体流程：
 *   1. elemInTensor = dataTensorSize / eleBytes（UB 数据预算的元素容量）
 *   2. sqrtedTensor = CalcSqrtedTensor(elemInTensor)（block 边长预算）
 *   3. CalcInUbPerm/CalcOutUbPerm：选择进入 UB 的输入/输出轴（≤3 根）
 *   4. CalcUbSplitInfo4Gather：十字布局 + 切分因子 + 硬件校验
 *   5. CalcUbSplitInfo4MTE：搬入/搬出跨步
 *   6. AdjustInUbAxesPosition：借轴调整（保证输出末维连续搬入）
 *
 * MTE/BC 校验失败（CalcUbSplitInfo4Gather 返回 FAIL）时返回失败，
 * 上层 DoTiling 会回退 NDDMA 主流程。
 *
 * @return GRAPH_SUCCESS 切轴完成；GRAPH_FAILED MTE 效率不足或 bank 冲突
 */
ge::graphStatus TransposeGatherTiling::CalcUbSplitInfo()
{
    int64_t elemInTensor = static_cast<int64_t>(dataTensorSize_ / shapeInfo_.eleLenInBytes);
    int64_t sqrtedTensor = CalcSqrtedTensor(elemInTensor);
    CalcInUbPerm(sqrtedTensor);
    CalcOutUbPerm(sqrtedTensor);
    OP_CHECK_IF(CalcUbSplitInfo4Gather(elemInTensor, sqrtedTensor) != ge::GRAPH_SUCCESS,
                OP_LOGD(context_->GetNodeName(), "MTE size is too small!"), return ge::GRAPH_FAILED);
    CalcUbSplitInfo4MTE();
    AdjustInUbAxesPosition();
    OP_LOGD(context_->GetNodeName(), "UB tiling is done!");
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief Gather 分核逻辑（块级循环轴 + 核数/负载均衡）
 *
 * 对 allUbPerm 之外（未进入单次 block）的轴，以及切分后仍剩余循环的轴，
 * 每个轴生成一个块级循环轴 blkAxes：
 *   - blkAxes[cnt]   = CeilDiv(shape[i], axisFactor)：该轴需循环的次数
 *   - blkAxesInAOffset[cnt]  = Π reducedInShape[j] (j>i) × axisFactor：输入地址跨步
 *   - blkAxesOutAOffset[cnt] = Π reducedOutShape[j] (j>perm逆位置) × axisFactor：输出地址跨步
 *   - 输入/输出切分轴若有尾块（size % factor != 0），记 blkInUbCutPos / blkOutUbCutPos
 *     （kernel UpdateUbAxes 据此切换 main/tail）
 *
 * 核数：totalElems = Π blkAxes
 *   usedCoreCnt = CeilDiv(totalElems, CeilDiv(totalElems, coreNum))
 *   若 usedCoreCnt < coreNum/2 → 块太少不划算，返回失败回退 NDDMA
 *   blkFactor = CeilDiv(totalElems, usedCoreCnt)
 *   blkTailFactor = totalElems - (usedCoreCnt-1) × blkFactor（尾核吃尾块）
 *
 * kernel GetCoreLoopRange：前 usedCoreCnt-1 个核各 blkFactor 块，最后核吃尾块。
 *
 * @return GRAPH_SUCCESS 分核完成；GRAPH_FAILED 块数太少回退 NDDMA
 */
ge::graphStatus TransposeGatherTiling::CalcBlockSplitInfo()
{
    int64_t totalElems = 1;
    CollectBlkAxes(totalElems);
    blkSplitInfo_.usedCoreCnt = Ops::Base::CeilDiv(totalElems, Ops::Base::CeilDiv(totalElems, platInfo_.coreNum));
    if (blkSplitInfo_.usedCoreCnt < static_cast<uint32_t>(platInfo_.coreNum / NUM_TWO)) {
        return ge::GRAPH_FAILED;
    }
    blkSplitInfo_.blkFactor = Ops::Base::CeilDiv(totalElems, static_cast<int64_t>(blkSplitInfo_.usedCoreCnt));
    blkSplitInfo_.blkTailFactor = totalElems - (blkSplitInfo_.usedCoreCnt - 1) * blkSplitInfo_.blkFactor;
    OP_LOGD(context_->GetNodeName(), "Block tiling is done!");
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 逐轴收集块级循环轴（CollectBlkAxes，从 CalcBlockSplitInfo 循环体提取）
 *
 * 循环体与原函数逐行同名：4 分支 dispatch 决定 axisFactor / 是否跳过；
 * in/out 切尾段对称参数化为 CalcBlkUbCutAxis（调用两次），共享偏移计算落入 CalcBlkAxesOffset。
 *
 * @param totalElems [in,out] 块级循环总块数乘积（初值 1，与原函数局部变量同名同序）
 */
void TransposeGatherTiling::CollectBlkAxes(int64_t& totalElems)
{
    int8_t dim = shapeInfo_.dim;
    int64_t axisFactor = 1;
    for (int8_t i = 0; i < dim; ++i) {
        if (allUbPerm_.find(i) == allUbPerm_.end()) {
            axisFactor = 1;
        } else if (i == inUbPerm_.perm[inUbPerm_.cnt - 1] &&
                   ubSplitInfo_.inUbCutAxisSize != ubSplitInfo_.inUbCutAxisFactor) {
            CalcBlkUbCutAxis(axisFactor, ubSplitInfo_.inUbCutAxisSize, ubSplitInfo_.inUbCutAxisFactor,
                             blkSplitInfo_.blkInUbCutPos, blkSplitInfo_.blkAxesCnt);
        } else if (i == outUbPerm_.perm[outUbPerm_.cnt - 1] &&
                   ubSplitInfo_.outUbCutAxisSize != ubSplitInfo_.outUbCutAxisFactor) {
            CalcBlkUbCutAxis(axisFactor, ubSplitInfo_.outUbCutAxisSize, ubSplitInfo_.outUbCutAxisFactor,
                             blkSplitInfo_.blkOutUbCutPos, blkSplitInfo_.blkAxesCnt);
        } else {
            continue;
        }
        CalcBlkAxesOffset(i, dim, axisFactor, totalElems);
    }
}

/**
 * @brief 设置块级切分轴因子并记录尾块位置（in/out 镜像段参数化，调用两次）
 *
 * 与原函数 in/out 分支逐行同名：
 *   axisFactor = cutAxisFactor;
 *   if (cutAxisSize % cutAxisFactor != 0) cutPos = blkAxesCnt;
 * 类型提升保留：cutAxisSize 为 int64_t、cutAxisFactor 为 int32_t（与 ubSplitInfo_ 字段同型）。
 *
 * @param axisFactor     [in,out] 块级循环轴因子（置为切分轴主块因子）
 * @param cutAxisSize    切分轴完整尺寸（in/outUbCutAxisSize）
 * @param cutAxisFactor  切分轴主块因子（in/outUbCutAxisFactor）
 * @param cutPos         [in,out] 块级切尾位置（blkIn/OutUbCutPos）
 * @param blkAxesCnt     当前块级轴计数（blkAxesCnt）
 */
void TransposeGatherTiling::CalcBlkUbCutAxis(int64_t& axisFactor, int64_t cutAxisSize, int32_t cutAxisFactor,
                                             int8_t& cutPos, int8_t blkAxesCnt)
{
    axisFactor = cutAxisFactor;
    if (cutAxisSize % cutAxisFactor != 0) {
        cutPos = blkAxesCnt;
    }
}

/**
 * @brief 计算单轴块级循环次数与输入/输出地址跨步（从 CalcBlockSplitInfo 偏移尾段提取）
 *
 * 与原函数 if-else-if 之后的共享尾段逐行同名：axisLpSize、blkAxes/InAOffset/OutAOffset、
 * ++blkAxesCnt、totalElems 累乘，算式（CeilDiv 参数、CalcShapeSize 区间、std::find 距离）逐字保留。
 *
 * @param i          当前轴索引
 * @param dim        简化后维度数（原函数局部变量 dim）
 * @param axisFactor 当前轴块级循环因子
 * @param totalElems [in,out] 块级循环总块数乘积
 */
void TransposeGatherTiling::CalcBlkAxesOffset(int8_t i, int8_t dim, int64_t axisFactor, int64_t& totalElems)
{
    int64_t axisLpSize = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[i], axisFactor);
    blkSplitInfo_.blkAxes[blkSplitInfo_.blkAxesCnt] = axisLpSize;
    blkSplitInfo_.blkAxesInAOffset[blkSplitInfo_.blkAxesCnt] = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim) *
                                                               axisFactor;
    auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(), i);
    int8_t gap = static_cast<int8_t>(std::distance(shapeInfo_.reducedPerm.begin(), iter));
    blkSplitInfo_.blkAxesOutAOffset[blkSplitInfo_.blkAxesCnt] = CalcShapeSize(shapeInfo_.reducedOutShape, gap + 1,
                                                                              dim) *
                                                                axisFactor;
    ++blkSplitInfo_.blkAxesCnt;
    totalElems *= axisLpSize;
}

ge::graphStatus TransposeGatherTiling::SetTilingKeyAndCore()
{
    OP_CHECK_IF(context_->SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set tiling key failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->SetBlockDim(blkSplitInfo_.usedCoreCnt) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set used core size failed!"), return ge::GRAPH_FAILED);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

void TransposeGatherTiling::WriteTilingData()
{
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_dataTensorSize(dataTensorSize_);
    tilingData_.set_indexTensorSize(indexTensorSize_);
    tilingData_.set_usedCoreCnt(blkSplitInfo_.usedCoreCnt);
    tilingData_.set_blkAxesCnt(blkSplitInfo_.blkAxesCnt);
    tilingData_.set_blkInUbCutPos(blkSplitInfo_.blkInUbCutPos);
    tilingData_.set_blkOutUbCutPos(blkSplitInfo_.blkOutUbCutPos);
    tilingData_.set_ubAxesCnt(ubSplitInfo_.ubAxesCnt);
    tilingData_.set_inUbInCutPos(ubSplitInfo_.inUbInCutPos);
    tilingData_.set_inUbOutCutPos(ubSplitInfo_.inUbOutCutPos);
    tilingData_.set_outUbInCutPos(ubSplitInfo_.outUbInCutPos);
    tilingData_.set_outUbOutCutPos(ubSplitInfo_.outUbOutCutPos);
    tilingData_.set_blkFactor(blkSplitInfo_.blkFactor);
    tilingData_.set_blkTailFactor(blkSplitInfo_.blkTailFactor);
    tilingData_.set_inUbCutAxisSize(ubSplitInfo_.inUbCutAxisSize);
    tilingData_.set_outUbCutAxisSize(ubSplitInfo_.outUbCutAxisSize);
    tilingData_.set_inUbCutAxisFactor(ubSplitInfo_.inUbCutAxisFactor);
    tilingData_.set_outUbCutAxisFactor(ubSplitInfo_.outUbCutAxisFactor);
    tilingData_.set_axis0InSrcStride(ubSplitInfo_.axis0InSrcStride);
    tilingData_.set_axis1InSrcStride(ubSplitInfo_.axis1InSrcStride);
    tilingData_.set_axis2InSrcStride(ubSplitInfo_.axis2InSrcStride);
    tilingData_.set_axis0OutDstStride(ubSplitInfo_.axis0OutDstStride);
    tilingData_.set_axis1OutDstStride(ubSplitInfo_.axis1OutDstStride);
    tilingData_.set_axis2OutDstStride(ubSplitInfo_.axis2OutDstStride);

    tilingData_.set_blkAxes(blkSplitInfo_.blkAxes);
    tilingData_.set_blkAxesInAOffset(blkSplitInfo_.blkAxesInAOffset);
    tilingData_.set_blkAxesOutAOffset(blkSplitInfo_.blkAxesOutAOffset);
    tilingData_.set_inUbAxes(ubSplitInfo_.inUbAxes);
    tilingData_.set_outUbAxes(ubSplitInfo_.outUbAxes);
    tilingData_.set_ubPerm(ubSplitInfo_.ubPerm);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

std::string TransposeGatherTiling::PrintTilingData()
{
    std::string tdStr;
    tdStr += std::to_string(static_cast<int64_t>(tilingKey_)) + ",";
    tdStr += std::to_string(static_cast<int32_t>(dataTensorSize_)) + ",";
    tdStr += std::to_string(static_cast<int32_t>(indexTensorSize_)) + ",";
    tdStr += std::to_string(static_cast<int32_t>(blkSplitInfo_.usedCoreCnt)) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkAxesCnt) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkInUbCutPos) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkOutUbCutPos) + ",";
    tdStr += std::to_string(ubSplitInfo_.ubAxesCnt) + ",";
    tdStr += std::to_string(ubSplitInfo_.inUbInCutPos) + ",";
    tdStr += std::to_string(ubSplitInfo_.inUbOutCutPos) + ",";
    tdStr += std::to_string(ubSplitInfo_.outUbInCutPos) + ",";
    tdStr += std::to_string(ubSplitInfo_.outUbOutCutPos) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkFactor) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkTailFactor) + ",";
    tdStr += std::to_string(ubSplitInfo_.inUbCutAxisSize) + ",";
    tdStr += std::to_string(ubSplitInfo_.outUbCutAxisSize) + ",";
    tdStr += std::to_string(ubSplitInfo_.inUbCutAxisFactor) + ",";
    tdStr += std::to_string(ubSplitInfo_.outUbCutAxisFactor) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis0InSrcStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis1InSrcStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis2InSrcStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis0OutDstStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis1OutDstStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis2OutDstStride) + ",";
    tdStr += "block axes:";
    for (int8_t i = 0; i < MAX_TRANS_AXIS_NUM; ++i) {
        tdStr += std::to_string(blkSplitInfo_.blkAxes[i]) + " ";
    }
    tdStr += ",block axes in offset:";
    for (int8_t i = 0; i < MAX_TRANS_AXIS_NUM; ++i) {
        tdStr += std::to_string(blkSplitInfo_.blkAxesInAOffset[i]) + " ";
    }
    tdStr += ",block axes out offset:";
    for (int8_t i = 0; i < MAX_TRANS_AXIS_NUM; ++i) {
        tdStr += std::to_string(blkSplitInfo_.blkAxesOutAOffset[i]) + " ";
    }
    tdStr += ",ub in axes:";
    for (int8_t i = 0; i < UB_MAX_DIM_NUM; ++i) {
        tdStr += std::to_string(ubSplitInfo_.inUbAxes[i]) + " ";
    }
    tdStr += ",ub out axes:";
    for (int8_t i = 0; i < UB_MAX_DIM_NUM; ++i) {
        tdStr += std::to_string(ubSplitInfo_.outUbAxes[i]) + " ";
    }
    tdStr += ",ub perm:";
    for (int8_t i = 0; i < UB_MAX_DIM_NUM; ++i) {
        tdStr += std::to_string(ubSplitInfo_.ubPerm[i]) + " ";
    }
    return tdStr;
}

/**
 * @brief Gather Tiling 主流程入口
 *
 * 依次执行：UB 预算 → 切轴（UB 内轴+切分因子）→ 分核 → 写 TilingData →
 * 设置 tilingKey 与 blockDim。
 *
 * 任一步失败（MTE 效率不足 / bank 冲突 / 块数太少）均返回 GRAPH_FAILED，
 * 上层 RunTranposelTiling 会继续走 NDDMA 决策树（EntryTilingTemplate）兜底。
 *
 * @return GRAPH_SUCCESS gather Tiling 完成；GRAPH_FAILED 回退 NDDMA
 */
ge::graphStatus TransposeGatherTiling::DoTiling()
{
    CalcTensorSize();
    OP_CHECK_IF(CalcUbSplitInfo() != ge::GRAPH_SUCCESS,
                OP_LOGD(context_->GetNodeName(), "Stop to run gather tiling, mte size is too small!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcBlockSplitInfo() != ge::GRAPH_SUCCESS,
                OP_LOGD(context_->GetNodeName(), "Stop to run gather tiling, block count is too small!"),
                return ge::GRAPH_FAILED);
    WriteTilingData();
    OP_LOGI(context_->GetNodeName(), "The tiling data is: %s", PrintTilingData().c_str());
    return SetTilingKeyAndCore();
}

} // namespace TransWithGather
} // namespace optiling
