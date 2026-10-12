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
 * \file transpose_tiling_arch35.cpp
 * \brief Transpose Tiling 核心计算实现
 *
 * 本文件实现了 TransposeNddmaTiling 类的所有方法，涵盖 Tiling 计算的完整流程：
 *
 * 主流程：
 *   1. 获取输入 shape、perm、dtype 信息
 *   2. 校验 shape/perm 有效性
 *   3. 消除 size=1 的轴
 *   4. 合并 perm 中连续的轴
 *   5. 校验简化后的 shape
 *   6. 尝试 VCONV/021 路径（DAV_5102 专用）
 *   7. 尾轴转置场景：判断尾轴是否参与转置并尝试 Gather 路径
 *   8. 策略选型（决策树）
 *   9. UB 切分计算
 *  10. 多核 Block 切分
 *  11. 5维扩展
 *  12. UB 内 shape 计算
 *  13. CUT_TWICE 区间计算
 *  14. 填充 TilingData 结构
 *
 * 策略选型决策树：
 *   dim==1 → TENSOR_MOVE
 *   totalVolume*eleBytes < threshold → SMALL_SHAPE（兜底SIMT）
 *   !isLastAxisTranspose && lastAxisSize>=32 → N_LAST_TRANSPOSE
 *   dim<=5 → NDDMA_BASE → 切轴 → CUT_ONCE/CUT_TWICE
 *   dim>5 → BIG_DIM
 */

#include <sstream>
#include "util/platform_util.h"
#include "transpose_tiling_base.h"
#include "transpose_tiling_arch35.h"
#include "transpose_tiling_with_gather_arch35.h"
#include "transpose_tiling_with_nchwconv_arch35.h"
#include "transpose_tiling_with_021vconv_arch35.h"
#include "common/inc/op_host/math_log.h"

namespace optiling {
static constexpr int32_t VCONV_DIM_NUM = 2;
static constexpr int32_t VCONV_DSIZE = 2;

ge::graphStatus TransposeNddmaTiling::Init(const int64_t& coreNum, const int64_t& ubSize)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init TransposeNddmaTiling.");
    coreNum_ = coreNum;
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    ubSize_ = ubSize;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    cacheLineSize_ = Ops::Base::GetCacheLineSize(tilingContext_);
    OP_CHECK_IF((cacheLineSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get cache line size."),
                return ge::GRAPH_FAILED);

    ubBlockSize_ = Ops::Base::GetUbBlockSize(tilingContext_);
    OP_CHECK_IF((ubBlockSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub block size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 主 Tiling 流程
 *
 * 执行完整的 Tiling 计算流程，包括 shape 预处理、策略选型、
 * UB 切分、Block 切分、5维扩展、TilingData 填充。
 *
 * @return ge::GRAPH_SUCCESS 成功；ge::GRAPH_FAILED 失败
 */
ge::graphStatus TransposeNddmaTiling::RunTranposelTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start running Tiling4Transpose.");
    if (!isRelatedTranspose_) {
        OP_CHECK_IF(GetShapeInfo() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "Failed to get shape info!"), return ge::GRAPH_FAILED);
    }

    auto ret = CheckShapeInfo();
    CHECK_RET_SUCC(ret);
    // 消除 size=1 的轴（减少不必要的维度）
    RemoveAxisV2(shapeInfo_);
    // 合并 perm 中连续的轴（进一步降低维度数）
    MergeAxisV2(shapeInfo_);
    // check reduced shape
    ret = CheckReducedShapeInfo();
    CHECK_RET_SUCC(ret);

    CalcTotalVolumeActual();
    OP_CHECK_IF(TryVCONVTiling() == ge::GRAPH_SUCCESS, OP_LOGD(tilingContext_->GetNodeName(), "Do convTiling success"),
                return ge::GRAPH_SUCCESS);

    SetIsLastAxisTranspose();
    if (!isRelatedTranspose_ && shapeInfo_.isLastAxisTranspose) {
        TransWithGather::PlatInfo platInfo{coreNum_, ubSize_, cacheLineSize_, ubBlockSize_};
        TransWithGather::TransposeGatherTiling gatherTiling(tilingContext_, platInfo, shapeInfo_);
        OP_CHECK_IF(gatherTiling.DoTiling() == ge::GRAPH_SUCCESS,
                    OP_LOGD(tilingContext_->GetNodeName(), "Do gather tiling done!"), return ge::GRAPH_SUCCESS);
    }

    // ensure tiling template
    EntryTilingTemplate();
    // UB split
    CalcUBSplitInfo();
    // block split
    CalcBlockSplitInfo();
    // dim expand
    NDDMADimExpand();
    // get in ub shape info
    GetInUbShapeInfo();
    // cut twice get interval info
    GetIntervalInfo();
    // fill data
    FillTilingData();
    // print data
    PrintTilingData();
    // set block dim and tilingKey
    tilingContext_->SetBlockDim(tilingData_.transposeOpTiling.get_realCoreNum());
    tilingContext_->SetTilingKey(tilingKey_);
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, workspaces);
    workspaces[0] = WORK_SPACE_SIZE;
    OP_LOGD(tilingContext_->GetNodeName(), "Tiling4Transpose success.");
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 尝试 VCONV/021 Tiling 路径（仅 DAV_5102）
 *
 * DAV_5102(Ascend950) 有专用的 TransDataTo5HD 硬件指令，可实现高效的2D/3D转置。
 * 两种 VCONV 路径
 *   1. VCONV_TRANSPOSE(10007)：2D perm=[1,0] + 16bit + shape[0]>5
 *   2. VCONV_021_TRANSPOSE(10008)：3D perm=[0,2,1] + 8/16/32bit + H>8,W>8,HW≥448
 *
 * @return ge::GRAPH_SUCCESS 命中 VCONV 路径；ge::GRAPH_FAILED 未命中
 */
ge::graphStatus TransposeNddmaTiling::TryVCONVTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start Try VCONVTiling.");
    auto platformInfo = tilingContext_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto arch = ascendcPlatform.GetCurNpuArch();
    if (!isRelatedTranspose_ && arch == NpuArch::DAV_5102) {
        SMALL_SHAPE_BYTES_THRES_HOLD = SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102;
        if (shapeInfo_.reducedPerm[0] == 1 && shapeInfo_.reducedPerm[1] == 0 && shapeInfo_.dim == VCONV_DIM_NUM &&
            shapeInfo_.eleLenInBytes == VCONV_DSIZE && shapeInfo_.reducedInShape[0] > DIM_FIVE) {
            TransposeWithVCONV::PlatInfo platInfo{coreNum_, ubSize_};
            TransposeWithVCONV::TransposeVCONVTiling vconvTiling(tilingContext_, platInfo, shapeInfo_);
            OP_CHECK_IF(vconvTiling.DoTiling() == ge::GRAPH_SUCCESS,
                        OP_LOGD(tilingContext_->GetNodeName(), "Do convTiling done"), return ge::GRAPH_SUCCESS);
        }
        if (Is021VConvValid()) {
            Transpose021WithVCONV::PlatInfo platInfo{coreNum_, ubSize_};
            Transpose021WithVCONV::Transpose021VCONVTiling vconv021Tiling(tilingContext_, platInfo, shapeInfo_);
            OP_CHECK_IF(vconv021Tiling.DoTiling() == ge::GRAPH_SUCCESS,
                        OP_LOGD(tilingContext_->GetNodeName(), "Do 021 convTiling done"), return ge::GRAPH_SUCCESS);
        }
    }
    return ge::GRAPH_FAILED;
}

/**
 * @brief 判断 021 VCONV 路径是否有效
 *
 * 校验条件：
 * 1. perm = [0, 2, 1]（H↔W转置，N维不变）
 * 2. dim == 3
 * 3. dtype 为 8/16/32 bit
 * 4. H > 8, W > 8
 * 5. H*W ≥ 448
 * 6. HW 填充效率 > 50%（避免对齐开销过大）
 * 7. 总数据量 ≥ 70KB
 *
 * @return true 满足 021 VCONV 条件；false 不满足
 */
bool TransposeNddmaTiling::Is021VConvValid()
{
    // check perm: 021 transpose
    if (!(shapeInfo_.reducedPerm[0] == 0 && shapeInfo_.reducedPerm[DIM_ONE] == VCONV_DIM_NUM &&
          shapeInfo_.reducedPerm[DIM_TWO] == 1)) {
        return false;
    }
    // check dim
    if (shapeInfo_.dim != DIM_THREE) {
        return false;
    }
    // check dtype: support B8、B16、B32
    if (!(shapeInfo_.eleLenInBytes == B8_BYTES || shapeInfo_.eleLenInBytes == B16_BYTES ||
          shapeInfo_.eleLenInBytes == B32_BYTES)) {
        return false;
    }
    // check HW shape
    int64_t H = shapeInfo_.reducedInShape[DIM_ONE];
    int64_t W = shapeInfo_.reducedInShape[DIM_TWO];
    if (H <= DIM_EIGHT || W <= DIM_EIGHT) {
        return false;
    }
    if (H * W < HW_MIN_PRODUCT) {
        return false;
    }
    int64_t hAlign = Ops::Base::CeilDiv(H, HW_ALIGN) * HW_ALIGN;
    int64_t wAlign = Ops::Base::CeilDiv(W, HW_ALIGN) * HW_ALIGN;
    if (H * W <= hAlign * wAlign / DIM_TWO) {
        return false;
    }
    // check total volume
    if (shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes < SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_021) {
        return false;
    }
    return true;
}

template <typename T>
bool TransposeNddmaTiling::GetPerm(const gert::Tensor* permTensor)
{
    const T* permValue = permTensor->GetData<T>();
    if (!permValue) {
        OP_LOGE(tilingContext_->GetNodeName(), "Perm GetData is nullptr");
        return false;
    }
    int64_t dims = permTensor->GetShapeSize();
    for (int64_t i = 0; i < dims; i++) {
        shapeInfo_.perm[i] = permValue[i] < 0 ? permValue[i] + dims : permValue[i];
    }
    return true;
}

void TransposeNddmaTiling::SetIsLastAxisTranspose()
{
    int64_t dim = shapeInfo_.dim;
    shapeInfo_.isLastAxisTranspose = shapeInfo_.reducedPerm[dim - 1] != dim - 1 ? true : false;
}

void TransposeNddmaTiling::CalcTotalVolumeActual()
{
    int64_t vol = 1;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        vol = vol * shapeInfo_.reducedInShape[i];
    }
    shapeInfo_.totalVolumeActual = vol;
}

ge::graphStatus TransposeNddmaTiling::GetShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering GetShapeInfo.");

    const gert::Tensor* permTensor = tilingContext_->GetInputTensor(INPUT_IDX_PERM);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, permTensor);
    shapeInfo_.permSize = permTensor->GetShapeSize();

    auto permDtype = tilingContext_->GetInputDesc(INPUT_IDX_PERM)->GetDataType();
    uint64_t permDtypeSize = ge::GetSizeByDataType(permDtype);
    if (permDtypeSize == B32_BYTES) {
        if (!GetPerm<int32_t>(permTensor)) {
            return ge::GRAPH_FAILED;
        }
    } else if (permDtypeSize == B64_BYTES) {
        if (!GetPerm<int64_t>(permTensor)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "perm",
                                  ge::TypeUtils::DataTypeToSerialString(permDtype).c_str(), "int32 or int64");
        return ge::GRAPH_FAILED;
    }

    auto outputY = tilingContext_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputY);
    auto yShape = outputY->GetStorageShape();
    auto yDims = yShape.GetDimNum();
    auto inputX = tilingContext_->GetInputTensor(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputX);
    auto xShape = inputX->GetStorageShape();
    auto xDims = xShape.GetDimNum();

    auto xDtype = tilingContext_->GetInputDesc(INPUT_IDX_X)->GetDataType();
    shapeInfo_.eleLenInBytes = ge::GetSizeByDataType(xDtype);
    shapeInfo_.inShapeSize = xDims;
    shapeInfo_.outShapeSize = yDims;
    shapeInfo_.dim = xDims;
    shapeInfo_.origDim = xDims;
    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        shapeInfo_.inShape[i] = xShape[i];
        shapeInfo_.outShape[i] = yShape[i];
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckShapeDims()
{
    int64_t inDims = shapeInfo_.inShapeSize;
    int64_t outDims = shapeInfo_.outShapeSize;
    int64_t permDims = shapeInfo_.permSize;
    OP_CHECK_IF(inDims < 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "x",
                                                         std::to_string(inDims).c_str(), "positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inDims != outDims,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(tilingContext_->GetNodeName(), "x and y",
                                                          Ops::Math::Join(inDims, outDims).c_str(),
                                                          "The shape dims of x and y must be the same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inDims != permDims,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    tilingContext_->GetNodeName(), "perm", std::to_string(permDims).c_str(),
                    "The total number of elements of perm must be equal to the shape dim of x"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CheckShapeInfo.");
    CHECK_RET_SUCC(CheckShapeDims());

    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        if (shapeInfo_.perm[i] >= shapeInfo_.inShapeSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "perm",
                                                  std::to_string(shapeInfo_.perm[i]).c_str(),
                                                  "The value of perm must be less than shape dim of x");
            return ge::GRAPH_FAILED;
        }
        if (shapeInfo_.inShape[shapeInfo_.perm[i]] != shapeInfo_.outShape[i]) {
            std::ostringstream oss;
            oss << "The shape of y must be the same as the shape consisting of the axes of x. "
                << "The " << i << "-th axis is determined by value " << shapeInfo_.perm[i] << " of perm. "
                << "When the value of perm is a negative number, the " << i << "-th axis is equal to the "
                << std::abs(shapeInfo_.perm[i]) << "-th axis counted from the end of shape of x";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                tilingContext_->GetNodeName(), "x and y",
                Ops::Math::Join(shapeInfo_.inShape[shapeInfo_.perm[i]], shapeInfo_.outShape[i]).c_str(),
                oss.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }

    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        if (shapeInfo_.inShape[i] <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "x",
                                                  std::to_string(shapeInfo_.inShape[i]).c_str(),
                                                  "All axes of x must be positive numbers");
            return ge::GRAPH_FAILED;
        }
        if (shapeInfo_.outShape[i] <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "y",
                                                  std::to_string(shapeInfo_.outShape[i]).c_str(),
                                                  "All axes of y must be positive numbers");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckReducedShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CheckReducedShapeInfo.");
    auto dim = shapeInfo_.dim;
    if (dim < 1) {
        OP_LOGE(tilingContext_->GetNodeName(), "The dim of reducedShape is invalid, dim = %ld", dim);
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < dim; i++) {
        if (shapeInfo_.reducedInShape[i] <= 0 || shapeInfo_.reducedOutShape[i] <= 0) {
            OP_LOGE(tilingContext_->GetNodeName(), "Invalid shape, index is %ld, inShape is %ld, outShape is %ld", i,
                    shapeInfo_.reducedInShape[i], shapeInfo_.reducedOutShape[i]);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 输入侧切轴：确定输入切分轴 inCutIndex 和切分因子 inUbFactor
 *
 * 切轴逻辑：
 * 从输入最末轴（连续性最好的轴）向前扫描，优先整体"吞掉"较短的靠后轴；
 * 遇到第一根 UB 预算放不下的轴时，将其切为主块+尾块，并停止扫描。
 *
 * 预算模型：
 *   - splitInfo_.ubElement  = 期望每次进 UB 的元素总数
 *   - splitInfo_.inUbElement = 输入侧剩余预算（循环中逐步扣除已被整体吞掉的轴）
 *   - 吞掉一个轴：inUbElement /= currentShapeDim（一个 block 中该轴全量出现）
 *   - 切分一个轴：inUbElement 直接作为 inUbFactor（该 block 中只搬这么多）
 *
 * 产物（写入 splitInfo_）：
 *   - inCutIndex   输入切分轴索引
 *   - inUbFactor   主块切分因子（一次搬入 UB 的元素数）
 *   - inTailFactor 尾块元素数 = curDim % inUbFactor
 *   - outUbElement = ubElement / inUbActual（留给输出侧切轴的剩余预算）
 *
 * @return 切分后剩余的总 block 基数（帮助上层判断是否需要更多切分/分核）
 */
int64_t TransposeNddmaTiling::DoSplitUBInput(bool tryAbsorb)
{
    int64_t remainingTotalElment = shapeInfo_.totalVolumeActual;
    bool hasAbsorbed = false;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        int64_t currentShapeDim = shapeInfo_.reducedInShape[shapeInfo_.dim - 1 - i];
        if (splitInfo_.inUbElement < currentShapeDim) {
            // 该轴放不下 → 它就是输入切分轴，主块一次搬 inUbElement 个元素
            // Try to absorb this axis using full UB budget and cut on a more major axis instead.
            // This helps shapes like [8000, 256] perm[1,0] where sqrt(UB) barely misses the minor axis,
            // avoiding unnecessary CUT_TWICE.
            if (tryAbsorb && !hasAbsorbed && (shapeInfo_.dim - 1 - i) > 0) {
                int64_t fullBudgetRemaining = splitInfo_.ubElement / splitInfo_.inUbActual;
                if (currentShapeDim <= fullBudgetRemaining) {
                    splitInfo_.inUbActual *= currentShapeDim;
                    splitInfo_.inUbElement = fullBudgetRemaining / currentShapeDim;
                    splitInfo_.inUbFactor = currentShapeDim;
                    remainingTotalElment /= currentShapeDim;
                    hasAbsorbed = true;
                    continue;
                }
            }
            splitInfo_.inCutIndex = shapeInfo_.dim - 1 - i;
            splitInfo_.inUbFactor = splitInfo_.inUbElement;
            splitInfo_.inTailFactor = currentShapeDim % splitInfo_.inUbFactor;
            splitInfo_.inUbActual *= splitInfo_.inUbElement;
            // 更新剩余 block 基数：该轴从"整根"变为"分成 CeilDiv 块"后需循环的总份数
            remainingTotalElment = remainingTotalElment / currentShapeDim *
                                   Ops::Base::CeilDiv(currentShapeDim, splitInfo_.inUbElement);
            break;
        } else {
            // 该轴整体放得下 → 吞掉它：一个 block 中该轴全量出现，预算除以该轴长度
            splitInfo_.inUbElement /= currentShapeDim;
            splitInfo_.inUbActual *= currentShapeDim;
            remainingTotalElment /= currentShapeDim;
            splitInfo_.inUbFactor = currentShapeDim;
        }
    }
    // 输出侧剩余预算：UB 总量扣除输入侧已占用的部分
    splitInfo_.outUbElement = splitInfo_.ubElement / splitInfo_.inUbActual;
    return remainingTotalElment;
}

/**
 * @brief 根据输入轴索引查找其在输出 perm 中的位置
 *
 * 输入第 index 轴对应输出 perm 顺序中的哪一根轴（FindOutIndex）。
 * 用于判断输出切分轴是否比输入切分轴的映射位置"更靠前"（决定 CUT_ONCE/TWICE）。
 *
 * @param index 输入轴索引
 * @return 输出 perm 中的位置索引
 */
int64_t TransposeNddmaTiling::FindOutIndex(int64_t index)
{
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        if (shapeInfo_.reducedPerm[i] == index) {
            return i;
        }
    }
    return 0;
}

/**
 * @brief UB 越界校验（CUT_ONCE/CUT_TWICE 分核因子回溯时使用）
 *
 * 校验"当前切的轴 currentSplitIndex 取 currentSplitValue，且最外层轴全量进入
 * 一个 block 时，进 UB 的元素数是否会超过预算 splitInfo_.ubElement"。
 *
 * 计算方式（把 burst 长度按输出维度 Block 对齐后累乘外层尺寸）：
 *   burstLenBlockAlign = Π outShape[j] (j>currentSplitIndex，即切分轴右侧的连续轴)
 *                      * currentSplitValue   （切分轴本次取的块长）
 *      若 calcIn 且该输出轴恰是输入切分轴 → 再乘 inUbFactor（输入切分块在 block 中占的份数）
 *      然后按 ubBlockSize 字节对齐
 *   inUbElements = burstLenBlockAlign * Π outShape[j] (j<currentSplitIndex 且 perm[j]>inCutIndex)
 *                 * inUbFactor（perm[j]==inCutIndex 时）
 *
 * @param currentSplitIndex  当前候选切分轴（输出 perm 视角）
 * @param currentSplitValue  该轴拟使用的块长
 * @param calcIn             是否把输入切分因子 inUbFactor 计入（切输入侧因子回调时传 true）
 * @return true 越界（该因子不可用）；false 通过
 */
bool TransposeNddmaTiling::UbOutOfBoundCheck(int64_t currentSplitIndex, int64_t currentSplitValue, bool calcIn)
{
    int64_t burstLenBlockAlign = 1;
    for (int64_t i = currentSplitIndex + 1; i < shapeInfo_.dim; i++) {
        burstLenBlockAlign *= shapeInfo_.reducedOutShape[i];
    }
    burstLenBlockAlign *= currentSplitValue;
    if (calcIn && shapeInfo_.reducedPerm[currentSplitIndex] == splitInfo_.inCutIndex) {
        burstLenBlockAlign *= splitInfo_.inUbFactor;
    }
    burstLenBlockAlign = Ops::Base::CeilAlign(burstLenBlockAlign * shapeInfo_.eleLenInBytes, ubBlockSize_) /
                         shapeInfo_.eleLenInBytes;
    int64_t inUbElements = burstLenBlockAlign;
    for (int64_t i = 0; i < currentSplitIndex; i++) {
        if (shapeInfo_.reducedPerm[i] > splitInfo_.inCutIndex) {
            inUbElements *= shapeInfo_.reducedOutShape[i];
        } else if (shapeInfo_.reducedPerm[i] == splitInfo_.inCutIndex) {
            inUbElements *= splitInfo_.inUbFactor;
        }
    }
    if (inUbElements > splitInfo_.ubElement) {
        return true;
    }
    return false;
}

bool TransposeNddmaTiling::UbOutOfBoundCheckNLast(int64_t currentSplitIndex, int64_t currentSplitValue)
{
    int64_t burstLenBlockAlign = 1;
    if (currentSplitIndex == shapeInfo_.dim - 1) {
        burstLenBlockAlign = currentSplitValue;
    } else {
        burstLenBlockAlign = shapeInfo_.reducedInShape[shapeInfo_.dim - 1];
    }
    burstLenBlockAlign = Ops::Base::CeilAlign(burstLenBlockAlign * shapeInfo_.eleLenInBytes, ubBlockSize_) /
                         shapeInfo_.eleLenInBytes;
    int64_t inUbElements = burstLenBlockAlign;
    for (int64_t i = currentSplitIndex; i < shapeInfo_.dim - 1; i++) {
        if (i == currentSplitIndex) {
            inUbElements *= currentSplitValue;
        } else {
            inUbElements *= shapeInfo_.reducedInShape[i];
        }
    }
    if (inUbElements > splitInfo_.ubElement) {
        return true;
    }
    return false;
}

/**
 * @brief N_LAST 场景：按核利用率阈值回溯输入切分因子（Rate 版）
 *
 * 当按当前 inUbFactor 计算出的总循环数不足以铺满 coreNum 核时，
 * 从预算上限 splitInfo_.inUbElement 向下递减 i，寻找满足：
 *   - 潜在核数 coreNumNew = remainingTotalElment * CeilDiv(curDim, i) 与 coreNum 的
 *     比值 rate >= VEC_CORE_USED_THRES_HOLD（0.9，即至少用满90%核）
 *   - 且按该因子切后数据不越界 UB（UbOutOfBoundCheckNLast）
 * 的最大因子，作为新的 inUbFactor，越小切得越细 → block 越多 → 核用得更满。
 *
 * @param currentSplitIndex    当前输入切分轴索引
 * @param currentInShapeDim    切分轴完整长度
 * @param remainingTotalElment 切分轴之外其他轴的循环基数
 */
void TransposeNddmaTiling::FindSplitFactorByRateNLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                                      int64_t remainingTotalElment)
{
    splitInfo_.inCutIndex = currentSplitIndex;
    splitInfo_.inUbFactor = 1;
    splitInfo_.inTailFactor = 0;
    for (int64_t i = splitInfo_.inUbElement; i >= DIM_TWO; i--) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, i);
        double rate = static_cast<double>(coreNumNew) / coreNum_;
        if ((rate >= VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            splitInfo_.inUbFactor = i;
            splitInfo_.inTailFactor = currentInShapeDim % i;
            splitInfo_.inUbActual *= i;
            break;
        }
    }
}

/**
 * @brief CUT_ONCE/CUT_TWICE 分核因子回溯（Multiples 版，输出侧因子求值）
 *
 * 目标是让块总数等于 coreNum 整数倍的同时尽量用满核：
 * 从候选因子 i=1 递增，若：
 *   - 潜在块数 coreNumNew = remainingTotalElment * CeilDiv(curDim, i) 除以 coreNum
 *     的商仍等于目标倍数 coreNumMultiples（即块数同量级，不跳跃）
 *   - 且按因子 i 切后不越界 UB（UbOutOfBoundCheck）
 * 取最小的合法因子（切得最细、块数最多；i 越小 CeilDiv 越大 → 核用得更满）。
 * 找不到完全等倍数的因子时，退而求其次选不越界的最大因子 bestI。
 *
 * 注意：此函数更新的是输出侧切分信息（outCutIndex/outUbFactor/outUbActual），
 * 因为 CUT_ONCE/TWICE 分核时优先回溯输出切分因子。
 *
 * @param currentSplitIndex    候选输出切分轴
 * @param currentInShapeDim    切分轴完整长度
 * @param remainingTotalElment 切分轴之外其他轴的循环基数
 * @param coreNumMultiples     目标块数/coreNum 的整数倍数
 */
void TransposeNddmaTiling::FindSplitFactorByMultiplesLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                                          int64_t remainingTotalElment, int64_t coreNumMultiples)
{
    splitInfo_.outCutIndex = currentSplitIndex;
    int64_t bestI = 1;
    for (int64_t i = 1; i <= splitInfo_.outUbElement; i++) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, i);
        if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples) &&
            !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
            splitInfo_.outUbFactor = i;
            splitInfo_.outTailFactor = currentInShapeDim % i;
            splitInfo_.outUbActual *= i;
            return;
        }
        if (!UbOutOfBoundCheck(currentSplitIndex, i, true)) {
            bestI = i;
        }
    }
    splitInfo_.outUbFactor = bestI;
    splitInfo_.outTailFactor = currentInShapeDim % bestI;
    splitInfo_.outUbActual *= bestI;
}

/**
 * @brief N_LAST 场景：分核因子回溯（Multiples 版，输入侧因子求值）
 *
 * 与 FindSplitFactorByRateNLast 不同，本函数在先满足"块数 ≥ coreNum"的前提下，
 * 进一步让块数保持为 coreNum 的整数倍（负载均衡）：
 *   1. 先取最大的合法因子 inUbFactor（从预算上限递减，不越界 UB 的最大值）；
 *   2. 计算该因子对应的块数 coreNumTmp 与 coreNum 的整数倍数 coreNumMultiples；
 *   3. 从 i=1 递增找最小合法因子，使得 CeilDiv(curDim, i) 产生的块数仍落在同一
 *      整数倍数（coreNumMultiples）内且不越界 → 块数几乎不变但切分更细，
 *      实际用核率更高。
 *
 * @param currentSplitIndex    输入切分轴索引
 * @param currentInShapeDim    切分轴完整长度
 * @param remainingTotalElment 切分轴之外其他轴的循环基数
 * @param coreNumMultiples     目标块数/coreNum 的整数倍数
 */
void TransposeNddmaTiling::FindSplitFactorByMultiplesNLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                                           int64_t remainingTotalElment, int64_t coreNumMultiples)
{
    splitInfo_.inCutIndex = currentSplitIndex;
    for (int64_t i = splitInfo_.inUbElement; i >= 1; i--) {
        if (!UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            splitInfo_.inUbFactor = i;
            splitInfo_.inTailFactor = currentInShapeDim % i;
            splitInfo_.inUbActual *= i;
            break;
        }
    }
    int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, splitInfo_.inUbFactor);
    coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
    for (int64_t i = 1; i < splitInfo_.inUbFactor; i++) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, i);
        if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples) &&
            !UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            splitInfo_.inUbFactor = i;
            splitInfo_.inTailFactor = currentInShapeDim % i;
            splitInfo_.inUbActual = splitInfo_.inUbActual / splitInfo_.inUbElement * i;
            break;
        }
    }
}

/**
 * @brief 核心切轴逻辑（DoSplitUB 主流程）
 *
 * 完成输入切轴后（DoSplitUBInput），继续在**输出 perm 顺序**上做切轴：
 * 从输出最末轴向前扫描，在 outUbElement 预算内决定是否切分输出侧，
 * 最终依据"输出切分轴 vs 输入切分轴映射位置"判定 CUT_ONCE 还是 CUT_TWICE。
 *
 * 决策逻辑：
 *   - 输出轴对应的输入轴在切分轴右侧（perm[i] > inCutIndex）：
 *     该轴已由输入切分预算覆盖，直接跳过不重复切
 *   - 输出轴恰是输入切分轴映射（perm[i] == inCutIndex）：
 *     每个 block 中该轴还剩 CeilDiv(inShape[inCutIndex], inUbFactor) 个"行"
 *   - 其余输出轴：按完整 outShape 推算
 *
 * 循环终止条件（找到 outCutIndex）：
 *   - outUbElement < currentShapeDim：预算放不下 → 在当前轴切分（outUbFactor 回溯求值）
 *   - currentShapeDim 全部吞掉则继续向前，同时记录 outCutIndex（退轴至吞不下的轴）
 *
 * 最终判定：
 *   outCutIndex > FindOutIndex(inCutIndex) → CUT_TWICE（输出切分轴比输入映射更靠前，
 *   需同时在输入/输出两侧切轴，否则最外层输出维度会溢出 UB）
 *   否则 → CUT_ONCE（输入切分已足够，输出侧由 NDDMA 自动重排）
 */
void TransposeNddmaTiling::DoSplitUBOutputScan(int64_t remainingTotalElment)
{
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        int64_t currentSplitIndex = shapeInfo_.dim - 1 - i;
        if (shapeInfo_.reducedPerm[currentSplitIndex] > splitInfo_.inCutIndex) { // skip axis full cut by input shape
            continue;
        }
        int64_t currentShapeDim = shapeInfo_.reducedOutShape[currentSplitIndex];
        if (shapeInfo_.reducedPerm[currentSplitIndex] == splitInfo_.inCutIndex) {
            currentShapeDim = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex],
                                                 splitInfo_.inUbFactor);
        }
        remainingTotalElment /= currentShapeDim;
        int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentShapeDim, splitInfo_.outUbElement);
        if (splitInfo_.outUbElement < currentShapeDim) {
            if (coreNumTmp > coreNum_) { // use full coreNum
                int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
                FindSplitFactorByMultiplesLast(currentSplitIndex, currentShapeDim, remainingTotalElment,
                                               coreNumMultiples);
            } else {
                splitInfo_.outCutIndex = currentSplitIndex;
                splitInfo_.outUbFactor = splitInfo_.outUbElement;
                splitInfo_.outTailFactor = currentShapeDim % splitInfo_.outUbElement;
                splitInfo_.outUbActual *= splitInfo_.outUbElement;
            }
            break;
        } else {
            splitInfo_.outUbElement /= currentShapeDim;
            splitInfo_.outUbActual *= currentShapeDim;
            splitInfo_.outCutIndex = currentSplitIndex;
            splitInfo_.outUbFactor = currentShapeDim;
        }
    }
    if (splitInfo_.outCutIndex > FindOutIndex(splitInfo_.inCutIndex)) {
        tilingKey_ = static_cast<int64_t>(SplitMode::CUT_TWICE);
    } else {
        tilingKey_ = static_cast<int64_t>(SplitMode::CUT_ONCE);
    }
}

void TransposeNddmaTiling::DoSplitUB()
{
    // Save initial state for potential retry with absorption strategy
    SplitInfo initialSplitInfo = splitInfo_;

    // Strategy 1: original (sqrt budget, no absorption)
    int64_t remainingTotalElment = DoSplitUBInput(false);
    DoSplitUBOutputScan(remainingTotalElment);

    // If CUT_TWICE, try absorption strategy to see if CUT_ONCE is achievable
    if (tilingKey_ == static_cast<int64_t>(SplitMode::CUT_TWICE)) {
        SplitInfo originalResult = splitInfo_;
        int64_t originalTilingKey = tilingKey_;

        // Reset to initial state and retry with absorption
        splitInfo_ = initialSplitInfo;
        remainingTotalElment = DoSplitUBInput(true);
        DoSplitUBOutputScan(remainingTotalElment);

        // If absorption didn't yield CUT_ONCE, revert to original strategy
        if (tilingKey_ != static_cast<int64_t>(SplitMode::CUT_ONCE)) {
            splitInfo_ = originalResult;
            tilingKey_ = originalTilingKey;
        } else {
            // Validate: CalcBlockSplitInfoForCutOnce will multiply outUbFactor by
            // inUbFactor when inCutIndex == reducedPerm[outCutIndex]. Check UB bounds
            // with the post-modification factor to avoid UB OOB in the kernel.
            int64_t effectiveOutUbFactor = splitInfo_.outUbFactor;
            if (splitInfo_.inCutIndex == shapeInfo_.reducedPerm[splitInfo_.outCutIndex]) {
                effectiveOutUbFactor *= splitInfo_.inUbFactor;
            }
            if (UbOutOfBoundCheck(splitInfo_.outCutIndex, effectiveOutUbFactor, true)) {
                splitInfo_ = originalResult;
                tilingKey_ = originalTilingKey;
            }
        }
    }
}

/**
 * @brief BIG_DIM（>5维）场景的切轴逻辑
 *
 * 超过 5 维时无法直接使用 NDDMA 5 维格式，切轴简化为只切一根输出轴：
 * 从输出最末轴向前扫描，遇到以下任一条件即确定 outCutIndex：
 *   1. UB 预算放不下该轴（ubElement < outShape[i]）：直接切分
 *   2. 吞掉该轴后累计元素数 ≤ coreNum（剩余维度已足够铺满核，无需再切更外层）
 *   3. 已经吞了 NDDMA_MAX_DIM_NUM-1=4 根轴（只剩最后一根可切的空间）
 *
 * 切分轴确定后，后续由 CalcBlockSplitInfoForBigDim 分核、FlushBaseNumForBigDim
 * 将 >5 维压缩到 5 维 NDDMA 表示。
 */
void TransposeNddmaTiling::DoSplitUBBigDim()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering DoSplitUBBigDim.");
    int64_t dimSize = NDDMA_MAX_DIM_NUM - 1;
    int64_t totalElment = shapeInfo_.totalVolumeActual;
    // search split index and calc base number
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; i--) {
        if (splitInfo_.ubElement < shapeInfo_.reducedOutShape[i]) {
            splitInfo_.outCutIndex = i;
            splitInfo_.outUbFactor = splitInfo_.ubElement;
            splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[i] % splitInfo_.outUbFactor;
            break;
        } else if (dimSize > 0) {
            totalElment = totalElment / shapeInfo_.reducedOutShape[i];
            if (totalElment <= coreNum_) {
                splitInfo_.outCutIndex = i;
                splitInfo_.outUbFactor = splitInfo_.ubElement <= shapeInfo_.reducedOutShape[i] ?
                                             splitInfo_.ubElement :
                                             shapeInfo_.reducedOutShape[i];
                splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[i] % splitInfo_.outUbFactor;
                break;
            }
            splitInfo_.ubElement = splitInfo_.ubElement / shapeInfo_.reducedOutShape[i];
            dimSize--;
        } else if (dimSize == 0) {
            splitInfo_.outCutIndex = i;
            splitInfo_.outUbFactor = splitInfo_.ubElement <= shapeInfo_.reducedOutShape[i] ?
                                         splitInfo_.ubElement :
                                         shapeInfo_.reducedOutShape[i];
            splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[i] % splitInfo_.outUbFactor;
            break;
        }
    }
}

/**
 * @brief BIG_DIM 场景：将 >5 维原始 shape 压缩为 NDDMA 5 维表示
 *
 * 生成 kernel 端 NDDMA SetLoopInfo 所需的三个数组：
 *   - nddmaIdx[i]：5 维压缩索引 i 对应的原始输入轴索引（按输出 perm 顺序从高维往低维排）
 *   - baseNddmaShape[i]：压缩后第 i 维的循环基数（该轴在 block 循环中的步长）
 *   - baseInShape_[k]：原始第 k 轴的输入地址基数（右侧轴的乘积）
 *
 * 循环基数＝该轴"一个 block 中占的重复份数"：
 *   - 切分轴右侧的轴：完整 outShape[i]（每 block 在该轴全量循环）
 *   - 切分轴：outUbFactor（主块因子）
 * 汇总到 totalNddmaNum_（一个 block 的总元素数，kernel CopyOut 用它设 blockLen）。
 *
 * 最后按输出维度顺序 sort 后回填 baseNddmaShape_（kernel 按 nddmaIdx 查找对应基数）。
 */
void TransposeNddmaTiling::FlushBaseNumForBigDim()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FlushBaseNumForBigDim.");
    int64_t idxNum = NDDMA_MAX_DIM_NUM - 1;
    int64_t baseInNum = 1;
    int64_t baseOutNum = 1;
    int64_t tmpNddmaShape[NDDMA_MAX_DIM_NUM] = {0};
    int64_t oriNddmaIdx[NDDMA_MAX_DIM_NUM] = {-1};
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; i--) {
        baseInShape_[i] = baseInNum;
        baseInNum *= shapeInfo_.reducedInShape[i];
        baseOutNum *= shapeInfo_.reducedOutShape[i];
        if (i > splitInfo_.outCutIndex) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            totalNddmaNum_ *= shapeInfo_.reducedOutShape[i];
            idxNum--;
        } else if (i == splitInfo_.outCutIndex) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            totalNddmaNum_ *= splitInfo_.outUbFactor;
            idxNum--;
        } else if (idxNum >= 0) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            idxNum--;
        }
    }

    std::sort(std::begin(nddmaIdx_), std::end(nddmaIdx_));
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        for (int64_t j = 0; j < NDDMA_MAX_DIM_NUM; j++) {
            if (nddmaIdx_[i] == oriNddmaIdx[j]) {
                baseNddmaShape_[i] = tmpNddmaShape[j];
            }
        }
    }
}

/**
 * @brief 策略选型决策树
 *
 * 按以下优先级选择 Tiling 策略：
 * 1. dim==1 → TENSOR_MOVE（纯搬运）
 * 2. totalVolume*eleBytes < threshold → SMALL_SHAPE（SIMT兜底）
 * 3. !isLastAxisTranspose && lastAxisSize>=32 → N_LAST_TRANSPOSE（连续行搬移）
 * 4. dim<=5 → NDDMA_BASE（后续 DoSplitUB 进一步判定 CUT_ONCE/CUT_TWICE）
 * 5. dim>5 → BIG_DIM（压缩到5维NDDMA）
 */
void TransposeNddmaTiling::EntryTilingTemplate()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering EntryTilingTemplate.");
    SetIsLastAxisTranspose();
    splitInfo_.ubElement = ubSize_ / shapeInfo_.eleLenInBytes;
    if (shapeInfo_.dim == 1) {
        // just tensor move
        tilingKey_ = static_cast<int64_t>(SplitMode::TENSOR_MOVE);
        return;
    }

    auto platformInfo = tilingContext_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto arch = ascendcPlatform.GetCurNpuArch();
    if (arch == NpuArch::DAV_5102 && !shapeInfo_.isLastAxisTranspose &&
        shapeInfo_.reducedInShape[shapeInfo_.dim - 1] >= MOVEALIGN_LAST_MIN_ELE) {
        SMALL_SHAPE_BYTES_THRES_HOLD = SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_NLAST;
    }
    if (shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes >= SMALL_SHAPE_BYTES_THRES_HOLD) {
        if (!shapeInfo_.isLastAxisTranspose &&
            shapeInfo_.reducedInShape[shapeInfo_.dim - 1] >= MOVEALIGN_LAST_MIN_ELE) {
            // MoveAlign场景 （大于等于32个元素）
            tilingKey_ = static_cast<int64_t>(SplitMode::N_LAST_TRANSPOSE);
            return;
        }
        if (shapeInfo_.dim <= NDDMA_MAX_DIM_NUM) {
            // nddma场景：尾轴不转置以及尾轴转置且dst[-1]>32（待优化）
            tilingKey_ = static_cast<int64_t>(SplitMode::NDDMA_BASE);
            return;
        } else {
            // BigShape场景,维度超过NDDMA_MAX_DIM_NUM且总元素超过threshold
            tilingKey_ = static_cast<int64_t>(SplitMode::BIG_DIM);
            return;
        }
    }
    // 兜底逻辑 simt
    tilingKey_ = static_cast<int64_t>(SplitMode::SMALL_SHAPE);
}

/**
 * @brief UB 切分信息计算入口（按 TilingKey 分派）
 *
 * 根据策略选择不同的 UB 预算模型：
 *   - TENSOR_MOVE / N_LAST_TRANSPOSE：双缓冲，ubElement = ubSize/2/eleBytes
 *     （一半 UB 用于 CopyIn，另一半用于 CopyOut）
 *   - NDDMA_BASE（CUT_ONCE/CUT_TWICE）：inUbElement = sqrt(ubElement)，
 *     近似保证输入+输出两份数据之和不超过 UB；随后 DoSplitUB 完成切轴
 *   - BIG_DIM：DoSplitUBBigDim 只切输出侧一根轴
 */
void TransposeNddmaTiling::CalcUBSplitInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcUBSplitInfo");
    switch (tilingKey_) {
        case static_cast<int64_t>(SplitMode::TENSOR_MOVE):
        case static_cast<int64_t>(SplitMode::N_LAST_TRANSPOSE):
            splitInfo_.ubElement = ubSize_ / BUFFER_NUM / shapeInfo_.eleLenInBytes;
            break;
        case static_cast<int64_t>(SplitMode::NDDMA_BASE):
            splitInfo_.inUbElement = sqrt(splitInfo_.ubElement);
            DoSplitUB();
            break;
        case static_cast<int64_t>(SplitMode::BIG_DIM):
            DoSplitUBBigDim();
            break;
        default:
            break;
    }
}

/**
 * @brief TENSOR_MOVE（1维纯搬运）分核逻辑
 *
 * 数据线性排布，无需处理转置。按总元素数均分：
 *   - 总元素 < coreNum：只用 1 个核，一次全部搬完（blkFactor = 总元素）
 *   - 否则：开满核，blkFactor = 总元素/coreNum，blkTailFactor 兜尾
 * inUbFactor = ubElement（单个 block 一次可搬运的元素数）。
 */
void TransposeNddmaTiling::CalcBlockSplitInfoForTensorMove()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForTensorMove.");
    if (shapeInfo_.totalVolumeActual < coreNum_) {
        realCoreNum_ = 1;
        blkFactor_ = shapeInfo_.totalVolumeActual;
        blkTailFactor_ = 0;
        splitInfo_.inUbFactor = splitInfo_.ubElement;
    } else {
        realCoreNum_ = coreNum_;
        blkFactor_ = shapeInfo_.totalVolumeActual / coreNum_;
        blkTailFactor_ = shapeInfo_.totalVolumeActual % coreNum_;
        splitInfo_.inUbFactor = splitInfo_.ubElement;
    }
}

/**
 * @brief SMALL_SHAPE 分核的多核切分辅助（每个候选轴的最小因数切分）
 *
 * 遍历形状轴 i，若整根轴搬移的字节超过 cacheLineSize（无法一次搬完），
 * 就在该轴按最小因数 j 切分 outCutIndex，使块级地址跳到下一条 cache line：
 *   - j == 轴全长 且是第 0 轴（素数轴且已到最后）：outUbFactor 取 ceil(cacheLine)，
 *     最终靠其他轴或外层循环开多核
 *   - j == 轴全长 且非第 0 轴：全切，剩余外层轴继续开多核
 *   - 其他：按最小因数 j 切分
 *
 * @param i            候选轴索引
 * @param shapeSizeByte 该轴完整尺寸（字节）
 * @param totalElment   [in,out] 累计 block 基数（切分后按 CeilDiv 更新）
 * @return 更新后的 totalElment
 */
int64_t TransposeNddmaTiling::CalcBlockSplitInfoForNoCutForMultiCore(int64_t i, int64_t shapeSizeByte,
                                                                     int64_t& totalElment)
{
    for (int64_t j = 2; j <= shapeInfo_.reducedOutShape[i]; j++) {
        if ((shapeInfo_.reducedOutShape[i] % j == 0) &&
            (shapeSizeByte / shapeInfo_.reducedOutShape[i] * j > cacheLineSize_)) {
            if (j == shapeInfo_.reducedOutShape[i] && i == 0) {
                // 素数且切到了最后，正常切
                splitInfo_.outCutIndex = i;
                splitInfo_.outUbFactor = Ops::Base::CeilDiv(cacheLineSize_ + 1,
                                                            shapeSizeByte / shapeInfo_.reducedOutShape[i]);
                splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[i] % splitInfo_.outUbFactor;
                totalElment *= Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[i], splitInfo_.outUbFactor);
                break;
            } else if (j == shapeInfo_.reducedOutShape[i] && i != 0) {
                // 素数但还没切到最后，全切，剩下的轴开多核
                splitInfo_.outCutIndex = i;
                splitInfo_.outUbFactor = shapeInfo_.reducedOutShape[i];
                splitInfo_.outTailFactor = 0;
                break;
            } else {
                // 其他场景，按最小因数切
                splitInfo_.outCutIndex = i;
                splitInfo_.outUbFactor = j;
                splitInfo_.outTailFactor = 0;
                totalElment *= Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[i], splitInfo_.outUbFactor);
                break;
            }
        }
    }
    return totalElment;
}

/**
 * @brief SMALL_SHAPE（SIMT 直读直写）分核逻辑
 *
 * SIMT 模式不经过 UB，每个核直接 GM→GM。分核目标是让每核处理的数据
 * 量对齐 128 字节（SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE），避免跨核访问
 * 边界不对齐：
 *   - 总元素 < coreNum：按元素数开核（每核 1 个元素）
 *   - 否则：先取 ceil（向上对齐 128B）与 floor（向下对齐 128B）两个候选，
 *     若 floor×核数仍能容纳全部数据则直接开满核，否则用 ceil 对齐后的核数
 *     （ceilAlignFactor 比 floor 大 → 实际核数略少但每核数据对齐）
 */
void TransposeNddmaTiling::CalcBlockSplitInfoForSmallShape()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForSmallShape.");
    int64_t totalElements = shapeInfo_.totalVolumeActual;
    int64_t alignElements = SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE / shapeInfo_.eleLenInBytes;
    if (totalElements < alignElements || totalElements < coreNum_) {
        realCoreNum_ = 1;
        blkFactor_ = totalElements;
        blkTailFactor_ = 0;
        return;
    }
    // simt every core elemets align to 128Byte
    int64_t blkFactor = totalElements / coreNum_;
    int64_t ceilAlignFactor = Ops::Base::CeilAlign(blkFactor, alignElements);
    int64_t floorAlignFactor = Ops::Base::FloorAlign(blkFactor, alignElements);
    if (totalElements - floorAlignFactor * (coreNum_ - 1) <= floorAlignFactor) {
        realCoreNum_ = coreNum_;
        blkFactor_ = floorAlignFactor;
        blkTailFactor_ = totalElements % floorAlignFactor;
    } else {
        realCoreNum_ = Ops::Base::CeilDiv(totalElements, ceilAlignFactor);
        blkFactor_ = ceilAlignFactor;
        blkTailFactor_ = totalElements % ceilAlignFactor;
    }
}

/**
 * @brief N_LAST_TRANSPOSE（尾轴不转置）分核逻辑
 *
 * 尾轴不转置时最后维在输入/输出中保持连续，可按"连续行搬移"分块。
 * 从输入最末轴向前做 UB 预算（inUbElement = ubSize/2/eleBytes，双缓冲）：
 *   - 预算放不下该轴（inUbElement < curDim）：
 *       · 若切后块数已足够 ≥coreNum → FindSplitFactorByMultiplesNLast（保整数倍负载均衡）
 *       · 否则 → FindSplitFactorByRateNLast（按 0.9 核利用率阈值回溯因子）
 *       · 若 inUbFactor 无效（0）→ CheckInUbFactorValid 退轴到上一层
 *   - 预算放得下：整根吞掉，预算除以该轴长度，记录 solvedTotalElment 便于退轴恢复
 *
 * 最终分核参数 = SetRealCoreNumAndBlkFactor(CeilDiv(inCutAxis, inUbFactor) × remaining)
 */
void TransposeNddmaTiling::CalcBlockSplitInfoForNLastTranspose()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForNLastTranspose.");
    splitInfo_.inUbElement = splitInfo_.ubElement;
    int64_t remainingTotalElment = shapeInfo_.totalVolumeActual;
    int64_t solvedTotalElment[MAX_AXIS_NUM_FOR_TRANSPOSE];
    solvedTotalElment[shapeInfo_.dim - 1] = 1;
    int64_t currentSplitIndex;
    int64_t currentInShapeDim;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        currentSplitIndex = shapeInfo_.dim - 1 - i;
        currentInShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
        remainingTotalElment /= currentInShapeDim;
        int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, splitInfo_.inUbElement);
        if (splitInfo_.inUbElement < currentInShapeDim) {
            if (coreNumTmp < coreNum_) { // use at least VEC_CORE_USED_THRES_HOLD * coreNum
                FindSplitFactorByRateNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment);
                break;
            } else { // use full coreNum
                int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
                FindSplitFactorByMultiplesNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment,
                                                coreNumMultiples);
                /* 检查inUbFactor是否合法，不合法则执行退轴逻辑 */
                CheckInUbFactorValid(currentSplitIndex, currentInShapeDim, remainingTotalElment, coreNumMultiples,
                                     solvedTotalElment);
                break;
            }
        } else if (coreNumTmp < coreNum_) { // use at least VEC_CORE_USED_THRES_HOLD * coreNum
            FindSplitFactorByRateNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment);
            break;
        } else {
            splitInfo_.inUbElement /= currentInShapeDim;
            solvedTotalElment[currentSplitIndex] = currentSplitIndex == shapeInfo_.dim - 1 ?
                                                       currentInShapeDim :
                                                       (currentInShapeDim * solvedTotalElment[currentSplitIndex + 1]);
        }
    }
    int64_t coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex], splitInfo_.inUbFactor) *
                      remainingTotalElment;
    SetRealCoreNumAndBlkFactor(coreNum);
}

/**
 * @brief N_LAST 退轴逻辑（inUbFactor 不可用时回退到上一根轴重切）
 *
 * 当在某根轴上切出的 inUbFactor==0（预算在该轴耗尽或因子查找失败）且
 * 尚未切到最末轴时，把切分位置**上移一层**（切更靠末的轴），并恢复该层
 * 的原始预算（inUbElement = 该轴完整长度、remaining 从 solvedTotalElment 重建），
 * 重新调用 FindSplitFactorByMultiplesNLast 求因子，直到找到合法 inUbFactor 或穷尽。
 *
 * @param currentSplitIndex    [in,out] 输入切分轴索引（会上移）
 * @param currentInShapeDim    [in,out] 当前轴长度
 * @param remainingTotalElment [in,out] 剩余循环基数
 * @param coreNumMultiples     [in,out] 目标块/coreNum 倍数
 * @param solvedTotalElment    记录已完整吞入的轴的容量（用于恢复）
 */
void TransposeNddmaTiling::CheckInUbFactorValid(int64_t& currentSplitIndex, int64_t& currentInShapeDim,
                                                int64_t& remainingTotalElment, int64_t& coreNumMultiples,
                                                int64_t* solvedTotalElment)
{
    if (splitInfo_.inUbFactor == 0 && currentSplitIndex < shapeInfo_.dim - 1) {
        while (currentSplitIndex < shapeInfo_.dim - 1) {
            currentSplitIndex++;
            currentInShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
            splitInfo_.inUbElement = shapeInfo_.reducedInShape[currentSplitIndex];
            remainingTotalElment = shapeInfo_.totalVolumeActual / solvedTotalElment[currentSplitIndex];
            coreNumMultiples = remainingTotalElment;
            FindSplitFactorByMultiplesNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment,
                                            coreNumMultiples);
            if (splitInfo_.inUbFactor > 0) {
                break;
            }
        }
    }
}

/**
 * @brief 根据 block 总数设置分核参数（realCoreNum/blkFactor/blkTailFactor）
 *
 * 核心规则：
 *   - block 总数 ≥ coreNum：开满全部核；每核基础 blkFactor 个 block，
 *     前 blkTailFactor 个核额外多处理 1 个（差额补偿均分）
 *   - block 总数 < coreNum：只用 blockTotal 个核，每核 1 个 block
 *
 * @param coreNum 当前分核路径计算出的 block 总数
 */
void TransposeNddmaTiling::SetRealCoreNumAndBlkFactor(int64_t coreNum)
{
    if (coreNum >= coreNum_) {
        realCoreNum_ = coreNum_;
        blkFactor_ = coreNum / coreNum_;
        blkTailFactor_ = coreNum % coreNum_;
    } else {
        realCoreNum_ = coreNum;
        blkFactor_ = 1;
        blkTailFactor_ = 0;
    }
}

/**
 * @brief CUT_ONCE 分核逻辑
 *
 * 输入切轴（DoSplitUBInput）＋输出切轴（DoSplitUB）完成后，计算把
 * block 摊到核上的分核参数（realCoreNum/blkFactor/blkTailFactor）。
 *
 * 前置修正（把 CUT_ONCE 的"单轴切分"统一到输入轴视角）：
 *   1. 若输入/输出切的是同一根轴（inCutIndex == perm[outCutIndex]）：
 *      outUbFactor *= inUbFactor —— 一个 block 同时覆盖输入的切分份数与输出的切分份数
 *   2. 若输出切分因子整好切满整个输出轴（outUbFactor == outShape[outCutIndex]）：
 *      实际有效切轴退化为输入轴，outCutIndex 改指 FindOutIndex(inCutIndex)
 *
 * block 总数 = CeilDiv(outShape[outCutIndex], outUbFactor)
 *            × Π outShape[i]（i < outCutIndex 且 perm[i] < inCutIndex 的外层轴）
 *
 * 核利用率不足（outUbAxis < coreNum）时，回调输出切分因子：
 * 从当前因子递减，找"潜在核数占比 ≥ 0.9 且不越界 UB"的最大因子；
 * 找不到则取 rate 最大的合法因子（bestRate/bestI 记录）。
 */
void TransposeNddmaTiling::CalcBlockSplitInfoForCutOnce()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForCutOnce.");
    // input and output split the same axis, the split factor is subject to the output
    if (splitInfo_.inCutIndex == shapeInfo_.reducedPerm[splitInfo_.outCutIndex]) {
        splitInfo_.outUbFactor *= splitInfo_.inUbFactor;
        splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[splitInfo_.outCutIndex] % splitInfo_.outUbFactor;
    }
    // input and output split different axis, but output split factor is the whole axis
    if (splitInfo_.outUbFactor == shapeInfo_.reducedOutShape[splitInfo_.outCutIndex]) {
        splitInfo_.outCutIndex = FindOutIndex(splitInfo_.inCutIndex);
        splitInfo_.outUbFactor = splitInfo_.inUbFactor;
        splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[splitInfo_.outCutIndex] % splitInfo_.outUbFactor;
    }
    int64_t outUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor);
    int64_t outUbAxisExceptSplitAxis = 1;
    for (int64_t i = 0; i < splitInfo_.outCutIndex; i++) {
        if (shapeInfo_.reducedPerm[i] < splitInfo_.inCutIndex) {
            outUbAxisExceptSplitAxis *= shapeInfo_.reducedOutShape[i];
        }
    }
    if (outUbAxis < coreNum_) {
        // use at least VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t currentSplitIndex = splitInfo_.outCutIndex;
        int64_t currentShapeDim = shapeInfo_.reducedOutShape[currentSplitIndex];
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = splitInfo_.outUbFactor; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outUbAxisExceptSplitAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheck(currentSplitIndex, i, false)) {
                splitInfo_.outUbFactor = i;
                splitInfo_.outTailFactor = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (!UbOutOfBoundCheck(currentSplitIndex, i, false) && rate > bestRate) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            splitInfo_.outUbFactor = bestI;
            splitInfo_.outTailFactor = currentShapeDim % bestI;
        }
    }
    outUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor) *
                outUbAxisExceptSplitAxis;
    SetRealCoreNumAndBlkFactor(outUbAxis);
}

/**
 * @brief CUT_TWICE 分核逻辑
 *
 * 双切场景 block 总数按"输出外层块数 × 输入轴块数"计算：
 *   outAxiseExceptSplitInAxis = Π outShape[i]（i<outCutIndex 且 perm[i]<inCutIndex）
 *                             × CeilDiv(outShape[outCutIndex], outUbFactor)
 *   inUbAxis = CeilDiv(inShape[inCutIndex], inUbFactor)
 *   blockTotal = outAxiseExceptSplitInAxis × inUbAxis
 *
 * 核利用率不足（< coreNum）时回调**输入**切分因子 inUbFactor
 * （输出切分轴已被外层循环枚举，提高并行度只能让输入侧切得更细），
 * 选取 rate≥0.9 且不越界 UB 的最大因子，找不到则取 rate 最大者。
 */
void TransposeNddmaTiling::CalcBlockSplitInfoForCutTwice()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForCutTwice.");
    int64_t outAxiseExceptSplitInAxis = 1;
    for (int64_t i = 0; i < splitInfo_.outCutIndex; i++) {
        if (shapeInfo_.reducedPerm[i] < splitInfo_.inCutIndex) {
            outAxiseExceptSplitInAxis *= shapeInfo_.reducedOutShape[i];
        }
    }
    outAxiseExceptSplitInAxis *= Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex],
                                                    splitInfo_.outUbFactor);
    int64_t inUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex], splitInfo_.inUbFactor);
    if (outAxiseExceptSplitInAxis * inUbAxis < coreNum_) {
        // use at least VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t currentSplitIndex = splitInfo_.inCutIndex;
        int64_t currentShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = splitInfo_.inUbFactor; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outAxiseExceptSplitInAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
                splitInfo_.inUbFactor = i;
                splitInfo_.inTailFactor = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (rate > bestRate && !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            splitInfo_.inUbFactor = bestI;
            splitInfo_.inTailFactor = currentShapeDim % bestI;
        }
    }
    inUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex], splitInfo_.inUbFactor) *
               outAxiseExceptSplitInAxis;
    SetRealCoreNumAndBlkFactor(inUbAxis);
}

/**
 * @brief BIG_DIM（>5维）分核逻辑
 *
 * 只回调输出切分因子 outUbFactor 来调节并行度：
 *   outUbAxisExceptSplitAxis = Π outShape[i]（i < outCutIndex）
 *   coreNum = CeilDiv(outShape[outCutIndex], outUbFactor) × outUbAxisExceptSplitAxis
 *
 * coreNum < coreNum_（块不足）时：
 *   递减 outUbFactor 找 rate≥0.9 的最大因子（无越界检查，BIG_DIM 无 UB 内切轴）
 * block 数充足时：找使 FloorDiv(块数,coreNum) 保持相同倍数的更细因子（负载均衡）
 * 最后 FlushBaseNumForBigDim 生成 5 维压缩映射并 SetRealCoreNumAndBlkFactor。
 */
void TransposeNddmaTiling::CalcBlockSplitInfoForBigDim()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForBigDim.");
    int64_t outUbAxisExceptSplitAxis = 1;
    for (int64_t i = 0; i < splitInfo_.outCutIndex; i++) {
        outUbAxisExceptSplitAxis *= shapeInfo_.reducedOutShape[i];
    }
    int64_t coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor) *
                      outUbAxisExceptSplitAxis;
    int64_t currentShapeDim = shapeInfo_.reducedOutShape[splitInfo_.outCutIndex];
    if (coreNum < coreNum_) {
        // use at least VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = splitInfo_.outUbFactor; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outUbAxisExceptSplitAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= VEC_CORE_USED_THRES_HOLD)) {
                splitInfo_.outUbFactor = i;
                splitInfo_.outTailFactor = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (rate > bestRate) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            splitInfo_.outUbFactor = bestI;
            splitInfo_.outTailFactor = currentShapeDim % bestI;
        }
    } else {
        // use full coreNum
        int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNum, coreNum_);
        for (int64_t i = 1; i <= splitInfo_.outUbFactor; i++) {
            int64_t coreNumNew = outUbAxisExceptSplitAxis * Ops::Base::CeilDiv(currentShapeDim, i);
            if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples)) {
                splitInfo_.outUbFactor = i;
                splitInfo_.outTailFactor = currentShapeDim % i;
                break;
            }
        }
    }
    FlushBaseNumForBigDim();
    coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor) *
              outUbAxisExceptSplitAxis;
    SetRealCoreNumAndBlkFactor(coreNum);
}

/**
 * @brief 分核逻辑入口（按 TilingKey 分派）
 *
 * 每种策略用不同方式计算 block 总数并设置 realCoreNum/blkFactor/blkTailFactor：
 *   - TENSOR_MOVE：按总元素数直接均分
 *   - SMALL_SHAPE：SIMT 直读直写，按 128B 对齐均分
 *   - CUT_ONCE / CUT_TWICE：由切分因子推断 block 数，必要时回调因子到 0.9 核利用率
 *   - N_LAST_TRANSPOSE：按连续行搬移的块数分核（可退轴）
 *   - BIG_DIM：按压缩后的输出侧块数分核
 */
void TransposeNddmaTiling::CalcBlockSplitInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfo.");
    switch (tilingKey_) {
        case static_cast<int64_t>(SplitMode::TENSOR_MOVE):
            CalcBlockSplitInfoForTensorMove();
            break;
        case static_cast<int64_t>(SplitMode::SMALL_SHAPE):
            CalcBlockSplitInfoForSmallShape();
            break;
        case static_cast<int64_t>(SplitMode::CUT_ONCE):
            CalcBlockSplitInfoForCutOnce();
            break;
        case static_cast<int64_t>(SplitMode::CUT_TWICE):
            CalcBlockSplitInfoForCutTwice();
            break;
        case static_cast<int64_t>(SplitMode::BIG_DIM):
            CalcBlockSplitInfoForBigDim();
            break;
        case static_cast<int64_t>(SplitMode::N_LAST_TRANSPOSE):
            CalcBlockSplitInfoForNLastTranspose();
            break;
        default:
            break;
    }
}

/**
 * @brief 5 维扩展：把简化后 ≤5 维的 shape/perm 左填充为 NDDMA 5 维
 *
 * NDDMA DataCopy<T,5> 固定使用 5 维参数，低维 shape（dim<5）需要在左侧
 * 补 1 维（offset = 5 - dim），使轴索引与 NDDMA 硬件语义一致，例如：
 *   reducedShape=[8,64,128] (dim=3) → expanded=[1,1,8,64,128]
 *   perm 同样 +offset：reducedPerm=[2,0,1] → expandedPerm=[4,2,3]
 *
 * kernel 端 NDDMA loopSize/stride 计算统一基于 expanded* 数组。
 */
void TransposeNddmaTiling::NDDMADimExpand()
{
    int64_t offset = (shapeInfo_.dim < NDDMA_MAX_DIM_NUM) ? (NDDMA_MAX_DIM_NUM - shapeInfo_.dim) : 0;

    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        expandedPerm_[i + offset] = shapeInfo_.reducedPerm[i] + offset;
        expandedInputShape_[i + offset] = shapeInfo_.reducedInShape[i];
        expandedOutputShape_[i + offset] = shapeInfo_.reducedOutShape[i];
    }
}

/**
 * @brief UB 内 shape 计算入口（按 TilingKey 分派）
 *
 * 生成 kernel 端 NDDMA SetupLoopInfo 所需的 inUb*SrcShape/inUb*DstShape：
 *   - SMALL_SHAPE：CalcInUbShapeInfoForNoNeedCut（无需 UB 切分的简单情形）
 *   - CUT_ONCE：单轴切分，Main/Tail 两组 shape
 *   - CUT_TWICE：双轴切分，Main/InputTail/OutputTail/Tail 四组 shape
 */
void TransposeNddmaTiling::GetInUbShapeInfo()
{
    switch (tilingKey_) {
        case static_cast<int64_t>(SplitMode::SMALL_SHAPE):
            CalcInUbShapeInfoForNoNeedCut();
            break;
        case static_cast<int64_t>(SplitMode::CUT_ONCE):
            CalcInUbShapeInfoForCutOnce();
            break;
        case static_cast<int64_t>(SplitMode::CUT_TWICE):
            CalcInUbShapeInfoForCutTwice();
            break;
        default:
            break;
    }
}

/**
 * @brief CUT_TWICE 区间信息计算入口（按 TilingKey 分派）
 *
 * 仅 CUT_TWICE 需要 4 种数据区间边界：
 *   Main / InputTail / OutputTail / Tail（定义见 GetIntervalInfoForCutTwice）。
 */
void TransposeNddmaTiling::GetIntervalInfo()
{
    switch (tilingKey_) {
        case static_cast<int64_t>(SplitMode::CUT_TWICE):
            GetIntervalInfoForCutTwice();
            break;
        default:
            break;
    }
}

/**
 * @brief 计算"无需切分"场景的 UB 内 shape（SMALL_SHAPE 等路径）
 *
 * 只有一个切分轴（outCutIndex）需要以 outUbFactor/outTailFactor 切分，
 * 其余进入 block 的轴要么取完整输出 shape、要么置 1（外层循环负责）：
 *   - i > outCutIndexExpand（切分轴右侧）：完整 expandedOutputShape
 *   - i == outCutIndexExpand：Main=outUbFactor，Tail=outTailFactor
 *   - i < outCutIndexExpand：1（不进入单次 block）
 */
void TransposeNddmaTiling::CalcInUbShapeInfoForNoNeedCut()
{
    int64_t outCutIndexExpand = splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        if (i > outCutIndexExpand) {
            inUbMainDstShape_[i] = expandedOutputShape_[i];
            inUbTailDstShape_[i] = expandedOutputShape_[i];
        } else if (i == outCutIndexExpand) {
            inUbMainDstShape_[i] = splitInfo_.outUbFactor;
            inUbTailDstShape_[i] = splitInfo_.outTailFactor;
        } else {
            inUbMainDstShape_[i] = 1;
            inUbTailDstShape_[i] = 1;
        }
    }
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        inUbMainSrcShape_[expandedPerm_[i]] = inUbMainDstShape_[i];
        inUbTailSrcShape_[expandedPerm_[i]] = inUbTailDstShape_[i];
    }
}

/**
 * @brief 计算 CUT_ONCE 的 UB 内源/目标 5 维 shape
 *
 * 一个 block 中进 UB 的数据在"源侧"（输入视角）和"目标侧"（输出视角）
 * 各有一份 5 维描述：
 *   - 输出切分轴（outCutIndex 展开位）：Main = outUbFactor，Tail = outTailFactor；
 *   - 输出切分轴右侧的轴：完整 expandedOutputShape（每 block 全量出现）；
 *   - 输出切分轴左侧、且 perm[i] < inCutIndex 的轴（外部循环轴）：Main/Tail = 1
 *     （该轴不进入单次 block，由分核外层循环负责）；
 *   - src 侧通过 expandedPerm 映射（输出第 i 轴 ↔ 输入第 expandedPerm[i] 轴）得到，
 *     kernel 端 SetupLoopInfo / GetLoopParams 据此构造 NDDMA loopSize / stride。
 */
void TransposeNddmaTiling::CalcInUbShapeInfoForCutOnce()
{
    int64_t outCutIndexExpand = splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    int64_t inCutIndexExpand = splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        inUbMainDstShape_[i] = expandedOutputShape_[i];
        inUbTailDstShape_[i] = expandedOutputShape_[i];
        if (i == outCutIndexExpand) {
            inUbMainDstShape_[i] = splitInfo_.outUbFactor;
            inUbTailDstShape_[i] = splitInfo_.outTailFactor;
        } else if (i < outCutIndexExpand && expandedPerm_[i] < inCutIndexExpand) {
            inUbMainDstShape_[i] = 1;
            inUbTailDstShape_[i] = 1;
        }
    }
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        inUbMainSrcShape_[expandedPerm_[i]] = inUbMainDstShape_[i];
        inUbTailSrcShape_[expandedPerm_[i]] = inUbTailDstShape_[i];
    }
}

/**
 * @brief 计算 CUT_TWICE 的 4 组 UB 内源/目标 5 维 shape
 *
 * CUT_TWICE 需同时切输入与输出轴，一个 block 的 UB 数据根据"输入/输出
 * 切分轴各自是主块还是尾块"分为 4 种相位，每种相位对应一组 src/dst shape：
 *
 *   inUbMain*         ：输入主块 × 输出主块
 *   inUbInputTail*    ：输入尾块 × 输出主块
 *   inUbOutputTail*   ：输入主块 × 输出尾块
 *   inUbTail*         ：输入尾块 × 输出尾块
 *
 * 规则（以 src 为例）：
 *   - 输入切分轴左侧外层轴：Main/InputTail/Tail = 1（外层循环负责）
 *   - 输入切分轴本身：Main = inUbFactor；InputTail/Tail = inTailFactor
 *   - 输出切分轴（通过 perm 映射到输入侧）：按 outUbFactor/outTailFactor 覆盖对应组
 *   - 其余进入 block 的轴保持完整 expandedInputShape
 * dst 侧 = src 侧按 expandedPerm 重排后得到（NDDMA 搬入后即完成维度重排）。
 */
void TransposeNddmaTiling::CalcInUbShapeInfoForCutTwice()
{
    // 双切分场景下，对于输入，输入切分轴右侧为UB内的轴；对于输出，输出切分轴右侧非由输入确定的UB内的轴也都为UB内的轴
    for (int64_t idx = 0; idx < NDDMA_MAX_DIM_NUM; idx++) {
        inUbMainSrcShape_[idx] = expandedInputShape_[idx];
        inUbInputTailSrcShape_[idx] = expandedInputShape_[idx];
        inUbOutputTailSrcShape_[idx] = expandedInputShape_[idx];
        inUbTailSrcShape_[idx] = expandedInputShape_[idx];
        if (idx < splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            inUbMainSrcShape_[idx] = 1;
            inUbInputTailSrcShape_[idx] = 1;
            inUbTailSrcShape_[idx] = 1;
        } else if (idx == splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            inUbMainSrcShape_[idx] = splitInfo_.inUbFactor;
            inUbInputTailSrcShape_[idx] = splitInfo_.inTailFactor;
            inUbTailSrcShape_[idx] = splitInfo_.inTailFactor;
        }
        inUbOutputTailSrcShape_[idx] = inUbInputTailSrcShape_[idx];
    }
    inUbOutputTailSrcShape_[splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim] = splitInfo_.inUbFactor;
    inUbOutputTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM -
                                          shapeInfo_.dim]] = splitInfo_.outTailFactor;
    inUbMainSrcShape_[expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim]] = splitInfo_
                                                                                                        .outUbFactor;
    inUbInputTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM -
                                         shapeInfo_.dim]] = splitInfo_.outUbFactor;
    inUbTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim]] = splitInfo_
                                                                                                        .outTailFactor;
    for (int64_t idx = splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim + 1; idx < NDDMA_MAX_DIM_NUM;
         idx++) {
        if (expandedPerm_[idx] == splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            continue;
        } else {
            inUbMainSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbInputTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbOutputTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
        }
    }
    for (int64_t idx = 0; idx < NDDMA_MAX_DIM_NUM; idx++) {
        inUbMainDstShape_[idx] = inUbMainSrcShape_[expandedPerm_[idx]];
        inUbInputTailDstShape_[idx] = inUbInputTailSrcShape_[expandedPerm_[idx]];
        inUbOutputTailDstShape_[idx] = inUbOutputTailSrcShape_[expandedPerm_[idx]];
        inUbTailDstShape_[idx] = inUbTailSrcShape_[expandedPerm_[idx]];
    }
}

/**
 * @brief 计算 CUT_TWICE 的 4 种数据区间边界（Main/InputTail/OutputTail/Tail）
 *
 * 双切后所有 block 按「输入切分轴 × 输出切分轴」二维排布，可分为 4 个连续区间。
 * 区间边界以全局 block 索引表示，供 kernel ProcessBlock 与自身核范围求交集。
 *
 * 排布顺序（共 inBlocks × outBlocks × outUbLoop 个 block）：
 *   Main        ：[0,                     inBlocks×outBlocks×outUbLoop-1]
 *   InputTail   ：Main 之后，长度 = outBlocks×outUbLoop（输入轴为尾、输出逐块）
 *   OutputTail  ：InputTail 之后，长度 = inBlocks×outUbLoop（输出轴为尾）
 *   Tail        ：最后 outUbLoop 个（两轴均为尾）
 *
 * 其中：
 *   inBlocks  = expandedInputShape[inCutIdx] / inUbMainSrcShape[inCutIdx]
 *   outBlocks = expandedInputShape[outCutIdx] / inUbMainSrcShape[outCutIdx]
 *   outUbLoop = Π (expandedInputShape[i]/inUbMainSrcShape[i])，i 为非切分轴
 *
 * 若某侧 tailFactor==0，对应区间长度为 0（start/end 保持 0，kernel 跳过）。
 */
void TransposeNddmaTiling::GetIntervalInfoForCutTwice()
{
    int64_t expandedInputCutIndex = splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    int64_t inputOutputCutIndex = expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim];

    int64_t outUbLoop = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        if (i != expandedInputCutIndex && i != inputOutputCutIndex) {
            outUbLoop = outUbLoop * (expandedInputShape_[i] / inUbMainSrcShape_[i]);
        }
    }

    offsetRangeMain_.end = (expandedInputShape_[expandedInputCutIndex] / inUbMainSrcShape_[expandedInputCutIndex]) *
                               (expandedInputShape_[inputOutputCutIndex] / inUbMainSrcShape_[inputOutputCutIndex]) *
                               outUbLoop -
                           1;

    if (splitInfo_.inTailFactor != 0 && splitInfo_.outTailFactor != 0) {
        offsetRangeInputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeInputTail_.end = offsetRangeInputTail_.start +
                                    (expandedInputShape_[inputOutputCutIndex] /
                                     inUbMainSrcShape_[inputOutputCutIndex]) *
                                        outUbLoop -
                                    1;
        offsetRangeOutputTail_.start = offsetRangeInputTail_.end + 1;
        offsetRangeOutputTail_.end = offsetRangeOutputTail_.start +
                                     (expandedInputShape_[expandedInputCutIndex] /
                                      inUbMainSrcShape_[expandedInputCutIndex]) *
                                         outUbLoop -
                                     1;
        offsetRangeTail_.start = offsetRangeOutputTail_.end + 1;
        offsetRangeTail_.end = offsetRangeTail_.start + outUbLoop - 1;
    }

    if (splitInfo_.inTailFactor != 0 && splitInfo_.outTailFactor == 0) {
        offsetRangeInputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeInputTail_.end = offsetRangeInputTail_.start +
                                    (expandedInputShape_[inputOutputCutIndex] /
                                     inUbMainSrcShape_[inputOutputCutIndex]) *
                                        outUbLoop -
                                    1;
    }

    if (splitInfo_.inTailFactor == 0 && splitInfo_.outTailFactor != 0) {
        offsetRangeOutputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeOutputTail_.end = offsetRangeOutputTail_.start +
                                     (expandedInputShape_[expandedInputCutIndex] /
                                      inUbMainSrcShape_[expandedInputCutIndex]) *
                                         outUbLoop -
                                     1;
    }
}

void TransposeNddmaTiling::FillTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData.");
    tilingData_.transposeOpTiling.set_permSize(shapeInfo_.dim);
    tilingData_.transposeOpTiling.set_inCutIndex(splitInfo_.inCutIndex);
    tilingData_.transposeOpTiling.set_outCutIndex(splitInfo_.outCutIndex);
    tilingData_.transposeOpTiling.set_inUbFactor(splitInfo_.inUbFactor);
    tilingData_.transposeOpTiling.set_outUbFactor(splitInfo_.outUbFactor);
    tilingData_.transposeOpTiling.set_inTailFactor(splitInfo_.inTailFactor);
    tilingData_.transposeOpTiling.set_outTailFactor(splitInfo_.outTailFactor);
    tilingData_.transposeOpTiling.set_realCoreNum(realCoreNum_);
    tilingData_.transposeOpTiling.set_blkFactor(blkFactor_);
    tilingData_.transposeOpTiling.set_blkTailFactor(blkTailFactor_);
    tilingData_.transposeOpTiling.set_ubSize(ubSize_);

    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        inputShape_[i] = shapeInfo_.reducedInShape[i];
        outputShape_[i] = shapeInfo_.reducedOutShape[i];
        perm_[i] = shapeInfo_.reducedPerm[i];
    }
    tilingData_.transposeOpTiling.set_inputShape(inputShape_);
    tilingData_.transposeOpTiling.set_outputShape(outputShape_);
    tilingData_.transposeOpTiling.set_perm(perm_);
    tilingData_.transposeOpTiling.set_baseInShape(baseInShape_);
    tilingData_.transposeOpTiling.set_baseNddmaShape(baseNddmaShape_);
    tilingData_.transposeOpTiling.set_nddmaIdx(nddmaIdx_);
    tilingData_.transposeOpTiling.set_totalNddmaNum(totalNddmaNum_);
    tilingData_.transposeOpTiling.set_rangeMainEnd(offsetRangeMain_.end);
    tilingData_.transposeOpTiling.set_rangeInputTailStart(offsetRangeInputTail_.start);
    tilingData_.transposeOpTiling.set_rangeInputTailEnd(offsetRangeInputTail_.end);
    tilingData_.transposeOpTiling.set_rangeOutputTailStart(offsetRangeOutputTail_.start);
    tilingData_.transposeOpTiling.set_rangeOutputTailEnd(offsetRangeOutputTail_.end);
    tilingData_.transposeOpTiling.set_rangeTailStart(offsetRangeTail_.start);
    tilingData_.transposeOpTiling.set_rangeTailEnd(offsetRangeTail_.end);

    tilingData_.transposeOpTiling.set_expandedPerm(expandedPerm_);
    tilingData_.transposeOpTiling.set_expandedInputShape(expandedInputShape_);
    tilingData_.transposeOpTiling.set_expandedOutputShape(expandedOutputShape_);

    tilingData_.transposeOpTiling.set_inUbMainSrcShape(inUbMainSrcShape_);
    tilingData_.transposeOpTiling.set_inUbMainDstShape(inUbMainDstShape_);
    tilingData_.transposeOpTiling.set_inUbInputTailSrcShape(inUbInputTailSrcShape_);
    tilingData_.transposeOpTiling.set_inUbInputTailDstShape(inUbInputTailDstShape_);
    tilingData_.transposeOpTiling.set_inUbOutputTailSrcShape(inUbOutputTailSrcShape_);
    tilingData_.transposeOpTiling.set_inUbOutputTailDstShape(inUbOutputTailDstShape_);
    tilingData_.transposeOpTiling.set_inUbTailSrcShape(inUbTailSrcShape_);
    tilingData_.transposeOpTiling.set_inUbTailDstShape(inUbTailDstShape_);

    if (!isRelatedTranspose_) {
        tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                                 tilingContext_->GetRawTilingData()->GetCapacity());
        tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    }
}

void TransposeNddmaTiling::PrintTilingData()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Entering PrintTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(tilingContext_->GetNodeName(),
                "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld. \
                baseInShape[%ld] is:%ld",
                i, inputShape_[i], i, outputShape_[i], i, perm_[i], i, baseInShape_[i]);
    }
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        OP_LOGI(tilingContext_->GetNodeName(), "baseNddmaShape_[%ld] is:%ld, nddmaIdx_[%ld]:%ld", i, baseNddmaShape_[i],
                i, nddmaIdx_[i]);
    }
    OP_LOGI(tilingContext_->GetNodeName(),
            "tilingData is permSize:%ld, inCutIndex:%ld, outCutIndex:%ld, inUbFactor:%ld, outUbFactor:%ld, \
            inTailFactor:%ld, outTailFactor:%ld, realCoreNum:%ld, blkFactor:%ld, blkTailFactor:%ld, \
            ubSize:%ld, totalNddmaNum:%ld, Tiling4Transpose ends. ",
            tilingData_.transposeOpTiling.get_permSize(), tilingData_.transposeOpTiling.get_inCutIndex(),
            tilingData_.transposeOpTiling.get_outCutIndex(), tilingData_.transposeOpTiling.get_inUbFactor(),
            tilingData_.transposeOpTiling.get_outUbFactor(), tilingData_.transposeOpTiling.get_inTailFactor(),
            tilingData_.transposeOpTiling.get_outTailFactor(), tilingData_.transposeOpTiling.get_realCoreNum(),
            tilingData_.transposeOpTiling.get_blkFactor(), tilingData_.transposeOpTiling.get_blkTailFactor(),
            tilingData_.transposeOpTiling.get_ubSize(), tilingData_.transposeOpTiling.get_totalNddmaNum());
}

ge::graphStatus TransposeNddmaTiling::TilingForRelatedTranspose(gert::TilingContext* context,
                                                                TransposeOpTilingData* tilingData,
                                                                TransposeCompilerInfo* compilerInfo, ShapeInfo& opInput)
{
    OP_LOGD(context->GetNodeName(), "Start TilingForRelatedTranspose.");
    TransposeNddmaTiling tilingObject(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingObject.tilingContext_ = context;
    tilingObject.tilingData_.transposeOpTiling = *tilingData;
    tilingObject.shapeInfo_ = opInput;

    tilingObject.isRelatedTranspose_ = true;
    if (tilingObject.Init(compilerInfo->coreNum, compilerInfo->ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "tilingObject.isRelatedTranspose_: %d", tilingObject.isRelatedTranspose_);
    return tilingObject.RunTranposelTiling();
}

static ge::graphStatus TransposeTilingForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingForTranspose");
    auto compilerInfo = context->GetCompileInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compilerInfo);
    TransposeNddmaTiling tilingObject(context);
    if (tilingObject.Init(compilerInfo->coreNum, compilerInfo->ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunTranposelTiling();
}

ge::graphStatus TilingPrepareTransposeForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepareTransposeForAscendC");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((ci->coreNum <= 0),
                OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, coreNum:%ld.", ci->coreNum),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, ubSize:%ld.", ci->ubSize),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Transpose Op get coreNum:%ld, ubSize:%ld.", ci->coreNum, ci->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Transpose)
    .Tiling(TransposeTilingForAscendC)
    .TilingParse<TransposeCompilerInfo>(TilingPrepareTransposeForAscendC);

} // namespace optiling
