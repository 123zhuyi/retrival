/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <sstream>
#include "register/op_impl_registry.h"
#include "conversion/batch_to_space_nd/op_kernel/arch35/batch_to_space_nd_tiling_data.h"
#include "conversion/batch_to_space_nd/op_kernel/arch35/batch_to_space_nd_tiling_key.h"
#include "batch_to_space_nd_tiling_dual_side_tiling.h"
#include "platform/platform_ascendc.h"
#include "op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"
#include "common/inc/op_host/math_log.h"

namespace optiling {
// 属性、索引
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_BLOCK_SHAPE = 1;
static constexpr size_t INPUT_IDX_CROPS = 2;
static constexpr size_t MIN_BLOCK_SHAPE_DIM = 1;
static constexpr size_t MIN_X_RANK = MIN_BLOCK_SHAPE_DIM + 1;
static constexpr size_t BLOCK_SHAPE_RANK = 1;
static constexpr size_t CROPS_RANK = 2;
static constexpr size_t CROPS_DIM_NUM_1 = 2;

// 公共常量
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_INT8,   ge::DT_UINT8,  ge::DT_INT16,  ge::DT_UINT16,    ge::DT_INT32,
    ge::DT_UINT32, ge::DT_INT64,  ge::DT_UINT64, ge::DT_BF16,      ge::DT_FLOAT16,
    ge::DT_FLOAT,  ge::DT_DOUBLE, ge::DT_BOOL,   ge::DT_COMPLEX32, ge::DT_COMPLEX64,
};
// 固定轴数量：batch轴 + remain轴
static constexpr size_t NUM_FIXED_AXES = 2;

// 大尾轴模板 常量
// BUFFER分割数量
static constexpr uint32_t LARGE_C_BUFFER_NUM = 2;
static constexpr uint32_t LARGE_C_MAX_BUFFER_SIZE = 64 * 1024U;
// 能切分的最外层的轴，正负数均可
static constexpr int32_t LARGE_C_OUTMOST_CUT_AXIS = -3;

// 小尾轴模板 常量
// BUFFER分割数量
static constexpr uint32_t SMALL_C_BUFFER_NUM = 2;
// 输入输出各分一半
static constexpr uint32_t SMALL_C_BUFFER_FACTOR = 2;
// 最大UB大小
static constexpr uint32_t SMALL_C_MAX_BUFFER_SIZE = 64 * 1024U;
// 最小UB大小
static constexpr uint32_t SMALL_C_MIN_BUFFER_SIZE = 4 * 1024U;
static constexpr double MIN_USED_CORES_RATIO = 0.6;
static constexpr int16_t MAX_TILING_TIME = 4;
static constexpr uint32_t UB_TILE_SIZE_REDUCTION_FACTOR = 2; // 迭代缩减UB tile尺寸的减半因子
// 每块预留大小
static constexpr uint32_t SMALL_C_RESERVE_BUFFER_SIZE = 256U;
// 被压缩的轴数量
static constexpr uint16_t SMALL_C_AXIS_COMPACT_CNT = 1U;

// SIMT 常量
static constexpr size_t MIN_RANK_FOR_SIMT = 6;
static constexpr int64_t MAX_UINT32_NUM = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t SIMT_DCACHE_SIZE = 32 * 1024U;
static constexpr uint32_t SIMT_MAX_UB_SIZE = 64 * 1024U;
static constexpr uint32_t SIMT_BUFFER_NUM = 2;
static constexpr uint32_t SIMT_THREAD_FACTOR = 1;

class BatchToSpaceNDTiling {
private:
    /* data */
    // soc info
    uint32_t ubSize_{0};
    uint32_t ubBlockSize_{0};
    uint32_t coreNum_{0};
    uint32_t cacheLineSize_{0};
    int32_t ubBlockElements_{0};
    int32_t cacheLineElements_{0};
    uint32_t vRegSize_{0};
    uint32_t simtMaxThreads_{0};

    // tiling key param
    uint8_t mode_{0};
    uint8_t blockShapeDimNum_{0};
    bool isBigShape_{false};

    // 输入参数
    int32_t dSize_{0};
    int64_t xShapeSize_{0};
    int64_t yShapeSize_{0};
    size_t originBlockShapeDim_{0};
    B2SNDInput originInput_;
    B2SNDInput mergedInput_;

    // 中间计算结果
    // 实际核数
    uint32_t realCoreNum_{0};

    // tiling context
    gert::TilingContext* context_;

    uint64_t lastDimSize_{0};

public:
    explicit BatchToSpaceNDTiling(gert::TilingContext* context) : context_(context) {};
    ~BatchToSpaceNDTiling() {};

    ge::graphStatus DoTiling();

private:
    // 参数检查，数据获取
    ge::graphStatus ParamCheck();
    ge::graphStatus GetSocInfo();
    ge::graphStatus CheckX();
    ge::graphStatus CheckBlockShape();
    ge::graphStatus CheckCrops();
    ge::graphStatus CheckY();
    ge::graphStatus MergeInput();

    // tiling 计算
    ge::graphStatus DoOpTiling();
    ge::graphStatus Tiling4LargeC();
    [[maybe_unused]] ge::graphStatus Tiling4SmallC();
    ge::graphStatus Tiling4SIMT();

    // 辅助函数
    // LargeC
    ge::graphStatus MoveAlignTilingBlock(uint32_t maxUBElements, const std::vector<uint64_t>& ubFactorAlign,
                                         const std::vector<uint64_t>& leftAlign, const std::vector<uint64_t>& dimValue,
                                         int32_t minCutAxis, B2SNDLargeCTilingData* tilingData);
    // SmallC
    void SmallCSetInput(B2SNDSmallCTilingData* tilingData);
    std::array<size_t, MAX_EXPAND_RANK> SmallCComputeOutputAxisPerm();
    int16_t SmallCComputeInputNeedAlign(uint64_t oriInShape[], uint64_t croppedInShape[]);
    // SIMT

    // 公共方法
    template <typename T>
    inline T CeilAlignBlockElement(T elementCount);

    // 打印
    void ShowBaseTilingData();
    void ShowLargeCTilingData();
    void ShowSmallCTilingData();
    void ShowSIMTTilingData();
};

ge::graphStatus BatchToSpaceNDTiling::DoTiling()
{
    // 校验属性
    CHECK_RET_SUCC(ParamCheck());
    // soc信息获取
    CHECK_RET_SUCC(GetSocInfo());

    CHECK_RET_SUCC(DoOpTiling());

    const uint64_t tilingKey = GET_TPL_TILING_KEY(mode_, blockShapeDimNum_, isBigShape_);
    OP_LOGI(context_, "tilingKey is %lu, mode %u, blockShapeDimNum %u, isBigShape %d", tilingKey, mode_,
            blockShapeDimNum_, isBigShape_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum_);
    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline T BatchToSpaceNDTiling::CeilAlignBlockElement(T elementCount)
{
    return Ops::Base::CeilAlign(elementCount, static_cast<T>(ubBlockElements_));
}

template <typename T>
static std::string ArrayToString(const T* v, size_t size)
{
    std::ostringstream oss;
    oss << "[";
    if (size > 0) {
        for (size_t i = 0; i < size - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[size - 1];
    }
    oss << "]";
    return oss.str();
}

void BatchToSpaceNDTiling::ShowBaseTilingData()
{
    if (unlikely(mergedInput_.rank <= NUM_FIXED_AXES)) {
        return;
    }
    // 输入信息
    OP_LOGI(context_, "input: x_shape %s, block_shape %s, crops %s, y_shape %s, data type size %d",
            ArrayToString(mergedInput_.inShape, mergedInput_.rank).c_str(),
            ArrayToString(mergedInput_.blockShape, mergedInput_.rank - NUM_FIXED_AXES).c_str(),
            ArrayToString(*mergedInput_.crops, (mergedInput_.rank - NUM_FIXED_AXES) * CROPS_DIM_NUM_1).c_str(),
            ArrayToString(mergedInput_.outShape, mergedInput_.rank).c_str(), dSize_);
    // soc 信息
    OP_LOGI(context_, "soc info: ubSize %lu, coreNum %u, cacheLineSize %lu, ubBlockSize %lu, simtMaxThreads %lu",
            ubSize_, coreNum_, cacheLineSize_, ubBlockSize_, simtMaxThreads_);
    // 中间计算结果
    OP_LOGI(context_, "middle data: realCoreNum %lu", realCoreNum_);
}

void BatchToSpaceNDTiling::ShowLargeCTilingData()
{
    ShowBaseTilingData();
    auto tilingData = context_->GetTilingData<B2SNDLargeCTilingData>();
    OP_LOGI(context_, "tiling data: ubAxis %lu, ubFactor %lu, outputBufferSize %lu", tilingData->ubAxis,
            tilingData->ubFactor, tilingData->outputBufferSize);
    OP_LOGI(context_, "\t: totalCount %u, perCoreCount %u", tilingData->totalCount, tilingData->perCoreCount);
}

void BatchToSpaceNDTiling::ShowSmallCTilingData()
{
    ShowBaseTilingData();
    auto tilingData = context_->GetTilingData<B2SNDSmallCTilingData>();
    OP_LOGI(context_, "tiling data: oriInShape %s, croppedInShape %s, crops %s",
            ArrayToString(tilingData->oriInShape, mergedInput_.rank + blockShapeDimNum_).c_str(),
            ArrayToString(tilingData->croppedInShape, mergedInput_.rank + blockShapeDimNum_).c_str(),
            ArrayToString(*tilingData->crops, blockShapeDimNum_ * CROPS_DIM_NUM_1).c_str());
    OP_LOGI(context_, "\t: coreNum %u, inUbAxis %u, outUbAxis %u, inUbFactor %u, outUbFactor %u", tilingData->coreNum,
            tilingData->inUbAxis, tilingData->outUbAxis, tilingData->inUbFactor, tilingData->outUbFactor);
    OP_LOGI(context_, "\t: ubTotalCount %u, ubPerCount %u, ubTileSize %u", tilingData->ubTotalCount,
            tilingData->ubPerCount, tilingData->ubTileSize);
}

void BatchToSpaceNDTiling::ShowSIMTTilingData()
{
    ShowBaseTilingData();
    auto tilingData = context_->GetTilingData<B2SNDSimtTilingData>();
    OP_LOGI(context_, "tiling data: totalBlock %lu, mainCoreBlock %lu", tilingData->totalBlock,
            tilingData->mainCoreBlock);
    OP_LOGI(context_, "\t: needCoreNum %u, mainCoreNum %u, blockSize %u, tailBlockSize %u", tilingData->needCoreNum,
            tilingData->mainCoreNum, tilingData->blockSize, tilingData->tailBlockSize);
}

ge::graphStatus BatchToSpaceNDTiling::CheckX()
{
    // 获取x
    auto inputValueDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
    auto inputDataType = inputValueDesc->GetDataType();
    dSize_ = ge::GetSizeByDataType(inputDataType);
    OP_CHECK_IF(dSize_ <= 0,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
                                          ge::TypeUtils::DataTypeToSerialString(inputDataType).c_str(),
                                          Ops::Math::JoinArray<Ops::Math::ItemConj::OR>(VALUE_DATA_TYPE_ALL).c_str()),
                return ge::GRAPH_FAILED);

    // 校验输入shape
    auto xInputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    auto xShape = xInputShape->GetStorageShape();
    originInput_.rank = xShape.GetDimNum();
    OP_CHECK_IF(originInput_.rank < MIN_X_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                         std::to_string(originInput_.rank).c_str(),
                                                         "The shape dim of x must be within the range [2, 8]"),
                return ge::GRAPH_FAILED);

    // 校验溢出
    xShapeSize_ = xShape.GetShapeSize();
    OP_CHECK_IF(xShapeSize_ <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "x", "overflow", "less than INT64_MAX"),
                return ge::GRAPH_FAILED);

    // 获取 shape
    for (size_t i = 0; i < originInput_.rank; ++i) {
        originInput_.inShape[i] = xShape.GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDTiling::CheckBlockShape()
{
    // 获取block shape
    auto bsInputShape = context_->GetInputShape(INPUT_IDX_BLOCK_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bsInputShape);

    // 校验shape
    auto bsShape = bsInputShape->GetStorageShape();
    size_t bsRank = bsShape.GetDimNum();
    OP_CHECK_IF(bsRank != BLOCK_SHAPE_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "block_shape", std::to_string(bsRank).c_str(),
                                             std::to_string(BLOCK_SHAPE_RANK).c_str()),
                return ge::GRAPH_FAILED);

    // 获取 block_shape 值
    gert::Shape blockShape;
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context_, INPUT_IDX_BLOCK_SHAPE, blockShape),
                OP_LOGE(context_, "get block_shape tensor data failed"), return ge::GRAPH_FAILED);

    // 校验维度
    originBlockShapeDim_ = blockShape.GetDimNum();
    OP_CHECK_IF(originBlockShapeDim_ < MIN_BLOCK_SHAPE_DIM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "block_shape",
                                                      std::to_string(originBlockShapeDim_).c_str(),
                                                      "All axes of block_shape must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        originBlockShapeDim_ >= originInput_.rank,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x and block_shape",
                                                  Ops::Math::Join(originInput_.rank, originBlockShapeDim_).c_str(),
                                                  "The shape dim of x must be > that of block_shape"),
        return ge::GRAPH_FAILED);

    // block_shape 为正数
    for (size_t i = 0; i < originBlockShapeDim_; ++i) {
        OP_CHECK_IF(blockShape[i] <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_shape",
                                                          std::to_string(blockShape[i]).c_str(),
                                                          "Each value of block_shape must be positive"),
                    return ge::GRAPH_FAILED);
        originInput_.blockShape[i] = static_cast<uint64_t>(blockShape[i]);
    }
    int64_t block_size = blockShape.GetShapeSize();
    OP_CHECK_IF(block_size <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_shape", "overflow",
                                                      "The value of block_shape must be less than INT64_MAX"),
                return ge::GRAPH_FAILED);

    // block_shape 能被batch整除
    int64_t batch = originInput_.inShape[0];
    OP_CHECK_IF(
        (batch % block_size) != 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(batch).c_str(),
                                              "0th axis of x must be exactly divided by product of block_shape"),
        return ge::GRAPH_FAILED);

    originInput_.outShape[0] = Ops::Base::FloorDiv(batch, block_size);
    yShapeSize_ = originInput_.outShape[0];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDTiling::CheckCrops()
{
    // 获取 crops
    auto cropsInputShape = context_->GetInputShape(INPUT_IDX_CROPS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cropsInputShape);
    auto cropsShape = cropsInputShape->GetStorageShape();
    size_t cropsRank = cropsShape.GetDimNum();
    OP_CHECK_IF(
        cropsRank != CROPS_RANK,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "crops", std::to_string(cropsRank).c_str(),
                                                 "The shape dim of crops must be 2"),
        return ge::GRAPH_FAILED);

    // crops 和 block_shape shape 相等
    OP_CHECK_IF(
        cropsShape.GetDim(0) != static_cast<int64_t>(originBlockShapeDim_),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "crops and block_shape",
                                               Ops::Math::Join(cropsShape.GetDim(0), originBlockShapeDim_).c_str(),
                                               "0th axis of crops must be equal to 0th axis of block_shape"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(cropsShape.GetDim(1) != static_cast<int64_t>(CROPS_DIM_NUM_1),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "crops",
                                                      std::to_string(cropsShape.GetDim(1)).c_str(),
                                                      "1st axis of crops must be equal to 2"),
                return ge::GRAPH_FAILED);

    // 获取 crops 值
    gert::Shape crops;
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context_, INPUT_IDX_CROPS, crops),
                OP_LOGE(context_, "get crops tensor data failed"), return ge::GRAPH_FAILED);

    // crops >= 0
    auto dims = crops.GetDimNum();
    for (size_t i = 0; i < dims; ++i) {
        OP_CHECK_IF(
            crops[i] < 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "crops", std::to_string(crops[i]).c_str(),
                                                  "Each value of crops must be non-negative"),
            return ge::GRAPH_FAILED);
        originInput_.crops[i / CROPS_DIM_NUM_1][i % CROPS_DIM_NUM_1] = static_cast<uint64_t>(crops[i]);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDTiling::CheckY()
{
    // spatial shape
    size_t i = 1;
    for (size_t j = 0; j < originBlockShapeDim_; ++i, ++j) {
        // x shape 已判断不会溢出，block_shape已判断整除batch，这里不会翻转
        int64_t cropedShape = originInput_.inShape[i] * originInput_.blockShape[j];
        // crops 是否溢出
        OP_CHECK_IF(originInput_.crops[j][0] > std::numeric_limits<int64_t>::max() - originInput_.crops[j][1],
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "crops", "overflow", "less than INT64_MAX"),
                    return ge::GRAPH_FAILED);
        // y shape 不能为负
        int64_t crops = originInput_.crops[j][0] + originInput_.crops[j][1];
        OP_CHECK_IF(
            cropedShape < crops,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y",
                                                  std::to_string(cropedShape - crops).c_str(), "y has negative axes"),
            return ge::GRAPH_FAILED);
        originInput_.outShape[i] = cropedShape - crops;
        // 比x shape size 小，不会溢出
        yShapeSize_ *= originInput_.outShape[i];
    }

    // remain shape
    for (; i < originInput_.rank; ++i) {
        originInput_.outShape[i] = originInput_.inShape[i];
        yShapeSize_ *= originInput_.outShape[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDTiling::MergeInput()
{
    size_t oldIdx = 0;
    size_t newIdx = 0;
    // batch不变
    mergedInput_.inShape[newIdx] = originInput_.inShape[oldIdx];
    mergedInput_.outShape[newIdx++] = originInput_.outShape[oldIdx++];

    // 合 block_shape/crops
    uint64_t x = 1, y = 1, crops0 = 0, crops1 = 0;
    uint64_t remainShape = 1;
    for (; oldIdx <= originBlockShapeDim_; ++oldIdx) {
        size_t j = oldIdx - 1;
        // 合并
        x *= originInput_.inShape[oldIdx];
        y *= originInput_.outShape[oldIdx];
        // crops如果上一维非0，则本维必为0，扩大 y[i] 倍
        // 如果上一维为0，则赋值为本维
        crops0 = crops0 * originInput_.outShape[oldIdx] + originInput_.crops[j][0];
        crops1 = crops1 * originInput_.outShape[oldIdx] + originInput_.crops[j][1];

        // 当前 block_shape 为 1
        if (originInput_.blockShape[j] == 1) {
            // block_shape 非最后一维，且下一个 crops 为 0，合并到下一维
            if (oldIdx < originBlockShapeDim_ && originInput_.crops[oldIdx][0] == 0 &&
                originInput_.crops[oldIdx][1] == 0) {
                continue;
            }
            // block_shape 最后一维，且当前 crops 为 0，合并到 remain_shape
            if (oldIdx == originBlockShapeDim_ && crops0 == 0 && crops1 == 0) {
                remainShape = x;
                continue;
            }
        }
        // 写入
        mergedInput_.inShape[newIdx] = x;
        mergedInput_.blockShape[newIdx - 1] = originInput_.blockShape[j];
        mergedInput_.crops[newIdx - 1][0] = crops0;
        mergedInput_.crops[newIdx - 1][1] = crops1;
        mergedInput_.outShape[newIdx++] = y;
        // 初始化
        x = 1;
        crops0 = 0;
        crops1 = 0;
        y = 1;
    }

    // space维度全被合并，保留1维
    if (newIdx == 1) {
        mergedInput_.inShape[newIdx] = 1;
        mergedInput_.outShape[newIdx++] = 1;
        mergedInput_.blockShape[0] = 1;
        mergedInput_.crops[0][0] = 0;
        mergedInput_.crops[0][1] = 0;
    }

    // 合 remain_shape
    for (; oldIdx < originInput_.rank; ++oldIdx) {
        remainShape *= originInput_.inShape[oldIdx];
    }
    mergedInput_.inShape[newIdx] = remainShape;
    mergedInput_.outShape[newIdx] = remainShape;
    mergedInput_.rank = newIdx + 1;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDTiling::ParamCheck()
{
    // 获取并校验参数
    CHECK_RET_SUCC(CheckX());
    CHECK_RET_SUCC(CheckBlockShape());
    CHECK_RET_SUCC(CheckCrops());
    CHECK_RET_SUCC(CheckY());
    // 合轴
    return MergeInput();
}

ge::graphStatus BatchToSpaceNDTiling::MoveAlignTilingBlock(uint32_t maxUBElements,
                                                           const std::vector<uint64_t>& ubFactorAlign,
                                                           const std::vector<uint64_t>& leftAlign,
                                                           const std::vector<uint64_t>& dimValue, int32_t minCutAxis,
                                                           B2SNDLargeCTilingData* tilingData)
{
    uint64_t totalCount = 1;
    uint32_t restFactor = maxUBElements;
    uint32_t ubAxis = 0, ubFactor = 0;

    int32_t rank = static_cast<int32_t>(dimValue.size());
    int32_t tailIdx = rank - 1;
    int32_t minIdx = std::max(0, (rank + minCutAxis) % rank);
    for (int32_t i = tailIdx; i >= minIdx; --i) {
        ubAxis = i;
        // 尾轴要ub block对齐
        uint64_t dimVal = (i == tailIdx) ? CeilAlignBlockElement(dimValue[i]) : dimValue[i];
        // 塞得下就直接算，但最小维要做对齐处理
        if (restFactor >= dimVal && i != minIdx) {
            // 剩余空间充足，按维度值对齐后赋值
            ubFactor = static_cast<uint32_t>(dimVal);
            restFactor /= ubFactor;
            continue;
        }
        // 塞不下
        // 头尾要单独处理
        // 对齐值不能为0，内部错误，无需上报
        OP_CHECK_IF(ubFactorAlign[i] == 0, OP_LOGE(context_, "the ub factor align must be non-zero"),
                    return ge::GRAPH_FAILED);
        uint64_t lastLeftAlign = leftAlign[i] % ubFactorAlign[i];
        uint64_t head = 0, tail = 0;
        if (lastLeftAlign + dimVal > ubFactorAlign[i]) {
            head = lastLeftAlign == 0 ? 0 : ubFactorAlign[i] - lastLeftAlign;
            tail = (lastLeftAlign + dimVal) % ubFactorAlign[i];
        } else {
            head = dimVal;
        }
        uint64_t middle = dimVal > (head + tail) ? dimVal - (head + tail) : 0;
        OP_LOGI(context_, "split axis %u into head %lu, middle %lu, tail %lu", ubAxis, head, middle, tail);
        restFactor = static_cast<uint32_t>(std::min(static_cast<uint64_t>(restFactor), dimVal));
        if (restFactor >= ubFactorAlign[i]) {
            // 超过对齐值，按对齐值倍数分组
            ubFactor = Ops::Base::FloorAlign(restFactor, static_cast<uint32_t>(ubFactorAlign[i]));
            // 中间部分对齐分组长度
            totalCount = Ops::Base::CeilDiv(middle, static_cast<uint64_t>(ubFactor));
            // 头尾单独分组
            totalCount += (head > 0) + (tail > 0);
        } else {
            // 不足对齐值，每个分组内再按 ub factor 分组
            ubFactor = restFactor;
            totalCount = Ops::Base::CeilDiv(ubFactorAlign[i], static_cast<uint64_t>(ubFactor)) *
                         Ops::Base::CeilDiv(middle, ubFactorAlign[i]);
            totalCount += Ops::Base::CeilDiv(head, static_cast<uint64_t>(ubFactor));
            totalCount += Ops::Base::CeilDiv(tail, static_cast<uint64_t>(ubFactor));
        }
        break;
    }
    // 非切分轴不需要对齐
    for (int32_t i = static_cast<int32_t>(ubAxis) - 1; i >= 0; --i) {
        totalCount *= dimValue[i];
    }

    tilingData->ubAxis = ubAxis;
    tilingData->totalCount = totalCount;
    tilingData->ubFactor = ubFactor;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDTiling::Tiling4LargeC()
{
    // tiling key
    mode_ = TPL_MODE_LARGE_C;

    // tiling data
    auto tilingData = context_->GetTilingData<B2SNDLargeCTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->input = mergedInput_;

    // ub 大小
    tilingData->outputBufferSize = ubSize_ / LARGE_C_BUFFER_NUM;
    tilingData->outputBufferSize = std::min(tilingData->outputBufferSize, LARGE_C_MAX_BUFFER_SIZE);
    uint32_t maxUBElements = tilingData->outputBufferSize / static_cast<uint32_t>(dSize_);

    // 各轴 ub factor 对齐值
    std::vector<uint64_t> ubFactorAlign;
    ubFactorAlign.resize(mergedInput_.rank);
    // batch 不对齐
    ubFactorAlign[0] = 1;
    // space 对齐 block shape
    std::copy(mergedInput_.blockShape, mergedInput_.blockShape + mergedInput_.rank - NUM_FIXED_AXES,
              ubFactorAlign.begin() + 1);
    // remain shape 对齐 ub block
    ubFactorAlign[mergedInput_.rank - 1] = ubBlockElements_;

    std::vector<uint64_t> dimValue = std::vector<uint64_t>(mergedInput_.outShape,
                                                           mergedInput_.outShape + mergedInput_.rank);
    std::vector<uint64_t> leftAlign{};
    leftAlign.resize(mergedInput_.rank);
    // batch 不对齐
    leftAlign[0] = 0;
    // space 要按crop前的大小对齐 block shape
    for (size_t i = 1; i < mergedInput_.rank - 1; ++i) {
        // 前面补齐crop前的部分
        leftAlign[i] = mergedInput_.crops[i - 1][0];
    }
    // remain shape 对齐 ub block
    leftAlign[mergedInput_.rank - 1] = 0;

    // 分块
    auto ret = MoveAlignTilingBlock(maxUBElements, ubFactorAlign, leftAlign, dimValue, LARGE_C_OUTMOST_CUT_AXIS,
                                    tilingData);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "large C tiling failed"), return ret);

    // 分核
    tilingData->perCoreCount = Ops::Base::CeilDiv(tilingData->totalCount, static_cast<uint64_t>(coreNum_));
    realCoreNum_ = Ops::Base::CeilDiv(tilingData->totalCount, tilingData->perCoreCount);

    // 打印 tiling data
    ShowLargeCTilingData();
    return ge::GRAPH_SUCCESS;
}

void BatchToSpaceNDTiling::SmallCSetInput(B2SNDSmallCTilingData* tilingData)
{
    // 展开 x shape
    size_t rank = blockShapeDimNum_ + mergedInput_.rank;
    // batch
    std::copy(mergedInput_.blockShape, mergedInput_.blockShape + blockShapeDimNum_, tilingData->oriInShape);
    tilingData->oriInShape[blockShapeDimNum_] = mergedInput_.outShape[0];
    // space + remain
    std::copy(mergedInput_.inShape + 1, mergedInput_.inShape + mergedInput_.rank,
              tilingData->oriInShape + blockShapeDimNum_ + 1);

    // 预crop
    for (size_t i = 0; i < blockShapeDimNum_; ++i) {
        uint64_t& bs = mergedInput_.blockShape[i];
        uint64_t& x = mergedInput_.inShape[i + 1];
        uint64_t yStart = mergedInput_.crops[i][0];
        uint64_t yEnd = x * bs - mergedInput_.crops[i][1] - 1;
        uint64_t xStart = yStart / bs; // 截取前的输出对应 x 的起始坐标
        uint64_t xEnd = yEnd / bs;     // 截取前的输出对应 x 的结束坐标
        // 截取后的 space
        tilingData->croppedInShape[blockShapeDimNum_ + 1 + i] = xEnd + 1 - xStart;
        if (xStart == xEnd) {
            // 截取后的 block shape
            tilingData->croppedInShape[i] = yEnd + 1 - yStart;
        } else {
            // 不截取 block shape
            tilingData->croppedInShape[i] = bs;
        }
    }
    // y batch
    tilingData->croppedInShape[blockShapeDimNum_] = mergedInput_.outShape[0];
    // remain
    tilingData->croppedInShape[rank - 1] = mergedInput_.inShape[mergedInput_.rank - 1];

    // crops
    std::copy(*mergedInput_.crops, (*mergedInput_.crops) + blockShapeDimNum_ * CROPS_DIM_NUM_1, *(tilingData->crops));
}

std::array<size_t, MAX_EXPAND_RANK> BatchToSpaceNDTiling::SmallCComputeOutputAxisPerm()
{
    // 输出轴映射输入轴
    size_t rank = blockShapeDimNum_ + mergedInput_.rank;
    std::array<size_t, MAX_EXPAND_RANK> yAxisPerm{};
    // batch
    yAxisPerm[0] = blockShapeDimNum_;
    // remain
    yAxisPerm[rank - 1] = rank - 1;
    for (size_t i = 0; i < blockShapeDimNum_; ++i) {
        // block shape
        yAxisPerm[2 + i * 2] = i;
        // space
        yAxisPerm[1 + i * 2] = blockShapeDimNum_ + 1 + i;
    }
    return yAxisPerm;
}

int16_t BatchToSpaceNDTiling::SmallCComputeInputNeedAlign(uint64_t oriInShape[], uint64_t croppedInShape[])
{
    size_t rank = blockShapeDimNum_ + mergedInput_.rank;
    // 从倒数第N个满足条件（有预裁剪）的轴开始需要对齐
    int16_t restCompactCnt = SMALL_C_AXIS_COMPACT_CNT;
    for (int16_t i = rank - 1; i >= 0; --i) {
        if (oriInShape[i] == croppedInShape[i]) {
            continue;
        }
        if ((restCompactCnt--) == 0) {
            return i;
        }
    }
    return INVALID_ALIGN_AXIS;
}

ge::graphStatus BatchToSpaceNDTiling::Tiling4SmallC()
{
    // tiling key
    mode_ = TPL_MODE_SMALL_C;
    blockShapeDimNum_ = mergedInput_.rank - NUM_FIXED_AXES;

    // tiling data
    auto tilingData = context_->GetTilingData<B2SNDSmallCTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    // 处理输入数据
    SmallCSetInput(tilingData);

    // 轴排列
    auto yAxisPerm = SmallCComputeOutputAxisPerm();
    // 是否对齐
    int16_t xNeedAlignAxis = SmallCComputeInputNeedAlign(tilingData->oriInShape, tilingData->croppedInShape);

    // 可用UB大小
    uint32_t validBufSize = ubSize_ / SMALL_C_BUFFER_NUM;
    tilingData->ubTileSize = std::min(validBufSize / SMALL_C_BUFFER_FACTOR, SMALL_C_MAX_BUFFER_SIZE);
    uint32_t inputElements = (tilingData->ubTileSize - SMALL_C_RESERVE_BUFFER_SIZE) / dSize_;

    // 输入输出双切分
    auto tiling = DualSideTiling(context_, ubBlockElements_, tilingData->croppedInShape, yAxisPerm.data(),
                                 xNeedAlignAxis, blockShapeDimNum_ + mergedInput_.rank);
    tiling.DoTiling(inputElements);
    tilingData->inUbAxis = tiling.inAxis;
    tilingData->inUbFactor = tiling.inFactor;
    tilingData->outUbAxis = tiling.outAxis;
    tilingData->outUbFactor = tiling.outFactor;
    tilingData->ubTotalCount = tiling.totalCount;
    int16_t tempCnt = 0;
    while (static_cast<double>(tilingData->ubTotalCount) / static_cast<double>(coreNum_) < MIN_USED_CORES_RATIO) {
        tempCnt++;
        if (tempCnt >= MAX_TILING_TIME) {
            break;
        }
        inputElements = inputElements / UB_TILE_SIZE_REDUCTION_FACTOR;
        auto tiling1 = DualSideTiling(context_, ubBlockElements_, tilingData->croppedInShape, yAxisPerm.data(),
                                      xNeedAlignAxis, blockShapeDimNum_ + mergedInput_.rank);
        tiling1.DoTiling(inputElements);
        if (tiling1.totalCount > coreNum_) {
            break;
        }
        tilingData->inUbAxis = tiling1.inAxis;
        tilingData->inUbFactor = tiling1.inFactor;
        tilingData->outUbAxis = tiling1.outAxis;
        tilingData->outUbFactor = tiling1.outFactor;
        tilingData->ubTotalCount = tiling1.totalCount;
    }

    // 分核
    tilingData->ubPerCount = Ops::Base::CeilDiv(tilingData->ubTotalCount, static_cast<uint64_t>(coreNum_));
    realCoreNum_ = Ops::Base::CeilDiv(tilingData->ubTotalCount, tilingData->ubPerCount);
    tilingData->coreNum = realCoreNum_;

    // 打印 tiling data
    ShowSmallCTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDTiling::Tiling4SIMT()
{
    // tiling key
    mode_ = TPL_MODE_SIMT;
    isBigShape_ = xShapeSize_ > MAX_UINT32_NUM;

    // 可用UB大小
    OP_CHECK_IF((ubSize_ < SIMT_DCACHE_SIZE), OP_LOGE(context_, "ub size invalid"), return ge::GRAPH_FAILED);
    uint32_t validBufSize = (ubSize_ - SIMT_DCACHE_SIZE) / SIMT_BUFFER_NUM;
    uint32_t usedBufSize = std::min(validBufSize, SIMT_MAX_UB_SIZE);
    auto ret = context_->SetLocalMemorySize(usedBufSize * SIMT_BUFFER_NUM);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "set local memory size failed."), return ret);

    // tiling data
    auto tilingData = context_->GetTilingData<B2SNDSimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->input = mergedInput_;

    // 按1维分块
    tilingData->blockSize = usedBufSize / dSize_;
    // 向下对齐线程数
    uint32_t threadNum = simtMaxThreads_ / SIMT_THREAD_FACTOR;
    tilingData->blockSize = std::max(1U, Ops::Base::FloorAlign(tilingData->blockSize, threadNum));
    // 防止输出为空，至少要1
    tilingData->totalBlock = std::max(
        1UL, Ops::Base::CeilDiv(static_cast<uint64_t>(yShapeSize_), static_cast<uint64_t>(tilingData->blockSize)));
    tilingData->tailBlockSize = static_cast<uint32_t>(static_cast<uint64_t>(yShapeSize_) % tilingData->blockSize);
    if (tilingData->tailBlockSize == 0) {
        tilingData->tailBlockSize = yShapeSize_ == 0 ? 0 : tilingData->blockSize;
    }

    // 均分分核
    realCoreNum_ = tilingData->totalBlock > coreNum_ ? coreNum_ : static_cast<uint32_t>(tilingData->totalBlock);
    realCoreNum_ = realCoreNum_ > 0 ? realCoreNum_ : 1U;
    tilingData->needCoreNum = realCoreNum_;
    tilingData->mainCoreBlock = Ops::Base::CeilDiv(tilingData->totalBlock, static_cast<uint64_t>(realCoreNum_));
    tilingData->mainCoreNum = static_cast<uint32_t>(tilingData->totalBlock % realCoreNum_);
    if (tilingData->mainCoreNum == 0) {
        tilingData->mainCoreNum = realCoreNum_;
    }

    // 打印 tiling data
    ShowSIMTTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDTiling::DoOpTiling()
{
    cacheLineElements_ = cacheLineSize_ / dSize_;
    ubBlockElements_ = ubBlockSize_ / dSize_;

    // 输出空tensor
    if (yShapeSize_ == 0) {
        return Tiling4SIMT();
    }

    // 维度较多
    if (mergedInput_.rank >= MIN_RANK_FOR_SIMT) {
        return Tiling4SIMT();
    }

    // 尾轴 >= cacheline
    if (mergedInput_.inShape[mergedInput_.rank - 1] >= static_cast<uint64_t>(cacheLineElements_)) {
        return Tiling4LargeC();
    }

    if (dSize_ == 1) {
        return Tiling4SIMT();
    }
    return Tiling4SmallC();
}

ge::graphStatus BatchToSpaceNDTiling::GetSocInfo()
{
    // 获取soc信息, 如ub大小, core数等
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    realCoreNum_ = coreNum_;
    OP_CHECK_IF((coreNum_ == 0U), OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF((ubSize == 0U), OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((ubSize > MAX_UINT32_NUM), OP_LOGE(context_, "ub size not support"), return ge::GRAPH_FAILED);
    ubSize_ = ubSize;
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF((cacheLineSize_ == 0U), OP_LOGE(context_, "Failed to get cache line size."), return ge::GRAPH_FAILED);
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ == 0U), OP_LOGE(context_, "Failed to get ub block size."), return ge::GRAPH_FAILED);
    vRegSize_ = Ops::Base::GetVRegSize(context_);
    OP_CHECK_IF((vRegSize_ == 0U), OP_LOGE(context_, "Failed to get vector register size."), return ge::GRAPH_FAILED);
    simtMaxThreads_ = Ops::Base::GetSimtMaxThreadNum(context_);
    OP_CHECK_IF((simtMaxThreads_ == 0U), OP_LOGE(context_, "Failed to get simt thread num."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4BatchToSpaceND(gert::TilingContext* context)
{
    // DoTiling
    BatchToSpaceNDTiling tiling{context};
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBatchToSpaceND([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BatchToSpaceND)
    .Tiling(Tiling4BatchToSpaceND)
    .TilingInputsDataDependency({INPUT_IDX_BLOCK_SHAPE, INPUT_IDX_CROPS})
    .TilingParse<B2SNDCompileInfo>(TilingPrepareForBatchToSpaceND);
} // namespace optiling
