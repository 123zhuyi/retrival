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
 * \file masked_select_v3_tiling_arch35.cpp
 * \brief masked_select_v3_tiling tiling function
 */
#include <cmath>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "masked_select_v3_tiling.h"

namespace optiling {
const gert::Shape g_vec_1_shape = {1};
constexpr static uint64_t BLOCK_SIZE = 256;
constexpr static uint32_t DOUBLE_BUFFER = 2;
constexpr static float UB_USAGE = 1.0f;
constexpr uint32_t INT64_LENGTH_IN_INT32 = 2;    // INT64 相当于 2个int32长
constexpr static uint64_t OFFSET_SELF_SIZE = 64; // kernel侧每个block的偏移量
inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
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

class MaskedSelectV3IsRegbaseSocVersionTiling {
public:
    explicit MaskedSelectV3IsRegbaseSocVersionTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();

private:
    MaskedSelectV3TilingData tiling;
    gert::TilingContext* tilingContext = nullptr;
    // 总元素个数
    uint64_t totalLength = 0;

    // ub对齐后长度
    uint64_t totalLengthAlignedWithBlock = 0;
    uint64_t tilingKey = 1;
    uint64_t formerNum = 0;
    uint64_t formerLength = 0;
    uint64_t formerTileNum = 0;
    uint64_t formerTileLength = 0;
    uint64_t formerLastTileLength = 0;
    uint64_t tailNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailTileNum = 0;
    uint64_t tailTileLength = 0;
    uint64_t tailLastTileLength = 0;
    uint64_t numBlocks = 0;

    // 求单个元素大小
    uint64_t sizeOfDataType = 1;
    ge::DataType dataType = ge::DT_UINT64;
};
ge::graphStatus MaskedSelectV3IsRegbaseSocVersionTiling::Init()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling init start.");
    auto compileInfo = reinterpret_cast<const MaskedSelectV3CompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    uint64_t aivNum = compileInfo->aivNum;                                    // Vector核数量
    uint64_t ubSize = compileInfo->ubSize / DOUBLE_BUFFER - OFFSET_SELF_SIZE; // ubSize大小（开启double buffer之后）
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetInputDesc(0));
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    sizeOfDataType = ge::GetSizeByDataType(dataType);
    if (sizeOfDataType == 0u) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "x",
            Ops::Base::ToString(dataType).c_str(), "The dtype size of x must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    // 一个block存放的元素
    uint64_t alignNum = BLOCK_SIZE / sizeOfDataType; // 256/<8>=32
    // ub对齐后长度
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetInputShape(0));
    totalLength = EnsureNotScalar(tilingContext->GetInputShape(0)->GetStorageShape()).GetShapeSize();
    if (totalLength == 0UL) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "x",
            std::to_string(totalLength).c_str(), "input shape_size must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    totalLengthAlignedWithBlock = ((totalLength + alignNum - 1UL) / alignNum) * alignNum;
    float totalFact = static_cast<float>(sizeof(uint8_t) + sizeOfDataType * 2UL);

    // 核内拆分，策略是尽可能的填满ub_size,最后一包单独处理，
    float tmp = static_cast<float>((sizeOfDataType) * 1.0 / totalFact);
    // ub_block在ascend C中不能全部被用来作为输入输出，给了13/20系数。计算出x 一次最多能处理多少block数量
    uint64_t ubBlockNum = static_cast<uint64_t>((ubSize * UB_USAGE * tmp) / BLOCK_SIZE);
    OP_LOGD(tilingContext->GetNodeName(), "ubBlockNum: %lu, totalLength: %lu.", ubBlockNum, totalLength);
    // ub能放的元素个数
    uint64_t ubLength = ubBlockNum * alignNum;
    // block数量
    uint64_t ubNum = (totalLengthAlignedWithBlock + ubLength - 1UL) / ubLength;
    // 运行核数
    numBlocks = (ubNum > aivNum) ? aivNum : ubNum;
    tilingContext->SetBlockDim(numBlocks);
    tilingKey = sizeOfDataType;
    tilingContext->SetTilingKey(tilingKey);
    OP_CHECK_IF(tilingContext->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext->GetNodeName(), "Failed to set ScheduleMode!"),
                return ge::GRAPH_FAILED);
    // 切分流程
    formerNum = totalLength % numBlocks;
    if (formerNum == 0UL) {
        formerNum = numBlocks;
    }
    tailNum = numBlocks - formerNum;
    formerLength = (totalLength + numBlocks - 1UL) / numBlocks;
    formerTileNum = (formerLength + ubLength - 1UL) / ubLength;
    formerTileLength = ubLength;
    formerLastTileLength = formerLength % ubLength;
    if (formerLastTileLength == 0UL) {
        formerLastTileLength = ubLength;
    }
    if (tailNum > 0UL) {
        tailLength = (totalLength - formerLength * formerNum) / tailNum; // 一定可能整出
        tailTileNum = (tailLength + ubLength - 1UL) / ubLength;
        tailTileLength = ubLength;
        tailLastTileLength = tailLength % ubLength;
        if (tailLastTileLength == 0UL) {
            tailLastTileLength = ubLength;
        }
    }

    OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MaskedSelectV3IsRegbaseSocVersionTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling start.");
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetInputShape(0));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetInputShape(1));
    auto storageShape0 = tilingContext->GetInputShape(0)->GetStorageShape();
    auto storageShape1 = tilingContext->GetInputShape(1)->GetStorageShape();
    if (!(storageShape0 == storageShape1)) {
        std::string shapesStr = Shape2String(storageShape0) + " and " + Shape2String(storageShape1);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x, mask", shapesStr.c_str(),
            "The shapes of x, mask must be the same");
        return ge::GRAPH_FAILED;
    }
    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_formertileNum(formerTileNum);
    tiling.set_formertileLength(formerTileLength);
    tiling.set_formerlasttileLength(formerLastTileLength);
    tiling.set_tailNum(tailNum);
    tiling.set_tailLength(tailLength);
    tiling.set_tailtileNum(tailTileNum);
    tiling.set_tailtileLength(tailTileLength);
    tiling.set_taillasttileLength(tailLastTileLength);
    tiling.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    auto compileInfo = reinterpret_cast<const MaskedSelectV3CompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    uint32_t sysWorkspaceSize = compileInfo->workSpaceSize;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaces入参所需workspace的块数。当前限制使用一块。
    size_t usrSize = totalLengthAlignedWithBlock * sizeOfDataType + numBlocks * 64UL;
    OP_LOGD(tilingContext->GetNodeName(), "usrWorkspaceSize: %lu.", usrSize);
    currentWorkspace[0] =
        usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间框架来申请并管理。
    TilingDataPrint();
    OP_LOGD(tilingContext->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
}
void MaskedSelectV3IsRegbaseSocVersionTiling::TilingDataPrint()
{
    OP_LOGD(tilingContext->GetNodeName(), "sizeOfDataType: %lu.", sizeOfDataType);
    OP_LOGD(tilingContext->GetNodeName(), "formerNum: %lu.", tiling.get_formerNum());
    OP_LOGD(tilingContext->GetNodeName(), "formerLength: %lu.", tiling.get_formerLength());
    OP_LOGD(tilingContext->GetNodeName(), "formerTileNum: %lu.", tiling.get_formertileNum());
    OP_LOGD(tilingContext->GetNodeName(), "formerTileLength: %lu.", tiling.get_formertileLength());
    OP_LOGD(tilingContext->GetNodeName(), "formerLastTileLength: %lu.", tiling.get_formerlasttileLength());
    OP_LOGD(tilingContext->GetNodeName(), "tailNum: %lu.", tiling.get_tailNum());
    OP_LOGD(tilingContext->GetNodeName(), "tailLength: %lu.", tiling.get_tailLength());
    OP_LOGD(tilingContext->GetNodeName(), "tailTileNum: %lu.", tiling.get_tailtileNum());
    OP_LOGD(tilingContext->GetNodeName(), "tailTileLength: %lu.", tiling.get_tailtileLength());
    OP_LOGD(tilingContext->GetNodeName(), "tailLastTileLength: %lu.", tiling.get_taillasttileLength());
}
ge::graphStatus TilingForMaskedSelectV3IsRegbaseSocVersion(gert::TilingContext* context)
{
    MaskedSelectV3IsRegbaseSocVersionTiling tilingObject(context);
    if (context == nullptr) {
        OP_LOGE("TilingForMaskedSelectV3IsRegbaseSocVersion", "Tiling context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Init tiling object return failed.");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}
} // namespace optiling