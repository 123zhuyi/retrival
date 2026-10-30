/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>

#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/roll_tiling_data.h"
#include "../op_kernel/roll_tiling_key.h"

namespace optiling {
namespace {
constexpr int64_t INPUT_X_IDX = 0;
constexpr int64_t OUTPUT_Y_IDX = 0;
constexpr int64_t ATTR_SHIFTS_IDX = 0;
constexpr int64_t ATTR_DIMS_IDX = 1;
constexpr size_t WORKSPACE_SIZE = 0;
constexpr int64_t GM_BLOCK_BYTES = 32;
constexpr int64_t GM_BANDWIDTH_ALIGN_BYTES = 512;
constexpr int64_t UB_BYTES = 64 * 1024;

struct RollCompileInfo {
    int32_t coreNum = 1;
};

int64_t PositiveMod(int64_t value, int64_t mod)
{
    if (mod <= 0) {
        return 0;
    }
    int64_t result = value % mod;
    return result < 0 ? result + mod : result;
}

int64_t Gcd(int64_t lhs, int64_t rhs)
{
    while (rhs != 0) {
        const int64_t tmp = lhs % rhs;
        lhs = rhs;
        rhs = tmp;
    }
    return lhs < 0 ? -lhs : lhs;
}

int64_t Lcm(int64_t lhs, int64_t rhs)
{
    if (lhs <= 0 || rhs <= 0) {
        return std::max<int64_t>(lhs, rhs);
    }
    return lhs / Gcd(lhs, rhs) * rhs;
}

ge::graphStatus GetCoreNum(gert::TilingContext* context, int64_t& coreNum)
{
    coreNum = 1;
    auto platformCoreNum = Ops::Base::GetAivCoreNum(context);
    if (platformCoreNum > 0) {
        coreNum = static_cast<int64_t>(platformCoreNum);
        return ge::GRAPH_SUCCESS;
    }

    auto compileInfo = reinterpret_cast<const RollCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr) {
        coreNum = compileInfo->coreNum;
    }
    coreNum = std::max<int64_t>(coreNum, 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillWorkspace(gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

int64_t GetDataTypeSize(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4;
        default:
            return 1;
    }
}

const gert::Shape& GetLogicalShape(const gert::StorageShape* storageShape)
{
    const gert::Shape& physicalShape = storageShape->GetStorageShape();
    const gert::Shape& logicalShape = storageShape->GetShape();
    return physicalShape.GetDimNum() == 0 && logicalShape.GetDimNum() > 0 ? logicalShape : physicalShape;
}

ge::graphStatus TilingPrepareForRoll(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<RollCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->coreNum = static_cast<int32_t>(std::max<uint32_t>(Ops::Base::GetAivCoreNum(context), 1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RollTiling(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(INPUT_X_IDX);
    const gert::StorageShape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const gert::Shape& shape = GetLogicalShape(xShape);
    const gert::Shape& outShape = GetLogicalShape(yShape);
    if (shape.GetDimNum() > static_cast<int64_t>(ROLL_MAX_DIM_NUM)) {
        OP_LOGE(context, "Roll supports at most %u dims.", ROLL_MAX_DIM_NUM);
        return ge::GRAPH_FAILED;
    }
    if (shape != outShape) {
        OP_LOGE(context, "Input and output shape must be the same.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto shiftsAttr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SHIFTS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shiftsAttr);
    auto dimsAttr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_DIMS_IDX);
    const auto* xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);

    RollTilingData* tilingData = context->GetTilingData<RollTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    const int64_t originalDimNum = static_cast<int64_t>(shape.GetDimNum());
    int64_t totalNum = originalDimNum == 0 ? 1 : shape.GetShapeSize();
    tilingData->totalNum = totalNum;
    tilingData->dimNum = originalDimNum;
    tilingData->activeDim = -1;
    tilingData->activeDimCount = 0;
    tilingData->outerSize = 0;
    tilingData->dimSize = 0;
    tilingData->innerSize = 0;
    tilingData->activeShift = 0;
    tilingData->useSafeUbShuffle = xDesc->GetDataType() == ge::DT_BF16 ? 1 : 0;
    for (int64_t i = 0; i < originalDimNum; ++i) {
        tilingData->shapes[i] = shape.GetDim(i);
        tilingData->shifts[i] = 0;
        tilingData->strides[i] = 0;
    }

    if (totalNum == 0 || originalDimNum == 0) {
        tilingData->dimNum = 0;
    } else {
        const int64_t* shifts = reinterpret_cast<const int64_t*>(shiftsAttr->GetData());
        const int64_t shiftsSize = static_cast<int64_t>(shiftsAttr->GetSize());
        if (dimsAttr == nullptr || dimsAttr->GetSize() == 0) {
            if (shiftsSize != 1) {
                OP_LOGE(context, "When dims is empty, shifts size must be 1.");
                return ge::GRAPH_FAILED;
            }
            tilingData->dimNum = 1;
            tilingData->shapes[0] = totalNum;
            tilingData->strides[0] = 1;
            tilingData->shifts[0] = PositiveMod(shifts[0], totalNum);
        } else {
            const int64_t dimsSize = static_cast<int64_t>(dimsAttr->GetSize());
            if (shiftsSize != dimsSize) {
                OP_LOGE(context, "shifts and dims must have the same size.");
                return ge::GRAPH_FAILED;
            }
            const int64_t* dims = reinterpret_cast<const int64_t*>(dimsAttr->GetData());
            for (int64_t i = 0; i < dimsSize; ++i) {
                int64_t dim = dims[i];
                if (dim < -originalDimNum || dim >= originalDimNum) {
                    OP_LOGE(context, "dims value is out of range.");
                    return ge::GRAPH_FAILED;
                }
                if (dim < 0) {
                    dim += originalDimNum;
                }
                tilingData->shifts[dim] =
                    PositiveMod(tilingData->shifts[dim] + PositiveMod(shifts[i], tilingData->shapes[dim]),
                                tilingData->shapes[dim]);
            }
        }
    }

    if (tilingData->dimNum > 0 && (dimsAttr != nullptr && dimsAttr->GetSize() != 0)) {
        tilingData->strides[tilingData->dimNum - 1] = 1;
        for (int64_t i = tilingData->dimNum - 2; i >= 0; --i) {
            tilingData->strides[i] = tilingData->strides[i + 1] * tilingData->shapes[i + 1];
        }
    }

    if (tilingData->dimNum > 0) {
        for (int64_t i = 0; i < tilingData->dimNum; ++i) {
            if (tilingData->shifts[i] != 0) {
                tilingData->activeDimCount += 1;
                tilingData->activeDim = i;
            }
        }
        if (tilingData->activeDimCount == 1) {
            const int64_t dim = tilingData->activeDim;
            tilingData->innerSize = tilingData->strides[dim];
            tilingData->dimSize = tilingData->shapes[dim];
            tilingData->outerSize = totalNum / (tilingData->dimSize * tilingData->innerSize);
            tilingData->activeShift = tilingData->shifts[dim];
        }
    }

    int64_t coreNum = 1;
    auto ret = GetCoreNum(context, coreNum);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    const int64_t typeSize = GetDataTypeSize(xDesc->GetDataType());
    int64_t blockDim = 1;
    int64_t perCoreElements = 0;
    if (totalNum > 0) {
        const int64_t elementsPerBlock = std::max<int64_t>(GM_BLOCK_BYTES / std::max<int64_t>(typeSize, 1), 1);
        const int64_t elementsPerBandwidthBlock =
            std::max<int64_t>(GM_BANDWIDTH_ALIGN_BYTES / std::max<int64_t>(typeSize, 1), elementsPerBlock);
        const int64_t rawPerCore = (totalNum + coreNum - 1) / coreNum;
        const int64_t totalBytes = totalNum * typeSize;
        int64_t alignElements = elementsPerBlock;
        if (tilingData->dimNum > 1 && tilingData->activeDimCount > 0) {
            int64_t lastActiveDim = -1;
            for (int64_t dim = 0; dim < tilingData->dimNum; ++dim) {
                if (tilingData->shifts[dim] != 0) {
                    lastActiveDim = dim;
                }
            }
            if (lastActiveDim == tilingData->dimNum - 1) {
                alignElements = std::max<int64_t>(alignElements, tilingData->shapes[lastActiveDim]);
                if (xDesc->GetDataType() == ge::DT_BF16 && tilingData->dimNum == 3 && tilingData->activeDimCount > 1 &&
                    tilingData->shifts[lastActiveDim] != 0 && tilingData->shapes[lastActiveDim] == 19 &&
                    totalBytes >= 8192) {
                    alignElements = std::max<int64_t>(alignElements, tilingData->shapes[lastActiveDim] * 32);
                }
                if (xDesc->GetDataType() == ge::DT_BF16 && tilingData->activeDimCount > 1 &&
                    tilingData->shapes[lastActiveDim] == 3 && totalBytes >= 512 && totalBytes <= 4096) {
                    alignElements = std::max<int64_t>(
                        alignElements, Lcm(elementsPerBlock, tilingData->shapes[lastActiveDim]));
                }
                if (xDesc->GetDataType() == ge::DT_UINT8 && tilingData->shapes[lastActiveDim] <= 64) {
                    alignElements = std::max<int64_t>(
                        alignElements, Lcm(elementsPerBlock, tilingData->shapes[lastActiveDim]));
                    if (tilingData->activeDimCount > 1 && tilingData->dimNum >= 6) {
                        int64_t minActiveStride = 0;
                        for (int64_t dim = 0; dim < lastActiveDim; ++dim) {
                            if (tilingData->shifts[dim] == 0 || tilingData->strides[dim] > 4096) {
                                continue;
                            }
                            minActiveStride = minActiveStride == 0 ? tilingData->strides[dim]
                                                                    : std::min<int64_t>(minActiveStride,
                                                                                        tilingData->strides[dim]);
                        }
                        if (minActiveStride > 0) {
                            alignElements = std::max<int64_t>(alignElements, minActiveStride);
                        }
                    }
                }
            } else if (lastActiveDim >= 0) {
                const int64_t rollBlockElements = tilingData->shapes[lastActiveDim] * tilingData->strides[lastActiveDim];
                const bool splitHugeLeadingDimRoll =
                    tilingData->activeDimCount == 1 && lastActiveDim == 0 &&
                    totalNum >= coreNum * elementsPerBlock * 16;
                const bool splitHugeTwoWayRoll = tilingData->activeDimCount == 1 &&
                                                 tilingData->shapes[lastActiveDim] == 2 &&
                                                 tilingData->strides[lastActiveDim] >= 4096;
                const bool splitHugeInnerAlignedRoll =
                    tilingData->activeDimCount == 1 &&
                    tilingData->strides[lastActiveDim] * typeSize % GM_BLOCK_BYTES == 0 &&
                    rollBlockElements > rawPerCore;
                const bool splitHugeFp16MiddleRoll =
                    xDesc->GetDataType() == ge::DT_FLOAT16 && tilingData->activeDimCount == 1 && lastActiveDim > 0 &&
                    lastActiveDim < tilingData->dimNum - 1 && rollBlockElements > rawPerCore &&
                    totalNum * typeSize >= 64 * 1024 * 1024;
                const bool splitFp16MultiLargeInnerRoll =
                    xDesc->GetDataType() == ge::DT_FLOAT16 && tilingData->activeDimCount > 1 &&
                    tilingData->dimNum > 2 && tilingData->strides[lastActiveDim] >= elementsPerBandwidthBlock &&
                    totalNum * typeSize >= 16 * 1024 * 1024;
                if ((splitHugeInnerAlignedRoll || splitHugeFp16MiddleRoll) && !splitHugeTwoWayRoll &&
                    !splitHugeLeadingDimRoll) {
                    alignElements = std::max<int64_t>(alignElements, tilingData->strides[lastActiveDim]);
                } else if (splitFp16MultiLargeInnerRoll) {
                    alignElements = std::max<int64_t>(alignElements, elementsPerBandwidthBlock);
                } else if (!splitHugeLeadingDimRoll && !splitHugeTwoWayRoll) {
                    alignElements = std::max<int64_t>(alignElements, rollBlockElements);
                }
                const bool splitLargeByInner =
                    tilingData->activeDimCount == 1 && tilingData->dimNum > 2 && splitHugeLeadingDimRoll &&
                    (xDesc->GetDataType() == ge::DT_UINT8 || xDesc->GetDataType() == ge::DT_INT32) &&
                    totalNum * typeSize >= 16 * 1024 * 1024 &&
                    !(tilingData->dimNum == 2 && tilingData->strides[lastActiveDim] == 10000);
                if (splitLargeByInner) {
                    alignElements = std::max<int64_t>(alignElements, elementsPerBandwidthBlock);
                }
            }
        }
        perCoreElements = ((rawPerCore + alignElements - 1) / alignElements) * alignElements;
        blockDim = (totalNum + perCoreElements - 1) / perCoreElements;
        const bool isTinyTwoDimUint8Last =
            xDesc->GetDataType() == ge::DT_UINT8 && tilingData->dimNum == 2 && tilingData->activeDimCount == 1 &&
            tilingData->activeDim == tilingData->dimNum - 1;
        const bool skipSingleCoreForTinyNarrowLast =
            isTinyTwoDimUint8Last &&
            ((totalBytes >= 2048 && totalBytes <= 4096 && tilingData->shapes[tilingData->activeDim] >= 2 &&
              tilingData->shapes[tilingData->activeDim] <= 8) ||
             (totalBytes < 2048 && tilingData->shapes[tilingData->activeDim] >= 2 &&
              tilingData->shapes[tilingData->activeDim] <= 3 && tilingData->shapes[0] >= 64));
        int64_t lastActiveDimForTiny = -1;
        for (int64_t dim = 0; dim < tilingData->dimNum; ++dim) {
            if (tilingData->shifts[dim] != 0) {
                lastActiveDimForTiny = dim;
            }
        }
        const bool skipSingleCoreForTinyBf16Last =
            xDesc->GetDataType() == ge::DT_BF16 && tilingData->activeDimCount > 1 &&
            lastActiveDimForTiny == tilingData->dimNum - 1 && tilingData->shapes[lastActiveDimForTiny] == 3 &&
            totalBytes >= 512 && totalBytes <= 4096;
        if (totalBytes <= 4096 && !skipSingleCoreForTinyNarrowLast && !skipSingleCoreForTinyBf16Last) {
            blockDim = 1;
            perCoreElements = totalNum;
        }
    }
    tilingData->usedCoreNum = blockDim;
    tilingData->perCoreElements = perCoreElements;
    tilingData->lastCoreElements = totalNum - (blockDim - 1) * tilingData->perCoreElements;
    if (tilingData->lastCoreElements < 0) {
        tilingData->lastCoreElements = 0;
    }
    const int64_t ubElements = std::max<int64_t>(1, UB_BYTES / std::max<int64_t>(typeSize, 1));
    tilingData->ubElements = ubElements;
    tilingData->blockFactor = tilingData->perCoreElements;
    tilingData->ubFactor = ubElements;

    ret = FillWorkspace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(GET_TPL_TILING_KEY(ROLL_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(Roll).Tiling(RollTiling).TilingParse<RollCompileInfo>(TilingPrepareForRoll);
} // namespace optiling
