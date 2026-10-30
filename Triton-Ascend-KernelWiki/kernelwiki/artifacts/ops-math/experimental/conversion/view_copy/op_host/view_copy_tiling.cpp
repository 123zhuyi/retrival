/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file view_copy_tiling.cpp
 * \brief ViewCopy tiling implementation.
 */

#include <limits>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/view_copy_tiling_data.h"
#include "../op_kernel/view_copy_tiling_key.h"

namespace optiling {
namespace {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t INDEX_DST = 0;
constexpr size_t INDEX_DST_SIZE = 1;
constexpr size_t INDEX_DST_STRIDE = 2;
constexpr size_t INDEX_DST_OFFSET = 3;
constexpr size_t INDEX_SRC = 4;
constexpr size_t INDEX_SRC_SIZE = 5;
constexpr size_t INDEX_SRC_STRIDE = 6;
constexpr size_t INDEX_SRC_OFFSET = 7;
constexpr size_t INDEX_Y = 0;
constexpr int64_t VIEWCOPY_DATA_BLOCK_BYTES = 64;

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static gert::Shape GetStorageShape(const gert::StorageShape* storageShape) { return storageShape->GetStorageShape(); }

static bool CalcNumel(const gert::Shape& shape, int64_t& numel)
{
    const size_t dimNum = shape.GetDimNum();
    if (dimNum == 0 || dimNum > VIEWCOPY_MAX_DIMS) {
        return false;
    }

    numel = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return false;
        }
        if (dim == 0) {
            numel = 0;
            return true;
        }
        if (numel > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        numel *= dim;
    }
    return true;
}

static bool GetTensorNumel(gert::TilingContext* context, size_t index, int64_t& numel)
{
    const gert::StorageShape* shape = context->GetInputShape(index);
    if (shape == nullptr) {
        return false;
    }
    return CalcNumel(GetStorageShape(shape), numel);
}

static bool GetOutputNumel(gert::TilingContext* context, size_t index, int64_t& numel)
{
    const gert::StorageShape* shape = context->GetOutputShape(index);
    if (shape == nullptr) {
        return false;
    }
    return CalcNumel(GetStorageShape(shape), numel);
}

static bool ValidateMetaTensorShape(gert::TilingContext* context, size_t index, int64_t expectedNumel, const char* name)
{
    int64_t actualNumel = 0;
    OP_CHECK_IF(!GetTensorNumel(context, index, actualNumel), OP_LOGE(context, "Failed to get %s shape.", name),
                return false);
    OP_CHECK_IF(actualNumel != expectedNumel, OP_LOGE(context, "%s shape length must match the view rank.", name),
                return false);
    return true;
}

static bool GetMetaTypeBytes(gert::TilingContext* context, int64_t& metaTypeBytes)
{
    auto dstSizeDesc = context->GetInputDesc(INDEX_DST_SIZE);
    auto dstStrideDesc = context->GetInputDesc(INDEX_DST_STRIDE);
    auto dstOffsetDesc = context->GetInputDesc(INDEX_DST_OFFSET);
    auto srcSizeDesc = context->GetInputDesc(INDEX_SRC_SIZE);
    auto srcStrideDesc = context->GetInputDesc(INDEX_SRC_STRIDE);
    auto srcOffsetDesc = context->GetInputDesc(INDEX_SRC_OFFSET);
    if (dstSizeDesc == nullptr || dstStrideDesc == nullptr || dstOffsetDesc == nullptr || srcSizeDesc == nullptr ||
        srcStrideDesc == nullptr || srcOffsetDesc == nullptr) {
        OP_LOGE(context, "Failed to get ViewCopy metadata descriptors.");
        return false;
    }

    const ge::DataType dtype = dstSizeDesc->GetDataType();
    if (dtype != dstStrideDesc->GetDataType() || dtype != dstOffsetDesc->GetDataType() ||
        dtype != srcSizeDesc->GetDataType() || dtype != srcStrideDesc->GetDataType() ||
        dtype != srcOffsetDesc->GetDataType()) {
        OP_LOGE(context, "All ViewCopy metadata inputs must use the same dtype.");
        return false;
    }

    switch (dtype) {
        case ge::DT_INT32:
            metaTypeBytes = 4;
            return true;
        case ge::DT_INT64:
            metaTypeBytes = 8;
            return true;
        default:
            OP_LOGE(context, "ViewCopy metadata dtype must be int32 or int64.");
            return false;
    }
}

static int64_t GetTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4;
        case ge::DT_INT64:
            return 8;
        default:
            return 0;
    }
}

static uint64_t GetTilingKeyByTypeBytes(int64_t typeBytes)
{
    switch (typeBytes) {
        case 1:
            return VIEWCOPY_TPL_SCH_MODE_0;
        case 2:
            return VIEWCOPY_TPL_SCH_MODE_1;
        case 4:
            return VIEWCOPY_TPL_SCH_MODE_2;
        case 8:
            return VIEWCOPY_TPL_SCH_MODE_3;
        default:
            return VIEWCOPY_TPL_SCH_MODE_0;
    }
}

static int64_t CeilDiv(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

static int64_t CalcDstSpan(const ViewCopyTilingData* tiling)
{
    if (tiling->metadataReady == 0 || tiling->ndim <= 0 || tiling->ndim > VIEWCOPY_MAX_DIMS) {
        return 0;
    }

    int64_t span = 1;
    for (int64_t dim = 0; dim < tiling->ndim; ++dim) {
        const int64_t size = tiling->sizes[dim];
        const int64_t stride = tiling->dstStrides[dim];
        if (size <= 0 || stride <= 0) {
            return 0;
        }
        const int64_t extent = size - 1;
        if (extent == 0) {
            continue;
        }
        if (stride > (std::numeric_limits<int64_t>::max() - span) / extent) {
            return 0;
        }
        span += extent * stride;
    }
    return span;
}

static bool IsDstMaybeOverlapped(const ViewCopyTilingData* tiling)
{
    if (tiling->metadataReady == 0 || tiling->ndim <= 0 || tiling->ndim > VIEWCOPY_MAX_DIMS) {
        return true;
    }

    int64_t sortedSizes[VIEWCOPY_MAX_DIMS] = {0};
    int64_t sortedStrides[VIEWCOPY_MAX_DIMS] = {0};
    int64_t dimCount = 0;
    for (int64_t dim = 0; dim < tiling->ndim; ++dim) {
        const int64_t size = tiling->sizes[dim];
        const int64_t stride = tiling->dstStrides[dim];
        if (size <= 0) {
            return false;
        }
        if (size == 1) {
            continue;
        }
        if (stride <= 0) {
            return true;
        }

        int64_t pos = dimCount;
        while (pos > 0 && sortedStrides[pos - 1] > stride) {
            sortedSizes[pos] = sortedSizes[pos - 1];
            sortedStrides[pos] = sortedStrides[pos - 1];
            --pos;
        }
        sortedSizes[pos] = size;
        sortedStrides[pos] = stride;
        ++dimCount;
    }

    int64_t requiredSpan = 1;
    for (int64_t idx = 0; idx < dimCount; ++idx) {
        if (sortedStrides[idx] < requiredSpan) {
            return true;
        }
        const int64_t extent = sortedSizes[idx] - 1;
        if (extent <= 0) {
            continue;
        }
        if (sortedStrides[idx] > (std::numeric_limits<int64_t>::max() - requiredSpan) / extent) {
            return true;
        }
        requiredSpan += extent * sortedStrides[idx];
    }
    return false;
}

static bool IsThreeDimTranspose2(const ViewCopyTilingData* tiling)
{
    return tiling->metadataReady != 0 && tiling->dstOverlap == 0 && tiling->ndim == 3 && tiling->sizes[2] == 2 &&
           tiling->srcStrides[1] == 2 && tiling->srcStrides[2] == 1 && tiling->dstStrides[1] == 1 &&
           tiling->dstStrides[2] == tiling->sizes[1];
}

static bool GetContiguousSliceInfo(const ViewCopyTilingData* tiling, int64_t& sliceDim, int64_t& sliceNum,
                                   int64_t& sliceElems)
{
    if (tiling->metadataReady == 0 || tiling->dstOverlap != 0 || tiling->ndim < 4 || tiling->sizes[0] <= 0 ||
        tiling->srcStrides[tiling->ndim - 1] != 1 || tiling->dstStrides[tiling->ndim - 1] != 1) {
        return false;
    }

    sliceElems = tiling->sizes[tiling->ndim - 1];
    int64_t dim = tiling->ndim - 2;
    for (; dim >= 0; --dim) {
        const int64_t expectedSrcStride = tiling->sizes[dim + 1] * tiling->srcStrides[dim + 1];
        const int64_t expectedDstStride = tiling->sizes[dim + 1] * tiling->dstStrides[dim + 1];
        if (tiling->srcStrides[dim] != expectedSrcStride || tiling->dstStrides[dim] != expectedDstStride) {
            break;
        }
        sliceElems *= tiling->sizes[dim];
    }
    sliceDim = dim + 1;
    if (sliceDim <= 0 || sliceDim >= tiling->ndim || sliceElems <= 0) {
        return false;
    }

    sliceNum = 1;
    for (int64_t outerDim = 0; outerDim < sliceDim; ++outerDim) {
        sliceNum *= tiling->sizes[outerDim];
    }
    return sliceNum > 0 && sliceNum * sliceElems == tiling->viewNum;
}

static int64_t GetThreeDimChunkElems(int64_t typeBytes, int64_t outerNum, int64_t midSize, int64_t coreNum)
{
    (void)outerNum;
    (void)coreNum;
    const int64_t baseBlockElems = 64 / typeBytes;
    int64_t chunkElems = baseBlockElems;
    if (typeBytes == 4) {
        chunkElems = baseBlockElems * 2;
        if (midSize >= 128) {
            chunkElems = baseBlockElems * 4;
        }
    }
    return chunkElems;
}

static int64_t GetContiguousSliceChunkElems(int64_t typeBytes, int64_t sliceNum, int64_t sliceElems, int64_t coreNum)
{
    const int64_t blockElems = 64 / typeBytes;
    if (sliceNum >= coreNum || sliceElems <= blockElems) {
        return sliceElems;
    }
    if (typeBytes != 4) {
        return sliceElems;
    }

    int64_t chunksPerSlice = CeilDiv(coreNum, sliceNum);
    if (chunksPerSlice <= 1) {
        return sliceElems;
    }

    int64_t chunkElems = CeilDiv(sliceElems, chunksPerSlice);
    chunkElems = CeilDiv(chunkElems, blockElems) * blockElems;
    return chunkElems < sliceElems ? chunkElems : sliceElems;
}

static int64_t GetBlockWork(const ViewCopyTilingData* tiling, int64_t typeBytes, int64_t coreNum, int64_t dstStorageNum)
{
    if (tiling->dstOverlap == 1) {
        const int64_t blockElems = VIEWCOPY_DATA_BLOCK_BYTES / typeBytes;
        if (tiling->dstSpan <= 0 || blockElems <= 0) {
            return 1;
        }
        const int64_t firstBlock = tiling->dstOffset / blockElems;
        const int64_t lastBlock = (tiling->dstOffset + tiling->dstSpan - 1) / blockElems;
        return lastBlock - firstBlock + 1;
    }
    if (IsThreeDimTranspose2(tiling)) {
        const int64_t outerNum = tiling->sizes[0];
        const int64_t midSize = tiling->sizes[1];
        const int64_t rowWork = outerNum;
        const int64_t chunkElems = GetThreeDimChunkElems(typeBytes, outerNum, midSize, coreNum);
        const int64_t chunkWork = outerNum * CeilDiv(midSize, chunkElems);
        return (rowWork * 4 < coreNum * 3) ? chunkWork : rowWork;
    }
    int64_t sliceDim = 0;
    int64_t sliceNum = 0;
    int64_t sliceElems = 0;
    if (GetContiguousSliceInfo(tiling, sliceDim, sliceNum, sliceElems)) {
        const int64_t chunkElems = GetContiguousSliceChunkElems(typeBytes, sliceNum, sliceElems, coreNum);
        return sliceNum * CeilDiv(sliceElems, chunkElems);
    }
    return tiling->metadataReady != 0 && tiling->viewNum > 0 ? tiling->viewNum : dstStorageNum;
}

template <typename T>
static bool ReadMetaTensor(gert::TilingContext* context, size_t index, int64_t expectedNumel, int64_t* values,
                           const char* name)
{
    const gert::Tensor* tensor = context->GetInputTensor(index);
    if (tensor == nullptr) {
        OP_LOGE(context, "Failed to get ViewCopy metadata tensor %s.", name);
        return false;
    }
    const T* data = tensor->GetData<T>();
    if (data == nullptr) {
        OP_LOGE(context, "Failed to get ViewCopy metadata tensor data %s.", name);
        return false;
    }
    for (int64_t i = 0; i < expectedNumel; ++i) {
        values[i] = static_cast<int64_t>(data[i]);
    }
    return true;
}

static bool ReadMetaTensorByType(gert::TilingContext* context, size_t index, int64_t expectedNumel,
                                 int64_t metaTypeBytes, int64_t* values, const char* name)
{
    if (metaTypeBytes == 4) {
        return ReadMetaTensor<int32_t>(context, index, expectedNumel, values, name);
    }
    if (metaTypeBytes == 8) {
        return ReadMetaTensor<int64_t>(context, index, expectedNumel, values, name);
    }
    return false;
}

static bool FillConstMetadata(gert::TilingContext* context, ViewCopyTilingData* tiling, int64_t ndim,
                              int64_t metaTypeBytes)
{
    int64_t srcSize[VIEWCOPY_MAX_DIMS] = {0};
    int64_t dstOffset[1] = {0};
    int64_t srcOffset[1] = {0};
    if (!ReadMetaTensorByType(context, INDEX_DST_SIZE, ndim, metaTypeBytes, tiling->sizes, "dst_size") ||
        !ReadMetaTensorByType(context, INDEX_SRC_SIZE, ndim, metaTypeBytes, srcSize, "src_size") ||
        !ReadMetaTensorByType(context, INDEX_DST_STRIDE, ndim, metaTypeBytes, tiling->dstStrides, "dst_stride") ||
        !ReadMetaTensorByType(context, INDEX_SRC_STRIDE, ndim, metaTypeBytes, tiling->srcStrides, "src_stride") ||
        !ReadMetaTensorByType(context, INDEX_DST_OFFSET, 1, metaTypeBytes, dstOffset, "dst_storage_offset") ||
        !ReadMetaTensorByType(context, INDEX_SRC_OFFSET, 1, metaTypeBytes, srcOffset, "src_storage_offset")) {
        return false;
    }

    int64_t viewNum = 1;
    for (int64_t dim = 0; dim < ndim; ++dim) {
        if (tiling->sizes[dim] != srcSize[dim] || tiling->sizes[dim] <= 0 || tiling->srcStrides[dim] <= 0 ||
            tiling->dstStrides[dim] <= 0 || viewNum > std::numeric_limits<int64_t>::max() / tiling->sizes[dim]) {
            return false;
        }
        viewNum *= tiling->sizes[dim];
    }
    tiling->srcOffset = srcOffset[0];
    tiling->dstOffset = dstOffset[0];
    tiling->viewNum = viewNum;
    tiling->metadataReady = 1;
    tiling->dstSpan = CalcDstSpan(tiling);
    return true;
}

} // namespace

static ge::graphStatus ViewCopyTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "ViewCopy tiling context is nullptr."), return ge::GRAPH_FAILED);

    ViewCopyTilingData* tiling = context->GetTilingData<ViewCopyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ViewCopyTilingData), 0, sizeof(ViewCopyTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    auto dstDesc = context->GetInputDesc(INDEX_DST);
    auto srcDesc = context->GetInputDesc(INDEX_SRC);
    auto yDesc = context->GetOutputDesc(INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    if (dstDesc->GetDataType() != srcDesc->GetDataType() || dstDesc->GetDataType() != yDesc->GetDataType()) {
        OP_LOGE(context, "The dtype of dst, src and y must be the same.");
        return ge::GRAPH_FAILED;
    }

    const int64_t typeBytes = GetTypeBytes(dstDesc->GetDataType());
    OP_CHECK_IF(typeBytes <= 0, OP_LOGE(context, "Unsupported ViewCopy dtype."), return ge::GRAPH_FAILED);

    int64_t ndim = 0;
    OP_CHECK_IF(!GetTensorNumel(context, INDEX_DST_SIZE, ndim), OP_LOGE(context, "Failed to get ViewCopy view rank."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ndim <= 0 || ndim > VIEWCOPY_MAX_DIMS,
                OP_LOGE(context, "ViewCopy only supports tensor rank from 1 to 8."), return ge::GRAPH_FAILED);

    if (!ValidateMetaTensorShape(context, INDEX_DST_STRIDE, ndim, "dst_stride") ||
        !ValidateMetaTensorShape(context, INDEX_SRC_SIZE, ndim, "src_size") ||
        !ValidateMetaTensorShape(context, INDEX_SRC_STRIDE, ndim, "src_stride") ||
        !ValidateMetaTensorShape(context, INDEX_DST_OFFSET, 1, "dst_storage_offset") ||
        !ValidateMetaTensorShape(context, INDEX_SRC_OFFSET, 1, "src_storage_offset")) {
        return ge::GRAPH_FAILED;
    }

    int64_t metaTypeBytes = 0;
    if (!GetMetaTypeBytes(context, metaTypeBytes)) {
        return ge::GRAPH_FAILED;
    }

    int64_t dstStorageNum = 0;
    OP_CHECK_IF(!GetOutputNumel(context, INDEX_Y, dstStorageNum),
                OP_LOGE(context, "Failed to get ViewCopy output storage size."), return ge::GRAPH_FAILED);
    int64_t srcStorageNum = 0;
    OP_CHECK_IF(!GetTensorNumel(context, INDEX_SRC, srcStorageNum),
                OP_LOGE(context, "Failed to get ViewCopy src storage size."), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = ascendcPlatform.GetCoreNumAic();
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }

    tiling->totalNum = 0;
    tiling->blockFactor = 0;
    tiling->ubFactor = 0;
    tiling->storageNum = dstStorageNum;
    tiling->srcStorageNum = srcStorageNum;
    tiling->ndim = ndim;
    tiling->metaTypeBytes = metaTypeBytes;
    const bool metadataReady = FillConstMetadata(context, tiling, ndim, metaTypeBytes);
    if (!metadataReady) {
        tiling->metadataReady = 0;
        tiling->dstOverlap = -1;
        tiling->viewNum = 0;
        tiling->dstSpan = 0;
    } else {
        tiling->dstOverlap = IsDstMaybeOverlapped(tiling) ? 1 : 0;
    }

    const int64_t blockWork = GetBlockWork(tiling, typeBytes, coreNum, dstStorageNum);
    const int64_t blockNum = blockWork > 0 ? (blockWork < coreNum ? blockWork : coreNum) : 1;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    context->SetTilingKey(GetTilingKeyByTypeBytes(typeBytes));
    OP_CHECK_IF(SetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Set workspace size failed."),
                return ge::GRAPH_FAILED);
    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    rawTilingData->SetDataSize(sizeof(ViewCopyTilingData));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForViewCopy([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ViewCopyCompileInfo {};

IMPL_OP_OPTILING(ViewCopy).Tiling(ViewCopyTilingFunc).TilingParse<ViewCopyCompileInfo>(TilingParseForViewCopy);

} // namespace optiling
