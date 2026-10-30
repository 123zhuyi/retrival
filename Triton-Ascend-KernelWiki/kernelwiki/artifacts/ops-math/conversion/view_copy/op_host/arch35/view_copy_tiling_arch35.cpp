/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file view_copy_tiling_arch35.cpp
 * \brief
 */

#include <cmath>
#include "log/log.h"
#include "util/math_util.h"
#include "view_copy_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "atvoss/reduce/reduce_tiling.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace Ops::Base;

namespace optiling {
constexpr int64_t INDEX_INPUT_DST = 0;
constexpr int64_t INDEX_INPUT_DST_SIZE = 1;
constexpr int64_t INDEX_INPUT_DST_STRIDE = 2;
constexpr int64_t INDEX_INPUT_DST_OFFSET = 3;
constexpr int64_t INDEX_INPUT_SRC = 4;
constexpr int64_t INDEX_INPUT_SRC_SIZE = 5;
constexpr int64_t INDEX_INPUT_SRC_STRIDE = 6;
constexpr int64_t INDEX_INPUT_SRC_OFFSET = 7;
constexpr int64_t BYTE_PER_DATA_8 = 8;
constexpr int64_t BYTE_PER_DATA_4 = 4;
constexpr int64_t BYTE_PER_DATA_2 = 2;
constexpr int64_t BYTE_PER_DATA_1 = 1;
constexpr int64_t BYTE_THRESHOLD_TAIL_AXIS = 64;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t WORKSPACE_SIZE = 32;
constexpr int64_t ONE_BLK_BYTES = 32;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t DIGIT_HUNDRED = 100;
constexpr int64_t DIM_SIX = 6;
constexpr int64_t DIM_SEVEN = 7;
constexpr int64_t LEAST_DEAL_NUMBER = 1024;
constexpr int64_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t MOV_ALGIN_THRESHOLD = 128;
constexpr int64_t INT32_MAX_BOUND = 2147483647;
constexpr int64_t SIMT_UB_RES_SIZE = 160;
constexpr int64_t PURE_MOVE_ALIGN_THRESHOLD = 4;     // 纯搬运模板输出的最大维度
constexpr int64_t PURE_MOVE_LAST_DIM_THRESHOLD = 64; // 纯搬运模板尾轴的最小数据64B
constexpr int64_t PURE_MOVE_TILINGKEY_BASE = 10000;  // 纯搬运模板tilingKey的基准值

template <typename T>
inline static ge::graphStatus ViewCopySetTilingData(gert::TilingContext* context, T& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(ViewCopyTilingData& tilingData, const ViewCopyTilingParam& tilingParam)
{
    OP_LOGI("ViewCopy", "bufferSize: %ld, dstStorageOffset: %ld, srcStorageOffset: %ld, ubFactor: %ld, srcUbDim: %d, \
dstUbDim: %d, blockFactor: %ld, fusedBlockDims: %ld, blockFusedDimsNumber: %ld, ubDimSize: %ld, uo: %ld,           \
nddmaSizeLen: %d, ubDstSizeLen: %d, enableMovAlign: %d, enableDstInt64: %d, usedCoreNum: %d, tilingKey: %ld",
            tilingData.get_bufferSize(), tilingData.get_dstStorageOffset(), tilingData.get_srcStorageOffset(),
            tilingData.get_ubFactor(), tilingData.get_srcUbDim(), tilingData.get_dstUbDim(),
            tilingData.get_blockFactor(), tilingData.get_fusedBlockDims(), tilingData.get_blockFusedDimsNumber(),
            tilingData.get_ubDimSize(), tilingData.get_uo(), tilingData.get_nddmaSizeLen(),
            tilingData.get_ubDstSizeLen(), tilingData.get_enableMovAlign(), tilingData.get_enableDstInt64(),
            tilingParam.usedCoreNum, tilingParam.tilingKey);
    OP_LOGI("ViewCopy", "blockStride: %d,blockSrcStride: %d,nddmaSize: %d,nddmaStride: %d,blockDstStride: %d, \
ubDstSize: %d,ubDstStride: %d,contiguousUbDstStride: %d,contiguousUbSrcStride: %d",
            tilingData.get_blockStride(), tilingData.get_blockSrcStride(), tilingData.get_nddmaSize(),
            tilingData.get_nddmaStride(), tilingData.get_blockDstStride(), tilingData.get_ubDstSize(),
            tilingData.get_ubDstStride(), tilingData.get_contiguousUbDstStride(),
            tilingData.get_contiguousUbSrcStride());
}

inline static bool IsInvalidType(const DataType dtype)
{
    bool isInvalidType = dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT &&
                         dtype != ge::DT_UINT8 && dtype != ge::DT_INT8 && dtype != ge::DT_UINT16 &&
                         dtype != ge::DT_INT16 && dtype != ge::DT_UINT32 && dtype != ge::DT_INT32 &&
                         dtype != ge::DT_UINT64 && dtype != ge::DT_INT64 && dtype != ge::DT_BOOL &&
                         dtype != ge::DT_HIFLOAT8 && dtype != ge::DT_FLOAT8_E5M2 && dtype != ge::DT_FLOAT8_E4M3FN;
    return isInvalidType;
}

inline static bool IsInvalidIndexType(const DataType dtype) { return dtype != ge::DT_INT32 && dtype != ge::DT_INT64; }

inline static bool checkNegative(const std::vector<int64_t>& vec)
{
    for (int64_t num : vec) {
        if (num < 0) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus CheckDstRelatedDtypes(const gert::TilingContext* context)
{
    auto inputDstPtr = context->GetInputDesc(INDEX_INPUT_DST);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDstPtr);
    auto dtype = inputDstPtr->GetDataType();
    OP_CHECK_IF(IsInvalidType(dtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "dst", Ops::Base::ToString(dtype).c_str(),
                    "The dtype of dst must be within the range [DT_BF16, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8, "
                    "DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, DT_INT64, DT_BOOL, DT_HIFLOAT8, "
                    "DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]."),
                return ge::GRAPH_FAILED);

    auto inputDstSizePtr = context->GetInputDesc(INDEX_INPUT_DST_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDstSizePtr);
    dtype = inputDstSizePtr->GetDataType();
    OP_CHECK_IF(
        IsInvalidIndexType(dtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "dst_size", Ops::Base::ToString(dtype).c_str(),
                                              "The dtype of dst_size must be within the range [DT_INT32, DT_INT64]."),
        return ge::GRAPH_FAILED);

    auto inputDstStridePtr = context->GetInputDesc(INDEX_INPUT_DST_STRIDE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDstStridePtr);
    dtype = inputDstStridePtr->GetDataType();
    OP_CHECK_IF(
        IsInvalidIndexType(dtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "dst_stride", Ops::Base::ToString(dtype).c_str(),
                                              "The dtype of dst_stride must be within the range [DT_INT32, DT_INT64]."),
        return ge::GRAPH_FAILED);

    auto inputDstOffsetPtr = context->GetInputDesc(INDEX_INPUT_DST_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDstOffsetPtr);
    dtype = inputDstOffsetPtr->GetDataType();
    OP_CHECK_IF(IsInvalidIndexType(dtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "dst_storage_offset", Ops::Base::ToString(dtype).c_str(),
                    "The dtype of dst_storage_offset must be within the range [DT_INT32, DT_INT64]."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSrcRelatedDtypes(const gert::TilingContext* context)
{
    auto inputSrcPtr = context->GetInputDesc(INDEX_INPUT_SRC);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSrcPtr);
    auto dtype = inputSrcPtr->GetDataType();
    OP_CHECK_IF(IsInvalidType(dtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "src", Ops::Base::ToString(dtype).c_str(),
                    "The dtype of src must be within the range [DT_BF16, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8, "
                    "DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, DT_INT64, DT_BOOL, DT_HIFLOAT8, "
                    "DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]."),
                return ge::GRAPH_FAILED);

    auto inputSrcSizePtr = context->GetInputDesc(INDEX_INPUT_SRC_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSrcSizePtr);
    dtype = inputSrcSizePtr->GetDataType();
    OP_CHECK_IF(
        IsInvalidIndexType(dtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "src_size", Ops::Base::ToString(dtype).c_str(),
                                              "The dtype of src_size must be within the range [DT_INT32, DT_INT64]."),
        return ge::GRAPH_FAILED);

    auto inputSrcStridePtr = context->GetInputDesc(INDEX_INPUT_SRC_STRIDE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSrcStridePtr);
    dtype = inputSrcStridePtr->GetDataType();
    OP_CHECK_IF(
        IsInvalidIndexType(dtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "src_stride", Ops::Base::ToString(dtype).c_str(),
                                              "The dtype of src_stride must be within the range [DT_INT32, DT_INT64]."),
        return ge::GRAPH_FAILED);

    auto inputSrcOffsetPtr = context->GetInputDesc(INDEX_INPUT_SRC_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSrcOffsetPtr);
    dtype = inputSrcOffsetPtr->GetDataType();
    OP_CHECK_IF(IsInvalidIndexType(dtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "src_storage_offset", Ops::Base::ToString(dtype).c_str(),
                    "The dtype of src_storage_offset must be within the range [DT_INT32, DT_INT64]."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDtype(const gert::TilingContext* context)
{
    if (CheckDstRelatedDtypes(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckSrcRelatedDtypes(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

inline static bool IsSameShape(const gert::Shape shape1, const gert::Shape shape2)
{
    size_t inputShapeSize = shape1.GetDimNum();
    if (shape2.GetDimNum() != inputShapeSize) {
        return false;
    }
    for (size_t i = 0; i < inputShapeSize; ++i) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}

inline static ge::graphStatus CheckInputShape(const gert::TilingContext* context)
{
    auto srcSizeShapePtr = context->GetInputShape(INDEX_INPUT_SRC_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcSizeShapePtr);
    auto srcSizeShape = srcSizeShapePtr->GetStorageShape();

    auto dstSizeShapePtr = context->GetInputShape(INDEX_INPUT_DST_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstSizeShapePtr);
    auto dstSizeShape = dstSizeShapePtr->GetStorageShape();

    OP_CHECK_IF(!IsSameShape(srcSizeShape, dstSizeShape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "dst_size, src_size",
                    (Ops::Base::ToString(dstSizeShape) + ", " + Ops::Base::ToString(srcSizeShape)).c_str(),
                    "The shapes of dst_size and src_size must be the same."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
static void FilterSizeAndStrideAxes(const T* dstSizeVal, const T* dstStrideVal, const T* srcSizeVal,
                                    const T* srcStrideVal, ViewCopyTilingParam& tilingParam)
{
    for (int i = 0; i < tilingParam.length; ++i) {
        if (dstSizeVal[i] != 1) {
            tilingParam.dstSize.push_back(dstSizeVal[i]);
            tilingParam.dstStride.push_back(dstStrideVal[i]);
        }
        if (srcSizeVal[i] != 1) {
            tilingParam.srcSize.push_back(srcSizeVal[i]);
            tilingParam.srcStride.push_back(srcStrideVal[i]);
        }
    }
    if (tilingParam.length > 0 && tilingParam.dstSize.size() == 0) {
        tilingParam.dstSize.push_back(1);
        tilingParam.dstStride.push_back(1);
    }
    if (tilingParam.length > 0 && tilingParam.srcSize.size() == 0) {
        tilingParam.srcSize.push_back(1);
        tilingParam.srcStride.push_back(1);
    }
}

template <typename T>
static ge::graphStatus GetSizeAndStrideValue(const gert::TilingContext* context, ViewCopyTilingParam& tilingParam)
{
    auto dstSizeTensor = context->GetInputTensor(INDEX_INPUT_DST_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstSizeTensor);
    auto dstStrideTensor = context->GetInputTensor(INDEX_INPUT_DST_STRIDE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstStrideTensor);
    auto dstOffsetTensor = context->GetInputTensor(INDEX_INPUT_DST_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstOffsetTensor);
    auto srcSizeTensor = context->GetInputTensor(INDEX_INPUT_SRC_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcSizeTensor);
    auto srcStrideTensor = context->GetInputTensor(INDEX_INPUT_SRC_STRIDE);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcStrideTensor);
    auto srcOffsetTensor = context->GetInputTensor(INDEX_INPUT_SRC_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcOffsetTensor);

    tilingParam.length = dstSizeTensor->GetStorageShape().GetDim(0);

    const T* dstSizeVal = dstSizeTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, dstSizeVal);
    const T* dstStrideVal = dstStrideTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, dstStrideVal);
    const T* dstOffsetVal = dstOffsetTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, dstOffsetVal);
    const T* srcSizeVal = srcSizeTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, srcSizeVal);
    const T* srcStrideVal = srcStrideTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, srcStrideVal);
    const T* srcOffsetVal = srcOffsetTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, srcOffsetVal);

    auto dstTensorPtr = context->GetInputTensor(INDEX_INPUT_DST);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstTensorPtr);
    auto dstTensorShape = dstTensorPtr->GetStorageShape();
    int64_t dstTensorSize = 1;
    for (size_t i = 0; i < dstTensorShape.GetDimNum(); ++i) {
        dstTensorSize *= dstTensorShape.GetDim(i);
    }
    tilingParam.enableDstInt64 = (dstTensorSize > INT32_MAX_BOUND) ? 1 : 0;

    FilterSizeAndStrideAxes(dstSizeVal, dstStrideVal, srcSizeVal, srcStrideVal, tilingParam);
    tilingParam.dstStorageOffset = dstOffsetVal[0];
    tilingParam.srcStorageOffset = srcOffsetVal[0];

    return ge::GRAPH_SUCCESS;
}

inline static int64_t GetTensorSize(const gert::Tensor* tensor)
{
    auto tensorShape = tensor->GetStorageShape();
    int64_t tensorSize = 1;
    for (size_t i = 0; i < tensorShape.GetDimNum(); ++i) {
        tensorSize *= tensorShape.GetDim(i);
    }
    return tensorSize;
}

inline static bool CheckSameDimensions(std::vector<int64_t> size, std::vector<int64_t> strides)
{
    return size.size() != strides.size();
}

inline static bool checkOffsetAndStride(int64_t offset, std::vector<int64_t> strides, std::vector<int64_t> size,
                                        int64_t tensorSize)
{
    for (size_t i = 0; i < strides.size(); i++) {
        if (size[i] > 0 && offset + (size[i] - 1) * strides[i] > tensorSize) {
            return true;
        }
    }
    return false;
}

inline static bool checkWrittenToSameDstPosition(const std::vector<int64_t>& dstSize,
                                                 const std::vector<int64_t>& dstStride)
{
    // 此时dstSize、dstStride大小必然是相同的。
    for (size_t i = 0; i < dstStride.size(); i++) {
        if (dstSize[i] > 1 && dstStride[i] == 0) {
            return true;
        }
    }
    return false;
}

inline static ge::graphStatus CheckSizeAndStrideBasic(const gert::TilingContext* context,
                                                      const ViewCopyTilingParam& tilingParam)
{
    OP_CHECK_IF(tilingParam.srcSize != tilingParam.dstSize,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "dst_size, src_size",
                    (VectorToString(tilingParam.dstSize) + ", " + VectorToString(tilingParam.srcSize)).c_str(),
                    "The values of dst_size and src_size must be the same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkNegative(tilingParam.srcSize),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "src_size",
                                                      VectorToString(tilingParam.srcSize).c_str(),
                                                      "The value of src_size cannot be a negative number."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkNegative(tilingParam.dstSize),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_size",
                                                      VectorToString(tilingParam.dstSize).c_str(),
                                                      "The value of dst_size cannot be a negative number."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkNegative(tilingParam.srcStride),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "src_stride",
                                                      VectorToString(tilingParam.srcStride).c_str(),
                                                      "The value of src_stride cannot be a negative number."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkNegative(tilingParam.dstStride),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_stride",
                                                      VectorToString(tilingParam.dstStride).c_str(),
                                                      "The value of dst_stride cannot be a negative number."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus CheckSizeAndStrideDims(const gert::TilingContext* context,
                                                     const ViewCopyTilingParam& tilingParam)
{
    OP_CHECK_IF(
        CheckSameDimensions(tilingParam.dstSize, tilingParam.dstStride),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "dst_size, dst_stride",
            (std::to_string(tilingParam.dstSize.size()) + ", " + std::to_string(tilingParam.dstStride.size())).c_str(),
            "The shape dims of dst_size and dst_stride must be the same."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckSameDimensions(tilingParam.srcSize, tilingParam.srcStride),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "src_size, src_stride",
            (std::to_string(tilingParam.srcSize.size()) + ", " + std::to_string(tilingParam.srcStride.size())).c_str(),
            "The shape dims of src_size and src_stride must be the same."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus CheckOffsetAndStrideBound(const gert::TilingContext* context,
                                                        const ViewCopyTilingParam& tilingParam, int64_t dstTensorSize,
                                                        int64_t srcTensorSize)
{
    OP_CHECK_IF(checkWrittenToSameDstPosition(tilingParam.dstSize, tilingParam.dstStride),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "dst_size, dst_stride",
                    (VectorToString(tilingParam.dstSize) + ", " + VectorToString(tilingParam.dstStride)).c_str(),
                    "If the value of an axis of dst_size is greater than 1, the value of this axis of dst_stride must "
                    "not be 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        checkOffsetAndStride(tilingParam.dstStorageOffset, tilingParam.dstStride, tilingParam.dstSize, dstTensorSize),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "dst_storage_offset, dst_stride, dst_size",
            (std::to_string(tilingParam.dstStorageOffset) + ", " + VectorToString(tilingParam.dstStride) + ", " +
             VectorToString(tilingParam.dstSize))
                .c_str(),
            (std::string(
                 "If the value of an axis of dst_size is greater than 0, the following formula must be met: Value of "
                 "dst_storage_offset + (Value of this axis of dst_size - 1) x Value of this axis of dst_stride ≤ ") +
             std::to_string(dstTensorSize))
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        checkOffsetAndStride(tilingParam.srcStorageOffset, tilingParam.srcStride, tilingParam.srcSize, srcTensorSize),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "src_storage_offset, src_stride, src_size",
            (std::to_string(tilingParam.srcStorageOffset) + ", " + VectorToString(tilingParam.srcStride) + ", " +
             VectorToString(tilingParam.srcSize))
                .c_str(),
            (std::string(
                 "If the value of an axis of src_size is greater than 0, the following formula must be met: Value of "
                 "src_storage_offset + (Value of this axis of src_size - 1) x Value of this axis of src_stride ≤ ") +
             std::to_string(srcTensorSize))
                .c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus CheckSizeAndStrideValue(const gert::TilingContext* context,
                                                      const ViewCopyTilingParam& tilingParam)
{
    auto dstTensorPtr = context->GetInputTensor(INDEX_INPUT_DST);
    auto srcTensorPtr = context->GetInputTensor(INDEX_INPUT_SRC);
    int64_t dstTensorSize = GetTensorSize(dstTensorPtr);
    int64_t srcTensorSize = GetTensorSize(srcTensorPtr);

    if (CheckSizeAndStrideBasic(context, tilingParam) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckSizeAndStrideDims(context, tilingParam) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOffsetAndStrideBound(context, tilingParam, dstTensorSize, srcTensorSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void CalUbFactorAndUbDim(ViewCopyTilingParam& tilingParam)
{
    // 对于指定的轴，放满UB时，轴的个数
    int64_t bytesForOneData = ge::GetSizeByDataType(tilingParam.dtype);
    int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER / bytesForOneData;
    // 要切分的轴以及ubFactor
    int16_t realUbDim = 0;
    int64_t ubFactor = 0;
    tilingParam.blockNum = ONE_BLK_BYTES / bytesForOneData;
    std::vector<int64_t> srcSizeCopy = tilingParam.srcSize;
    srcSizeCopy.back() = CeilAlign(srcSizeCopy.back(), tilingParam.blockNum);
    tilingParam.coreData = 1;
    for (int16_t i = static_cast<int16_t>(srcSizeCopy.size()) - 1; i >= 0; i--) {
        if (srcSizeCopy[i] >= maxUbAvailable) {
            realUbDim = i;
            ubFactor = maxUbAvailable;
            tilingParam.coreData *= maxUbAvailable;
            break;
        } else {
            realUbDim = i;
            ubFactor = tilingParam.srcSize[i];
            tilingParam.coreData *= srcSizeCopy[i];
            maxUbAvailable = FloorDiv(maxUbAvailable, srcSizeCopy[i]);
        }
    }
    tilingParam.realUbDim = realUbDim;
    tilingParam.ubFactor = ubFactor;
    tilingParam.uo = CeilDiv(tilingParam.srcSize[tilingParam.realUbDim], tilingParam.ubFactor);
}

inline static int64_t FindLastDiscoutiguousAxis(const ViewCopyTilingParam& tilingParam)
{
    // 找到最后一个非连续轴，用于合轴
    if (tilingParam.ubDstStride.back() != 1) {
        return static_cast<int64_t>(tilingParam.ubDstSize.size()) - 1;
    }
    for (int64_t i = static_cast<int64_t>(tilingParam.ubDstSize.size()) - 2; i >= 0; i--) {
        if (tilingParam.ubDstSize[i + 1] * tilingParam.ubDstStride[i + 1] != tilingParam.ubDstStride[i]) {
            return i;
        }
    }
    return -1;
}

static void MergeAxisForDst(ViewCopyTilingParam& tilingParam)
{
    if (tilingParam.ubDstSize.size() <= 1) {
        return;
    }
    // dst连续轴合轴
    for (int64_t i = 0; i < static_cast<int64_t>(tilingParam.ubDstSize.size()) - 1; i++) {
        if (tilingParam.ubDstSize[i + 1] * tilingParam.ubDstStride[i + 1] == tilingParam.ubDstStride[i]) {
            tilingParam.ubDstSize[i + 1] = tilingParam.ubDstSize[i] * tilingParam.ubDstSize[i + 1];
            tilingParam.ubDstSize[i] = 1;
        }
    }

    std::vector<int32_t> mergeUbDstSize;
    std::vector<int64_t> mergeUbDstStride;
    std::vector<int64_t> mergeContiguousUbDstStride;
    // 去掉轴大小为1的轴
    for (size_t i = 0; i < tilingParam.ubDstSize.size(); i++) {
        if (tilingParam.ubDstSize[i] != 1) {
            mergeUbDstSize.push_back(tilingParam.ubDstSize[i]);
            mergeUbDstStride.push_back(tilingParam.ubDstStride[i]);
            mergeContiguousUbDstStride.push_back(tilingParam.contiguousUbDstStride[i]);
        }
    }
    if (mergeUbDstSize.size() == 0 && tilingParam.ubDstSize.size() > 0) {
        mergeUbDstSize.push_back(1);
        mergeUbDstStride.push_back(1);
        mergeContiguousUbDstStride.push_back(1);
    }
    tilingParam.ubDstSize = mergeUbDstSize;
    tilingParam.ubDstStride = mergeUbDstStride;
    tilingParam.contiguousUbDstStride = mergeContiguousUbDstStride;
}

static void MergeAxisForSrc(ViewCopyTilingParam& tilingParam)
{
    if (tilingParam.nddmaSize.size() <= 1) {
        return;
    }
    // src连续轴合轴
    for (int64_t i = 0; i < static_cast<int64_t>(tilingParam.nddmaSize.size()) - 1; i++) {
        if ((tilingParam.nddmaSize[i + 1] * tilingParam.nddmaStride[i + 1] == tilingParam.nddmaStride[i]) &&
            (tilingParam.nddmaStride[i] == tilingParam.contiguousUbSrcStride[i])) {
            tilingParam.nddmaSize[i + 1] = tilingParam.nddmaSize[i] * tilingParam.nddmaSize[i + 1];
            tilingParam.nddmaSize[i] = 1;
        }
    }

    std::vector<int32_t> mergeNddmaSize;
    std::vector<int64_t> mergemergeNddmaStride;
    std::vector<int64_t> mergeContiguousUbSrcStride;
    // 去掉轴大小为1的轴
    for (size_t i = 0; i < tilingParam.nddmaSize.size(); i++) {
        if (tilingParam.nddmaSize[i] != 1) {
            mergeNddmaSize.push_back(tilingParam.nddmaSize[i]);
            mergemergeNddmaStride.push_back(tilingParam.nddmaStride[i]);
            mergeContiguousUbSrcStride.push_back(tilingParam.contiguousUbSrcStride[i]);
        }
    }
    if (mergeNddmaSize.size() == 0 && tilingParam.nddmaSize.size() > 0) {
        mergeNddmaSize.push_back(1);
        mergemergeNddmaStride.push_back(1);
        mergeContiguousUbSrcStride.push_back(1);
    }
    tilingParam.nddmaSize = mergeNddmaSize;
    tilingParam.nddmaStride = mergemergeNddmaStride;
    tilingParam.contiguousUbSrcStride = mergeContiguousUbSrcStride;
}

static void MergeAxis(ViewCopyTilingParam& tilingParam)
{
    MergeAxisForDst(tilingParam);
    MergeAxisForSrc(tilingParam);
}

static void ComplementDim(ViewCopyTilingParam& tilingParam)
{
    int8_t srcPadOneNum = TILING_ARRAY_LEN_EIGHT - tilingParam.nddmaSize.size();
    tilingParam.srcUbDim = srcPadOneNum;
    if (srcPadOneNum != 0) {
        tilingParam.nddmaSize.insert(tilingParam.nddmaSize.begin(), srcPadOneNum, 1);
        tilingParam.nddmaStride.insert(tilingParam.nddmaStride.begin(), srcPadOneNum, 0);
        tilingParam.contiguousUbSrcStride.insert(tilingParam.contiguousUbSrcStride.begin(), srcPadOneNum, 0);
    }

    int8_t dstPadOneNum = TILING_ARRAY_LEN_EIGHT - tilingParam.ubDstSize.size();
    tilingParam.dstUbDim = dstPadOneNum;
    if (dstPadOneNum != 0) {
        tilingParam.ubDstSize.insert(tilingParam.ubDstSize.begin(), dstPadOneNum, 1);
        tilingParam.ubDstStride.insert(tilingParam.ubDstStride.begin(), dstPadOneNum, 0);
        tilingParam.contiguousUbDstStride.insert(tilingParam.contiguousUbDstStride.begin(), dstPadOneNum, 0);
    }
}

static bool IsEnableSimt(const ViewCopyTilingParam& tilingParam)
{
    if (tilingParam.dstStride.size() == 0 || tilingParam.dstStride.back() != 1) {
        return true;
    }
    auto byteNum = ge::GetSizeByDataType(tilingParam.dtype);
    int64_t tailAxisByteNum = tilingParam.dstSize.back();
    for (int64_t i = static_cast<int64_t>(tilingParam.dstSize.size()) - 2; i >= 0; i--) {
        if (tilingParam.dstSize[i + 1] * tilingParam.dstStride[i + 1] == tilingParam.dstStride[i]) {
            tailAxisByteNum *= tilingParam.dstSize[i];
        } else {
            break;
        }
    }
    bool isSimt = tailAxisByteNum * byteNum < BYTE_THRESHOLD_TAIL_AXIS;
    return isSimt;
}

static bool IsEnableMovAlign(const ViewCopyTilingParam& tilingParam)
{
    if (tilingParam.nddmaSize.size() == 0 || tilingParam.nddmaStride.size() == 0 ||
        tilingParam.nddmaStride.back() != 1 || tilingParam.contiguousUbDstStride.size() == 0 ||
        tilingParam.contiguousUbDstStride.back() != 1) {
        return false;
    }
    int64_t byteNum = ge::GetSizeByDataType(tilingParam.dtype);
    if (static_cast<int64_t>(tilingParam.nddmaSize.back() * byteNum) < MOV_ALGIN_THRESHOLD) {
        return false;
    }
    int64_t num = tilingParam.nddmaSize.size();
    if (num == 1 || (num == DIGIT_TWO && tilingParam.contiguousUbDstStride.front() * byteNum % ONE_BLK_BYTES == 0)) {
        return true;
    }
    return false;
}

static bool IsEmpty(const ViewCopyTilingParam& tilingParam)
{
    if (tilingParam.srcSize.size() == 0) {
        return true;
    }
    int64_t totalSize = 1;
    for (size_t i = 0; i < tilingParam.srcSize.size(); i++) {
        totalSize *= tilingParam.srcSize[i];
    }
    return (totalSize <= 0);
}

static void UseLowerDtypeForInt64Nddma(ViewCopyTilingParam& tilingParam)
{
    if (tilingParam.dtype != ge::DT_INT64 && tilingParam.dtype != ge::DT_UINT64) {
        return;
    }
    if (tilingParam.srcSize.size() == TILING_ARRAY_LEN_EIGHT || tilingParam.dstSize.size() == TILING_ARRAY_LEN_EIGHT) {
        return;
    }
    tilingParam.srcSize.push_back(DIGIT_TWO);
    tilingParam.dstSize.push_back(DIGIT_TWO);
    for (size_t i = 0; i < tilingParam.srcStride.size(); ++i) {
        tilingParam.srcStride[i] = tilingParam.srcStride[i] * DIGIT_TWO;
        tilingParam.dstStride[i] = tilingParam.dstStride[i] * DIGIT_TWO;
    }
    tilingParam.srcStride.push_back(1);
    tilingParam.dstStride.push_back(1);
    tilingParam.dtype = ge::DT_INT32;
    tilingParam.dstStorageOffset *= DIGIT_TWO;
    tilingParam.srcStorageOffset *= DIGIT_TWO;
}

static int64_t CalcOutSizeAndAdjustUb(ViewCopyTilingParam& tilingParam)
{
    int64_t outSize = 1;
    for (int64_t i = 0; i < tilingParam.realUbDim; i++) {
        outSize *= tilingParam.srcSize[i];
    }
    outSize *= tilingParam.uo;
    int64_t bytesForOneData = ge::GetSizeByDataType(tilingParam.dtype);
    if (bytesForOneData > 0) {
        while (outSize <= static_cast<int64_t>(tilingParam.totalCoreNum) / DIGIT_TWO &&
               tilingParam.coreData > LEAST_DEAL_NUMBER / bytesForOneData) {
            tilingParam.ubSize /= DIGIT_TWO;
            CalUbFactorAndUbDim(tilingParam);
            outSize = 1;
            for (int16_t i = 0; i < tilingParam.realUbDim; i++) {
                outSize *= tilingParam.srcSize[i];
            }
            outSize *= tilingParam.uo;
        }
    }
    return outSize;
}

static void CalcContiguousStrides(ViewCopyTilingParam& tilingParam, int64_t disContiguousAxis)
{
    tilingParam.contiguousUbDstStride = std::vector<int64_t>(tilingParam.ubDstSize.size(), 1);
    for (int64_t i = static_cast<int64_t>(tilingParam.ubDstSize.size()) - 2; i >= 0; i--) {
        if (i != disContiguousAxis) {
            tilingParam.contiguousUbDstStride[i] = tilingParam.contiguousUbDstStride[i + 1] *
                                                   tilingParam.ubDstSize[i + 1];
        } else {
            tilingParam.contiguousUbDstStride[i] = CeilDiv(tilingParam.contiguousUbDstStride[i + 1] *
                                                               tilingParam.ubDstSize[i + 1],
                                                           tilingParam.blockNum) *
                                                   tilingParam.blockNum;
        }
    }

    tilingParam.contiguousUbSrcStride = std::vector<int64_t>(tilingParam.nddmaSize.size(), 1);
    for (int64_t i = static_cast<int64_t>(tilingParam.nddmaSize.size()) - 2; i >= 0; i--) {
        if (i != disContiguousAxis) {
            tilingParam.contiguousUbSrcStride[i] = tilingParam.contiguousUbSrcStride[i + 1] *
                                                   tilingParam.nddmaSize[i + 1];
        } else {
            tilingParam.contiguousUbSrcStride[i] = CeilDiv(tilingParam.contiguousUbSrcStride[i + 1] *
                                                               tilingParam.nddmaSize[i + 1],
                                                           tilingParam.blockNum) *
                                                   tilingParam.blockNum;
        }
    }
}

static void DoTiling(ViewCopyTilingParam& tilingParam)
{
    if (IsEmpty(tilingParam)) {
        tilingParam.usedCoreNum = 0;
        return;
    }
    UseLowerDtypeForInt64Nddma(tilingParam);
    // 计算要切分的轴以及ubFactor
    CalUbFactorAndUbDim(tilingParam);
    int64_t outSize = CalcOutSizeAndAdjustUb(tilingParam);

    tilingParam.blockSrcStride = tilingParam.srcStride;
    tilingParam.blockSrcStride.insert(tilingParam.blockSrcStride.begin() + tilingParam.realUbDim,
                                      tilingParam.blockSrcStride[tilingParam.realUbDim] * tilingParam.ubFactor);

    tilingParam.blockFactor = CeilDiv(outSize, static_cast<int64_t>(tilingParam.totalCoreNum));
    tilingParam.usedCoreNum = static_cast<int16_t>(CeilDiv(outSize, tilingParam.blockFactor));

    tilingParam.blockStride.push_back(1);
    for (int64_t i = tilingParam.realUbDim; i > 0; i--) {
        if (i == tilingParam.realUbDim) {
            tilingParam.blockStride.insert(tilingParam.blockStride.begin(),
                                           tilingParam.blockStride[0] * tilingParam.uo);
        } else {
            tilingParam.blockStride.insert(tilingParam.blockStride.begin(),
                                           tilingParam.blockStride[0] * tilingParam.srcSize[i]);
        }
    }

    tilingParam.blockFusedDimsNumber = tilingParam.blockStride.size();
    tilingParam.fusedBlockDims = outSize;

    for (int64_t i = static_cast<int64_t>(tilingParam.srcSize.size()) - 1; i > tilingParam.realUbDim - 1; i--) {
        if (i == tilingParam.realUbDim) {
            tilingParam.nddmaSize.insert(tilingParam.nddmaSize.begin(), tilingParam.ubFactor);
        } else {
            tilingParam.nddmaSize.insert(tilingParam.nddmaSize.begin(), tilingParam.srcSize[i]);
        }
        tilingParam.nddmaStride.insert(tilingParam.nddmaStride.begin(), tilingParam.srcStride[i]);
    }
    tilingParam.ubDimSize = tilingParam.srcSize[tilingParam.realUbDim];

    tilingParam.blockDstStride = tilingParam.dstStride;
    tilingParam.blockDstStride.insert(tilingParam.blockDstStride.begin() + tilingParam.realUbDim,
                                      tilingParam.blockDstStride[tilingParam.realUbDim] * tilingParam.ubFactor);

    for (int64_t i = static_cast<int64_t>(tilingParam.dstSize.size()) - 1; i > tilingParam.realUbDim - 1; i--) {
        if (i == tilingParam.realUbDim) {
            tilingParam.ubDstSize.insert(tilingParam.ubDstSize.begin(), tilingParam.ubFactor);
        } else {
            tilingParam.ubDstSize.insert(tilingParam.ubDstSize.begin(), tilingParam.dstSize[i]);
        }
        tilingParam.ubDstStride.insert(tilingParam.ubDstStride.begin(), tilingParam.dstStride[i]);
    }

    // 找到最后一个非连续轴
    int64_t disContiguousAxis = FindLastDiscoutiguousAxis(tilingParam);

    CalcContiguousStrides(tilingParam, disContiguousAxis);

    MergeAxis(tilingParam);
    tilingParam.enableMovAlign = IsEnableMovAlign(tilingParam) ? 1 : 0;
    ComplementDim(tilingParam);
}

static void GetTilingKey(ViewCopyTilingParam& tilingParam)
{
    // tilingkey使用3位数表示，个位数由数据类型所占字节决定(1/2/4/8)，十位数由维度决定(1/2/3/4/5/8)，百位数由是否走SIMT决定(1/2,SIMT为2，非SIMT为1)
    auto byteNum = ge::GetSizeByDataType(tilingParam.dtype);
    int64_t dims = tilingParam.srcSize.size();

    int64_t hundredDigit = tilingParam.isSimt ? DIGIT_TWO : DIGIT_ONE;
    int64_t tenDigit = (dims == DIM_SIX || dims == DIM_SEVEN || dims == 0) ? DIGIT_EIGHT : dims;
    int64_t digit = byteNum;
    tilingParam.tilingKey = hundredDigit * DIGIT_HUNDRED + tenDigit * DIGIT_TEN + digit;
}

static void SetTilingData(ViewCopyTilingData& tilingData, const ViewCopyTilingParam& tilingParam)
{
    tilingData.set_bufferSize(tilingParam.ubSize / N_BUFFER);
    tilingData.set_dstStorageOffset(tilingParam.dstStorageOffset);
    tilingData.set_srcStorageOffset(tilingParam.srcStorageOffset);
    tilingData.set_ubFactor(tilingParam.ubFactor);
    tilingData.set_srcUbDim(tilingParam.srcUbDim);
    tilingData.set_dstUbDim(tilingParam.dstUbDim);
    tilingData.set_blockFactor(tilingParam.blockFactor);
    tilingData.set_fusedBlockDims(tilingParam.fusedBlockDims);
    tilingData.set_blockFusedDimsNumber(tilingParam.blockFusedDimsNumber);
    tilingData.set_ubDimSize(tilingParam.ubDimSize);
    tilingData.set_uo(tilingParam.uo);
    int64_t lengthIsTenArray[TILING_ARRAY_LEN_TEN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::copy(tilingParam.blockStride.begin(), tilingParam.blockStride.end(), lengthIsTenArray);
    tilingData.set_blockStride(lengthIsTenArray);

    std::copy(tilingParam.blockSrcStride.begin(), tilingParam.blockSrcStride.end(), lengthIsTenArray);
    tilingData.set_blockSrcStride(lengthIsTenArray);

    int64_t lengthIsEightArray[TILING_ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};

    std::copy(tilingParam.nddmaStride.begin(), tilingParam.nddmaStride.end(), lengthIsEightArray);
    tilingData.set_nddmaStride(lengthIsEightArray);

    std::copy(tilingParam.blockDstStride.begin(), tilingParam.blockDstStride.end(), lengthIsTenArray);
    tilingData.set_blockDstStride(lengthIsTenArray);

    std::copy(tilingParam.ubDstStride.begin(), tilingParam.ubDstStride.end(), lengthIsEightArray);
    tilingData.set_ubDstStride(lengthIsEightArray);

    std::copy(tilingParam.contiguousUbDstStride.begin(), tilingParam.contiguousUbDstStride.end(), lengthIsEightArray);
    tilingData.set_contiguousUbDstStride(lengthIsEightArray);

    std::copy(tilingParam.contiguousUbSrcStride.begin(), tilingParam.contiguousUbSrcStride.end(), lengthIsEightArray);
    tilingData.set_contiguousUbSrcStride(lengthIsEightArray);

    int32_t lengthIsEightArrayForSize[TILING_ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::copy(tilingParam.nddmaSize.begin(), tilingParam.nddmaSize.end(), lengthIsEightArrayForSize);
    tilingData.set_nddmaSize(lengthIsEightArrayForSize);
    tilingData.set_nddmaSizeLen(static_cast<int8_t>(tilingParam.nddmaSize.size()));

    std::copy(tilingParam.ubDstSize.begin(), tilingParam.ubDstSize.end(), lengthIsEightArrayForSize);
    tilingData.set_ubDstSize(lengthIsEightArrayForSize);
    tilingData.set_ubDstSizeLen(static_cast<int8_t>(tilingParam.ubDstSize.size()));
    tilingData.set_enableMovAlign(tilingParam.enableMovAlign);
    tilingData.set_enableDstInt64(tilingParam.enableDstInt64);
}

static ge::graphStatus SetBaseTilingInfo(gert::TilingContext* context, ViewCopyTilingParam& tilingParam)
{
    tilingParam.isSimt = IsEnableSimt(tilingParam);
    if (tilingParam.isSimt) {
        OP_CHECK_IF((tilingParam.ubSize <= DCACHE_SIZE),
                    OP_LOGE(context->GetNodeName(), "ub size less than Dcache Size"), return ge::GRAPH_FAILED);
        tilingParam.ubSize = tilingParam.ubSize - DCACHE_SIZE;
        context->SetLocalMemorySize(tilingParam.ubSize + SIMT_UB_RES_SIZE);
    }
    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(ViewCopyTilingDataPureMoveAlign& tilingData, const ViewCopyTilingParam& tilingParam)
{
    tilingData.set_ubDim(tilingParam.ubDim);
    tilingData.set_blockFactor(tilingParam.blockFactor);
    tilingData.set_tailBlockFactor(tilingParam.tailBlockFactor);
    tilingData.set_ubFactor(tilingParam.ubFactor);
    tilingData.set_tailUbFactor(tilingParam.tailUbFactor);
    tilingData.set_uo(tilingParam.uo);
    tilingData.set_dstStorageOffset(tilingParam.dstStorageOffset);
    tilingData.set_srcStorageOffset(tilingParam.srcStorageOffset);

    int64_t pureMoveAlignArrayForSize[TILING_ARRAY_LEN_PURE_MOVE_DST] = {0, 0, 0, 0};
    std::copy(tilingParam.pureDstSize.begin(), tilingParam.pureDstSize.end(), pureMoveAlignArrayForSize);
    tilingData.set_pureDstSize(pureMoveAlignArrayForSize);

    std::copy(tilingParam.pureDstStride.begin(), tilingParam.pureDstStride.end(), pureMoveAlignArrayForSize);
    tilingData.set_pureDstStride(pureMoveAlignArrayForSize);
}

static void PrintTilingData(ViewCopyTilingDataPureMoveAlign& tilingData)
{
    OP_LOGI("ViewCopy", "ubDim: %ld, blockFactor: %ld, tailBlockFactor: %ld, ubFactor: %ld, \
tailUbFactor: %ld, uo: %ld, dstStorageOffset: %ld, srcStorageOffset: %ld",
            tilingData.get_ubDim(), tilingData.get_blockFactor(), tilingData.get_tailBlockFactor(),
            tilingData.get_ubFactor(), tilingData.get_tailUbFactor(), tilingData.get_uo(),
            tilingData.get_dstStorageOffset(), tilingData.get_srcStorageOffset());
    OP_LOGI("ViewCopy", "pureDstSize: %d, pureDstStride: %d", tilingData.get_pureDstSize(),
            tilingData.get_pureDstStride());
}

static bool IsSrcContinuous(ViewCopyTilingParam& param)
{
    // 判断输入数据的连续性
    std::vector<int64_t> srcContinuousStride = {DIGIT_ONE};
    int64_t strideDim = DIGIT_ONE;
    for (size_t i = param.srcSize.size() - 1; i > 0; --i) {
        strideDim *= param.srcSize[i];
        srcContinuousStride.insert(srcContinuousStride.begin(), strideDim);
    }
    for (size_t i = 0; i < srcContinuousStride.size(); i++) {
        if (srcContinuousStride[i] != param.srcStride[i]) {
            return false;
        }
    }
    return true;
}

static void MergeAxisForPureDst(ViewCopyTilingParam& tilingParam)
{
    // 纯搬运模板，对dstSize相邻连续轴进行合轴
    std::vector<int64_t> mergeDstSize = tilingParam.dstSize;
    std::vector<int64_t> mergeDstStride = tilingParam.dstStride;

    // dstSize的size等于1场景
    if (mergeDstSize.size() == 1) {
        tilingParam.pureDstSize.push_back(tilingParam.dstSize[0]);
        tilingParam.pureDstStride.push_back(tilingParam.dstStride[0]);
        return;
    }

    // 将dstSize相邻连续的轴进行合并
    for (int64_t i = static_cast<int64_t>(mergeDstSize.size() - 1); i > 0; --i) {
        if (tilingParam.dstSize[i] * tilingParam.dstStride[i] == tilingParam.dstStride[i - 1]) {
            mergeDstSize[i - 1] = mergeDstSize[i] * mergeDstSize[i - 1];
            mergeDstSize[i] = 1;
            mergeDstStride[i - 1] = mergeDstStride[i];
        }
    }

    // 去掉轴大小为1的轴
    for (size_t i = 0; i < mergeDstSize.size(); i++) {
        if (mergeDstSize[i] != 1) {
            tilingParam.pureDstSize.push_back(mergeDstSize[i]);
            tilingParam.pureDstStride.push_back(mergeDstStride[i]);
        }
    }
    // 去掉所有1轴后，处理pureDstSize == 0 场景
    if (tilingParam.pureDstSize.size() == 0 && mergeDstSize.size() > 0) {
        tilingParam.pureDstSize.push_back(1);
        tilingParam.pureDstStride.push_back(1);
    }
}

static bool IsUsePureMoveAlign(ViewCopyTilingParam& tilingParam)
{
    // 使用纯搬运模板条件: 输入连续; 输出合轴后为4维(3及3以内非连续),且尾轴大于64B,且尾轴连续,即尾轴的stride是1
    MergeAxisForPureDst(tilingParam);
    if (IsSrcContinuous(tilingParam) && tilingParam.pureDstSize.size() <= PURE_MOVE_ALIGN_THRESHOLD &&
        (tilingParam.dstDtypeSize * tilingParam.pureDstSize.back()) > PURE_MOVE_LAST_DIM_THRESHOLD &&
        tilingParam.pureDstStride.back() == 1) {
        return true;
    }
    return false;
}

static void CalcUbFactorAndUbDimForPureMoveAlign(ViewCopyTilingParam& tilingParam)
{
    // 对于指定的轴，放满UB时，数据的个数
    int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER / tilingParam.dstDtypeSize;

    // 使用合轴的dstSize要切分的轴和UbFactor
    std::vector<int64_t> pureDstSizeCopy = tilingParam.pureDstSize;
    int64_t ubDim = 0;
    int64_t ubFactor = 0;
    tilingParam.blockNum = ONE_BLK_BYTES / tilingParam.dstDtypeSize;
    pureDstSizeCopy.back() = CeilAlign(pureDstSizeCopy.back(), tilingParam.blockNum);
    tilingParam.coreData = 1;

    // 计算切分轴
    for (int16_t i = static_cast<int16_t>(pureDstSizeCopy.size() - 1); i >= 0; --i) {
        if (pureDstSizeCopy[i] >= maxUbAvailable) {
            ubDim = i;
            ubFactor = maxUbAvailable;
            tilingParam.coreData *= maxUbAvailable;
            break;
        } else {
            ubDim = i;
            ubFactor = tilingParam.pureDstSize[i];
            tilingParam.coreData *= pureDstSizeCopy[i];
            maxUbAvailable = FloorDiv(maxUbAvailable, pureDstSizeCopy[i]);
        }
    }

    tilingParam.ubDim = ubDim;
    tilingParam.ubFactor = ubFactor;
    tilingParam.uo = CeilDiv(tilingParam.pureDstSize[ubDim], ubFactor);
    tilingParam.tailUbFactor = tilingParam.pureDstSize[ubDim] - (tilingParam.uo - 1) * ubFactor;
}

static void DoTilingForPureMoveAlign(ViewCopyTilingParam& tilingParam)
{
    // 计算dst要切分的轴以及ubFactor
    CalcUbFactorAndUbDimForPureMoveAlign(tilingParam);

    // 计算tiling中blockFactor,tailBlockFactor及使用的核数
    int64_t outSize = tilingParam.uo;
    for (int16_t i = 0; i < tilingParam.ubDim; i++) {
        outSize *= tilingParam.pureDstSize[i];
    }

    // 剩余数据不够开多核的情况下调整ubSize, 重新计算ubFactor
    while (outSize <= static_cast<int64_t>(tilingParam.totalCoreNum) / DIGIT_TWO &&
           tilingParam.coreData > LEAST_DEAL_NUMBER / tilingParam.dstDtypeSize) {
        tilingParam.ubSize /= DIGIT_TWO;
        CalcUbFactorAndUbDimForPureMoveAlign(tilingParam);
        outSize = tilingParam.uo;
        for (int16_t i = 0; i < tilingParam.ubDim; i++) {
            outSize *= tilingParam.pureDstSize[i];
        }
    }

    tilingParam.blockFactor = CeilDiv(outSize, static_cast<int64_t>(tilingParam.totalCoreNum));
    tilingParam.usedCoreNum = static_cast<int16_t>(CeilDiv(outSize, tilingParam.blockFactor));
    tilingParam.tailBlockFactor = outSize - (tilingParam.usedCoreNum - 1) * tilingParam.blockFactor;
}

static ge::graphStatus TilingForViewCopyPureMoveAlign(gert::TilingContext* context, ViewCopyTilingParam& tilingParam)
{
    ViewCopyTilingDataPureMoveAlign tilingData;
    DoTilingForPureMoveAlign(tilingParam);
    tilingParam.tilingKey = PURE_MOVE_TILINGKEY_BASE + tilingParam.dstDtypeSize;

    SetTilingData(tilingData, tilingParam);
    OP_CHECK_IF(ViewCopySetTilingData<ViewCopyTilingDataPureMoveAlign>(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ViewCopySetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingParam.usedCoreNum);
    context->SetTilingKey(tilingParam.tilingKey);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    PrintTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ViewCopyTilingForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "ViewCopyTilingForAscendC running begin.");

    OP_CHECK_IF(CheckInputDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "input dtype check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputShape(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "input shape check failed."), return ge::GRAPH_FAILED);

    ViewCopyTilingParam tilingParam;
    auto compileInfo = static_cast<const ViewCopyCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    tilingParam.totalCoreNum = compileInfo->CoreNum;
    tilingParam.ubSize = compileInfo->ubSize;

    // get dst and src's dtype.
    auto inputDstPtr = context->GetInputDesc(INDEX_INPUT_DST);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDstPtr);
    tilingParam.dtype = inputDstPtr->GetDataType();
    tilingParam.dstDtypeSize = ge::GetSizeByDataType(inputDstPtr->GetDataType());

    auto inputDstSizePtr = context->GetInputDesc(INDEX_INPUT_DST_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDstSizePtr);
    tilingParam.strideDtype = inputDstSizePtr->GetDataType();

    // get dst_size, dst_stride, dst_size_stride and dst_offset.
    ge::graphStatus resOfGetValue;
    if (tilingParam.strideDtype == ge::DT_INT32) {
        resOfGetValue = GetSizeAndStrideValue<int32_t>(context, tilingParam);
    } else {
        resOfGetValue = GetSizeAndStrideValue<int64_t>(context, tilingParam);
    }
    OP_CHECK_IF(resOfGetValue != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "get size or stride fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckSizeAndStrideValue(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "check size and stride fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetBaseTilingInfo(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "set base tiling info fail."), return ge::GRAPH_FAILED);

    if (IsUsePureMoveAlign(tilingParam)) {
        return TilingForViewCopyPureMoveAlign(context, tilingParam);
    }

    DoTiling(tilingParam);

    GetTilingKey(tilingParam);

    ViewCopyTilingData tilingData;
    SetTilingData(tilingData, tilingParam);
    OP_CHECK_IF(ViewCopySetTilingData<ViewCopyTilingData>(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ViewCopySetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingParam.usedCoreNum);
    context->SetTilingKey(tilingParam.tilingKey);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    PrintTilingData(tilingData, tilingParam);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ViewCopyTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "ViewCopyTiling running begin.");
    // get compile info.
    const auto compile_info = static_cast<const ViewCopyCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    return ViewCopyTilingForAscendC(context);
}

ge::graphStatus TilingPrepareViewCopyForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareViewCopyForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<ViewCopyCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->CoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->CoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ViewCopy(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForViewCopy running.");
    auto compile_info = context->GetCompiledInfo<ViewCopyCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    return TilingPrepareViewCopyForAscendC(context);
}

IMPL_OP_OPTILING(ViewCopy)
    .TilingInputsDataDependency({INDEX_INPUT_DST_SIZE, INDEX_INPUT_DST_STRIDE, INDEX_INPUT_DST_OFFSET,
                                 INDEX_INPUT_SRC_SIZE, INDEX_INPUT_SRC_STRIDE, INDEX_INPUT_SRC_OFFSET})
    .Tiling(ViewCopyTiling)
    .TilingParse<ViewCopyCompileInfo>(TilingPrepare4ViewCopy);
} // namespace optiling
