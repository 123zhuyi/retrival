/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file strided_slice_v3.cc
 * \brief
 */
#include "strided_slice_v3_tiling_arch35.h"
#include <numeric>

namespace {
const std::string OP_NAME = "StridedSliceV3";
const int INDEX_X = 0;
const int INDEX_BEGIN = 1;
const int INDEX_END = 2;
const int INDEX_AXES = 3;
const int INDEX_STRIDES = 4;
const int INDEX_Y = 0;
}  // namespace

namespace optiling {

static constexpr int32_t BYTE_BLOCK = 32;
static constexpr size_t MAX_SUPPORTED_DIMS = 8;
static constexpr int32_t SHAPE_LEN = 2;
static constexpr int32_t TILING_FACTOR_2 = 2;
static constexpr int32_t TILING_FACTOR_16 = 16;
static constexpr int64_t UB_BLOCK_SIZE = 32;
static constexpr int32_t ELEMENT_DOUBLE = 2;
static constexpr int32_t TILING_MODE_1 = 1;
static constexpr int32_t TILING_MODE_2 = 2;
static constexpr int32_t TILING_MODE_3 = 3;
static constexpr int32_t TILING_MODE_4 = 4;
static constexpr int32_t TILING_MODE_5 = 5;
static constexpr int32_t TILING_MODE_6 = 6;
static constexpr int32_t TILING_MODE_7 = 7;
static constexpr int32_t TILING_MODE_8 = 8;
static constexpr int32_t TILING_ONLY_LAST_STRIDE_LARGER1_OUT_CONTINUAL_VNCHWCONV = 9;
static constexpr int32_t TILING_LAST_STRIDE_LARGER1_SMALL_INOUT_INNER_VNCHWCONV = 10;
static constexpr int32_t TILING_LAST_STRIDE_LARGER1_LE_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV = 11;
static constexpr int32_t TILING_LAST_STRIDE_LARGER1_FUNCTIONAL = 12;
static constexpr int32_t TILING_LAST_STRIDE_LARGER_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV = 13;
static constexpr int32_t TILING_LAST_STRIDE_32B_ALIGN_LARGE_OUT_INNER_VNCHWCONV = 14;
static constexpr int32_t TILING_LAST_STRIDE_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV = 15;
static constexpr int32_t TILING_LAST_STRIDE_LARGER_BLOCK_ELE_NOT_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV = 16;
static constexpr int32_t TILING_NOT_LAST_STRIDE_LARGER1_FUNCTIONAL = 17;
static constexpr int32_t TILING_NOT_LAST_STRIDE_LARGER1_SMALL_OUT_INNER_VNCHWCONV = 18;
static constexpr int32_t VNCHWCONV_ELE4FP16 = 256;
static constexpr int32_t NUM_THREE = 3;
static constexpr int64_t TILING_UB_SIZE = 382;
static constexpr int64_t VNCHWCONV_COLUMNS_FP16 = 16;
static constexpr int64_t VNCHWCONV_COLUMNS_INT8 = 32;
static constexpr int64_t DTYPE_SIZE_FP16 = 2;
static constexpr int64_t VNCHWCONV_FACTOR = 2;
static constexpr int64_t VNCHWCONV_ROWS = 16;
static const std::pair<int64_t, std::string> BEGIN_MASK_ATTR_INFO{0, "begin_mask"};
static const std::pair<int64_t, std::string> END_MASK_ATTR_INFO{1, "end_mask"};
static const std::pair<int64_t, std::string> ELLIPSIS_MASK_ATTR_INFO{2, "ellipsis_mask"};
static const std::pair<int64_t, std::string> NEW_AXIS_MASK_ATTR_INFO{3, "new_axis_mask"};
static const std::pair<int64_t, std::string> SHRINK_AXIS_MASK_ATTR_INFO{4, "shrink_axis_mask"};
static const uint32_t MASK_ATTR_DEFAULT_VALUE = 0;
// tilingData的size和StridedSlice保持一致
constexpr size_t MAX_DIM_SUPPORTED = 8;
constexpr size_t TILING_HEAD_LEN = 2;
constexpr size_t TILING_PARAMS_COUNT = 5; // input_shape, output_shape, begin, end, stride
constexpr size_t MAX_TILING_DATA_SIZE = MAX_DIM_SUPPORTED * TILING_PARAMS_COUNT + TILING_HEAD_LEN + 1;

template<typename T>
std::string to_string_ops(const std::vector<T> &items) {
    std::ostringstream oss;
    oss << "[";
    for (const auto &item: items) {
        oss << item << ", ";
    }
    oss << "]";
    return oss.str();
}

std::string SliceParameters::to_string() const {
    std::string result = "input_shape:" + to_string_ops(input);
    result += " output_shape:" + to_string_ops(output_shape);
    result += " core_num:" + std::to_string(core_num);
    result += " begin:" + to_string_ops(begin_list);
    result += " end:" + to_string_ops(end_list);
    result += " stride:" + to_string_ops(stride_list);
    result += " tiling_mode:" + std::to_string(tiling_mode);
    return result;
}

void VectorToShape(const std::vector<int64_t>& vec, gert::Shape& shape)
{
    shape.SetDimNum(static_cast<size_t>(vec.size()));
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        shape.SetDim(i, vec[i]);
    }
}

void ConvertSliceParameters(const SliceParameters& param, SliceParametersRuntime2& sliceParam)
{
    VectorToShape(param.input, sliceParam.inputShape);
    VectorToShape(param.output_shape, sliceParam.outputShape);
    VectorToShape(param.begin_list, sliceParam.beginList);
    VectorToShape(param.end_list, sliceParam.endList);
    VectorToShape(param.stride_list, sliceParam.strideList);

    sliceParam.tilingMode = param.tiling_mode;
    sliceParam.coreNum = param.core_num;
}

ge::graphStatus StridedSliceV3TilingForAscendC(gert::TilingContext* context, int64_t coreNum, int64_t ubSize,
                                               int64_t cacheLineSize, SliceParameters& param,
                                               const ge::DataType dtype)
{
    OP_LOGD(context->GetNodeName(), "Enter StridedSliceV3TilingForAscendC.");

    StridedSliceV3Tiling tilingImpl(context);
    SliceParametersRuntime2 sliceParam;
    ConvertSliceParameters(param, sliceParam);
    if (tilingImpl.Init(coreNum, ubSize, cacheLineSize, sliceParam, dtype) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "StridedSliceV3TilingForAscendC init failed.");
        return ge::GRAPH_FAILED;
    }
    return tilingImpl.RunStrideSliceTiling();
}

static int64_t CalShapeMul(const std::vector<int64_t>& shape, int64_t start, int64_t end) {
    int64_t res = 1;
    for (; start <= end; start += 1) {
        res *= shape[start];
    }
    return res;
}

static int64_t CalVnchwUbSize(int64_t ubSize, int64_t dtypeSize) {
    OP_CHECK_IF(dtypeSize == 0, OP_LOGE("StridedSlice", "dtypeSize = 0 is not supported."),
                    return ubSize);
    int64_t blockElement = BYTE_BLOCK / dtypeSize;
    return (ubSize / dtypeSize - blockElement) / ELEMENT_DOUBLE / blockElement * blockElement;
}

static bool IsShapeEqualExceptLast(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& outputShape,
                                   int64_t end) {
    for (int64_t i = 0; i <= end; i++) {
        if (inputShape[i] != outputShape[i]) {
        return false;
        }
    }
    return true;
}

static int64_t Get32BytesAlignRows(int64_t column, int64_t blockElement) {
    if (column == 0 || blockElement == 0) {
        return UB_BLOCK_SIZE;
    }

    if (column % blockElement == 0) {
        return 1;
    }

    if (blockElement % column == 0) {
        return blockElement / column;
    }

    return blockElement;
}

static bool CanOptimStrideLargerThanOne(const int64_t &rowsEachRepeat, const int64_t &validLen,
    const int64_t &dstStride, const int64_t &vconvUbByteSize) {
    if (rowsEachRepeat <= 1) {
        return false;
    }
    return UB_BLOCK_SIZE * (rowsEachRepeat * validLen + (rowsEachRepeat - 1) * dstStride) < vconvUbByteSize;
}

static int64_t ComputeRowsEachRepeat(const int64_t &maxRowsInUb, const int64_t &coreNum,
    const int64_t &outDim, const int64_t &output32BytesAlignRows) {
    int64_t rowsEachCore = Ops::Base::CeilAlign(Ops::Base::CeilDiv(outDim, coreNum), output32BytesAlignRows);
    int64_t repeatTimes = Ops::Base::CeilDiv(rowsEachCore, maxRowsInUb);
    int64_t rowsEachRepeat = Ops::Base::CeilAlign(Ops::Base::CeilDiv(rowsEachCore, repeatTimes), output32BytesAlignRows);
    rowsEachCore = repeatTimes * rowsEachRepeat;
    if (rowsEachCore > outDim) {
        rowsEachCore = outDim;
        repeatTimes = Ops::Base::CeilDiv(rowsEachCore, maxRowsInUb);
        rowsEachRepeat = Ops::Base::CeilAlign(Ops::Base::CeilDiv(rowsEachCore, repeatTimes), output32BytesAlignRows);
        if (rowsEachRepeat) {
            rowsEachRepeat = outDim;
        }
    }
    return rowsEachRepeat;
}

static int64_t GetTilingMode4StridesLargerThanOne(const SliceParameters& parameters,
                                                  const ge::DataType& dtype, int32_t ubSize,
                                                  const std::string& opType, int32_t coreNum) {
    const int64_t dtypeSize = static_cast<int64_t>(GetSizeByDataType(dtype));
    const int64_t blockElement = Ops::Base::FloorDiv(UB_BLOCK_SIZE, dtypeSize);
    const auto& strides = parameters.stride_list;
    const int64_t lastStride = *strides.crbegin();
    const bool isLastStrideLargerThanOne = lastStride > 1;
    const int64_t needUbCount = 2;
    const int64_t ubUsefulFactor = dtypeSize % 2 == 0 ? 1 : 2;
    const int64_t reserveUbSize = 512;
    const int64_t vnchwconvUbSize =
        Ops::Base::FloorAlign(ubSize / needUbCount / ubUsefulFactor, UB_BLOCK_SIZE) - reserveUbSize;
    const int64_t inputInner = *parameters.input.crbegin();
    const int64_t outputInner = *parameters.output_shape.crbegin();
    const int64_t output32bytesAlignRows = Get32BytesAlignRows(outputInner, blockElement);
    const int64_t inputInnerBytes = inputInner * dtypeSize;
    const int64_t outputInnerBytes = outputInner * dtypeSize;
    OP_LOGD(opType, "outputInnerBytes:%ld, output32bytesAlignRows:%ld, vnchwconvUbSize:%ld.",
            outputInnerBytes, output32bytesAlignRows, vnchwconvUbSize);
    if (!isLastStrideLargerThanOne) {
        if (Ops::Base::CeilAlign(outputInnerBytes, UB_BLOCK_SIZE) * output32bytesAlignRows * UB_BLOCK_SIZE <=
            vnchwconvUbSize) {
            return TILING_NOT_LAST_STRIDE_LARGER1_SMALL_OUT_INNER_VNCHWCONV;
        }

        return TILING_NOT_LAST_STRIDE_LARGER1_FUNCTIONAL;
    }

    const bool isOnlyLastStrideLargerThanOne =
        isLastStrideLargerThanOne &&
        std::find_if(strides.crbegin() + 1, strides.crend(), [](int64_t x) -> bool { return x > 1; }) == strides.crend();
    if (isOnlyLastStrideLargerThanOne && parameters.output_shape.size() > 1 &&
        Ops::Base::CeilAlign(inputInnerBytes * output32bytesAlignRows, UB_BLOCK_SIZE) * VNCHWCONV_ROWS <= vnchwconvUbSize) {
        bool outContinues = true;
        int64_t outDim = parameters.output_shape[0];
        for (size_t i = 1; i < parameters.output_shape.size() - 1; i++) {
            outDim = static_cast<int64_t>(outDim * parameters.output_shape[i]);
            if (parameters.output_shape[i] != parameters.input[i]) {
                outContinues = false;
                break;
            }
        }

        int64_t alignUbSize = static_cast<int64_t>((ubSize - TILING_UB_SIZE) /
            dtypeSize / blockElement * blockElement);
        int64_t multiTimes = dtypeSize / DTYPE_SIZE_FP16 > 1 ? dtypeSize / DTYPE_SIZE_FP16 : 1;
        int64_t vconvUbSize = alignUbSize * multiTimes / 2;
        int64_t multiBlockElement = blockElement * multiTimes;
        int64_t vconvColumn = dtypeSize % DTYPE_SIZE_FP16 == 0 ? VNCHWCONV_COLUMNS_FP16 : VNCHWCONV_COLUMNS_INT8;
        int64_t alignColSize = Ops::Base::CeilAlign(inputInner * output32bytesAlignRows, multiBlockElement);
        int64_t maxRowsInUb = Ops::Base::FloorDiv(vconvUbSize / vconvColumn, alignColSize) * output32bytesAlignRows;
        bool isValidLen = outputInner - multiTimes > 0;
        int64_t validLen = (outputInner - multiTimes) * lastStride + multiTimes;
        int64_t dstStride = (lastStride - 1) * multiTimes;
        auto rowsEachRepeat = ComputeRowsEachRepeat(maxRowsInUb, coreNum, outDim, output32bytesAlignRows);
        int64_t vconvUbByte = vconvUbSize * dtypeSize;

        if (outContinues && isValidLen && CanOptimStrideLargerThanOne(rowsEachRepeat, validLen,
            dstStride, vconvUbByte)) {
            return TILING_ONLY_LAST_STRIDE_LARGER1_OUT_CONTINUAL_VNCHWCONV;
        }
    }

    if (Ops::Base::CeilAlign(outputInnerBytes * lastStride, UB_BLOCK_SIZE) * output32bytesAlignRows * VNCHWCONV_ROWS <=
        vnchwconvUbSize) {
        return TILING_LAST_STRIDE_LARGER1_SMALL_INOUT_INNER_VNCHWCONV;
    }

    if (lastStride <= blockElement) {
        if (lastStride * dtypeSize * output32bytesAlignRows * VNCHWCONV_ROWS > vnchwconvUbSize) {
            return TILING_LAST_STRIDE_LARGER1_FUNCTIONAL;
        }
        if (outputInner >= blockElement) {
            return TILING_LAST_STRIDE_LARGER1_LE_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV;
        }

        return TILING_LAST_STRIDE_LARGER1_FUNCTIONAL;
    }

    if (outputInnerBytes * blockElement * output32bytesAlignRows * VNCHWCONV_ROWS <= vnchwconvUbSize) {
        if (lastStride % blockElement == 0) {
            return TILING_LAST_STRIDE_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV;
        }

        return TILING_LAST_STRIDE_LARGER_BLOCK_ELE_NOT_32B_ALIGN_SMALL_OUT_INNER_VNCHWCONV;
    }

    if ((lastStride * dtypeSize) % UB_BLOCK_SIZE == 0) {
        if (outputInner >= blockElement) {
            return TILING_LAST_STRIDE_32B_ALIGN_LARGE_OUT_INNER_VNCHWCONV;
        }

        return TILING_LAST_STRIDE_LARGER1_FUNCTIONAL;
    }

    if (outputInner >= blockElement) {
        return TILING_LAST_STRIDE_LARGER_BLOCK_ELE_LARGE_OUT_INNER_VNCHWCONV;
    }

    return TILING_LAST_STRIDE_LARGER1_FUNCTIONAL;
}

void SetTilingMode(SliceParameters& parameters, int32_t coreNum, const ge::DataType& dtype, int32_t ubSize,
                   const std::string& opType) {
    const auto& strides = parameters.stride_list;
    auto found = std::find_if(strides.cbegin(), strides.cend(), [](int64_t x) -> bool { return x > 1; });
    if (found != strides.cend()) {
        parameters.tiling_mode = GetTilingMode4StridesLargerThanOne(parameters, dtype, ubSize, opType, coreNum);
        return;
    }

    int64_t dtypeSize = GetSizeByDataType(dtype);
    const int32_t uint16Max = 65535;
    const int32_t STRIDE_LIMIT = uint16Max * BYTE_BLOCK;
    OP_LOGD(opType.c_str(), "param input/output tensor's data type: %s, dtype size: %ld", Ops::Base::ToString(dtype).c_str(),
            dtypeSize);
    OP_LOGD(opType.c_str(), "param CalVnchwUbSize: %ld", CalVnchwUbSize(ubSize, dtypeSize));
    int64_t shapeLen = parameters.output_shape.size();
    int64_t float16TypeSize = 2;

    if (parameters.output_shape[shapeLen - 1] * dtypeSize < BYTE_BLOCK) {
        parameters.tiling_mode = TILING_MODE_1;
    } else {
        parameters.tiling_mode = TILING_MODE_2;
    }

    const int64_t b16Bytes = 2;
    const int64_t minDimLen = 16;
    const int64_t twoDim = 2;
    if (parameters.output_shape[shapeLen - 1] * dtypeSize < BYTE_BLOCK && shapeLen >= SHAPE_LEN &&
        dtypeSize == b16Bytes && CalShapeMul(parameters.output_shape, 0, shapeLen - NUM_THREE) % coreNum == 0 &&
        parameters.output_shape[shapeLen - twoDim] >= minDimLen &&
        parameters.input[shapeLen - 1] * VNCHWCONV_ELE4FP16 <= CalVnchwUbSize(ubSize, dtypeSize)) {
        parameters.tiling_mode = TILING_MODE_3;
    }

    const int64_t minShapeLen = 2;
    if (shapeLen >= minShapeLen && parameters.output_shape[shapeLen - 1] * dtypeSize % BYTE_BLOCK == 0 &&
        parameters.input[shapeLen - 1] * dtypeSize % BYTE_BLOCK == 0 &&
        IsShapeEqualExceptLast(parameters.input, parameters.output_shape, shapeLen - SHAPE_LEN) &&
        ubSize >= ELEMENT_DOUBLE * parameters.output_shape[shapeLen - 1] * dtypeSize &&
        (parameters.input[shapeLen - 1] - parameters.output_shape[shapeLen - 1]) * dtypeSize <= STRIDE_LIMIT) {
        parameters.tiling_mode = TILING_MODE_4;
    }

    if (shapeLen == SHAPE_LEN && IsShapeEqualExceptLast(parameters.input, parameters.output_shape, shapeLen - \
        SHAPE_LEN) && parameters.output_shape[shapeLen - 1] * dtypeSize > UB_BLOCK_SIZE &&
        parameters.input[shapeLen - 1] * BYTE_BLOCK * TILING_FACTOR_2 <= ubSize &&
        parameters.input[shapeLen - 1] * dtypeSize > UB_BLOCK_SIZE) {
        parameters.tiling_mode = TILING_MODE_6;
    }

    if (shapeLen == SHAPE_LEN && IsShapeEqualExceptLast(parameters.input, parameters.output_shape, shapeLen - \
        SHAPE_LEN) && parameters.output_shape[shapeLen - 1] == 1 && BYTE_BLOCK * \
        (parameters.input[shapeLen - 1] + 1) < ubSize) {
        parameters.tiling_mode = TILING_MODE_8;
    }

    int64_t multiTimes = dtypeSize / float16TypeSize;
    int64_t inputInnerDims = parameters.input[shapeLen - 1] * multiTimes;
    int64_t outputInnerDims = parameters.output_shape[shapeLen - 1] * multiTimes;
    int64_t output32bytesAlignRows = BYTE_BLOCK / float16TypeSize;
    if (outputInnerDims > 0 && output32bytesAlignRows % outputInnerDims == 0) {
        output32bytesAlignRows = output32bytesAlignRows / outputInnerDims;
    } else if (outputInnerDims % output32bytesAlignRows == 0) {
        output32bytesAlignRows = 1;
    }
    int64_t needUbSize = inputInnerDims * TILING_FACTOR_16 * TILING_FACTOR_2;
    OP_LOGD(opType.c_str(), "needUbSize: %ld", needUbSize);
    if (shapeLen == SHAPE_LEN && IsShapeEqualExceptLast(parameters.input, parameters.output_shape, shapeLen - \
        SHAPE_LEN) && needUbSize * BYTE_BLOCK / dtypeSize * output32bytesAlignRows < ubSize &&
        dtypeSize % float16TypeSize == 0) {
        parameters.tiling_mode = TILING_MODE_5;
    }

    if (shapeLen == 1 && parameters.stride_list[0] == 1) {
        parameters.tiling_mode = TILING_MODE_7;
    }

    OP_LOGD(opType.c_str(), "parameters.tiling_mode: %ld", parameters.tiling_mode);
}

static void MakePerformanceParamsNeg(SliceParameters &param)
{
    OP_LOGI("", "before handle negative perf slice params: %s", param.to_string().c_str());
    SliceParameters perfParams;
    size_t perfSize = 0;
    for (size_t i = 0; i < param.input.size(); i++) {
        const auto inputShapeI = param.input[i];
        const auto outputShapeI = param.output_shape[i];
        const auto beginI = param.begin_list[i];
        const auto endI = param.end_list[i];
        const auto stride_i = endI > beginI ? std::min(param.stride_list[i], endI - beginI) :
                                              std::max(param.stride_list[i], endI - beginI);
        if (inputShapeI == 1 && outputShapeI == 1 && i != 0) {
            continue;
        }
        // Continuous stride=1 axis fused already. Continuous stride=-1 can fuse too.
        if (i == 0 || inputShapeI != outputShapeI || stride_i != -1 ||
            (!perfParams.stride_list.empty() && perfParams.stride_list[perfSize - 1] != -1)) {
            perfParams.input.push_back(inputShapeI);
            perfParams.output_shape.push_back(outputShapeI);
            perfParams.begin_list.push_back(beginI);
            perfParams.end_list.push_back(endI);
            perfParams.stride_list.push_back(stride_i);
            perfSize++;
            continue;
        }
        const auto perfIndex = perfSize - 1;
        perfParams.input[perfIndex] *= inputShapeI;
        perfParams.output_shape[perfIndex] *= outputShapeI;
        perfParams.begin_list[perfIndex] = perfParams.begin_list[perfIndex] * inputShapeI + inputShapeI - 1;
        perfParams.end_list[perfIndex] = perfParams.end_list[perfIndex] * inputShapeI + inputShapeI - 1;
        perfParams.stride_list[perfIndex] = -1;
    }

    param = perfParams;

    for (size_t i = 0; i < param.input.size(); i++) {
        if (param.output_shape[i] == 1 && param.stride_list[i] < 0) {
            param.stride_list[i] = 1;
            param.end_list[i] = param.begin_list[i] + 1;
        }
    }
}

static int64_t CanonicalIndex(int64_t dimValue, int64_t inputShapeI, const std::array<int64_t, 2>& validRange) {
    int64_t dimValuePos = dimValue < 0 ? inputShapeI + dimValue : dimValue;
    if (dimValuePos < validRange[0]) {
        return validRange[0];
    }

    return std::min(dimValuePos, validRange[1]);
}

void MakePerformanceParams(SliceParameters& para, bool isAscendc) {
    if (para.output_shape.empty()) {
        return;
    }

    bool hasNegStride = false;
    for (size_t i = 0; i < para.input.size(); i++) {
        if (para.stride_list[i] < 0) {
            hasNegStride = true;
            break;
        }
    }

    SliceParameters perfParams;
    size_t perfSize = 0;
    for (size_t i = 0; i < para.input.size(); i++) {
        const auto inputShapeI = para.input[i];
        const auto outputShapeI = para.output_shape[i];
        auto beginI = para.begin_list[i];
        auto endI = para.end_list[i];

        const std::array<int64_t, 2> validRange = {
            {para.stride_list[i] > 0 ? 0 : -1, para.stride_list[i] > 0 ? inputShapeI : inputShapeI - 1}};
        beginI = CanonicalIndex(beginI, inputShapeI, validRange);
        endI = CanonicalIndex(endI, inputShapeI, validRange);

        const auto strideI = endI > beginI ? std::min(para.stride_list[i], endI - beginI) : para.stride_list[i];
        if (i == 0 || inputShapeI != outputShapeI || strideI != 1 || perfParams.stride_list[perfSize - 1] != 1) {
            perfParams.input.push_back(inputShapeI);
            perfParams.output_shape.push_back(outputShapeI);
            perfParams.begin_list.push_back(beginI);
            perfParams.end_list.push_back(endI);
            perfParams.stride_list.push_back(strideI);
            perfSize++;
            continue;
        }

        const auto perf_index = perfSize - 1;
        perfParams.input[perf_index] *= inputShapeI;
        perfParams.output_shape[perf_index] *= outputShapeI;
        perfParams.begin_list[perf_index] *= inputShapeI;
        perfParams.end_list[perf_index] *= inputShapeI;
        perfParams.stride_list[perf_index] = 1;
    }

    para = perfParams;

    if (isAscendc && hasNegStride) {
        MakePerformanceParamsNeg(para);
    }
}

template <typename T>
static void PositiveAxisImpl(int32_t inputDims, const gert::Tensor* axesTensor, std::vector<int64_t>& newAxes) {
    int32_t axesSize = static_cast<int32_t>(axesTensor->GetShapeSize());
    const T* data = axesTensor->GetData<T>();
    for (int32_t i = 0; i < axesSize; i++) {
        int32_t value = static_cast<int32_t>(data[i]);
        if (value >= 0 && value < inputDims) {
            newAxes.push_back(value);
        } else if (value < 0 && value >= -inputDims) {
            newAxes.push_back(value + inputDims);
        }
    }
    return;
}

static std::vector<int64_t> ConstructValidAxis(const gert::Tensor* axesTensor, int32_t inputDims) {
    std::vector<int64_t> newAxes;
    if (!axesTensor || axesTensor->GetShapeSize() == 0) {
        newAxes.resize(inputDims);
        std::iota(newAxes.begin(), newAxes.end(), 0);
        return newAxes;
    }
    if (axesTensor->GetDataType() == ge::DT_INT32) {
        PositiveAxisImpl<int32_t>(inputDims, axesTensor, newAxes);
    } else {
        PositiveAxisImpl<int64_t>(inputDims, axesTensor, newAxes);
    }
    return newAxes;
}

static void ConstructSliceShape(const gert::Shape& shape, int32_t dimNum, std::vector<int64_t>& param) {
    param.resize(dimNum);
    for (int32_t i = 0; i < dimNum; i++) {
        param[i] = shape.GetDim(i);
    }
    return;
}

static int64_t GetConstIndexValue(const gert::Tensor* tensor, int32_t idx) {
    // idx must be valid
    int64_t value = 0;
    if (tensor->GetDataType() == ge::DT_INT32) {
        const int32_t* data = tensor->GetData<int32_t>();
        value = static_cast<int64_t>(data[idx]);
    } else {
        const int64_t* data = tensor->GetData<int64_t>();
        value = data[idx];
    }
    OP_LOGD(OP_NAME.c_str(), "const tensor[%d] is %ld.", idx, value);
    return value;
}

static int64_t GetConstIndexValue(const gert::Tensor* tensor, int32_t idx, int64_t inputSize) {
    // idx must be valid
    int64_t value = 0;
    if (tensor->GetDataType() == ge::DT_INT32) {
        const int32_t* data = tensor->GetData<int32_t>();
        value = static_cast<int64_t>(data[idx]);
    } else {
        const int64_t* data = tensor->GetData<int64_t>();
        value = data[idx];
    }
    if (value < 0) {
        value += inputSize;
    }

    // clamp value
    if (value < 0) {
        value = 0;
    } else if (value > inputSize) {
        value = inputSize;
    }
    OP_LOGD(OP_NAME.c_str(), "const tensor[%d] is %ld.", idx, value);
    return value;
}

static void ConstructStrideList(const gert::Tensor* strideTensor, int64_t dimNum, const std::vector<int64_t>& axes,
                                std::vector<int64_t>& strideVec) {
    // init strideVec with 1
    strideVec.assign(dimNum, 1);
    if (!strideTensor) {
        OP_LOGD(OP_NAME.c_str(), "Stride tensor is null. Set stride as 1.");
        return;
    }

    // update strideVec with const value of strideTensor
    const int32_t strideSize = static_cast<int32_t>(strideTensor->GetShapeSize());
    const int32_t axesSize = static_cast<int32_t>(axes.size());
    for (int32_t i = 0; i < axesSize && i < strideSize; i++) {
        int64_t axesValue = axes[i];
        strideVec[axesValue] = GetConstIndexValue(strideTensor, i);
    }
    return;
}

static void ConstructBeginList(const gert::Tensor* beginTensor, const gert::Shape& xShape,
                               const std::vector<int64_t>& axes, std::vector<int64_t>& beginVec) {
    // init beginVec with 0
    const int32_t dimNum = static_cast<int32_t>(xShape.GetDimNum());
    beginVec.assign(dimNum, 0);

    // update beginVec with const value of beginTensor
    const int32_t beginsSize = static_cast<int32_t>(beginTensor->GetShapeSize());
    const int32_t axesSize = static_cast<int32_t>(axes.size());
    for (int32_t i = 0; i < axesSize && i < beginsSize; i++) {
        int64_t axesValue = axes[i];
        beginVec[axesValue] = GetConstIndexValue(beginTensor, i, xShape.GetDim(axesValue));
    }
    return;
}

static void ConstructEndList(const gert::Tensor* endTensor, const gert::Shape& xShape,
                             const std::vector<int64_t>& axes, std::vector<int64_t>& endVec) {
    // init endVec with input_shape
    const int32_t dimNum = static_cast<int32_t>(xShape.GetDimNum());
    endVec.resize(dimNum);
    for (int32_t i = 0; i < dimNum; i++) {
        endVec[i] = xShape.GetDim(i);
    }

    // update endVec with const value of endTensor
    const int32_t endSize = static_cast<int32_t>(endTensor->GetShapeSize());
    const int32_t axesSize = static_cast<int32_t>(axes.size());
    for (int32_t i = 0;  i < axesSize && i < endSize; i++) {
        int64_t axesValue = axes[i];
        endVec[axesValue] = GetConstIndexValue(endTensor, i, xShape.GetDim(axesValue));
    }
    return;
}

static ge::graphStatus TilingPrepareForStridedSliceV3AscendC(gert::TilingParseContext* context,
    StridedSliceV3CompileInfo* compileInfo)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepareForStridedSliceV3AscendC.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->block_dim = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->block_dim <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "core number", std::to_string(compileInfo->block_dim),
                "The value of core number cannot <= 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int64_t>(ubSize);
    if (compileInfo->ub_size <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "ubSize", std::to_string(compileInfo->ub_size),
                "The value of ubSize cannot <= 0.");
        return ge::GRAPH_FAILED;
    } 

    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    if (compileInfo->cacheLineSize == 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "cacheLineSize", std::to_string(compileInfo->cacheLineSize),
                "The value of cacheLineSize cannot be 0.");
        return ge::GRAPH_FAILED;
    }  
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForStridedSliceV3(gert::TilingParseContext* context) {
    auto compileInfo = context->GetCompiledInfo<StridedSliceV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    return TilingPrepareForStridedSliceV3AscendC(context, compileInfo);
}

static ge::graphStatus TilingForStridedSliceV3(gert::TilingContext* context) {
    const gert::StorageShape* xStorage = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    const gert::StorageShape* yStorage = context->GetOutputShape(INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);

    struct SliceParameters sliceParam;
    const gert::Shape& xShape = Ops::Base::EnsureNotScalar(xStorage->GetOriginShape());
    int32_t inputDimNum = static_cast<int32_t>(xShape.GetDimNum());
    ConstructSliceShape(xShape, inputDimNum, sliceParam.input);
    const gert::Shape& yShape = Ops::Base::EnsureNotScalar(yStorage->GetOriginShape());
    ConstructSliceShape(yShape, inputDimNum, sliceParam.output_shape);

    std::vector<int64_t> newAxes = ConstructValidAxis(context->GetOptionalInputTensor(INDEX_AXES), inputDimNum);
    const gert::Tensor* beginTensor = context->GetInputTensor(INDEX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, beginTensor);
    ConstructBeginList(beginTensor, xShape, newAxes, sliceParam.begin_list);
    const gert::Tensor* endTensor = context->GetInputTensor(INDEX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, endTensor);
    ConstructEndList(endTensor, xShape, newAxes, sliceParam.end_list);
    ConstructStrideList(context->GetOptionalInputTensor(INDEX_STRIDES), inputDimNum, newAxes,
                        sliceParam.stride_list);

    auto compileInfo = context->GetCompileInfo<StridedSliceV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(OP_NAME.c_str(), "before make performance, slice params: %s", sliceParam.to_string().c_str());
    MakePerformanceParams(sliceParam, true);
    OP_LOGD(OP_NAME.c_str(), "perf slice params: %s", sliceParam.to_string().c_str());

    const auto xTensor = context->GetInputDesc(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);

    return StridedSliceV3TilingForAscendC(
        context, compileInfo->block_dim, compileInfo->ub_size, compileInfo->cacheLineSize, sliceParam,
        xTensor->GetDataType());
}

IMPL_OP_OPTILING(StridedSliceV3)
    .Tiling(TilingForStridedSliceV3)
    .TilingParse<StridedSliceV3CompileInfo>(TilingPrepareForStridedSliceV3);
}  // namespace optiling