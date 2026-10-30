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
 * \file concat_tiling_arch35.cpp
 * \brief concat tiling for ascendC impl
 */

#include "concat_tiling_arch35.h"
#include "log/log.h"
#include <cmath>
#include <sstream>
#include <cctype>
#include "op_common/op_host/util/shape_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_api/op_util.h"
#include <algorithm>

using namespace std;
using namespace ge;

namespace optiling {

constexpr size_t CONCAT_DIM_IDX = 0;
constexpr int64_t INVLID_CONCAT_DIM_IDX = static_cast<int64_t>(-1);
constexpr size_t PACK_ATTR_AXIS_IDX = 0;
constexpr size_t PACK_INPUT_IDX = 0;
constexpr int64_t PACK_AXIS_DEFAULT_VALUE = 1;
constexpr int64_t DIM0 = 0;
constexpr int64_t DIM1 = 1;
constexpr int64_t DIM2 = 2;
constexpr int64_t HALF = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DIM1_ALIGN_THRESHOLD = 128;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t MIN_RESERVED_SIZE = 2048;        // 2k
constexpr size_t SYSTEM_WORKSPACE_SIZE = 16777216; // 16M
constexpr int64_t INDEX_USE_UB = 1024;             // 预留1k空间给索引
constexpr int64_t TENS_DIGITS = 10;
constexpr int64_t HUNDREDS_DIGITS = 100;
constexpr int64_t THOUSANDS_DIGITS = 1000;
constexpr int64_t TEN_THOUSANDS_DIGITS = 10000;
constexpr int64_t LEAST_ROWS = 64; // ub切分的最小行数
constexpr int64_t LEAST_COLS = 256;
constexpr bool ENABLE_DB = true;
constexpr int64_t B64_BYTES = 8;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B8_BYTES = 1;
constexpr int64_t B4_BYTES = 1004; // ge::GetSizeByDataType 对 FP4 类型的返回值（枚举值，非实际字节数）
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t FP4_TO_B8_RATIO = 2; //用于FP4到B8模板的转换 2个FP4= 1字节
constexpr int64_t GATHER_MODE = 3;
constexpr int64_t EVERY_CORE_THRESHOLD = 2048; // 2k
constexpr int64_t LEAST_BLOCK_BYTES = 512;
constexpr int64_t PURE_COPY_COL_THRESHOLD_BASE = 256;
constexpr int64_t PURE_COPY_COL_THRESHOLD_ALIGN = 1024;
constexpr int64_t PURE_COPY_COL_THRESHOLD_NOALIGN = 2048;
constexpr int64_t BLOCK_THRESHOLD = 49152;    // 48k
constexpr double LARGE_TENSOR_RATIO_THRESHOLD = 0.9;
constexpr int64_t PURE_COPY_NO_SPLIT_DIM1_TILINGKEY = 20001;
constexpr int64_t PURE_COPY_SPLIT_DIM1_TILINGKEY = 20002;
constexpr int64_t SIMT_PER_CORE_THRESHOLD = 65536; // 64k
constexpr int64_t SIMT_TILINGKEY_PREFIX = 30000;
constexpr int64_t SIMT_COMPARE_THRESHOLD = 1024;
constexpr int64_t SMALL_BAG = 128;
constexpr int64_t ALL_DATA_SMALL = 8192;

constexpr int32_t NUM_2 = 2;
constexpr int32_t NUM_3 = 3;

template <typename T>
inline static ge::graphStatus ConcatSetTilingData(gert::TilingContext* context, T& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

template <typename T>
static inline void PrintTilingDataList(T &tilingData)
{
    auto strideList = tilingData.arrays.get_strideList();
    for (int32_t i = 0; i < tilingData.get_tensorNum(); i++) {
        OP_LOGI("[Concat list]","tensor: %d, stride: %ld", i, strideList[i]);
    }
}

template <typename T>
static inline void PrintTilingData(T& tilingData, int64_t tilingKey, int64_t usedCoreNum)
{
    OP_LOGI(
        "[Concat]",
        "ubSplitDim1: %d, dim: %d, blockFactor: %ld,tailBlockFactor: %ld,\
ubFactorDim0: %d,ubFactorDim1: %d,tailUbFactorDim0: %d, tailUbFactorDim1: %d,uoDim0: %ld,uoDim1: %ld,\
tensorNum: %d,catDim1: %ld,isnon: %d,tilingKey: %ld,usedCoreNum: %ld",
        tilingData.get_ubSplitDim1(), tilingData.get_dim(), tilingData.get_blockFactor(),
        tilingData.get_tailBlockFactor(), tilingData.get_ubFactorDim0(), tilingData.get_ubFactorDim1(),
        tilingData.get_tailUbFactorDim0(), tilingData.get_tailUbFactorDim1(), tilingData.get_uoDim0(),
        tilingData.get_uoDim1(), tilingData.get_tensorNum(), tilingData.get_catDim1(), tilingData.get_isNonContiguous(), tilingKey, usedCoreNum);
    PrintTilingDataList(tilingData);
}

inline static ge::graphStatus GetTensorList(
    const gert::TilingContext* context, ConcatTilingParam& param, int64_t inputIdx)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfo);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();
    for (uint32_t i = 0; i < inputNum; ++i) {
        gert::Shape inputTensorShape = GetShapeByAll(context, param.isNonContiguous, inputIdx, i);
        size_t inputTensorDimNum = inputTensorShape.GetDimNum();
        vector<int64_t> inputShapeList(inputTensorDimNum, 0);
        for (size_t j = 0; j < inputTensorDimNum; j++) {
            inputShapeList[j] = inputTensorShape.GetDim(j);
        }
        param.tensorList.push_back(inputShapeList);
    }
    return ge::GRAPH_SUCCESS;
}

inline static int64_t MergeDim(const vector<int64_t>& tensorSize, int64_t startIdx, int64_t endIdx)
{
    int64_t ans = 1;
    for (int64_t i = startIdx; i < endIdx; i++) {
        ans *= tensorSize[i];
    }
    return ans;
}

inline static void GetTensorListDim(ConcatTilingParam& param)
{
    vector<int64_t> tmpTensor(DIM2);
    for (const auto& tensorSize : param.tensorList) {
        tmpTensor[DIM0] = MergeDim(tensorSize, 0, param.dim);
        tmpTensor[DIM1] = MergeDim(tensorSize, param.dim, tensorSize.size());
        param.mergeTensorList.push_back(tmpTensor);
    }

    for (const auto& tensorSize : param.mergeTensorList) {
        param.tensorListDim0.push_back(tensorSize[0]);
        param.tensorListDim1.push_back(tensorSize[1]);
    }
}

inline static void GetTensorSameDim1(ConcatTilingParam& param)
{
    if (static_cast<int64_t>(param.tensorListDim1.size()) > 0) {
        if (param.inputShapeSame == 1) {
            param.sameShapeTensorDim1 = param.tensorListDim1[0];
        } else {
            // shape 不同时只保存concat轴之后的相同部分
            param.sameShapeTensorDim1 = MergeDim(param.tensorList[0], param.dim + 1, param.tensorList[0].size());
        }
    }
}

inline static int64_t CalcSum(const vector<int64_t>& vec)
{
    int64_t sum = 0;
    for (const auto& it : vec) {
        sum += it;
    }
    return sum;
}

inline static void GenerateOutputShape(ConcatTilingParam& param)
{
    if (static_cast<int64_t>(param.tensorListDim0.size()) > 0) {
        param.catDim0 = param.tensorListDim0[0];
    } else {
        param.catDim0 = 0;
    }
    param.catDim1 = CalcSum(param.tensorListDim1);
    param.isEmpty = (param.catDim0 * param.catDim1) == 0;
}

inline static void CalcNoAlignTensorNum(ConcatTilingParam& param, int64_t dtypeSize)
{
    int64_t num = 0;
    for (const auto& tensorSize : param.tensorListDim1) {
        if (tensorSize * dtypeSize % BLOCK_SIZE != 0) {
            num += 1;
        }
    }
    param.noAlignTensorNum = num;
    OP_LOGD("[Concat]", "noAlignTensorNum: %ld", param.noAlignTensorNum);
}

inline static bool CheckCatDimAlign(vector<vector<int64_t>>& mergeTensorList, int64_t dtypeSize)
{
    // 用合轴之后的1轴去判断是否对齐
    for (int64_t i = 0; i < static_cast<int64_t>(mergeTensorList.size()); i++) {
        if (mergeTensorList[i][DIM1] * dtypeSize % BLOCK_SIZE != 0) {
            return false;
        }
    }
    return true;
}

inline static bool CheckDim1Align(vector<vector<int64_t>>& mergeTensorList, int64_t dtypeSize)
{
    // 用合轴之后的1轴去判断是否128对齐
    for (int64_t i = 0; i < static_cast<int64_t>(mergeTensorList.size()); i++) {
        if (mergeTensorList[i][DIM1] * dtypeSize % DIM1_ALIGN_THRESHOLD != 0) {
            return false;
        }
    }
    return true;
}

inline static bool CheckInputShapeSame(vector<vector<int64_t>>& tensorList)
{
    for (int64_t i = 0; i < static_cast<int64_t>(tensorList.size()) - 1; i++) {
        if (tensorList[i] != tensorList[i + 1]) {
            return false;
        }
    }
    return true;
}

inline static ge::graphStatus CheckFP4Dim1Even(const ConcatTilingParam& param)
{
    for (const auto& tensorSize : param.tensorListDim1) {
        if (tensorSize % FP4_TO_B8_RATIO != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Concat", "tensor_dim1",
                std::to_string(tensorSize).c_str(),
                "The value of tensor_dim1 must be an even number for FP4 dtype.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertFP4DimsToB8(ConcatTilingParam& param)
{
    if (!param.isFP4Type) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus ret = CheckFP4Dim1Even(param);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    param.catDim1 /= FP4_TO_B8_RATIO;
    if (param.inputShapeSame == 1) {
        param.sameShapeTensorDim1 /= FP4_TO_B8_RATIO;
    }
    for (auto& tensorSize : param.tensorListDim1) {
        tensorSize /= FP4_TO_B8_RATIO;
    }
    for (auto& tensorSize : param.mergeTensorList) {
        tensorSize[1] /= FP4_TO_B8_RATIO;
    }
    if (param.isNonContiguous) {
        for (int16_t i = 0; i < param.tensorNum; ++i) {
            param.strideList[i] /= FP4_TO_B8_RATIO;
        }
    }
    param.dtypeSize = B8_BYTES;
    param.orgDtypeSize = B8_BYTES;
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus CalcBaseTilingParam(const gert::TilingContext* context, ConcatTilingParam& param)
{
    auto compileInfo = reinterpret_cast<const ConcatDCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    param.totalCoreNum = min(static_cast<int64_t>(compileInfo->totalCoreNum), TILING_ARRAY_LENGTH);
    if (compileInfo->totalCoreNum > TILING_ARRAY_LENGTH) {
        OP_LOGW("[Concat]", "Currently, more than 72 cores are not supported, Only 72 cores are used.");
    }
    param.ubSize = compileInfo->ubSize;
    param.tensorNum = param.tensorList.size();
    param.gatherThreshold = compileInfo->vectorLen / DIGIT_TWO;
    GetTensorListDim(param);
    GenerateOutputShape(param);
    param.orgDtypeSize = param.dtypeSize;
    param.inputShapeSame = CheckInputShapeSame(param.mergeTensorList) ? 1 : 0;
    GetTensorSameDim1(param);
    // FP4 预处理：在对齐判断之前，将 FP4(4bit) 转换为 B8(1byte) 视角
    param.isFP4Type = (param.orgDtypeSize == B4_BYTES) ? 1 : 0;
    OP_CHECK_IF(
        ConvertFP4DimsToB8(param) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "ConvertFP4DimsToB8 failed."), return ge::GRAPH_FAILED);
    param.isAllTensorAlign = CheckCatDimAlign(param.mergeTensorList, param.dtypeSize) ? 1 : 0;
    param.isDim1AllAlign = CheckDim1Align(param.mergeTensorList, param.dtypeSize) ? 1 : 0;
    OP_CHECK_IF(
        param.dtypeSize <= 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "input",
            std::to_string(param.dtypeSize).c_str(),
            "The dtype size of input must be greater than 0."),
        return ge::GRAPH_FAILED);
    param.leastCopyNumber = MIN_RESERVED_SIZE / param.dtypeSize;
    param.everyBlockNumber = BLOCK_SIZE / param.dtypeSize;
    CalcNoAlignTensorNum(param, param.dtypeSize);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static ge::graphStatus GetConcatDimInput(
    const gert::TilingContext* context, ConcatTilingParam& param, int64_t dimIdx)
{
    auto concatDimTensor = context->GetRequiredInputTensor(dimIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimTensor);
    const T* concatDimValPtr = concatDimTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimValPtr);
    param.dim = concatDimValPtr[0];
    return ge::GRAPH_SUCCESS;
}

inline static bool IsInvalidType(const DataType dtype)
{
    std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT16, ge::DT_INT16,
        ge::DT_UINT32, ge::DT_INT32,   ge::DT_UINT64, ge::DT_INT64, ge::DT_BOOL, ge::DT_DOUBLE, ge::DT_COMPLEX64,
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E2M1};
    bool isInvalidType = (supportedDtype.count(dtype) == 0);

    return isInvalidType;
}

inline static ge::graphStatus GetDtypeSize(
    const gert::TilingContext* context, ConcatTilingParam& param, size_t inputIndex)
{
    auto inputDesc = context->GetDynamicInputDesc(inputIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputDataType = inputDesc->GetDataType();
    param.dtypeSize = ge::GetSizeByDataType(inputDataType);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static void DupTensor(vector<T>& dst, const vector<T>& src, int64_t num)
{
    int64_t index = 0;
    for (int i = 0; i < num; i++) {
        for (int64_t j = 0; (j < static_cast<int64_t>(src.size()) && index < TILING_ARRAY_LENGTH); j++) {
            dst[index] = src[j];
            index++;
        }
    }
}

inline static bool IsEnableGather(const ConcatTilingParam& param)
{
    if (param.isAllTensorAlign == 0 && param.inputShapeSame == 1 &&
        param.sameShapeTensorDim1 * param.dtypeSize < param.gatherThreshold) {
        return true;
    }
    return false;
}

inline static bool IsEnableScatter(const ConcatTilingParam& param)
{
    if (param.isAllTensorAlign == 0 && param.inputShapeSame == 0) {
        return true;
    }
    return false;
}

inline static bool IsEnableRowConcat(const ConcatTilingParam& param)
{
    if (param.isAllTensorAlign == 1 && param.dim == 0 && !param.isEmpty) {
        return true;
    }
    return false;
}

inline static void CalcLargeTensorNum(
    const ConcatTilingParam& param, int64_t tensorCol, int64_t rowsUsedCoreNum, int64_t& largeInputNum,
    int64_t& totalInputNum)
{
    if (tensorCol * param.ubFactorDim0 * param.dtypeSize >= BLOCK_THRESHOLD) {
        largeInputNum += (rowsUsedCoreNum - 1);
    }
    if (tensorCol * param.tailUbFactorDim0 * param.dtypeSize >= BLOCK_THRESHOLD) {
        largeInputNum += 1;
    }
    totalInputNum += rowsUsedCoreNum;
}

inline static bool IsEnablePureCopyTemplate(
    const ConcatTilingParam& param, int64_t rowsUsedCoreNum, int64_t colsUsedCoreNum)
{
    int64_t threshold = 0;
    if (param.isDim1AllAlign == 1 && param.inputShapeSame == 1) {
        threshold = PURE_COPY_COL_THRESHOLD_BASE;
    } else if (param.isDim1AllAlign == 1 || param.inputShapeSame == 1) {
        threshold = PURE_COPY_COL_THRESHOLD_ALIGN;
    } else {
        threshold = PURE_COPY_COL_THRESHOLD_NOALIGN;
    }
    for (const auto& tensorSize : param.tensorListDim1) {
        if (tensorSize * param.dtypeSize < threshold) {
            return false;
        }
    }
    int64_t totalInputNum = 0;
    int64_t largeInputNum = 0;
    if (param.blockSplitAxis == 0) {
        for (const auto& tensorSize : param.tensorListDim1) {
            CalcLargeTensorNum(param, tensorSize, param.usedCoreNum, largeInputNum, totalInputNum);
        }
    } else {
        for (int64_t i = 0; i < colsUsedCoreNum; i++) {
            if (param.startTensorIdx[i] == param.endTensorIdx[i]) {
                int64_t tensorCol = param.endTensorOffset[i] - param.startTensorOffset[i];
                CalcLargeTensorNum(param, tensorCol, rowsUsedCoreNum, largeInputNum, totalInputNum);
                continue;
            }
            int16_t startIdx = param.startTensorIdx[i];
            CalcLargeTensorNum(
                param, param.tensorListDim1[startIdx] - param.startTensorOffset[i], rowsUsedCoreNum,
                largeInputNum, totalInputNum);
            for (int16_t k = param.startTensorIdx[i] + 1; k < param.endTensorIdx[i]; k++) {
                CalcLargeTensorNum(param, param.tensorListDim1[k], rowsUsedCoreNum, largeInputNum, totalInputNum);
            }
            int64_t lastTensorCol = param.endTensorOffset[i];
            CalcLargeTensorNum(param, lastTensorCol, rowsUsedCoreNum, largeInputNum, totalInputNum);
        }
    }
    if (totalInputNum <= 0) {
        return false;
    }
    double largeRatio = static_cast<double>(largeInputNum) / static_cast<double>(totalInputNum);
    if (largeRatio >= LARGE_TENSOR_RATIO_THRESHOLD) {
        return true;
    }
    return false;
}

inline static void GenTilingKey(ConcatTilingParam& param)
{
    // tilingKey按5位设计：个位->字节数(1/2/4/8),十位->input shape相同/不相同(1/2)
    // 百位->输入cat部分全对齐/不对齐/不对齐gather模板(1/2/3),千位->首轴cat/非首轴cat(1/2),万位->是否使用定制tilingData
    // 0 表示空tensor
    if (param.isEmpty) {
        param.tilingKey = 0;
        return;
    }
    bool shapeSame = param.inputShapeSame == 1;
    bool isAllTensorAlign = param.isAllTensorAlign == 1;

    int64_t isCatDimAlign = isAllTensorAlign ? 1 : 2;
    int64_t dtypeSize = param.dtypeSize;
    if (IsEnableScatter(param)) {
        // scatter b64降2个b32处理，但仍需新的tilingkey
        dtypeSize = param.orgDtypeSize;
    }
    if (IsEnableGather(param)) {
        isCatDimAlign = GATHER_MODE;
    }
    int64_t isInputShapeSame = shapeSame ? 1 : 2;
    int64_t isFirstDim = DIGIT_TWO;
    int64_t isUseSpcTilingData = param.blockSplitAxis == 0 ? 1 : 0;
    param.tilingKey = dtypeSize + isInputShapeSame * TENS_DIGITS + isCatDimAlign * HUNDREDS_DIGITS +
                      isFirstDim * THOUSANDS_DIGITS + isUseSpcTilingData * TEN_THOUSANDS_DIGITS;
}

inline static ge::graphStatus IsDimValid(const gert::TilingContext* context, int64_t& dim, int64_t inputIdx, bool isNonContiguous, int64_t& strideDim)
{
    gert::Shape inputShape = GetShapeByAll(context, isNonContiguous, inputIdx, 0);
    int64_t shapeSize = static_cast<int64_t>(inputShape.GetDimNum());

    int64_t minDim = shapeSize * static_cast<int64_t>(-1);
    int64_t maxDim = shapeSize - 1;
    if (!(dim >= minDim && dim <= maxDim)) {
        return ge::GRAPH_FAILED;
    }
    // convert negative dim to positive dim
    if (dim < 0) {
        dim += shapeSize;
    }
    strideDim = dim - 1;
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus IsShapeValid(
    const gert::TilingContext* context, vector<vector<int64_t>>& tensorList, int64_t realDim)
{
    if (tensorList.size() < 1) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t dimSize = tensorList[0].size();
    auto shape0 = tensorList[0];
    for (const auto& tensorSize : tensorList) {
        int64_t curDimSize = tensorSize.size();
        OP_CHECK_IF(
            curDimSize != dimSize, 
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "input_tensors",
                (std::to_string(dimSize) + ", " + std::to_string(curDimSize)).c_str(),
                "The shape dims of input tensors must be the same."),
            return ge::GRAPH_FAILED);
        for (int64_t j = 0; j < dimSize; j++) {
            if (realDim == j) {
                continue;
            }
            OP_CHECK_IF(
                shape0[j] != tensorSize[j],
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input_tensors",
                    (std::to_string(shape0[j]) + ", " + std::to_string(tensorSize[j])).c_str(),
                    ("Shape [" + std::to_string(j) + "] of input tensors must be the same.").c_str()),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingUbForNosplitDim1(
    gert::TilingContext* context, int64_t maxAvaliableUb, ConcatTilingParam& param)
{
    OP_CHECK_IF(
        param.catDim1 <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "output_tensor",
            std::to_string(param.catDim1).c_str(),
            "Shape concat_axis of output_tensor must be greater than 0."),
        return ge::GRAPH_FAILED);
    param.ubFactorDim0 = min(maxAvaliableUb / param.catDim1, param.catDim0);
    OP_CHECK_IF(
        param.ubFactorDim0 <= 0,
        OP_LOGE(context->GetNodeName(), "ubFactorDim0 must be greater than 0, ubFactorDim0: %ld", param.ubFactorDim0),
        return ge::GRAPH_FAILED);
    param.uoDim0 = (param.catDim0 + param.ubFactorDim0 - 1) / param.ubFactorDim0;
    param.uoDim1 = 1;
    param.ubFactorDim1 = param.catDim1;
    param.tailUbFactorDim0 = param.catDim0 % param.ubFactorDim0;
    if (param.tailUbFactorDim0 == 0) {
        param.tailUbFactorDim0 = param.ubFactorDim0;
    }
    param.tailUbFactorDim1 = param.catDim1;
    param.ubSplitDim1 = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingUbForSplitDim1(
    gert::TilingContext* context, int64_t maxAvaliableUb, int64_t storageAlignUsed, int64_t maxDim1Factor,
    ConcatTilingParam& param)
{
    int64_t realFactorDim1 = maxDim1Factor;
    if (param.isAllTensorAlign == 0 && param.inputShapeSame == 1) {
        // tensor不对齐且需要切列的场景需要kernel侧重新进行ub切分，此处不再预留storage_align空间
        realFactorDim1 = maxAvaliableUb / std::min(LEAST_ROWS, param.catDim0);
        OP_CHECK_IF(
            param.everyBlockNumber <= 0,
            OP_LOGE(
                context->GetNodeName(), "everyBlockNumber must be greater than 0, everyBlockNumber: %ld",
                param.everyBlockNumber),
            return ge::GRAPH_FAILED);
        int64_t alignFactorDim1 = param.everyBlockNumber;
        if (param.inputShapeSame == 1 && param.sameShapeTensorDim1 * param.dtypeSize <= param.gatherThreshold) {
            alignFactorDim1 = param.sameShapeTensorDim1;
        }
        realFactorDim1 = realFactorDim1 / alignFactorDim1 * alignFactorDim1;
    } else {
        maxAvaliableUb -= storageAlignUsed;
    }
    param.ubFactorDim1 = min(realFactorDim1, param.catDim1);
    OP_CHECK_IF(
        param.ubFactorDim1 <= 0,
        OP_LOGE(
            context->GetNodeName(), "param.ubFactorDim1 must be greater than 0, param.ubFactorDim1: %ld",
            param.ubFactorDim1),
        return ge::GRAPH_FAILED);
    param.ubFactorDim0 = min(maxAvaliableUb / param.ubFactorDim1, param.catDim0);
    OP_CHECK_IF(
        param.ubFactorDim0 <= 0,
        OP_LOGE(
            context->GetNodeName(), "param.ubFactorDim0 must be greater than 0, param.ubFactorDim0: %ld",
            param.ubFactorDim0),
        return ge::GRAPH_FAILED);
    param.uoDim1 = (param.catDim1 + param.ubFactorDim1 - 1) / param.ubFactorDim1;
    param.uoDim0 = (param.catDim0 + param.ubFactorDim0 - 1) / param.ubFactorDim0;
    param.tailUbFactorDim0 = param.catDim0 % param.ubFactorDim0;
    if (param.tailUbFactorDim0 == 0) {
        param.tailUbFactorDim0 = param.ubFactorDim0;
    }
    param.tailUbFactorDim1 = param.catDim1 % param.ubFactorDim1;
    if (param.tailUbFactorDim1 == 0) {
        param.tailUbFactorDim1 = param.ubFactorDim1;
    }
    param.ubSplitDim1 = 1;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingUb(gert::TilingContext* context, ConcatTilingParam& param)
{
    OP_CHECK_IF(
        param.dtypeSize <= 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "x",
            Ops::Base::ToString(context->GetDynamicInputDesc(0, 0)->GetDataType()).c_str(),
            "The dtype size of x must be greater than 0."),
        return ge::GRAPH_FAILED);
    int64_t maxAvaliableUb = (param.ubSize - INDEX_USE_UB) / param.dtypeSize;
    if (param.isAllTensorAlign == 0) {
        // tensor不对齐的场景下，需要在UB中拼接，内存分成输入输出2部分
        maxAvaliableUb = maxAvaliableUb / BUFFER_NUM;
        // 非对齐场景scatter/gather索引为u16/u32,需确保ub内每个tensor的元素个数不超过U16上限
        maxAvaliableUb = std::min(maxAvaliableUb, static_cast<int64_t>(std::numeric_limits<uint16_t>::max()));
    }
    param.bufferSize = maxAvaliableUb;
    int64_t realFactorDim1 = maxAvaliableUb / std::min(LEAST_ROWS, param.catDim0);
    int64_t storageAlignUsed = 0;
    if (param.isAllTensorAlign == 0) {
        // tensor不对齐的场景下，预留输入和输出storage_align空间
        storageAlignUsed = param.everyBlockNumber * (param.noAlignTensorNum + 1);
        realFactorDim1 = (maxAvaliableUb - storageAlignUsed) / std::min(LEAST_ROWS, param.catDim0);
    }
    OP_CHECK_IF(
        param.everyBlockNumber <= 0,
        OP_LOGE(
            context->GetNodeName(), "param.everyBlockNumber must be greater than 0, param.everyBlockNumber: %ld",
            param.everyBlockNumber),
        return ge::GRAPH_FAILED);
    realFactorDim1 = realFactorDim1 / param.everyBlockNumber * param.everyBlockNumber;

    if (param.catDim1 < realFactorDim1) {
        maxAvaliableUb = maxAvaliableUb - storageAlignUsed;
        OP_CHECK_IF(
            TilingUbForNosplitDim1(context, maxAvaliableUb, param) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "TilingUbForNosplitDim1 failed"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            TilingUbForSplitDim1(context, maxAvaliableUb, storageAlignUsed, realFactorDim1, param) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "TilingUbForSplitDim1 failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline static ge::graphStatus TilingBlock(gert::TilingContext* context, ConcatTilingParam& param)
{
    if (param.uoDim0 > (param.totalCoreNum / HALF)) {
        OP_CHECK_IF(
            param.totalCoreNum <= 0,
            OP_LOGE(
                context->GetNodeName(), "param.totalCoreNum must be greater than 0, param.totalCoreNum: %ld",
                param.totalCoreNum),
            return ge::GRAPH_FAILED);
        param.blockFactor = (param.uoDim0 + param.totalCoreNum - 1) / param.totalCoreNum;
        OP_CHECK_IF(
            param.blockFactor <= 0,
            OP_LOGE(
                context->GetNodeName(), "param.blockFactor must be greater than 0, param.blockFactor: %ld",
                param.blockFactor),
            return ge::GRAPH_FAILED);
        param.usedCoreNum = (param.uoDim0 + param.blockFactor - 1) / param.blockFactor;
        param.tailBlockFactor = param.uoDim0 - (param.usedCoreNum - 1) * param.blockFactor;
        param.blockSplitAxis = 0;
    } else {
        int64_t rowsUsedCoreNum = param.uoDim0;
        OP_CHECK_IF(
            rowsUsedCoreNum <= 0,
            OP_LOGE(
                context->GetNodeName(), "rowsUsedCoreNum must be greater than 0, rowsUsedCoreNum: %ld",
                rowsUsedCoreNum),
            return ge::GRAPH_FAILED);
        int64_t leftCore = param.totalCoreNum / rowsUsedCoreNum;
        int64_t alignFactorDim1 = param.everyBlockNumber;
        if (param.inputShapeSame == 1 && param.sameShapeTensorDim1 * param.dtypeSize <= param.gatherThreshold) {
            alignFactorDim1 = param.sameShapeTensorDim1;
        }
        OP_CHECK_IF(
            alignFactorDim1 <= 0,
            OP_LOGE(
                context->GetNodeName(), "alignFactorDim1 must be greater than 0, alignFactorDim1: %ld",
                alignFactorDim1),
            return ge::GRAPH_FAILED);
        // 核未开满时，dim1借轴
        while (param.uoDim1 < leftCore && param.ubFactorDim1 * param.dtypeSize >= LEAST_COLS &&
               param.ubFactorDim1 * param.ubFactorDim0 >= HALF * param.leastCopyNumber) {
            param.ubFactorDim1 = (param.ubFactorDim1 / HALF) / alignFactorDim1 * alignFactorDim1;
            OP_CHECK_IF(
                param.ubFactorDim1 <= 0,
                OP_LOGE(
                    context->GetNodeName(), "param.ubFactorDim1 must be greater than 0, param.ubFactorDim1: %ld",
                    param.ubFactorDim1),
                return ge::GRAPH_FAILED);
            param.uoDim1 = (param.catDim1 + param.ubFactorDim1 - 1) / param.ubFactorDim1;
            param.tailUbFactorDim1 = param.catDim1 % param.ubFactorDim1;
            if (param.tailUbFactorDim1 == 0) {
                param.tailUbFactorDim1 = param.ubFactorDim1;
            }
        }
        if (param.uoDim1 > 1) {
            param.blockSplitAxis = 1;
            OP_CHECK_IF(
                leftCore <= 0,
                OP_LOGE(context->GetNodeName(), "leftCore must be greater than 0, leftCore: %ld", leftCore),
                return ge::GRAPH_FAILED);
            param.blockFactor = (param.uoDim1 + leftCore - 1) / leftCore;
            OP_CHECK_IF(
                param.blockFactor <= 0,
                OP_LOGE(
                    context->GetNodeName(), "param.blockFactor must be greater than 0, param.blockFactor: %ld",
                    param.blockFactor),
                return ge::GRAPH_FAILED);
            int64_t colsUsedCoreNum = (param.uoDim1 + param.blockFactor - 1) / param.blockFactor;
            param.usedCoreNum = rowsUsedCoreNum * colsUsedCoreNum;
            param.tailBlockFactor = param.uoDim1 - (colsUsedCoreNum - 1) * param.blockFactor;
        } else {
            param.blockSplitAxis = 0;
            param.blockFactor = 1;
            param.tailBlockFactor = 1;
            param.usedCoreNum = rowsUsedCoreNum;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline static void SetTensorListTilingData(ConcatTilingData& tilingData, ConcatTilingParam& param)
{
    std::copy(param.endTensorIdx.begin(), param.endTensorIdx.end(), param.endIdxArr);
    tilingData.arrays.set_endTensorIdx(param.endIdxArr);

    std::copy(param.endTensorOffset.begin(), param.endTensorOffset.end(), param.endOffsetArr);
    tilingData.arrays.set_endTensorOffset(param.endOffsetArr);
}

template <typename T>
inline static void SetTilingData(T& tilingData, ConcatTilingParam& param)
{
    // set tiling data
    tilingData.set_ubSplitDim1(param.ubSplitDim1);
    tilingData.set_dim(static_cast<int16_t>(param.dim));
    tilingData.set_blockFactor(param.blockFactor);
    tilingData.set_tailBlockFactor(param.tailBlockFactor);
    tilingData.set_ubFactorDim0(static_cast<int32_t>(param.ubFactorDim0));
    tilingData.set_ubFactorDim1(static_cast<int32_t>(param.ubFactorDim1));
    tilingData.set_tailUbFactorDim0(static_cast<int32_t>(param.tailUbFactorDim0));
    tilingData.set_tailUbFactorDim1(static_cast<int32_t>(param.tailUbFactorDim1));
    tilingData.set_uoDim0(param.uoDim0);
    tilingData.set_uoDim1(param.uoDim1);
    tilingData.set_tensorNum(param.tensorNum);
    tilingData.set_catDim1(param.catDim1);
    tilingData.set_sameShapeTensorDim1(param.sameShapeTensorDim1);
    tilingData.set_isFP4Type(param.isFP4Type);
    tilingData.set_bufferSize(static_cast<int32_t>(param.bufferSize));
    tilingData.set_dtypeSize(static_cast<int16_t>(param.dtypeSize));

    int64_t preLoadSize = std::min(TILING_PRELOAD_DIM1_LENGTH, static_cast<int64_t>(param.tensorListDim1.size()));
    std::copy(param.tensorListDim1.begin(), param.tensorListDim1.begin() + preLoadSize, param.preLoadDim1Arr);
    tilingData.arrays.set_preLoadDim1(param.preLoadDim1Arr);
    tilingData.set_isNonContiguous(static_cast<int16_t>(param.isNonContiguous ? 1 : 0));
    if (param.isNonContiguous) {
        uint64_t strideList[NON_CON_TENSOR_SIZE];
        std::copy(param.strideList.begin(), param.strideList.end(), strideList);
        tilingData.arrays.set_strideList(strideList);
        std::copy(param.concatDimList.begin(), param.concatDimList.end(), strideList);
        tilingData.arrays.set_concatDimList(strideList);
    }
}

inline static void CalcTensorList(ConcatTilingParam& param, int64_t everyCoreData, int64_t rowsUsedCoreNum)
{
    int64_t curOffset = 0;
    int64_t curTensorOffset = 0;

    for (int16_t i = 0; i < param.tensorNum; i++) {
        if (param.blockStartTensorIdx.size() == param.blockEndTensorIdx.size()) {
            param.blockStartTensorIdx.push_back(i);
            param.blockStartTensorOffset.push_back(0);
        }
        while (curOffset + param.tensorListDim1[i] - curTensorOffset >= everyCoreData) {
            param.blockEndTensorIdx.push_back(i);
            curTensorOffset = curTensorOffset + everyCoreData - curOffset;
            param.blockEndTensorOffset.push_back(curTensorOffset);
            if (curTensorOffset == param.tensorListDim1[i]) {
                curOffset = 0;
                break;
            } else {
                param.blockStartTensorIdx.push_back(i);
                param.blockStartTensorOffset.push_back(curTensorOffset);
                curOffset = 0;
            }
        }
        if (curTensorOffset == 0) {
            curOffset += param.tensorListDim1[i];
        } else {
            curOffset = param.tensorListDim1[i] - curTensorOffset;
        }
        curTensorOffset = 0;
    }
    if (curOffset != 0) {
        param.blockEndTensorIdx.push_back(param.tensorNum - 1);
        param.blockEndTensorOffset.push_back(param.tensorListDim1[param.tensorNum - 1]);
    }

    DupTensor(param.startTensorIdx, param.blockStartTensorIdx, rowsUsedCoreNum);
    DupTensor(param.endTensorIdx, param.blockEndTensorIdx, rowsUsedCoreNum);
    DupTensor(param.startTensorOffset, param.blockStartTensorOffset, rowsUsedCoreNum);
    DupTensor(param.endTensorOffset, param.blockEndTensorOffset, rowsUsedCoreNum);
}

inline static bool IsEnableb8ToB16(const ConcatTilingParam& param)
{
    // b8 dim1为偶数 不对齐场景可升为b16处理
    if (param.isNonContiguous) {
        if (param.dtypeSize != B8_BYTES || param.inputShapeSame != 1 || param.sameShapeTensorDim1 % DIGIT_TWO != 0 || param.strideList[0] % DIGIT_TWO != 0) {
            return false;
        }
        for (const auto& tensorSize : param.tensorListDim1) {
            if (tensorSize % DIGIT_TWO != 0) {
                return false;
            };
        }
        for (int16_t i = 0; i < param.tensorNum; ++i) {
            if (param.strideList[i] % DIGIT_TWO != 0) {
                return false;
            }
        }
    } else {
        if (param.dtypeSize != B8_BYTES || param.inputShapeSame != 1 || param.sameShapeTensorDim1 % DIGIT_TWO != 0) {
            return false;
        }
        for (const auto& tensorSize : param.tensorListDim1) {
            if (tensorSize % DIGIT_TWO != 0) {
                return false;
            };
        }
    }
    return true;
}

static ge::graphStatus PreProcessForNoAlign(ConcatTilingParam& param)
{
    if (param.isAllTensorAlign == 1) {
        return ge::GRAPH_SUCCESS;
    }
    if (param.dtypeSize == B64_BYTES) {
        // b64 不对齐场景降为b32处理
        param.sameShapeTensorDim1 *= DIGIT_TWO;
        param.dtypeSize = B32_BYTES;
        param.leastCopyNumber = MIN_RESERVED_SIZE / param.dtypeSize;
        param.everyBlockNumber = BLOCK_SIZE / param.dtypeSize;
        param.catDim1 *= DIGIT_TWO;
        for (auto& tensorSize : param.tensorListDim1) {
            tensorSize *= DIGIT_TWO;
        }
        for (auto& tensorSize : param.mergeTensorList) {
            tensorSize[1] *= DIGIT_TWO;
        }
        if (param.isNonContiguous) {
            for (int16_t i = 0; i < param.tensorNum; ++i) {
                param.strideList[i] *= DIGIT_TWO;
            }
        }
        return ge::GRAPH_SUCCESS;
    }
    if (IsEnableb8ToB16(param)) {
        // b8 dim1为偶数 不对齐场景升为b16处理
        param.sameShapeTensorDim1 /= DIGIT_TWO;
        param.dtypeSize = B16_BYTES;
        param.leastCopyNumber = MIN_RESERVED_SIZE / param.dtypeSize;
        param.everyBlockNumber = BLOCK_SIZE / param.dtypeSize;
        param.catDim1 /= DIGIT_TWO;
        for (auto& tensorSize : param.tensorListDim1) {
            tensorSize /= DIGIT_TWO;
        }
        for (auto& tensorSize : param.mergeTensorList) {
            tensorSize[1] /= DIGIT_TWO;
        }
        if (param.isNonContiguous) {
            for (int16_t i = 0; i < param.tensorNum; ++i) {
                param.strideList[i] /= DIGIT_TWO;
            }
        }
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}


inline static std::vector<int64_t> FindUniqueCut(int64_t coreNum)
{
    std::vector<int64_t> candidateSet;
    int64_t upBound = static_cast<int64_t>(std::ceil(std::sqrt(coreNum) + 1.0));
    for (int64_t m = 1; m < upBound; m++) {
        int64_t y = coreNum / m;
        candidateSet.push_back(m);
        candidateSet.push_back(y);
    }
    return candidateSet;
}

static std::pair<int64_t, int64_t> AutoBlockTiling(int64_t rows, int64_t cols, int64_t coreNum)
{
    std::vector<int64_t> candidateSet = FindUniqueCut(coreNum);
    std::vector<std::vector<int64_t>> allTiling;
    for (int64_t m : candidateSet) {
        if (m > rows) {
            continue;
        }
        int64_t mPart = (rows + m - 1) / m;
        int64_t n = coreNum / m;
        if (n > cols) {
            continue;
        }
        int64_t nPart = (cols + n - 1) / n;
        int64_t delta = mPart * nPart;
        if (m * n == coreNum) {
            if (rows % m == 0 && cols % n == 0) {
                delta = 0;
            } else if (rows % m == 0) {
                delta = delta - mPart * (cols - nPart * (n - 1));
            } else if (cols % n == 0) {
                delta = delta - nPart * (rows - mPart * (m - 1));
            } else {
                delta = delta - (cols - nPart * (n - 1)) * (rows - mPart * (m - 1));
            }
        }
        allTiling.push_back({m, n, m * n, delta});
    }
    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        return std::make_pair(a[DIGIT_THREE], -a[DIGIT_ONE]) < std::make_pair(b[DIGIT_THREE], -b[DIGIT_ONE]);
    });
    if (allTiling.size() == 0) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(allTiling[0][0], allTiling[0][DIGIT_ONE]);
}

static bool IsEnableUsedSimt(ConcatTilingParam& param)
{
    int64_t totalDataNum = param.catDim0 * param.catDim1 * param.dtypeSize;
    int64_t useCoreNum = std::min(static_cast<int64_t>(param.tensorNum), param.totalCoreNum);
    int64_t maxDim1 = param.tensorListDim1[0];
    int64_t minDim1 = param.tensorListDim1[0];

    if (param.isNonContiguous) {
        return false;
    }

    // 总数据量大于 64K * 使用核数，不使用simt模板
    if (totalDataNum >= useCoreNum * SIMT_PER_CORE_THRESHOLD) {
        return false;
    }

    // 合轴后对齐场景和相同shape场景，不使用simt模板
    if (param.isAllTensorAlign == 1 || param.inputShapeSame == 1) {
        return false;
    }

    // 输入tensor的个数对核数取模，余数小于等于核数的一半场景，不使用simt模板
    if (static_cast<int64_t>(param.tensorNum) % (param.totalCoreNum + 1) <= (param.totalCoreNum / DIGIT_TWO)) {
        return false;
    }

    // tensor数目大于128不使用simt模板
    if (param.tensorNum > TILING_COLS_OFFSET_LENGTH) {
        return false;
    }

    // 数据量大于单个tensor数据量大于1024，波动在2倍以上的数据不使用simt模板
    for (auto dim1Ptr = param.tensorListDim1.begin(); dim1Ptr != param.tensorListDim1.end(); ++dim1Ptr) {
        maxDim1 = std::max(maxDim1, *dim1Ptr);
        minDim1 = std::min(minDim1, *dim1Ptr);
        if (maxDim1 / minDim1 >= DIGIT_TWO && maxDim1 * param.catDim0 * param.dtypeSize > SIMT_COMPARE_THRESHOLD) {
            return false;
        }
    }

    return true;
}

inline static void CalcTensorColsOffset(ConcatTilingParam& param)
{
    // 计算每个tensor行方向的数据偏移个数，从第0个开始记录，即如果是tensor 1的偏移要使用tensorColsOffset[0]数值
    int32_t curConcatDimOffset = 0;
    int32_t digitSize = param.dtypeSize > B64_BYTES ? param.dtypeSize / B64_BYTES : DIGIT_ONE;
    for (auto dimPtr = param.tensorListDim1.begin(); dimPtr != param.tensorListDim1.end(); ++dimPtr) {
        curConcatDimOffset = static_cast<int32_t>(curConcatDimOffset + *dimPtr * digitSize);
        param.tensorColsOffset.push_back(curConcatDimOffset);
    }
}

inline static void SetTensorColsOffset(ConcatTilingDataForSimt& tilingData, ConcatTilingParam& param)
{
    int32_t tilingList[TILING_COLS_OFFSET_LENGTH];

    std::copy(param.tensorColsOffset.begin(), param.tensorColsOffset.end(), tilingList);
    tilingData.arrays.set_tensorColsOffset(tilingList);
}

static inline void PrintSimtTilingData(ConcatTilingDataForSimt& tilingData)
{
    OP_LOGI(
        "[Concat]", "tensorNumPerCore: %d, get_tensorNum: %d,catDim0: %d,catDim1: %d",
        tilingData.get_tensorNumPerCore(), tilingData.get_tensorNum(), tilingData.get_catDim0(),
        tilingData.get_catDim1());
}

static ge::graphStatus TilingForConcatDSimt(gert::TilingContext* context, ConcatTilingParam& param)
{
    ConcatTilingDataForSimt tilingData;
    int64_t dtypeSize = param.dtypeSize;
    int64_t catDim1 = static_cast<int64_t>(param.catDim1);
    // 大于B64的数据类型使用B64类型计算,concat轴数据总量不变，数据个数需翻倍
    catDim1 = dtypeSize > B64_BYTES ? (dtypeSize / B64_BYTES) * catDim1 : catDim1;
    dtypeSize = dtypeSize > B64_BYTES ? B64_BYTES : dtypeSize;

    // 计算simt模板使用的核数，每核处理的tensor个数，计算tilingKey
    param.usedCoreNum = std::min(static_cast<int64_t>(param.tensorNum), param.totalCoreNum);
    param.tensorNumPerCore = (static_cast<int64_t>(param.tensorNum) + param.usedCoreNum - 1) / param.usedCoreNum;
    param.tilingKey = SIMT_TILINGKEY_PREFIX + dtypeSize;

    // 计算每个tensor行方向上的偏移
    CalcTensorColsOffset(param);

    // 设置tilingData的值
    tilingData.set_tensorNumPerCore(param.tensorNumPerCore);
    tilingData.set_tensorNum(static_cast<int32_t>(param.tensorNum));
    tilingData.set_catDim0(static_cast<int32_t>(param.catDim0));
    tilingData.set_catDim1(catDim1);
    SetTensorColsOffset(tilingData, param);

    PrintSimtTilingData(tilingData);
    context->SetBlockDim(param.usedCoreNum);
    context->SetTilingKey(param.tilingKey);
    // set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = SYSTEM_WORKSPACE_SIZE;
    OP_CHECK_IF(
        ConcatSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "SimtSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool TilingForPureCopy(ConcatTilingParam& param)
{
    param.usedCoreNum =
        std::min(param.totalCoreNum, (param.catDim0 * param.catDim1 + EVERY_CORE_THRESHOLD - 1) / EVERY_CORE_THRESHOLD);
    int64_t nCols = (param.catDim1 + LEAST_BLOCK_BYTES - 1) / LEAST_BLOCK_BYTES;
    int64_t mRows = param.catDim0;
    int64_t rowsCutPart = 0;
    int64_t colsCutPart = 0;
    std::tie(rowsCutPart, colsCutPart) = AutoBlockTiling(mRows, nCols, param.usedCoreNum);
    if (rowsCutPart == 0 || colsCutPart == 0) {
        return false;
    }
    param.ubFactorDim0 = (param.catDim0 + rowsCutPart - 1) / rowsCutPart;
    param.tailUbFactorDim0 = param.catDim0 - param.ubFactorDim0 * (rowsCutPart - 1);
    param.ubFactorDim1 = (param.catDim1 + colsCutPart - 1) / colsCutPart;
    param.tailUbFactorDim1 = param.catDim1 - param.ubFactorDim1 * (colsCutPart - 1);
    if (param.tailUbFactorDim0 < 0 || param.tailUbFactorDim1 < 0) {
        return false;
    }
    int64_t everyBlockCols = (param.catDim1 + colsCutPart - 1) / colsCutPart;
    if (colsCutPart > 1) {
        param.blockSplitAxis = 1;
        CalcTensorList(param, everyBlockCols, rowsCutPart);
    } else {
        param.blockSplitAxis = 0;
    }
    bool isUsedPureCopy = IsEnablePureCopyTemplate(param, rowsCutPart, colsCutPart);
    if (isUsedPureCopy) {
        if (ENABLE_DB) {
            param.ubSize = param.ubSize / DIGIT_TWO;
        }
        param.bufferSize = param.ubSize / param.dtypeSize;
        if (param.blockSplitAxis == 0) {
            param.tilingKey = PURE_COPY_NO_SPLIT_DIM1_TILINGKEY;
        } else {
            param.tilingKey = PURE_COPY_SPLIT_DIM1_TILINGKEY;
        }
        param.uoDim0 = rowsCutPart; // 行方向使用几个核
        param.uoDim1 = colsCutPart; // 列方向使用几个核
        param.inputShapeSame = 0;
        GetTensorSameDim1(param);
    } else {
        param.blockStartTensorIdx.clear();
        param.blockEndTensorIdx.clear();
        param.blockStartTensorOffset.clear();
        param.blockEndTensorOffset.clear();
    }
    return isUsedPureCopy;
}

inline static ge::graphStatus DoTiling(gert::TilingContext* context, ConcatTilingParam& param)
{
    if (param.isEmpty) {
        param.usedCoreNum = 0;
        return ge::GRAPH_SUCCESS;
    }
    if (TilingForPureCopy(param)) {
        // 先尝试纯搬运模板
        return ge::GRAPH_SUCCESS;
    }
    if (param.isAllTensorAlign == 0 && (param.dtypeSize == B64_BYTES || param.dtypeSize == B8_BYTES)) {
        OP_CHECK_IF(
            PreProcessForNoAlign(param) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "check PreProcessForNoAlign failed"), return ge::GRAPH_FAILED);
    }
    if (ENABLE_DB) {
        param.ubSize = param.ubSize / HALF;
    }
    // ub切分,不切列
    OP_CHECK_IF(
        TilingUb(context, param) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "check tiling_ub failed"),
        return ge::GRAPH_FAILED);
    // block切分
    OP_CHECK_IF(
        TilingBlock(context, param) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "check tiling_block failed"),
        return ge::GRAPH_FAILED);
    if (param.blockSplitAxis == 1) {
        CalcTensorList(param, param.blockFactor * param.ubFactorDim1, param.uoDim0);
    }
    GenTilingKey(param);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static int64_t GetAxis(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axis = attrs->GetAttrPointer<T>(PACK_ATTR_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    return *axis;
}

inline static ge::graphStatus IsPackDimValid(gert::TilingContext* context, int64_t& dim)
{
    auto inputShapePtr = context->GetDynamicInputShape(PACK_INPUT_IDX, 0);
    gert::Shape inputShape = inputShapePtr->GetStorageShape();
    int64_t shapeSize = static_cast<int64_t>(inputShape.GetDimNum());

    int64_t minAxis = (shapeSize + PACK_AXIS_DEFAULT_VALUE) * (-1);
    int64_t maxAxis = shapeSize;
    if (!(dim >= minAxis && dim <= maxAxis)) {
        return ge::GRAPH_FAILED;
    }
    // convert negative dim to positive dim
    if (dim < 0) {
        dim = dim + shapeSize + PACK_AXIS_DEFAULT_VALUE;
    }
    return ge::GRAPH_SUCCESS;
}

bool IsInvalidTypeForPack(const DataType dtype)
{
    std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT,  ge::DT_FLOAT16,   ge::DT_BF16,     ge::DT_UINT8,
                                             ge::DT_INT8,   ge::DT_UINT16,    ge::DT_INT16,    ge::DT_UINT32,
                                             ge::DT_INT32,  ge::DT_UINT64,    ge::DT_INT64,    ge::DT_BOOL,
                                             ge::DT_DOUBLE, ge::DT_COMPLEX64, ge::DT_COMPLEX32};
    bool isInvalidType = (supportedDtype.count(dtype) == 0);

    return isInvalidType;
}

ge::graphStatus CheckInputShapeSameForPack(gert::TilingContext* context)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(PACK_INPUT_IDX);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();
    if (inputNum < 1) {
        return ge::GRAPH_FAILED;
    }
    auto firstInputTensorShapePtr = context->GetDynamicInputShape(PACK_INPUT_IDX, 0);
    gert::Shape firstInputTensorShape = firstInputTensorShapePtr->GetStorageShape();
    size_t firstInputTensorDimNum = firstInputTensorShape.GetDimNum();
    vector<int64_t> fisrtInputShapeList(firstInputTensorDimNum, 0);
    for (size_t i = 0; i < firstInputTensorDimNum; i++) {
        fisrtInputShapeList[i] = firstInputTensorShape.GetDim(i);
    }

    for (uint32_t i = 1; i < inputNum; ++i) {
        auto inputTensorShapePtr = context->GetDynamicInputShape(PACK_INPUT_IDX, i);
        gert::Shape inputTensorShape = inputTensorShapePtr->GetStorageShape();
        size_t inputTensorDimNum = inputTensorShape.GetDimNum();
        vector<int64_t> inputShapeList(inputTensorDimNum, 0);
        for (size_t j = 0; j < inputTensorDimNum; j++) {
            inputShapeList[j] = inputTensorShape.GetDim(j);
            if (inputTensorShape.GetDim(j) != fisrtInputShapeList[j]) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

void GetTensorListForPack(gert::TilingContext* context, ConcatTilingParam& param)
{
    auto inputTensorShapePtr = context->GetDynamicInputShape(PACK_INPUT_IDX, 0);
    gert::Shape inputTensorShape = inputTensorShapePtr->GetStorageShape();
    size_t inputTensorDimNum = inputTensorShape.GetDimNum();
    vector<int64_t> inputShapeList(inputTensorDimNum, 0);
    for (size_t i = 0; i < inputTensorDimNum; i++) {
        inputShapeList[i] = inputTensorShape.GetDim(i);
    }

    auto computeNodeInfo = context->GetComputeNodeInfo();
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(PACK_INPUT_IDX);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();

    for (uint32_t i = 0; i < inputNum; ++i) {
        param.tensorList.push_back(inputShapeList);
    }
}

ge::graphStatus Tiling4PackToConcatForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4PackToConcatForAscendC running begin");
    ConcatTilingParam param;
    param.dim = GetAxis<int64_t>(context);
    OP_CHECK_IF(
        IsPackDimValid(context, param.dim) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check pack_axis failed, please check pack_axis."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckInputShapeSameForPack(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input_tensors",
            "different_shapes",
            "The shapes of input tensors must be the same."),
        return ge::GRAPH_FAILED);
    auto inputDesc = context->GetDynamicInputDesc(PACK_INPUT_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputDataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        IsInvalidTypeForPack(inputDataType),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "input",
            Ops::Base::ToString(inputDataType).c_str(),
            "The dtype of input must be within the range [DT_UINT8, DT_INT8, DT_BOOL, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_FLOAT16, DT_BF16, DT_UINT16, DT_INT64, DT_UINT64, DT_DOUBLE, DT_COMPLEX32, DT_COMPLEX64]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetDtypeSize(context, param, PACK_INPUT_IDX) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "GetDtypeSize failed."), return ge::GRAPH_FAILED);
    GetTensorListForPack(context, param);
    OP_CHECK_IF(
        CalcBaseTilingParam(context, param) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CalcBaseTilingParam failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        DoTiling(context, param) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "DoTiling failed."),
        return ge::GRAPH_FAILED);
    context->SetTilingKey(param.tilingKey);
    context->SetBlockDim(param.usedCoreNum);
    // set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYSTEM_WORKSPACE_SIZE;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (param.blockSplitAxis == 1) {
        ConcatTilingData tilingData;
        SetTilingData<ConcatTilingData>(tilingData, param);
        SetTensorListTilingData(tilingData, param);
        PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
        ret = ConcatSetTilingData<ConcatTilingData>(context, tilingData);
    } else {
        ConcatTilingDataNoArray tilingData;
        SetTilingData<ConcatTilingDataNoArray>(tilingData, param);
        PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
        ret = ConcatSetTilingData<ConcatTilingDataNoArray>(context, tilingData);
    }
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "PackSetTilingData set tiling data fail."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetConcatDim(gert::TilingContext* context, ConcatTilingParam& param, int64_t dimIdx)
{
    if (dimIdx == INVLID_CONCAT_DIM_IDX) {
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const int64_t* axis = attrs->GetAttrPointer<int64_t>(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, axis);
        param.dim = *axis;
    } else {
        auto concatDimPtr = context->GetRequiredInputDesc(dimIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, concatDimPtr);
        ge::DataType concatDimType = concatDimPtr->GetDataType();
        if (concatDimType == ge::DT_INT32) {
            OP_CHECK_IF(
                GetConcatDimInput<int32_t>(context, param, dimIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get concat_dim failed."), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                GetConcatDimInput<int64_t>(context, param, dimIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get concat_dim failed."), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

gert::Shape GetShapeByAll(const gert::TilingContext* context, bool isNonContiguous, int inputIdx, int index)
{
    auto inputTensorShapePtr = context->GetDynamicInputShape(inputIdx, index);
    if (isNonContiguous) {
        return inputTensorShapePtr->GetShape();
    } else {
        return inputTensorShapePtr->GetStorageShape();
    }
}

// 校验是否为全连续
bool IsAllContiguous(gert::TilingContext* context, ConcatTilingParam &param, int64_t inputIdx)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfo);
    param.tensorNum = anchorInstanceInfo->GetInstanceNum();
    for (int16_t i = 0; i < param.tensorNum; ++i) {
        bool isViewI = context->DynamicInputIsView(inputIdx, i);
        auto nonStrideI = context->GetDynamicInputStride(inputIdx, i);
        if (isViewI && nonStrideI != nullptr && nonStrideI->GetDimNum() > 0) {
            return false;
        } 
    }
    return true;
}

ge::graphStatus CheckNonConBasic(gert::TilingContext* context, ConcatTilingParam &param)
{
    OP_CHECK_IF(param.tensorNum <= 1 || param.tensorNum > NON_CON_TENSOR_SIZE,
        OP_LOGE_FOR_INVALID_TENSORNUM(context->GetNodeName(), "input_tensors",
            static_cast<int64_t>(param.tensorNum),
            ("within the range [2, " + std::to_string(NON_CON_TENSOR_SIZE) + "]").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(param.strideDim < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "stride_dim",
            std::to_string(param.strideDim).c_str(),
            "The value of stride_dim must be greater than or equal to 0."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateDtypeConsistency(
    gert::TilingContext* context, int64_t inputIdx,
    int64_t tensorIdx, ge::DataType input0DataType)
{
    auto inputIDesc = context->GetDynamicInputDesc(inputIdx, tensorIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputIDesc);
    OP_CHECK_IF(inputIDesc->GetDataType() != input0DataType,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "input_tensors",
            (Ops::Base::ToString(input0DataType) + ", " + Ops::Base::ToString(inputIDesc->GetDataType())).c_str(),
            "The dtypes of input_tensors must be the same."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateAndSetStride(
    gert::TilingContext* context, ConcatTilingParam& param, int64_t inputIdx, int64_t i)
{
    bool isViewI = context->DynamicInputIsView(inputIdx, i);
    auto nonStrideI = context->GetDynamicInputStride(inputIdx, i);
    if (isViewI && nonStrideI != nullptr && nonStrideI->GetDimNum() > 0) {
        OP_CHECK_IF(nonStrideI->GetStride(param.tensorList[i].size() - 1) != 1,
            OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "input_stride",
                std::to_string(nonStrideI->GetStride(param.tensorList[i].size() - 1)).c_str(),
                "1"),
            return ge::GRAPH_FAILED);
        for (int64_t j = param.tensorList[i].size() - 2; j >= 0; j--) {
            if (param.strideDim != j) {
                OP_CHECK_IF(nonStrideI->GetStride(j) != nonStrideI->GetStride(j + 1) * param.tensorList[i][j + 1],
                    OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "input_stride",
                        std::to_string(nonStrideI->GetStride(j)).c_str(),
                        std::to_string(nonStrideI->GetStride(j + 1) * param.tensorList[i][j + 1]).c_str()),
                    return ge::GRAPH_FAILED);
            } 
        }
        param.strideList[i] = static_cast<uint64_t>(nonStrideI->GetStride(param.strideDim));
    } else {
        param.strideList[i] = MergeDim(param.tensorList[i], param.strideDim + 1, param.tensorList[i].size());
    }
    param.concatDimList[i] = static_cast<uint32_t>(param.tensorList[i][param.dim]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateSmallBagScenario(
    gert::TilingContext* context, const ConcatTilingParam& param, int64_t i)
{
    int64_t allData = MergeDim(param.tensorList[i], 0, param.tensorList[i].size());
    if (!(param.tensorListDim1[i] * param.dtypeSize >= SMALL_BAG) 
        && !(param.strideList[i] * param.dtypeSize > SMALL_BAG) 
        && !(allData * param.dtypeSize < param.totalCoreNum * ALL_DATA_SMALL)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tensorListDim1",
            std::to_string(param.tensorListDim1[i] * param.dtypeSize).c_str(),
            ("The combined size of the concat dim and subsequent dim must be at least " + 
             std::to_string(SMALL_BAG) + " bytes.").c_str());
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "strideList",
            std::to_string(param.strideList[i] * param.dtypeSize).c_str(),
            ("The stride of the non contiguous axis must be greater than " + 
             std::to_string(SMALL_BAG) + " bytes.").c_str());
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "allData",
            std::to_string(allData * param.dtypeSize).c_str(),
            ("The total data size must be less than " + 
             std::to_string(param.totalCoreNum * ALL_DATA_SMALL) + " bytes.").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckNonContiguous(gert::TilingContext* context, ConcatTilingParam &param, int64_t inputIdx)
{
    if (CheckNonConBasic(context, param) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto input0Desc = context->GetDynamicInputDesc(inputIdx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0Desc);
    auto input0DataType = input0Desc->GetDataType();
    
    for (int64_t i = 0; i < param.tensorNum; i++) {
        if (ValidateDtypeConsistency(context, inputIdx, i, input0DataType) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        
        if (ValidateAndSetStride(context, param, inputIdx, i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        
        if (ValidateSmallBagScenario(context, param, i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    
    param.isNonContiguous = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCommon(gert::TilingContext* context, int64_t inputIdx, int64_t dimIdx)
{
    ConcatTilingParam param;
    // get dim
    OP_CHECK_IF(
        GetConcatDim(context, param, dimIdx) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get concat_dim failed."), return ge::GRAPH_FAILED);
    param.isNonContiguous = !(IsAllContiguous(context, param, inputIdx));
    OP_CHECK_IF(
        IsDimValid(context, param.dim, inputIdx, param.isNonContiguous, param.strideDim) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check concat_dim failed, please check concat_dim."), return ge::GRAPH_FAILED);
    auto inputDesc = context->GetDynamicInputDesc(inputIdx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputDataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        IsInvalidType(inputDataType),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "input",
            Ops::Base::ToString(inputDataType).c_str(),
            "The dtype of input must be within the range [DT_UINT8, DT_INT8, DT_BOOL, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_FLOAT16, DT_BF16, DT_UINT16, DT_INT64, DT_UINT64, DT_DOUBLE, DT_COMPLEX64, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetDtypeSize(context, param, inputIdx) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "GetDtypeSize failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetTensorList(context, param, inputIdx) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "GetTensorList failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        IsShapeValid(context, param.tensorList, param.dim) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check input shape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcBaseTilingParam(context, param) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CalcBaseTilingParam failed."), return ge::GRAPH_FAILED);
     if (param.isNonContiguous) {
        OP_CHECK_IF(CheckNonContiguous(context, param, inputIdx) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "input tensor non contiguous validation failed."), return ge::GRAPH_FAILED);
    }
    // 处理simt模板
    if (IsEnableUsedSimt(param)) {
        return TilingForConcatDSimt(context, param);
    }
    OP_CHECK_IF(
        DoTiling(context, param) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "DoTiling failed."),
        return ge::GRAPH_FAILED);
    context->SetTilingKey(param.tilingKey);
    context->SetBlockDim(param.usedCoreNum);
    // set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYSTEM_WORKSPACE_SIZE;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (param.blockSplitAxis == 1) {
        ConcatTilingData tilingData;
        SetTilingData<ConcatTilingData>(tilingData, param);
        SetTensorListTilingData(tilingData, param);
        PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
        ret = ConcatSetTilingData<ConcatTilingData>(context, tilingData);
    } else {
        ConcatTilingDataNoArray tilingData;
        SetTilingData<ConcatTilingDataNoArray>(tilingData, param);
        PrintTilingData(tilingData, param.tilingKey, param.usedCoreNum);
        ret = ConcatSetTilingData<ConcatTilingDataNoArray>(context, tilingData);
    }
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "ConcatSetTilingData set tiling data fail."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ConcatForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ConcatForAscendC running begin");
    return TilingCommon(context, 1, 0);
}

ge::graphStatus TilingPrepareForConcat(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ConcatDCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepareForConcat fail to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "TilingPrepareForConcat fail to get ub size."),
        return ge::GRAPH_FAILED);
    compileInfo->vectorLen = static_cast<int64_t>(Ops::Base::GetVRegSize(context));
    OP_CHECK_IF(
        (compileInfo->vectorLen <= 0), OP_LOGE(context->GetNodeName(), "TilingPrepareForConcat fail to get vectorLen."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Concat)
    .TilingInputsDataDependency({CONCAT_DIM_IDX})
    .Tiling(Tiling4ConcatForAscendC)
    .TilingParse<ConcatDCompileInfo>(TilingPrepareForConcat);
} // namespace optiling