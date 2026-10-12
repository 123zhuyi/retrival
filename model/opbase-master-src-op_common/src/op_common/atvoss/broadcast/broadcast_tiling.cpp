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
 * \file broadcast_tiling.cpp
 * \brief atvoss broadcast template tiling
 */

#include <algorithm>
#include "op_common/atvoss/broadcast/broadcast_tiling.h"

namespace Ops {
namespace Base {

static std::string ShapeToString(const gert::Shape& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << shape.GetDim(i);
    }
    oss << "]";
    return oss.str();
}

static std::string StrideToString(const gert::Stride& stride)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < stride.GetDimNum(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << stride.GetStride(i);
    }
    oss << "]";
    return oss.str();
}

static std::string VectorToString(const std::vector<int64_t>& dims)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

void BrcPrintShape(const gert::Shape& shape, const std::string& prefix)
{
    std::string shapeStr = ShapeToString(shape);
    OP_LOGI("BroadcastTiling", "%s: %s", prefix.c_str(), shapeStr.c_str());
}

void BrcPrintShapes(const std::vector<gert::Shape>& shapes, const std::string& prefix)
{
    OP_LOGI("BroadcastTiling", "%s", prefix.c_str());
    for (size_t i = 0; i < shapes.size(); ++i) {
        std::string shapeStr = ShapeToString(shapes[i]);
        std::ostringstream lineOss;
        lineOss << "shape[" << i << "]: " << shapeStr;
        OP_LOGI("BroadcastTiling", "%s", lineOss.str().c_str());
    }
}

void BrcPrintStride(const gert::Stride& stride, const std::string& prefix)
{
    std::string strideStr = StrideToString(stride);
    OP_LOGI("BroadcastTiling", "%s: %s", prefix.c_str(), strideStr.c_str());
}

void BrcPrintStrides(const std::vector<gert::Stride>& strides, const std::string& prefix)
{
    OP_LOGI("BroadcastTiling", "%s", prefix.c_str());
    for (size_t i = 0; i < strides.size(); ++i) {
        std::string strideStr = StrideToString(strides[i]);
        std::ostringstream lineOss;
        lineOss << "stride[" << i << "]: " << strideStr;
        OP_LOGI("BroadcastTiling", "%s", lineOss.str().c_str());
    }
}

void BrcPrintVector(const std::vector<int64_t>& dims, const std::string& prefix)
{
    std::string vectorStr = VectorToString(dims);
    OP_LOGI("BroadcastTiling", "%s: %s", prefix.c_str(), vectorStr.c_str());
}

void BrcPrintVectors(const std::vector<std::vector<int64_t>>& allDims, const std::string& prefix)
{
    OP_LOGI("BroadcastTiling", "%s", prefix.c_str());
    for (size_t i = 0; i < allDims.size(); ++i) {
        std::string vectorStr = VectorToString(allDims[i]);
        std::ostringstream lineOss;
        lineOss << "vector[" << i << "]: " << vectorStr;
        OP_LOGI("BroadcastTiling", "%s", lineOss.str().c_str());
    }
}

ge::graphStatus IsTensorContiguous(const gert::Shape& viewShape, const gert::Stride* viewStride, bool& isContiguous)
{
    isContiguous = true;
    OP_CHECK_IF((viewStride == nullptr), OP_LOGI("BroadcastTiling", "IsContiguous Check viewStride is nullptr."),
                return ge::GRAPH_SUCCESS);
    size_t shapeDim = viewShape.GetDimNum();
    size_t strideDim = viewStride->GetDimNum();
    if (strideDim == 0) {
        OP_LOGI("BroadcastTiling", "IsContiguous Check strideDim is 0.");
        return ge::GRAPH_SUCCESS;
    }
    if (shapeDim != strideDim) {
        OP_LOGE("BroadcastTiling", "IsContiguous Check ViewShape DimNum %zu != ViewStride DimNum %zu.", shapeDim,
                strideDim);
        return ge::GRAPH_FAILED;
    }

    int64_t validStride = 1;
    for (int64_t i = shapeDim - 1; i >= 0; i--) {
        if (viewShape.GetDim(i) == 1) {
            continue;
        }
        if (validStride != viewStride->GetStride(i)) {
            isContiguous = false;
            return ge::GRAPH_SUCCESS;
        }
        validStride *= viewShape.GetDim(i);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 *  合轴逻辑
 * @param inShapes 输入shape
 * @param outShapes 输出shape
 * @param dims 合轴后轴大小
 * @param strides 合轴后stride大小
 * @return
 */
ge::graphStatus DimensionCollapse(const std::vector<gert::Shape>& inShapes, const gert::Shape& outShapes,
                                  std::vector<std::vector<int64_t>>& dims, std::vector<std::vector<int64_t>>& strides)
{
    // 获取输出shape的轴数量，并封装输出shape
    uint64_t maxDim = outShapes.GetDimNum();
    std::vector<int64_t> outputShapes;
    for (uint64_t i = 0; i < outShapes.GetDimNum(); i++) {
        outputShapes.push_back(outShapes.GetDim(i));
    }

    // 获取输入shape的轴数量，并封装输入shape
    // 对维度不足输出shape的输入进行补维操作
    std::vector<std::vector<int64_t>> inputShapes;
    for (uint64_t i = 0; i < inShapes.size(); i++) {
        uint64_t inputDim = inShapes[i].GetDimNum();
        if (maxDim < inputDim) {
            OP_LOGE("BroadcastTiling", "The %lu input's dim num is not same with output's dim num", i);
            return ge::GRAPH_FAILED;
        }
        int64_t diff = maxDim - inputDim;
        std::vector<int64_t> tmp(diff, 1);
        for (uint64_t j = 0; j < inputDim; j++) {
            tmp.push_back(inShapes[i].GetDim(j));
        }
        inputShapes.push_back(tmp);
    }

    // 设置默认flag大小跟第一个输入大小一致，并初始化为0
    std::vector<int64_t> flags(inputShapes[0].size(), 0);
    // 遍历所有输入的所有维度，校验轴的合法性。
    // 将某个输入的brc轴对应的二进制位置设置为1
    for (uint64_t i = 0; i < inputShapes[0].size(); ++i) {
        int64_t flag = 0;
        for (uint64_t j = 0; j < inputShapes.size(); ++j) {
            flag <<= 1;
            if (inputShapes[j][i] != 1 && inputShapes[j][i] != outputShapes[i]) {
                OP_LOGE("BroadcastTiling", "The %lu input's dim index(%lu) is not same with out, and not 1", j, i);
                return ge::GRAPH_FAILED;
            }
            if (inputShapes[j][i] <= 0) {
                OP_LOGE("BroadcastTiling", "The %lu input's dim index(%lu) must be a positive number", j, i);
                return ge::GRAPH_FAILED;
            }
            if (inputShapes[j][i] == 1) {
                flag++;
            }
        }
        flags[i] = flag;
    }

    // 做输入shape合轴逻辑
    // 遍历所有的输入，遍历所有的轴
    // 1.如果所有输入的相邻两根轴大小都一样，则可以合轴。
    // 2.如果所有输入输出都是1，则可以合轴
    int64_t target = (1 << inputShapes.size()) - 1;
    for (uint64_t i = 0; i < inputShapes.size(); i++) {
        int64_t prevValue = inputShapes[i][0];
        int64_t prevFlag = flags[0];
        std::vector<int64_t> tmp{prevValue};
        for (uint64_t j = 1; j < inputShapes[i].size(); j++) {
            int64_t curValue = inputShapes[i][j];
            int64_t curFlag = flags[j];
            bool isValid = (prevFlag == curFlag) || (prevFlag == target && outputShapes[j - 1] == 1);
            if (isValid) {
                int64_t product = curValue * tmp.back();
                tmp.pop_back();
                tmp.push_back(product);
                prevFlag = curFlag;
                continue;
            }
            if (curFlag == target && outputShapes[j] == 1) {
                continue;
            }
            prevFlag = curFlag;
            tmp.push_back(curValue);
        }
        dims.push_back(tmp);
    }

    // 做输出shape合轴逻辑
    int64_t prevValue = outputShapes[0];
    int64_t prevFlag = flags[0];
    std::vector<int64_t> outputDims{prevValue};
    for (uint64_t j = 1; j < outputShapes.size(); j++) {
        int64_t curValue = outputShapes[j];
        int64_t curFlag = flags[j];
        bool isValid = (prevFlag == curFlag) || (prevFlag == target && outputShapes[j - 1] == 1);
        if (isValid) {
            int64_t product = curValue * outputDims.back();
            outputDims.pop_back();
            outputDims.push_back(product);
            prevFlag = curFlag;
            continue;
        }
        if (curFlag == target && outputShapes[j] == 1) {
            continue;
        }
        prevFlag = curFlag;
        outputDims.push_back(curValue);
    }
    dims.push_back(outputDims);

    // 计算stride信息
    for (uint64_t i = 0; i < dims.size(); i++) {
        std::vector<int64_t> tmp;
        int64_t base = 1;
        for (int64_t j = dims[i].size() - 1; j >= 0; j--) {
            if (dims[i][j] == 1 && i != dims.size() - 1 && dims[dims.size() - 1][j] != 1) {
                tmp.push_back(0);
            } else {
                tmp.push_back(base);
                base *= dims[i][j];
            }
        }
        std::reverse(tmp.begin(), tmp.end());
        strides.push_back(tmp);
    }
    return ge::GRAPH_SUCCESS;
}

enum class CollapseAction : uint8_t {
    KEEP,
    MERGE,
    ABSORB,
};

static bool IsAxesPairJointContiguous(const std::vector<std::vector<int64_t>>& inputShapes,
                                      const std::vector<std::vector<int64_t>>& inputStrides,
                                      uint64_t lastKeptAxis, uint64_t currentAxis)
{
    for (uint64_t i = 0; i < inputStrides.size(); i++) {
        if (inputStrides[i].empty()) {
            continue;
        }
        if (inputShapes[i][currentAxis] == 1 || inputShapes[i][lastKeptAxis] == 1) {
            continue;
        }
        if (inputStrides[i][lastKeptAxis] != inputStrides[i][currentAxis] * inputShapes[i][currentAxis]) {
            return false;
        }
    }
    return true;
}

/**
 *  合轴决策：对每根轴只算一次 KEEP/MERGE/ABSORB，输入侧与输出侧共用同一份决策
 *  1.如果所有输入的相邻两根轴大小都一样，则可以合轴。
 *  2.如果所有输入输出都是1，则可以合轴
 * @param inputShapes 补维后的输入shape
 * @param inputStrides 补维后的原始输入stride，要求未被标记过-1
 * @param outputShapes 输出shape
 * @param flags 每根轴的brc标记
 * @param target 全1轴flag值
 * @return 每根轴的合轴动作
*/
static std::vector<CollapseAction> ComputeBroadcastCollapsePlan(
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<std::vector<int64_t>>& inputStrides,
    const std::vector<int64_t>& outputShapes,
    const std::vector<int64_t>& flags, int64_t target)
{
    std::vector<CollapseAction> plan(outputShapes.size(), CollapseAction::KEEP);
    int64_t prevFlag = flags[0];
    uint64_t lastKeptAxis = 0;
    for (uint64_t j = 1; j < outputShapes.size(); j++) {
        int64_t curFlag = flags[j];
        // 场景1 或者 场景2
        bool isValid = (prevFlag == curFlag) || (prevFlag == target && outputShapes[j - 1] == 1);
        if (isValid && (prevFlag == target || IsAxesPairJointContiguous(inputShapes, inputStrides, lastKeptAxis, j))) {
            plan[j] = CollapseAction::MERGE;
            prevFlag = curFlag;
            lastKeptAxis = j;
        } else if (curFlag == target && outputShapes[j] == 1) {
            // 当前维度为全1，且输出也为1，可以直接合轴，跳过处理
            plan[j] = CollapseAction::ABSORB;
        } else {
            plan[j] = CollapseAction::KEEP;
            prevFlag = curFlag;
            lastKeptAxis = j;
        }
    }
    return plan;
}

/**
 *  合轴逻辑
 * @param inShapes 输入shape
   @param inStrides 输入stride
 * @param outShapes 输出shape
 * @param dims 合轴后轴大小
 * @param strides 合轴后stride大小
 * @return
*/
ge::graphStatus NonContiguousDimensionCollapse(const std::vector<gert::Shape>& inShapes,
                                               const std::vector<gert::Stride>& inStrides, const gert::Shape& outShapes,
                                               std::vector<std::vector<int64_t>>& dims,
                                               std::vector<std::vector<int64_t>>& strides)
{
    // 获取输出shape的轴数量，并封装输出shape
    uint64_t maxDim = outShapes.GetDimNum();
    std::vector<int64_t> outputShapes;
    for (uint64_t i = 0; i < outShapes.GetDimNum(); i++) {
        outputShapes.push_back(outShapes.GetDim(i));
    }

    // 获取输入shape的轴数量，并封装输入shape
    // 对维度不足输出shape的输入进行补维操作
    // 对维度不足输出shape对应stride进行补维操作
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> inputStrides;
    std::vector<int64_t> emptyStride;
    for (uint64_t i = 0; i < inShapes.size(); i++) {
        uint64_t inputDim = inShapes[i].GetDimNum();
        uint64_t strideDim = inStrides[i].GetDimNum();
        if (maxDim < inputDim) {
            OP_LOGE("BroadcastTiling", "The %lu input's dim num is not same with output's dim num", i);
            return ge::GRAPH_FAILED;
        }
        int64_t diff = maxDim - inputDim;
        std::vector<int64_t> tmpShape(diff, 1);
        std::vector<int64_t> tmpStride(diff, 0);
        for (uint64_t j = 0; j < inputDim; j++) {
            tmpShape.push_back(inShapes[i].GetDim(j));
            if (strideDim != 0) {
                tmpStride.push_back(inStrides[i].GetStride(j));
            }
        }
        inputShapes.push_back(tmpShape);
        if (strideDim != 0) {
            inputStrides.push_back(tmpStride);
        } else {
            inputStrides.push_back(emptyStride);
        }
    }

    // 设置默认flag大小跟第一个输入大小一致，并初始化为0
    std::vector<int64_t> flags(inputShapes[0].size(), 0);
    // 遍历所有输入的所有维度，校验轴的合法性。
    // 将某个输入的brc轴对应的二进制位置设置为1
    for (uint64_t i = 0; i < inputShapes[0].size(); ++i) {
        int64_t flag = 0;
        for (uint64_t j = 0; j < inputShapes.size(); ++j) {
            flag <<= 1;
            if (inputShapes[j][i] != 1 && inputShapes[j][i] != outputShapes[i]) {
                OP_LOGE("BroadcastTiling", "The %lu input's dim index(%lu) is not same with out, and not 1", j, i);
                return ge::GRAPH_FAILED;
            }
            if (inputShapes[j][i] <= 0) {
                OP_LOGE("BroadcastTiling", "The %lu input's dim index(%lu) must be a positive number", j, i);
                return ge::GRAPH_FAILED;
            }
            if (inputShapes[j][i] == 1) {
                flag++;
            }
        }
        flags[i] = flag;
    }

    int64_t target = (1 << inputShapes.size()) - 1;
    std::vector<CollapseAction> collapsePlan = ComputeBroadcastCollapsePlan(
        inputShapes, inputStrides, outputShapes, flags, target);

    // 做输入shape合轴逻辑
    // 遍历所有的输入，遍历所有的轴
    for (uint64_t i = 0; i < inputShapes.size(); i++) {
        bool isTensorContiguous = inStrides[i].GetDimNum() == 0;
        std::vector<int64_t> tmp{inputShapes[i][0]};
        for (uint64_t j = 1; j < inputShapes[i].size(); j++) {
            if (collapsePlan[j] == CollapseAction::MERGE) {
                tmp.back() *= inputShapes[i][j];
                // shape合轴，对应stride置为-1，后续将-1的stride消除即为合轴后对应的strides
                if (!isTensorContiguous) {
                    for (int64_t k = j - 1; k >= 0; k--) {
                        if (inputStrides[i][k] >= 0) {
                            inputStrides[i][k] = -1;
                            break;
                        }
                    }
                }
            } else if (collapsePlan[j] == CollapseAction::ABSORB) {
                // shape合轴，对应stride置为-1，后续将-1的stride消除即为合轴后对应的strides
                if (!isTensorContiguous) {
                    inputStrides[i][j] = -1;
                }
            } else {
                tmp.push_back(inputShapes[i][j]);
            }
        }
        dims.push_back(tmp);
    }

    // 做输出shape合轴逻辑
    std::vector<int64_t> outputDims{outputShapes[0]};
    for (uint64_t j = 1; j < outputShapes.size(); j++) {
        if (collapsePlan[j] == CollapseAction::MERGE) {
            outputDims.back() *= outputShapes[j];
        } else if (collapsePlan[j] == CollapseAction::ABSORB) {
            continue;
        } else {
            outputDims.push_back(outputShapes[j]);
        }
    }
    dims.push_back(outputDims);
    // 计算stride信息
    for (uint64_t i = 0; i < dims.size(); i++) {
        std::vector<int64_t> tmp;
        int64_t base = 1;
        for (int64_t j = dims[i].size() - 1; j >= 0; j--) {
            if (dims[i][j] == 1 && i != dims.size() - 1 && dims[dims.size() - 1][j] != 1) {
                tmp.push_back(0);
            } else {
                tmp.push_back(base);
                base *= dims[i][j];
            }
        }
        std::reverse(tmp.begin(), tmp.end());
        strides.push_back(tmp);
    }
    // 非连续stride刷入strides中
    for (uint64_t i = 0; i < inputStrides.size(); i++) {
        if (inputStrides[i].size() != 0) {
            std::vector<int64_t> newStride;
            for (uint64_t j = 0; j < inputStrides[i].size(); j++) {
                if (inputStrides[i][j] >= 0) {
                    newStride.push_back(inputStrides[i][j]);
                }
            }
            strides[i] = newStride;
        }
    }

    // brc轴的stride设置为0
    for (uint64_t i = 0; i < (dims.size() - 1); i++) {
        for (uint64_t j = 0; j < dims[i].size(); j++) {
            if (dims[i][j] == 1 && dims[dims.size() - 1][j] != 1) {
                strides[i][j] = 0;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t GetBlockSplitFactor(BroadcastTilingData& broadcastTilingData, ubSplitInfo& ubInfo, uint64_t maxElemNum)
{
    // 做ub切分
    uint64_t curProduct = 1;
    uint64_t ubSplitAxes = 0;
    bool flag = true;
    for (int64_t i = broadcastTilingData.dims.back().size() - 1; i >= 0; i--) {
        curProduct *= broadcastTilingData.dims.back()[i];
        if (curProduct > maxElemNum) {
            curProduct = curProduct / broadcastTilingData.dims.back()[i];
            ubSplitAxes = i;
            flag = false;
            break;
        }
    }
    // 全部能放下，则去掉第一个维度
    if (flag) {
        curProduct = curProduct / broadcastTilingData.dims.back()[0];
    }

    uint32_t ubFormer = 0; // 表示当前切分轴的切分因子
    if (broadcastTilingData.dims.back().size() == 1) {
        ubFormer = maxElemNum;
    } else {
        ubFormer = maxElemNum / curProduct;
    }

    uint64_t ubOuter = (broadcastTilingData.dims.back()[ubSplitAxes] + ubFormer - 1) / ubFormer;
    uint64_t ubTail = broadcastTilingData.dims.back()[ubSplitAxes] - (ubOuter - 1) * ubFormer;

    // 计算ub外轴乘积
    uint64_t fusedProduct = ubOuter;
    for (uint64_t i = 0; i < ubSplitAxes; i++) {
        fusedProduct *= broadcastTilingData.dims.back()[i];
    }

    ubInfo.ubFormer = ubFormer;
    ubInfo.ubSplitAxis = ubSplitAxes;
    ubInfo.ubOuter = ubOuter;
    ubInfo.ubTail = ubTail;

    return fusedProduct;
}

ge::graphStatus DoBrodcastTiling(const BroadcastTilingParams& broadcastTilingParams,
                                 BroadcastTilingData& broadcastTilingData)
{
    uint64_t computeKey = BroadcastGetComputeKey();
    auto iter = broadcastTilingParams.computeMap.find(computeKey);
    BroadcastComputeParams computeParams;
    if (iter != broadcastTilingParams.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("BroadcastTiling", "can not find computeKey: %lu", computeKey);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(broadcastTilingParams.ubSize < computeParams.extraSize[0],
                OP_LOGE("BroadcastTiling", "ubSize is smaller than extra size."), return ge::GRAPH_FAILED);

    // 获取最大存活空间大小
    uint64_t maxElemNum = BroadcastGetMaxElemNum(broadcastTilingParams.ubSize, computeParams);
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. origin maxElemNum: %lu ubSize: %ld", maxElemNum,
            broadcastTilingParams.ubSize);
    OP_CHECK_IF((broadcastTilingParams.ubSize <= 0), OP_LOGE("BroadcastTiling", "ubSize can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((broadcastTilingParams.coreNum <= 0), OP_LOGE("BroadcastTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((maxElemNum == 0), OP_LOGE("BroadcastTiling", "maxElemNum can not be 0"), return ge::GRAPH_FAILED);

    ubSplitInfo ubInfo;
    uint64_t fusedProduct = GetBlockSplitFactor(broadcastTilingData, ubInfo, maxElemNum);
    uint64_t blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
    uint64_t blockNum = (fusedProduct + blockFormer - 1) / blockFormer;

    // 当preferMultiCore为true且当前使用的核数少于总核数，尝试降低UB分块以充分利用更多核心
    if (broadcastTilingParams.preferMultiCore && blockNum < static_cast<uint64_t>(broadcastTilingParams.coreNum)) {
        // dstFusedProduct为尽量切多核时的理想多核切分因子
        uint64_t dstFusedProduct = blockFormer * broadcastTilingParams.coreNum;
        uint64_t tmpFusedProduct = fusedProduct;
        while (tmpFusedProduct < dstFusedProduct) {
            maxElemNum = maxElemNum - CACHE_LINE;
            if (maxElemNum <= CACHE_LINE_512) {
                break;
            }
            tmpFusedProduct = GetBlockSplitFactor(broadcastTilingData, ubInfo, maxElemNum);
        }
        // 计算最终调整后的多核切分因子
        maxElemNum = (maxElemNum + CACHE_LINE + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;
        fusedProduct = GetBlockSplitFactor(broadcastTilingData, ubInfo, maxElemNum);
        // 更新blockFormer和blockNum
        blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
        blockNum = (fusedProduct + blockFormer - 1) / blockFormer;
    }

    uint64_t blockTail = fusedProduct - (blockNum - 1) * blockFormer;
    uint64_t dimProductBeforeUbInner = fusedProduct;
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. maxElemNum: %lu fusedProduct: %lu ubFormer: %ld ", maxElemNum,
            fusedProduct, ubInfo.ubFormer);

    broadcastTilingData.ubSplitAxis = ubInfo.ubSplitAxis;
    broadcastTilingData.ubFormer = ubInfo.ubFormer;
    broadcastTilingData.ubOuter = ubInfo.ubOuter;
    broadcastTilingData.ubTail = ubInfo.ubTail;

    broadcastTilingData.blockFormer = blockFormer;
    broadcastTilingData.blockNum = blockNum;
    broadcastTilingData.blockTail = blockTail;
    broadcastTilingData.dimProductBeforeUbInner = dimProductBeforeUbInner;
    broadcastTilingData.elemNum = maxElemNum;

    uint64_t scheduleKey = BroadcastGetScheduleKey(broadcastTilingData.shapeLen - broadcastTilingData.ubSplitAxis);
    broadcastTilingData.innerKey = computeKey * BROADCAST_COMPUTE_KEY_OFFSET + scheduleKey;
    return ge::GRAPH_SUCCESS;
}

/**
 *  合轴逻辑
 * @param broadcastTilingParams tiling参数
 * @param broadcastTilingData 临时tilingData缓存
 *
 * @return
 */
ge::graphStatus DoDimensionCollapse(const BroadcastTilingParams& broadcastTilingParams,
                                    BroadcastTilingData& broadcastTilingData)
{
    std::vector<std::vector<int64_t>> dims;
    std::vector<std::vector<int64_t>> strides;
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    if (broadcastTilingParams.inputAllContiguous) {
        status = DimensionCollapse(broadcastTilingParams.inShape, broadcastTilingParams.outShape, dims, strides);
    } else {
        status = NonContiguousDimensionCollapse(broadcastTilingParams.inShape, broadcastTilingParams.inStride,
                                                broadcastTilingParams.outShape, dims, strides);
    }
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    broadcastTilingData.shapeLen = dims.back().size();
    OP_CHECK_IF((broadcastTilingData.shapeLen > static_cast<int64_t>(BROADCAST_MAX_DIMS)),
                OP_LOGE("BroadcastTiling", "broadcast can't support dim size greater than 8."),
                return ge::GRAPH_FAILED);

    broadcastTilingData.dims = dims;
    broadcastTilingData.strides = strides;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BroadcastTiling(const BroadcastTilingParams& broadcastTilingParams,
                                BroadcastTilingData& broadcastTilingData)
{
    auto status = DoDimensionCollapse(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE("BroadcastTiling", "dimension collapse failed");
        return ge::GRAPH_FAILED;
    }

    status = DoBrodcastTiling(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE("BroadcastTiling", "inner broadcast tiling failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace Base
} // namespace Ops
