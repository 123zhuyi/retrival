/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "math/common/op_api/broadcast_util.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <algorithm>
#include <vector>

namespace op {
namespace {

constexpr int64_t CACHE_LINE = 128;
constexpr int64_t SINGLE_CORE_SIZE_LIMIT = 8 * 1024;
constexpr int64_t NON_CONTIGUOUS_LAST_DIM_LIMIT = 8192;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr size_t DIM_FOUR = 4;

struct CollapsedTensor {
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    bool isContiguous;
    int64_t dtypeSize;
};

bool IsOnlyLastTwoAxesTransposed(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& strides)
{
    size_t shapeDim = viewShape.size();
    size_t stridesDim = strides.size();
    if (shapeDim < DIM_TWO || shapeDim != stridesDim) {
        return false;
    }
    size_t lastDim = shapeDim - 1;
    size_t secondLastDim = stridesDim - DIM_TWO;
    bool transposedStride = (strides[lastDim] == viewShape[secondLastDim]) && (strides[secondLastDim] == 1);
    bool othersContiguous = true;
    if (shapeDim > DIM_TWO) {
        int64_t expectedStride = viewShape[lastDim] * viewShape[secondLastDim];
        for (int64_t i = static_cast<int64_t>(shapeDim) - static_cast<int64_t>(DIM_THREE); i >= 0; i--) {
            if (strides[i] != expectedStride) {
                othersContiguous = false;
                break;
            }
            expectedStride = expectedStride * viewShape[i];
        }
    }
    return transposedStride && othersContiguous;
}

bool IsCollapsedContiguous(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    if (dims.empty() || strides.empty() || dims.size() != strides.size()) {
        return false;
    }
    int64_t expectedStride = 1;
    for (int64_t i = static_cast<int64_t>(dims.size()) - 1; i > 0; i--) {
        if (dims[i] != 1 && strides[i] != expectedStride) {
            return false;
        }
        if (dims[i] != 1) {
            expectedStride *= dims[i];
        }
    }
    if (dims[0] != 1 && strides[0] != expectedStride) {
        return false;
    }
    return true;
}

struct PaddedInput {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    bool isContiguous;
};

// 前置条件：dims.size() <= maxDim（调用方已按广播 shape 对齐维度）；
// isContiguous 为 true 时 strides 为空（补 0 占位，不参与合轴判定），
// 为 false 时 strides 与 dims 等长（由 CollectInput 的 shapeDim == stridesDim 检查保证），
// 否则补维后 shape 与 strides 会错位
bool PadSingleInput(const CollapsedTensor& input, int64_t maxDim, PaddedInput& out)
{
    if (static_cast<int64_t>(input.dims.size()) > maxDim) {
        return false;
    }
    int64_t diff = maxDim - static_cast<int64_t>(input.dims.size());
    for (int64_t d = 0; d < diff; d++) {
        out.shape.push_back(1);
        out.strides.push_back(0);
    }
    for (int64_t d = 0; d < static_cast<int64_t>(input.dims.size()); d++) {
        out.shape.push_back(input.dims[d]);
    }
    if (!input.isContiguous) {
        for (int64_t d = 0; d < static_cast<int64_t>(input.strides.size()); d++) {
            out.strides.push_back(input.strides[d]);
        }
    }
    out.isContiguous = input.isContiguous;
    return true;
}

std::vector<int64_t> ComputeBroadcastFlags(const std::vector<PaddedInput>& padded, int64_t maxDim)
{
    std::vector<int64_t> flags(static_cast<size_t>(maxDim), 0);
    for (int64_t i = 0; i < maxDim; i++) {
        int64_t flag = 0;
        for (int64_t j = 0; j < static_cast<int64_t>(padded.size()); j++) {
            flag <<= 1;
            if (padded[j].shape[i] == 1) {
                flag++;
            }
        }
        flags[i] = flag;
    }
    return flags;
}

void MarkStrideForMerge(std::vector<int64_t>& workingStrides, int64_t mergeEnd)
{
    for (int64_t k = mergeEnd; k >= 0; k--) {
        if (workingStrides[static_cast<size_t>(k)] >= 0) {
            workingStrides[static_cast<size_t>(k)] = -1;
            break;
        }
    }
}

enum class CollapseAction : uint8_t {
    KEEP,
    MERGE,
    ABSORB,
};

bool IsAxesPairJointContiguous(const std::vector<PaddedInput>& padded, int64_t lastKeptAxis, int64_t currentAxis)
{
    for (int64_t i = 0; i < static_cast<int64_t>(padded.size()); i++) {
        if (padded[i].isContiguous) {
            continue;
        }
        if (padded[i].shape[currentAxis] == 1 || padded[i].shape[lastKeptAxis] == 1) {
            continue;
        }
        if (padded[i].strides[lastKeptAxis] != padded[i].strides[currentAxis] * padded[i].shape[currentAxis]) {
            return false;
        }
    }
    return true;
}

/**
 *  合轴决策：对每根轴只算一次 KEEP/MERGE/ABSORB，输入侧与输出侧共用同一份决策
 *  1.如果所有输入的相邻两根轴大小都一样，则可以合轴。
 *  2.如果所有输入输出都是1，则可以合轴
 * @param padded 补维后的输入
 * @param flags 每根轴的brc标记
 * @param outputShapes 输出shape
 * @param target 全1轴flag值
 * @return 每根轴的合轴动作
 */
std::vector<CollapseAction> ComputeBroadcastCollapsePlan(const std::vector<PaddedInput>& padded,
                                                         const std::vector<int64_t>& flags,
                                                         const std::vector<int64_t>& outputShapes, int64_t target)
{
    std::vector<CollapseAction> plan(outputShapes.size(), CollapseAction::KEEP);
    // 空 shape（0 维标量）无轴可合，直接返回全 KEEP，避免 flags[0] 越界
    if (outputShapes.empty() || flags.empty()) {
        return plan;
    }
    int64_t prevFlag = flags[0];
    int64_t lastKeptAxis = 0;
    for (int64_t j = 1; j < static_cast<int64_t>(outputShapes.size()); j++) {
        int64_t curFlag = flags[j];
        // 场景1 或者 场景2
        bool canMerge = (prevFlag == curFlag) || (prevFlag == target && outputShapes[j - 1] == 1);
        if (canMerge && (prevFlag == target || IsAxesPairJointContiguous(padded, lastKeptAxis, j))) {
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

CollapsedTensor BuildCollapsedTensor(const std::vector<int64_t>& dims, std::vector<int64_t>& workingStrides,
                                     bool isContiguous)
{
    CollapsedTensor ct;
    ct.dims = dims;
    ct.isContiguous = isContiguous;
    if (!isContiguous) {
        for (int64_t j = 0; j < static_cast<int64_t>(workingStrides.size()); j++) {
            if (workingStrides[j] >= 0) {
                ct.strides.push_back(workingStrides[j]);
            }
        }
        if (IsCollapsedContiguous(ct.dims, ct.strides)) {
            ct.isContiguous = true;
            ct.strides.clear();
        }
    }
    return ct;
}

CollapsedTensor CollapseSingleTensor(const PaddedInput& padded, const std::vector<CollapseAction>& plan)
{
    std::vector<int64_t> workingStrides = padded.strides;
    std::vector<int64_t> collapsedDims{padded.shape[0]};

    for (int64_t j = 1; j < static_cast<int64_t>(padded.shape.size()); j++) {
        if (plan[j] == CollapseAction::MERGE) {
            collapsedDims.back() *= padded.shape[j];
            if (!padded.isContiguous) {
                MarkStrideForMerge(workingStrides, j - 1);
            }
        } else if (plan[j] == CollapseAction::ABSORB) {
            if (!padded.isContiguous) {
                workingStrides[j] = -1;
            }
        } else {
            collapsedDims.push_back(padded.shape[j]);
        }
    }
    return BuildCollapsedTensor(collapsedDims, workingStrides, padded.isContiguous);
}

bool DoDimensionCollapse(const std::vector<CollapsedTensor>& inputs, const op::Shape& outputShape,
                         std::vector<CollapsedTensor>& collapsed, std::vector<int64_t>& collapsedOutputDims)
{
    int64_t maxDim = static_cast<int64_t>(outputShape.GetDimNum());
    // 0 维输出（标量）无轴可合：直接透传原始输入，保持 dims/strides 原样
    if (maxDim == 0) {
        collapsed = inputs;
        collapsedOutputDims.clear();
        return true;
    }
    std::vector<int64_t> outputShapes;
    for (int64_t i = 0; i < maxDim; i++) {
        outputShapes.push_back(outputShape.GetDim(i));
    }

    std::vector<PaddedInput> paddedInputs;
    for (int64_t i = 0; i < static_cast<int64_t>(inputs.size()); i++) {
        PaddedInput pi;
        if (!PadSingleInput(inputs[i], maxDim, pi)) {
            return false;
        }
        paddedInputs.push_back(pi);
    }

    std::vector<int64_t> flags = ComputeBroadcastFlags(paddedInputs, maxDim);

    int64_t target = (1 << paddedInputs.size()) - 1;

    std::vector<CollapseAction> plan = ComputeBroadcastCollapsePlan(paddedInputs, flags, outputShapes, target);

    for (int64_t i = 0; i < static_cast<int64_t>(paddedInputs.size()); i++) {
        collapsed.push_back(CollapseSingleTensor(paddedInputs[i], plan));
    }

    CollapsedTensor outputCt;
    outputCt.isContiguous = true;
    for (int64_t i = 0; i < maxDim; i++) {
        outputCt.dims.push_back(outputShape.GetDim(i));
    }
    PaddedInput outputPadded;
    if (!PadSingleInput(outputCt, maxDim, outputPadded)) {
        return false;
    }
    CollapsedTensor collapsedOutput = CollapseSingleTensor(outputPadded, plan);
    collapsedOutputDims = collapsedOutput.dims;

    for (int64_t i = 0; i < static_cast<int64_t>(collapsed.size()); i++) {
        if (collapsed[i].isContiguous) {
            continue;
        }
        for (int64_t j = 0;
             j < static_cast<int64_t>(collapsed[i].dims.size()) && j < static_cast<int64_t>(collapsedOutputDims.size());
             j++) {
            if (collapsed[i].dims[j] == 1 && collapsedOutputDims[j] != 1) {
                collapsed[i].strides[j] = 0;
            }
        }
    }
    return true;
}

int64_t ComputeDataSize(const std::vector<int64_t>& dims, int64_t typeSize)
{
    int64_t product = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(dims.size()); i++) {
        product *= dims[i];
    }
    return product * typeSize;
}

bool IsBroadcastTo(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    for (int64_t i = 0; i < static_cast<int64_t>(dims.size()); i++) {
        if (dims[i] != 1 && strides[i] == 0) {
            return true;
        }
    }
    return false;
}

// 单个输入的 schMode 分类结果，对应 opbase broadcast_tiling 的模板选择逻辑
struct InputClassifyResult {
    bool isLastTranspose;
    bool isNLast;
    bool needsNonContiguousBase;
    bool isLastAxisContiguous; // 末轴 stride 是否连续（用于 post-loop 兜底）
    bool inputInvalid;         // 异常输入，整体回退
};

// 对单个非连续输入做 schMode 场景分类，判定逻辑逐条对齐 opbase 的决策树
// （dimLimit 按该输入自身 dtype 计算，与 opbase 逐输入取 inputDtype 一致；
//   sizeLimit = single_core_limit * coreNum 为循环不变量，由调用方计算传入）
InputClassifyResult ClassifySingleInput(const CollapsedTensor& ct, int64_t sizeLimit)
{
    InputClassifyResult result = {false, false, false, true, false};

    // 异常输入：维度/步长为空或不匹配，直接回退
    if (ct.dims.empty() || ct.strides.empty() || ct.dims.size() != ct.strides.size()) {
        OP_LOGD("Broadcast Template NonContiguous UnSupported. empty or mismatched dims/strides");
        result.inputInvalid = true;
        return result;
    }
    size_t shapeDim = ct.dims.size();
    size_t strideDim = ct.strides.size();

    // expand 形态输入（dim != 1 且 stride == 0，合轴后存在广播拷贝轴）
    if (IsBroadcastTo(ct.dims, ct.strides)) {
        result.needsNonContiguousBase = true;
        return result;
    }

    // 记录末轴是否连续，供 post-loop 兜底判定
    result.isLastAxisContiguous = (ct.strides[strideDim - 1] <= 1);

    // 维度超过 4D（NLast/LastTranspose 模板最多支持 4 维）
    if (shapeDim > DIM_FOUR) {
        result.needsNonContiguousBase = true;
        return result;
    }

    // 末尾两轴转置且维数 <= 3 走 LastTranspose
    if (shapeDim <= DIM_THREE && IsOnlyLastTwoAxesTransposed(ct.dims, ct.strides)) {
        result.isLastTranspose = true;
        return result;
    }

    // opbase 门限元素：dimLimit = cache_line / dtype_size
    int64_t dimLimit = (ct.dtypeSize > 0) ? CACHE_LINE / ct.dtypeSize : 0;

    // 场景 a: 末轴 >= dimLimit 且末轴 stride==1（连续大末轴）→ NLast
    if (ct.dims[shapeDim - 1] >= dimLimit && ct.strides[strideDim - 1] == 1) {
        result.isNLast = true;
        return result;
    }

    // 场景 b: 末轴 < dimLimit 且倒数第二轴 stride > dimLimit（小末轴大步幅）→ NLast
    if (shapeDim > 1 && ct.dims[shapeDim - 1] < dimLimit && ct.strides[strideDim - DIM_TWO] > dimLimit) {
        result.isNLast = true;
        return result;
    }

    // 场景 c: 末轴 < dimLimit 且总数据量 <= sizeLimit（单核可处理）→ NLast
    int64_t inputSize = ComputeDataSize(ct.dims, ct.dtypeSize);
    if (ct.dims[shapeDim - 1] < dimLimit && inputSize <= sizeLimit) {
        result.isNLast = true;
        return result;
    }

    // 以上场景均不满足走 NonContiguousBase（末轴不连续等常规非连续形态）
    result.needsNonContiguousBase = true;
    return result;
}

bool CheckNonContiguousSupport(const std::vector<CollapsedTensor>& collapsed,
                               const std::vector<int64_t>& collapsedOutputDims)
{
    // schMode 分类标志，对应 opbase broadcast_tiling 的模板选择逻辑
    bool isLastTranspose = false;
    bool isNLast = false;
    bool needsNonContiguousBase = false;
    bool isLastAxisContiguous = true; // 末轴 stride 是否连续（用于 post-loop 兜底）

    int64_t coreNum = static_cast<int64_t>(GetCurrentPlatformInfo().GetVectorCoreNum());
    if (coreNum == 0) {
        coreNum = 1;
    }
    // opbase 门限元素：sizeLimit = single_core_limit * coreNum（循环不变量，一次计算）
    int64_t sizeLimit = SINGLE_CORE_SIZE_LIMIT * coreNum;

    // 逐输入判定 schMode 类别
    for (const auto& ct : collapsed) {
        // 连续输入跳过，不影响任何标志
        if (ct.isContiguous) {
            continue;
        }
        InputClassifyResult r = ClassifySingleInput(ct, sizeLimit);
        if (r.inputInvalid) {
            return false;
        }
        if (r.needsNonContiguousBase) {
            // early break: 对齐 opbase early return，不再处理后续输入
            needsNonContiguousBase = true;
            break;
        }
        isLastTranspose = isLastTranspose || r.isLastTranspose;
        isNLast = isNLast || r.isNLast;
        isLastAxisContiguous = isLastAxisContiguous && r.isLastAxisContiguous;
    }

    // Post-loop 兜底: 末轴不连续且不是 LastTranspose → 无法走 NLast，降级为 NonContiguousBase
    // 同时清除 isNLast，防止混合输入误判为 NLast
    if (!isLastTranspose && !isLastAxisContiguous) {
        needsNonContiguousBase = true;
        isNLast = false;
    }

    // NLast 路径，始终支持 NonContiguous
    if (isNLast && !isLastTranspose && !needsNonContiguousBase) {
        return true;
    }

    // 无输出维度信息时默认支持
    if (collapsedOutputDims.empty()) {
        return true;
    }
    int64_t lastDim = collapsedOutputDims.back();

    // 末轴 > NON_CONTIGUOUS_LAST_DIM_LIMIT 时回退 Contiguous 转换路径
    if ((isLastTranspose || needsNonContiguousBase) && lastDim > NON_CONTIGUOUS_LAST_DIM_LIMIT) {
        OP_LOGD("Broadcast Template NonContiguous Fallback. schMode lastDim %ld > %ld", lastDim,
                NON_CONTIGUOUS_LAST_DIM_LIMIT);
        return false;
    }

    return true;
}

bool CollectInput(const aclTensor* input, CollapsedTensor& ct)
{
    if (input == nullptr) {
        return false;
    }
    int64_t typeSize = static_cast<int64_t>(op::TypeSize(input->GetDataType()));
    if (typeSize == 0) {
        OP_LOGD("Broadcast Template NonContiguous UnSupported. typeSize is 0");
        return false;
    }
    auto viewShape = input->GetViewShape();
    auto viewStride = input->GetViewStrides();
    if (viewShape.GetDimNum() != viewStride.size()) {
        OP_LOGD("Broadcast Template NonContiguous UnSupported. shapeDim %zu != stridesDim %zu", viewShape.GetDimNum(),
                viewStride.size());
        return false;
    }
    ct.isContiguous = op::IsContiguous(input);
    ct.dtypeSize = typeSize;
    for (int64_t i = 0; i < static_cast<int64_t>(viewShape.GetDimNum()); i++) {
        ct.dims.push_back(viewShape.GetDim(i));
    }
    if (!ct.isContiguous) {
        for (int64_t i = 0; i < static_cast<int64_t>(viewStride.size()); i++) {
            ct.strides.push_back(viewStride[i]);
        }
    }
    return true;
}

} // anonymous namespace

bool IsBroadcastTemplateNonContiguousSupport(const std::vector<const aclTensor*>& inputs, const op::Shape& outputShape)
{
    if (inputs.empty()) {
        return false;
    }

    std::vector<CollapsedTensor> collectedInputs;
    collectedInputs.reserve(inputs.size());
    bool allContiguous = true;
    for (int64_t i = 0; i < static_cast<int64_t>(inputs.size()); i++) {
        CollapsedTensor ct;
        if (!CollectInput(inputs[i], ct)) {
            return false;
        }
        if (!ct.isContiguous) {
            allContiguous = false;
        }
        collectedInputs.push_back(ct);
    }

    if (allContiguous) {
        OP_LOGD("IsBroadcastTemplateNonContiguousSupport: all inputs contiguous, skip collapse");
        return true;
    }

    std::vector<CollapsedTensor> collapsed;
    std::vector<int64_t> collapsedOutputDims;
    if (!DoDimensionCollapse(collectedInputs, outputShape, collapsed, collapsedOutputDims)) {
        OP_LOGD("Broadcast Template NonContiguous UnSupported. dimension collapse failed");
        return false;
    }

    return CheckNonContiguousSupport(collapsed, collapsedOutputDims);
}

} // namespace op
