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
 * \file reduce_tiling.cpp
 * \brief atvoss reduce template tiling
 */

#include "op_common/atvoss/reduce/reduce_tiling.h"
#include "op_common/op_host/util/math_util.h"
#include "reduce_tiling_batch_invariant.h"

namespace Ops {
namespace Base {
using namespace ReduceOpTmpl;

// float数据类型, UB间reduce缓存为16K时,最大能支持的A, 公式CACHE_BUF_SIZE = MAX_INNER_A * log2(Ro)
// log2(Ro)支持最大取值为32
constexpr static int32_t MAX_INNER_A = 512; // 单位字节
constexpr static int32_t A_STEP_LEN = 4;
/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape,
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param in_shape input shape
 * @return non-scalar shape
 */
inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        static const gert::Shape scalarShape = {1};
        return scalarShape;
    }
    return inShape;
}

namespace ReduceOpTmpl {
static bool CheckAllReduce(gert::TilingContext* context, int32_t outIdx)
{
    auto out = context->GetOutputShape(outIdx);
    OP_CHECK_IF(out == nullptr, OP_LOGE(context, "out is nullptr"), return false);
    gert::Shape outShape = EnsureNotScalar(out->GetStorageShape());
    return (outShape.GetShapeSize() == 1L);
}

static bool CheckIsContiguous(ReduceOpInputParam& opInput)
{
    bool isContiguous{true};
    for (size_t i = 0; i < opInput.shape.size(); i++) {
        if (i != opInput.shape.size() - 1) {
            isContiguous = (opInput.dimStrides[i] == opInput.dimStrides[i + 1] * opInput.shape[i + 1]);
            if (!isContiguous) {
                break;
            }
        } else {
            isContiguous = (opInput.dimStrides[i] == 1UL);
            if (!isContiguous) {
                break;
            }
        }
    }
    return isContiguous;
}

static bool checkTailAIsOne(ReduceOpInputParam& opInput)
{
    int32_t len = opInput.shape.size();
    auto iter = std::find_if(opInput.axes.begin(), opInput.axes.end(),
                             [len](int64_t axis) { return axis == len - 1 || axis == -1; });
    if (opInput.shape[len - 1] == 1 && iter == opInput.axes.end()) {
        return 1;
    } else {
        return 0;
    }
}

ge::graphStatus GetInputShape(gert::TilingContext* context, int32_t idx, std::vector<int64_t>& shape)
{
    auto xInput = context->GetInputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInput);
    gert::Shape xInputShape;
    if (context->InputIsView(idx)) {
        xInputShape = EnsureNotScalar(xInput->GetShape());
    } else {
        xInputShape = EnsureNotScalar(xInput->GetStorageShape());
    }
    size_t shapeSize = xInputShape.GetDimNum();
    OP_CHECK_IF((shapeSize >= static_cast<size_t>(MAX_DIM)),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(shapeSize).c_str(),
                                                         "shape dim size is over max dim, cannot support"),
                return ge::GRAPH_FAILED);

    shape.resize(shapeSize);
    for (size_t i = 0; i < shapeSize; i++) {
        shape[i] = xInputShape.GetDim(i);
        if (shape[i] < 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(shape[i]).c_str(),
                                                     "shape dim cannot support dynamic");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputStride(const gert::TilingContext* context, int32_t idx, std::vector<int64_t>& dimStrides)
{
    auto xStride = context->GetInputStride(idx);
    size_t shapeSize = 0UL;
    if (xStride != nullptr) {
        shapeSize = xStride->GetDimNum();
        OP_CHECK_IF(
            (shapeSize >= static_cast<size_t>(MAX_DIM)),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(shapeSize).c_str(),
                                                     "slice dim size is over max slice dim, cannot support"),
            return ge::GRAPH_FAILED);
    }
    if (shapeSize == 0UL || !context->InputIsView(idx)) {
        // contiguous case, xStride == nullptr or xStride->GetDimNum() == 0
        auto xInput = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, xInput);
        gert::Shape xInputShape = EnsureNotScalar(xInput->GetStorageShape());
        dimStrides.resize(xInputShape.GetDimNum());
        int64_t stride = 1L;
        for (int32_t i = xInputShape.GetDimNum() - 1; i >= 0; i--) {
            dimStrides[i] = stride;
            if (dimStrides[i] < 0) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(dimStrides[i]).c_str(),
                                                         "dimStrides dim size cannot support dynamic");
                return ge::GRAPH_FAILED;
            }
            stride = stride * xInputShape.GetDim(i);
        }
    } else {
        dimStrides.resize(shapeSize);
        for (size_t i = 0; i < shapeSize; i++) {
            dimStrides[i] = xStride->GetStride(i);
            if (dimStrides[i] < 0) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(dimStrides[i]).c_str(),
                                                         "dimStrides dim size cannot support dynamic");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputDtype(gert::TilingContext* context, int32_t idx, ge::DataType& dtype)
{
    auto inputDesc = context->GetInputDesc(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dtype = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputParam(gert::TilingContext* context, ReduceOpInputParam& opInput, int32_t inputIdx)
{
    OP_CHECK_IF((GetInputDtype(context, inputIdx, opInput.inputDtype) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input dtype failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF((GetInputShape(context, inputIdx, opInput.shape) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input shape failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF((GetInputStride(context, inputIdx, opInput.dimStrides) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input stride failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputParam(gert::TilingContext* context, ReduceOpInputParam& opInput, int32_t inputIdx,
                              const ReduceTilingKey& key)
{
    OP_CHECK_IF((GetInputDtype(context, inputIdx, opInput.inputDtype) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input dtype failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF((GetInputShape(context, inputIdx, opInput.shape) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input shape failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF((GetInputStride(context, inputIdx, opInput.dimStrides) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input stride failed"), return ge::GRAPH_FAILED);
    // batch一致性场景，尾轴是A且为1时置为1，用来标识这个尾轴1不能被去掉
    if (key.batchInvariant) {
        opInput.isTailAOne = checkTailAIsOne(opInput);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputParam(gert::TilingContext* context, ReduceOpInputParam& opInput, int32_t inputIdx,
                              int32_t axesIdx, int32_t outIdx)
{
    ge::DataType axesDtype;
    ge::graphStatus status = GetInputDtype(context, axesIdx, axesDtype);
    OP_CHECK_IF((status == ge::GRAPH_FAILED), OP_LOGE(context, "ReduceOpTmpl get axes dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetInputDtype(context, inputIdx, opInput.inputDtype) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetInputShape(context, inputIdx, opInput.shape) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input shape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetInputStride(context, inputIdx, opInput.dimStrides) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input stride failed"), return ge::GRAPH_FAILED);
    if (CheckAllReduce(context, outIdx) && CheckIsContiguous(opInput)) {
        // all reduce 场景不读取const data
        opInput.axes.resize(opInput.shape.size());
        for (size_t i = 0; i < opInput.shape.size(); i++) {
            opInput.axes[i] = i;
        }
    } else {
        if (axesDtype == ge::DT_INT32) {
            status = GetConstInputData<int32_t>(context, axesIdx, opInput.axes);
        } else if (axesDtype == ge::DT_INT64) {
            status = GetConstInputData<int64_t>(context, axesIdx, opInput.axes);
        } else {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "axes",
                                                  Ops::Base::ToString(axesDtype).c_str(),
                                                  "only supports const input dtype in [int32, int64]");
            status = ge::GRAPH_FAILED;
        }
        OP_CHECK_IF((status == ge::GRAPH_FAILED), OP_LOGE(context, "ReduceOpTmpl get axes const input failed"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputParam(gert::TilingContext* context, ReduceOpInputParam& opInput, int32_t inputIdx,
                              int32_t axesIdx, int32_t outIdx, const ReduceTilingKey& key)
{
    ge::DataType axesDtype;
    ge::graphStatus status = GetInputDtype(context, axesIdx, axesDtype);
    OP_CHECK_IF((status == ge::GRAPH_FAILED), OP_LOGE(context, "ReduceOpTmpl get axes dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetInputDtype(context, inputIdx, opInput.inputDtype) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetInputShape(context, inputIdx, opInput.shape) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input shape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetInputStride(context, inputIdx, opInput.dimStrides) == ge::GRAPH_FAILED),
                OP_LOGE(context, "ReduceOpTmpl get input stride failed"), return ge::GRAPH_FAILED);
    if (axesDtype == ge::DT_INT32) {
        status = GetConstInputData<int32_t>(context, axesIdx, opInput.axes);
    } else if (axesDtype == ge::DT_INT64) {
        status = GetConstInputData<int64_t>(context, axesIdx, opInput.axes);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "axes", Ops::Base::ToString(axesDtype).c_str(),
                                              "only supports const input dtype in [int32, int64]");
        status = ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((status == ge::GRAPH_FAILED), OP_LOGE(context, "ReduceOpTmpl get axes const input failed"),
                return ge::GRAPH_FAILED);
    // batch一致性场景，尾轴是A且为1时置为1，用来标识这个尾轴1不能被去掉
    if (key.batchInvariant) {
        opInput.isTailAOne = checkTailAIsOne(opInput);
    }
    if (CheckAllReduce(context, outIdx) && CheckIsContiguous(opInput) && !opInput.isTailAOne) {
        // all reduce 场景不读取const data
        opInput.axes.resize(opInput.shape.size());
        for (size_t i = 0; i < opInput.shape.size(); i++) {
            opInput.axes[i] = i;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace ReduceOpTmpl

static void MakeWrapDim(const std::vector<int64_t>& shape, std::vector<int64_t>& axes)
{
    // EnsureNotScalar at least return 1-D Tensor, so shapeSize cannot be 0
    size_t shapeSize = shape.size();
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] < 0) {
            axes[i] += shapeSize;
        }
    }
    std::sort(axes.begin(), axes.end());
}

template <class Pattern>
bool ReduceOpTiling::IsAxisA(int32_t idx)
{
    if (Pattern::FirstA) {
        return idx % CONST2 == 0;
    } else {
        return idx % CONST2 == 1;
    }
}

void ReduceOpTiling::PrintTilingData()
{
    OP_LOGI(context_,
            "TilingData: factorACntPerCore:%lu, factorATotalCnt:%lu, ubFactorA:%lu, factorRCntPerCore:%lu, "
            "factorRTotalCnt:%lu, ubFactorR:%lu, groupR:%lu, outSize:%lu, basicBlock:%lu, resultBlock:%lu, "
            "coreNum:%u, useNddma:%u, meanVar:%lf",
            tilingData_->factorACntPerCore, tilingData_->factorATotalCnt, tilingData_->ubFactorA,
            tilingData_->factorRCntPerCore, tilingData_->factorRTotalCnt, tilingData_->ubFactorR, tilingData_->groupR,
            tilingData_->outSize, tilingData_->basicBlock, tilingData_->resultBlock, context_->GetBlockDim(),
            tilingData_->useNddma, tilingData_->meanVar);

    OP_LOGI(
        context_,
        "TilingData: shape is:%s, stride is:%s, dstStride is:%s, sliceNum is:%s, sliceShape is:%s, sliceStride is:%s",
        ReduceOpTmpl::VectorToString(tilingData_->shape, MAX_DIM).c_str(),
        ReduceOpTmpl::VectorToString(tilingData_->stride, MAX_DIM).c_str(),
        ReduceOpTmpl::VectorToString(tilingData_->dstStride, MAX_DIM).c_str(),
        ReduceOpTmpl::VectorToString(tilingData_->sliceNum, MAX_DIM).c_str(),
        ReduceOpTmpl::VectorToString(tilingData_->sliceShape, MAX_DIM).c_str(),
        ReduceOpTmpl::VectorToString(tilingData_->sliceStride, MAX_DIM).c_str());
}

uint64_t ReduceOpTiling::Ratio()
{
    if (opDag_.memLevel == MemLevel::LEVEL_2) {
        return CeilDiv(opDag_.maxInputBytes, static_cast<uint64_t>(ge::GetSizeByDataType(opInput_.inputDtype)));
    }
    return 1UL;
}

// check axes value range in [-dimNum, dimNum)
ge::graphStatus ReduceOpTiling::AxesCheck(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes)
{
    int64_t shapeSize = static_cast<int64_t>(shape.size());
    int64_t axesSize = static_cast<int64_t>(axes.size());
    OP_CHECK_IF(
        (axesSize > shapeSize),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "axes", std::to_string(axesSize).c_str(),
                                                  "axes size over x shape size is illegal"),
        return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < axesSize; i++) {
        if (axes[i] >= shapeSize || axes[i] < 0) {
            std::string reasonMsg = "illegal axis: " + std::to_string(i) + " dim: " + std::to_string(axes[i]) +
                                    " out of shape range:[0, " + std::to_string(shapeSize) + ")";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "axes", std::to_string(axes[i]).c_str(),
                                                     reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReduceOpTiling::ParamCheck(ReduceOpInputParam& opInput)
{
    int32_t dtypeSize = ge::GetSizeByDataType(opInput.inputDtype);
    OP_CHECK_IF(dtypeSize <= 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dtypeSize).c_str(),
                                                      "dtype of input x is illegal"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_, "view shape is:%s, strides:%s, axes:%s", ReduceOpTmpl::VectorToString(opInput.shape).c_str(),
            ReduceOpTmpl::VectorToString(opInput.dimStrides).c_str(),
            ReduceOpTmpl::VectorToString(opInput.axes).c_str());
    MakeWrapDim(opInput.shape, opInput.axes);
    OP_CHECK_IF((AxesCheck(opInput.shape, opInput.axes) == ge::GRAPH_FAILED), OP_LOGE(context_, "illegal axes"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReduceOpTiling::PreProcessOptionalParam()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<ReduceOpTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(ReduceOpTilingData), 0, sizeof(ReduceOpTilingData)) != EOK),
                OP_LOGE(context_, "memset tilingdata failed"), return ge::GRAPH_FAILED);

    if (opInput_.dimStrides.empty()) {
        size_t shapeSize = opInput_.shape.size();
        opInput_.dimStrides.resize(shapeSize);
        int64_t stride = 1L;
        for (int32_t i = shapeSize - 1; i >= 0; i--) {
            opInput_.dimStrides[i] = stride;
            stride = stride * opInput_.shape[i];
        }
    }

    if (compileInfo_ == nullptr) {
        compileInfoPtr_ = std::make_unique<ReduceOpCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr_);
        compileInfo_ = compileInfoPtr_.get();
    }
    compileInfo_->vectorCoreNum = GetAivCoreNum(context_);
    OP_CHECK_IF(compileInfo_->vectorCoreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "vectorCoreNum",
                                                      std::to_string(compileInfo_->vectorCoreNum).c_str(),
                                                      "The value of vector core num must be greater than 0"),
                return ge::GRAPH_FAILED);
    compileInfo_->ubSize = GetUbSize(context_);
    OP_CHECK_IF(compileInfo_->ubSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize",
                                                      std::to_string(compileInfo_->ubSize).c_str(),
                                                      "The value of ub size must be greater than 0"),
                return ge::GRAPH_FAILED);
    compileInfo_->cacheLineSize = GetCacheLineSize(context_);
    OP_CHECK_IF(compileInfo_->cacheLineSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cacheLineSize",
                                                      std::to_string(compileInfo_->cacheLineSize).c_str(),
                                                      "The value of cacheline size must be greater than 0"),
                return ge::GRAPH_FAILED);
    compileInfo_->ubBlockSize = GetUbBlockSize(context_);
    OP_CHECK_IF(compileInfo_->ubBlockSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubBlockSize",
                                                      std::to_string(compileInfo_->ubBlockSize).c_str(),
                                                      "The value of ub block size must be greater than 0"),
                return ge::GRAPH_FAILED);
    compileInfo_->vRegSize = GetVRegSize(context_);
    OP_CHECK_IF(compileInfo_->vRegSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "vRegSize",
                                                      std::to_string(compileInfo_->vRegSize).c_str(),
                                                      "The value of vReg size must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// elimniate dim and axes where dim = 1
void ReduceOpTiling::EliminateOne(uint64_t* viewShape, int64_t* viewStride, int32_t& shapeSize)
{
    bool contiguousFlag = isContiguous_;
    int32_t dstIdx = 1; // shape中第一个数给了1, 跳过第一个数
    for (size_t i = 0; i < opInput_.axes.size(); i++) {
        // 前面补了一维，所有的axes需要加1
        opInput_.axes[i] = opInput_.axes[i] + 1;
    }
    int32_t eraseNum = 0;
    for (size_t i = 0; i < opInput_.shape.size(); i++) {
        auto iter = std::find(opInput_.axes.begin(), opInput_.axes.end(), i + 1);
        bool isContiguous{false};
        if (i != opInput_.shape.size() - 1) {
            isContiguous = (opInput_.dimStrides[i] == opInput_.dimStrides[i + 1] * opInput_.shape[i + 1]);
        } else {
            isContiguous = (opInput_.dimStrides[i] == 1UL);
        }
        // 连续场景、开启batch一致性、尾轴为A且为1时，不消除该1轴
        if (((i == opInput_.shape.size() - 1) && opInput_.isTailAOne && contiguousFlag) || opInput_.shape[i] != 1L ||
            !isContiguous) {
            viewShape[dstIdx] = opInput_.shape[i];
            viewStride[dstIdx] = opInput_.dimStrides[i];
            if (iter != opInput_.axes.end()) {
                *iter = *iter - eraseNum;
            }
            dstIdx++;
        } else {
            eraseNum++;
            if (iter != opInput_.axes.end()) {
                opInput_.axes.erase(iter);
            }
        }
    }
    shapeSize = dstIdx;
    viewStride[0] = opInput_.dimStrides[0] * opInput_.shape[0];
    OP_LOGI(context_, "after EliminateOne, viewShape is:%s, viewStride is:%s, axes:%s",
            ReduceOpTmpl::VectorToString(viewShape, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(viewStride, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(opInput_.axes).c_str());
}

// For NonContiguous scene, pad dim one after non-contiguous axis
void ReduceOpTiling::PadDimForNonContiguous(uint64_t* viewShape, int64_t* viewStride, int32_t& shapeSize)
{
    int32_t nonContiguousNum = 0;
    // last dim non-contiguous, no need pad
    for (int32_t i = 0; i < shapeSize - 1; i++) {
        bool isContiguous = (viewStride[i] == viewStride[i + 1] * static_cast<int64_t>(viewShape[i + 1]));
        if (!isContiguous) {
            nonContiguousNum++;
        }
    }
    int32_t j = nonContiguousNum + shapeSize - 1;
    for (int32_t i = shapeSize - 1; i >= 0; i--) {
        bool isContiguous{false};
        if (i != shapeSize - 1) {
            isContiguous = (viewStride[i] == viewStride[i + 1] * static_cast<int64_t>(viewShape[i + 1]));
        } else {
            isContiguous = (viewStride[i] == 1UL);
        }
        auto iter = std::find(opInput_.axes.begin(), opInput_.axes.end(), i);
        bool isReduce = (iter != opInput_.axes.end());
        if (!isContiguous && i != shapeSize - 1) {
            // 非尾轴的非连续轴后加1
            viewShape[j] = 1UL;
            viewStride[j] = viewStride[j + 1] * static_cast<int64_t>(viewShape[j + 1]);
            if (!isReduce) {
                opInput_.axes.emplace_back(j);
            }
            j--;
        }
        viewShape[j] = viewShape[i];
        viewStride[j] = viewStride[i];
        if (isReduce) {
            *iter = j;
        }
        j--;
    }
    std::sort(opInput_.axes.begin(), opInput_.axes.end());
    shapeSize = nonContiguousNum + shapeSize;
    OP_LOGI(context_, "after PadDimForNonContiguous, viewShape is:%s, viewStride is:%s, axes:%s",
            ReduceOpTmpl::VectorToString(viewShape, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(viewStride, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(opInput_.axes).c_str());
}

// merge continuous r axes and a axes
void ReduceOpTiling::MergeAxis(uint64_t* viewShape, int64_t* viewStride, int32_t& shapeSize)
{
    int32_t tmpSize = 0;
    for (int32_t i = 0; i < shapeSize;) {
        sliceNum_[tmpSize] = 1UL;
        sliceShape_[tmpSize] = viewShape[i];
        sliceStride_[tmpSize] = viewStride[i];
        viewStride_[tmpSize] = viewStride[i];
        auto iter0 = std::find(opInput_.axes.begin(), opInput_.axes.end(), i);
        bool isRAxis0 = iter0 != opInput_.axes.end();
        uint64_t s = viewShape[i];
        int32_t j = i + 1;
        for (; j < shapeSize; j++) {
            int64_t curContiguousStride = (j == shapeSize - 1 ? 1L : viewStride[j + 1] * viewShape[j + 1]);
            bool isContiguous1 = (viewStride[j] == curContiguousStride);
            auto iter1 = std::find(opInput_.axes.begin(), opInput_.axes.end(), j);
            bool isRAxis1 = iter1 != opInput_.axes.end();
            if (isRAxis0 != isRAxis1) {
                break;
            }
            if (!isContiguous1) {
                sliceNum_[tmpSize] = s; // sliceNum 更新为非连续的非尾轴大小
            }
            s *= viewShape[j];
            sliceShape_[tmpSize] = isContiguous1 ? sliceShape_[tmpSize] * viewShape[j] :
                                                   viewShape[j]; // sliceShape 更新为非连续的尾轴大小
            sliceStride_[tmpSize] = isContiguous1 ? curContiguousStride : sliceStride_[tmpSize];
            viewStride_[tmpSize] = viewStride[j];
            if (isRAxis1) {
                // 连续的R轴, 需要擦除后续R轴的索引
                opInput_.axes.erase(iter1);
            }
        }
        i = j;
        viewShape[tmpSize] = s;
        tmpSize++;
        if (isRAxis0) {
            *iter0 = tmpSize - 1;
        }
    }
    for (int32_t i = tmpSize; i < shapeSize; i++) {
        viewShape[i] = 0;
    }
    shapeSize = tmpSize;
    OP_LOGI(context_,
            "after merge axis, viewShape:%s, viewStride:%s, sliceNum:%s, sliceShape:%s, sliceStride:%s, axes:%s",
            ReduceOpTmpl::VectorToString(viewShape, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(viewStride_, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(sliceNum_, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(sliceShape_, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(sliceStride_, shapeSize).c_str(),
            ReduceOpTmpl::VectorToString(opInput_.axes).c_str());
}

template <class Pattern>
bool ReduceOpTiling::IsEmtpyTensor(const uint64_t* shape)
{
    for (int32_t i = 0; i < Pattern::Dim; i++) {
        if (shape[i] == 0) {
            return true;
        }
    }
    return false;
}

template <class Pattern>
uint64_t ReduceOpTiling::CaculateReduceSize(const uint64_t* shape)
{
    uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / dSize;
    int32_t dim = Pattern::TailA ? Pattern::Dim - AXES_STEP : Pattern::Dim - CONST1;
    uint64_t r = 1;
    for (int32_t i = dim; i > -1; i = i - AXES_STEP) {
        if (i == Pattern::Dim - 1) {
            r = r * CeilAlign(shape[i], ubBlockSize);
        } else {
            r = r * shape[i];
        }
    }
    return r;
}

// pad dim = 1 in front of actual dim
template <class Pattern>
void ReduceOpTiling::PadDimOne(uint64_t* shape)
{
    // TailA pad to ARARARARA, TailR pad to ARARARAR
    int32_t padNum = Pattern::TailA ? CONST9 - Pattern::Dim : CONST8 - Pattern::Dim;
    int32_t maxDim = Pattern::TailA ? CONST9 : CONST8;
    if (padNum == 0) {
        return;
    }
    for (int32_t i = 0; i < Pattern::Dim; ++i) {
        shape[maxDim - 1 - i] = shape[Pattern::Dim - 1 - i];
        viewStride_[maxDim - 1 - i] = viewStride_[Pattern::Dim - 1 - i];
        sliceNum_[maxDim - 1 - i] = sliceNum_[Pattern::Dim - 1 - i];
        sliceShape_[maxDim - 1 - i] = sliceShape_[Pattern::Dim - 1 - i];
        sliceStride_[maxDim - 1 - i] = sliceStride_[Pattern::Dim - 1 - i];
    }
    for (int32_t i = 0; i < padNum; ++i) {
        shape[i] = 1UL;
        sliceNum_[i] = 1UL;
        sliceShape_[i] = 1UL;
        viewStride_[i] = viewStride_[padNum] * shape[padNum];
        sliceStride_[i] = sliceStride_[padNum] * sliceNum_[padNum];
    }
}

void ReduceOpTiling::TransformShape(uint64_t* shape, int64_t* viewStride, int32_t& shapeSize)
{
    shape[0] = 1UL;

    EliminateOne(shape, viewStride, shapeSize);

    PadDimForNonContiguous(shape, viewStride, shapeSize);

    MergeAxis(shape, viewStride, shapeSize);
}

void ReduceOpTiling::DoReduceTiling(ReduceTilingKey& key)
{
    uint64_t newShape[MAX_DIM] = {0};
    int64_t newStride[MAX_DIM] = {0};
    int32_t newShapeSize = 0;
    tilingKey_.batchInvariant = key.batchInvariant;
    isContiguous_ = CheckIsContiguous(opInput_);
    TransformShape(newShape, newStride, newShapeSize);

    DoTilingMatchPattern(newShape, newShapeSize);

    CalcUserWorkSpace();

    GetTilingKey(key);

    PrintTilingData();
}

/*
 * cacheLine切分找到硬件cacheLine大小的位置
 * 例如: float32, shape:(2, 35, 7), cacheLine:256B
 * cacheLine切分找到axis:1, step:10, outer:4
 */
template <class Pattern>
void ReduceOpTiling::ComputeCacheLineBlock(const uint64_t* shape)
{
    uint64_t cacheSize = compileInfo_->cacheLineSize / ge::GetSizeByDataType(opInput_.inputDtype);
    // 尾轴为A的场景，按照最小字节数的UBBlock元素个数对齐
    uint64_t dSize = (Pattern::TailA ? opDag_.minInputBytes : ge::GetSizeByDataType(opInput_.inputDtype));
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / dSize;
    uint64_t cacheLineShape = 1UL;
    uint64_t cacheLineStep = 1UL;
    uint64_t cacheLineOuter = 1UL;
    uint64_t aInCacheLine = 1UL;
    uint64_t rInCacheLine = 1UL;

    for (int32_t i = Pattern::Dim - 1; i > -1; --i) {
        cacheLineShape *= shape[i];
        if (cacheLineShape > cacheSize) {
            cacheLineShape /= shape[i];
            cacheLineStep = CeilDiv(cacheSize, cacheLineShape); // sliceshape
            // 非连续与sliceShape对齐
            cacheLineStep = cacheLineStep > sliceShape_[i] ? FloorAlign(cacheLineStep, sliceShape_[i]) : cacheLineStep;
            if (cacheLineStep >= sliceShape_[i]) {
                cacheLineOuter = CeilDiv(shape[i], cacheLineStep);
            } else {
                cacheLineOuter = sliceNum_[i] * CeilDiv(sliceShape_[i], cacheLineStep);
            }
            cBlock_.axis = i;
            break;
        } else {
            cacheLineStep = shape[i];
            cBlock_.axis = i;
        }
    }

    for (int32_t i = Pattern::Dim - 1; i > cBlock_.axis; --i) {
        if (i == Pattern::Dim - 1) {
            if (IsAxisA<Pattern>(i)) {
                aInCacheLine = aInCacheLine * CeilAlign(shape[i], ubBlockSize);
            } else {
                rInCacheLine = rInCacheLine * CeilAlign(shape[i], ubBlockSize);
            }
        } else {
            if (IsAxisA<Pattern>(i)) {
                aInCacheLine = aInCacheLine * shape[i];
            } else {
                rInCacheLine = rInCacheLine * shape[i];
            }
        }
    }

    cBlock_.cacheLineStep = cacheLineStep;
    cBlock_.cacheLineOuter = cacheLineOuter;
    cBlock_.aSize = aInCacheLine;
    cBlock_.rSize = rInCacheLine;
    OP_LOGD(context_, "cacheLine Block axis:%d, cacheLineStep:%lu, cacheLineOuter:%lu, aSize:%lu, rSize:%lu",
            cBlock_.axis, cBlock_.cacheLineStep, cBlock_.cacheLineOuter, cBlock_.aSize, cBlock_.rSize);
}

template <class Pattern>
void ReduceOpTiling::InitUnit(const uint64_t* shape)
{
    int32_t axisInCacheLine = cBlock_.axis;
    for (int32_t i = Pattern::FirstA ? 0 : 1; i < axisInCacheLine; i += AXES_STEP) {
        unitA_.outer *= shape[i];
    }
    for (int32_t i = Pattern::FirstA ? 1 : 0; i < axisInCacheLine; i += AXES_STEP) {
        unitR_.outer *= shape[i];
    }
    bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
    if (basicSplitA) {
        unitA_.outer *= cBlock_.cacheLineOuter;
    } else {
        unitR_.outer *= cBlock_.cacheLineOuter;
    }
}

template <class Pattern>
void ReduceOpTiling::ComputeCacheLineBlockAndUnit(const uint64_t* shape)
{
    ComputeCacheLineBlock<Pattern>(shape);
    InitUnit<Pattern>(shape);
}

/*
 * 计算UB内A轴的切分大小，最大512B或者按A轴分核低于85%
 * 例如: float32, shape:(12800, 8), pattern:AR, cacheLine:256B, coreNum:64
 * cacheLine切分后(1600, cacheLine(8, 8))
 * 对1600做A轴切分，找到uintA inner:16(cacheline中A轴为8, 与16相乘后达到512B上限), outer:100
 */
template <class Pattern>
void ReduceOpTiling::ComputeUnitA(const uint64_t* shape)
{
    int32_t axisInCacheLine = cBlock_.axis;
    bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
    uint64_t outerA = (basicSplitA ? unitA_.outer / cBlock_.cacheLineOuter * shape[axisInCacheLine] : unitA_.outer);
    uint64_t innerA = unitA_.inner;
    uint64_t maxCacheA = MAX_INNER_A / opDag_.maxInputBytes;
    uint64_t maxInnerA = Pattern::ID == PATTERN_A ? basicBlock_ * Ratio() / opDag_.maxInputBytes : maxCacheA;
    uint64_t stepLen = Pattern::ID == PATTERN_A ? A_STEP_LEN : 1; // 纯A的步长为4, 减少循环次数
    uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / opDag_.minInputBytes;
    uint64_t step = 1;
    int32_t iA;
    for (iA = basicSplitA ? axisInCacheLine : axisInCacheLine - 1; iA > -1; iA -= AXES_STEP) {
        uint64_t axisLen = shape[iA];
        // cacheline切分轴从cacheLineStep开始找，非cacheline切分轴从1开始
        step = (iA == axisInCacheLine ? cBlock_.cacheLineStep : 1UL);
        uint64_t maxStep = (iA == axisInCacheLine ? cBlock_.cacheLineStep : 1UL);
        bool splitHere = false;
        double maxRate = 0.0f;
        for (; step <= axisLen; step = step + stepLen) {
            uint64_t s = step > sliceShape_[iA] ? FloorAlign(step, sliceShape_[iA]) : step; // 非连续与sliceShape对齐
            if (iA == Pattern::Dim - 1 && s <= sliceShape_[iA]) {
                s = FloorAlign(s, ubBlockSize);
            }
            uint64_t tmpInnerA = innerA * s;
            uint64_t tmpOuterA = (s >= sliceShape_[iA] ?
                                      outerA / axisLen * CeilDiv(axisLen, s) :
                                      outerA / axisLen * sliceNum_[iA] * CeilDiv(sliceShape_[iA], s));
            uint64_t aSize = tmpInnerA * cBlock_.aSize;
            if (aSize <= maxInnerA && aSize * cBlock_.rSize <= bBlockNum) {
                uint64_t tempCoreNum = (tmpOuterA * unitR_.outer) /
                                       CeilDiv(tmpOuterA * unitR_.outer, compileInfo_->vectorCoreNum);
                tempCoreNum = tempCoreNum > tmpOuterA ? FloorAlign(tempCoreNum, tmpOuterA) : tempCoreNum;
                double rate = static_cast<double>(tempCoreNum) / static_cast<double>(compileInfo_->vectorCoreNum);
                if (Pattern::TailA) {
                    maxStep = (rate > THRES_HOLD) ? s : maxStep;
                } else {
                    maxStep = (rate > THRES_HOLD && rate > maxRate) ? s : maxStep;
                }
                maxRate = rate > maxRate ? rate : maxRate;
            } else {
                splitHere = true;
                break;
            }
        }
        if (splitHere || maxStep != axisLen || iA - AXES_STEP < 0) {
            // A轴最大、分核最大或者无法继续迭代
            step = maxStep == 0 ? 1UL : maxStep;
            innerA = innerA * step;
            outerA = (step >= sliceShape_[iA] ? outerA / axisLen * CeilDiv(axisLen, step) :
                                                outerA / axisLen * sliceNum_[iA] * CeilDiv(sliceShape_[iA], step));
            break;
        }
        innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
        outerA /= axisLen;
    }
    AssembleUnit(unitA_, iA, innerA, outerA, step);
}

/*
 * 根据BasicBlock大小和UB内A轴的切分大小，计算UB内R轴的切分大小
 * 用BasicBlock / UB内A的切分大小，找到满BasicBlock的R轴切分位置
 */
template <class Pattern>
void ReduceOpTiling::ComputeUnitR(const uint64_t* shape)
{
    int32_t axisInCacheLine = cBlock_.axis;
    bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
    uint64_t outerR = (basicSplitA ? unitR_.outer : unitR_.outer / cBlock_.cacheLineOuter * shape[axisInCacheLine]);
    uint64_t innerR = unitR_.inner;
    uint64_t outerA = unitA_.outer;
    uint64_t innerA = unitA_.inner;
    uint64_t step = 1UL;
    uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
    uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / dSize;
    int32_t iR;
    for (iR = basicSplitA ? axisInCacheLine - 1 : axisInCacheLine; iR > -1; iR -= AXES_STEP) {
        uint64_t axisLen = shape[iR];
        uint64_t effectiveLen = (iR == Pattern::Dim - 1) ? CeilAlign(axisLen, ubBlockSize) : axisLen;
        if (innerA * innerR * cBlock_.aSize * cBlock_.rSize * effectiveLen <= bBlockNum) {
            outerR = outerR / axisLen;
            innerR *= effectiveLen;
            continue;
        }

        // maybe bBlockNum not full
        step = std::min(bBlockNum / (innerA * innerR * cBlock_.aSize * cBlock_.rSize), axisLen);
        if (step > sliceShape_[iR]) {
            step = FloorAlign(step, sliceShape_[iR]);
        }
        if (iR == Pattern::Dim - 1) {
            step = FloorAlign(step, ubBlockSize);
        }
        uint64_t minStep = (iR == axisInCacheLine ? cBlock_.cacheLineStep : 1UL);
        for (uint64_t s = step; s > minStep; s--) {
            uint64_t tmpS = (s > sliceShape_[iR] ? FloorAlign(s, sliceShape_[iR]) : s); // 非连续与sliceShape对齐
            if (iR == Pattern::Dim - 1 && tmpS != sliceShape_[iR]) {
                // 尾轴R场景，尽可能保证尾轴切分block对齐，减少kernel侧补pad
                tmpS = FloorAlign(tmpS, ubBlockSize);
            }
            auto tmpOuterR = (tmpS >= sliceShape_[iR] ?
                                  outerR / axisLen * CeilDiv(axisLen, tmpS) :
                                  outerR / axisLen * sliceNum_[iR] * CeilDiv(sliceShape_[iR], tmpS));

            uint64_t tempCoreNum = (outerA * tmpOuterR) / CeilDiv(outerA * tmpOuterR, compileInfo_->vectorCoreNum);
            tempCoreNum = tempCoreNum > outerA ? FloorAlign(tempCoreNum, outerA) : tempCoreNum;
            double rate = static_cast<double>(tempCoreNum) / static_cast<double>(compileInfo_->vectorCoreNum);
            if (rate > THRES_HOLD) {
                // 找不到合适的rate，就不刷新step
                step = tmpS;
                break;
            }
        }
        AdjustStepToDivisor<Pattern>(shape, iR, step, bBlockNum, innerA, innerR, ubBlockSize);
        innerR = innerR * step;
        outerR = (step >= sliceShape_[iR] ? outerR / axisLen * CeilDiv(axisLen, step) :
                                            outerR / axisLen * sliceNum_[iR] * CeilDiv(sliceShape_[iR], step));
        break;
    }
    AssembleUnit(unitR_, iR, innerR, outerR, step);
}

/*
 * 针对R轴过小，UB内R轴全载，BasicBlock不能满载时，调整UB内A轴切分，上限Reduce计算后输出buffer大小
 */
template <class Pattern>
void ReduceOpTiling::ComputeProgressUnitA(const uint64_t* shape)
{
    // if RAxis is fully loaded in UB, and basicBlock is not enough, we need to calculate extra unitA
    if (unitR_.idx != -1) {
        return;
    }
    uint64_t innerA = unitA_.inner / unitA_.step; // restore original innerA
    uint64_t outerA = unitA_.outer / CeilDiv(shape[unitA_.idx], unitA_.step) * shape[unitA_.idx];
    if (unitA_.step < sliceShape_[unitA_.idx]) {
        outerA = unitA_.outer / sliceNum_[unitA_.idx] / CeilDiv(sliceShape_[unitA_.idx], unitA_.step) *
                 shape[unitA_.idx];
    }
    uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
    uint64_t maxInnerA = resultBlock_ / opDag_.maxInputBytes;
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / opDag_.minInputBytes;
    uint64_t cacheSize = compileInfo_->cacheLineSize / opDag_.minInputBytes;
    uint64_t innerR = unitR_.inner;
    uint64_t step = 1;
    int32_t iA;
    for (iA = unitA_.idx; iA > -1; iA -= AXES_STEP) {
        uint64_t axisLen = shape[iA];
        bool splitHere = false;
        uint64_t s = (iA == unitA_.idx ? unitA_.step + 1UL : 1UL);
        uint64_t maxStep = (iA == unitA_.idx ? unitA_.step : 1UL);
        for (; s <= axisLen; s++) {
            uint64_t tmpS = s > sliceShape_[iA] ? FloorAlign(s, sliceShape_[iA]) : s; // 非连续与sliceShape对齐
            if (iA == Pattern::Dim - 1) {
                tmpS = CeilAlign(tmpS, ubBlockSize);
            }
            uint64_t tmpInnerA = innerA * tmpS;
            uint64_t tmpOuterA = (tmpS >= sliceShape_[iA] ?
                                      outerA / axisLen * CeilDiv(axisLen, tmpS) :
                                      outerA / axisLen * sliceNum_[iA] * CeilDiv(sliceShape_[iA], tmpS));
            double rate = static_cast<double>(tmpOuterA) /
                          static_cast<double>(CeilAlign(tmpOuterA, compileInfo_->vectorCoreNum));
            bool isContinue = (tmpInnerA * innerR * cBlock_.aSize * cBlock_.rSize <= bBlockNum &&
                               tmpInnerA * cBlock_.aSize <= maxInnerA);
            if (isContinue) {
                if (tmpInnerA * cBlock_.aSize <= cacheSize) {
                    maxStep = tmpS;
                } else {
                    maxStep = rate >= THRES_HOLD ? tmpS : maxStep;
                }
                continue;
            } else {
                splitHere = true;
                break;
            }
        }
        if (splitHere || iA - AXES_STEP < 0) {
            step = maxStep;
            innerA *= step;
            outerA = (step >= sliceShape_[iA] ? outerA / axisLen * CeilDiv(axisLen, step) :
                                                outerA / axisLen * sliceNum_[iA] * CeilDiv(sliceShape_[iA], step));
            break;
        }
        innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
        outerA /= axisLen;
    }
    AssembleUnit(unitA_, iA, innerA, outerA, step);
}

/*
 * 非质数且连续场景下，调整step为shape整除的因子，保证R轴均匀切分，减少kernel侧补pad
 */
template <class Pattern>
void ReduceOpTiling::AdjustStepToDivisor(const uint64_t* shape, int32_t iR, uint64_t& step, uint64_t bBlockNum,
                                         uint64_t innerA, uint64_t innerR, uint64_t ubBlockSize)
{
    if (!isPrime(shape[iR]) && isContiguous_) {
        uint64_t maxStep = bBlockNum / (innerA * innerR * cBlock_.aSize * cBlock_.rSize) / 2;
        for (uint64_t i = step; i >= maxStep; i--) {
            if (shape[iR] % i == 0) {
                step = i;
                break;
            }
        }
        if (iR == Pattern::Dim - 1) {
            step = FloorAlign(step, ubBlockSize);
        }
    }
}

template <class Pattern>
uint64_t ReduceOpTiling::TryGetReduceBlock(const uint64_t* shape, uint64_t preInputBufferNum, uint64_t preRestBufferNum,
                                           uint64_t postBufferNum)
{
    uint64_t r = CaculateReduceSize<Pattern>(shape);
    // basic * (preInputBufferNum + preRestBufferNum * ratio) + (basic / r) * postBufferNum * ratio = ub - reserved
    uint64_t ubAvilSize = compileInfo_->ubSize - (CACHE_BUF_SIZE + opInput_.reservedSize);
    uint64_t ratio = Ratio();

    double basicBlock = static_cast<double>(ubAvilSize) /
                        (static_cast<double>(preInputBufferNum + preRestBufferNum * ratio) +
                         static_cast<double>(postBufferNum * ratio) / static_cast<double>(r));
    return static_cast<uint64_t>(static_cast<double>(basicBlock * ratio) / static_cast<double>(r));
}

template <class Pattern>
ge::graphStatus ReduceOpTiling::CalcBasicBlock(const uint64_t* shape)
{
    if (compileInfo_->ubSize <= CACHE_BUF_SIZE + opInput_.reservedSize) {
        std::string reasonMsg = "ubSize: " + std::to_string(compileInfo_->ubSize) +
                                " is smaller than size: " + std::to_string(CACHE_BUF_SIZE + opInput_.reservedSize) +
                                ", not supported";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize",
                                              std::to_string(compileInfo_->ubSize).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // reduce前的计算图是开启DB的，对于输入输出需要乘以2,
    // 且对于内存策略3而言，reduce算子的输入和计算内存是按照cast前后比例计算
    uint64_t preInputBufferNum = opDag_.preInput * CONST2;
    // 用户自定义存活节点在reduce前计算，确保申请足够大的内存
    uint64_t preRestBufferNum = opDag_.preTempCalc + opDag_.preOutput * CONST2 + opInput_.reservedNode;
    if (opDag_.reduceOpPos == 1) {
        // 对于reduce前没有计算的算子，需要额外的inputNum确保搬运流水
        preInputBufferNum = preInputBufferNum * CONST2;
    }
    // reduce后的存活节点加1，用于存放reduce的输出
    uint64_t postBufferNum = opDag_.postInput + opDag_.postTempCalc + opDag_.postOutput + CONST1;

    uint64_t ratio = Ratio();
    uint64_t ubAvilSize = compileInfo_->ubSize - (CACHE_BUF_SIZE + opInput_.reservedSize);
    if (Pattern::ID == PATTERN_A) {
        basicBlock_ = ubAvilSize / (preInputBufferNum + (preRestBufferNum + postBufferNum) * ratio);
        basicBlock_ = FloorAlign(basicBlock_, compileInfo_->vRegSize);
        resultBlock_ = basicBlock_ * ratio;
    } else {
        uint64_t resBlock = TryGetReduceBlock<Pattern>(shape, preInputBufferNum, preRestBufferNum, postBufferNum);
        resultBlock_ = FloorAlign(resBlock, compileInfo_->cacheLineSize);
        if (resultBlock_ < static_cast<uint64_t>(MAX_INNER_A)) {
            resultBlock_ = MAX_INNER_A;
        } else if (resultBlock_ > CACHE_BUF_SIZE) {
            // 输出大小不能超过cache缓存大小, 否则计算额外A轴搬入时会越界
            resultBlock_ = CACHE_BUF_SIZE;
        }
        if (ubAvilSize <= resultBlock_ * postBufferNum) {
            std::string reasonMsg = "ubSize: " + std::to_string(compileInfo_->ubSize) + " is smaller than size: " +
                                    std::to_string(CACHE_BUF_SIZE + opInput_.reservedSize +
                                                   resultBlock_ * postBufferNum) +
                                    ", not supported";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize",
                                                  std::to_string(compileInfo_->ubSize).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        uint64_t preBufSize = ubAvilSize - resultBlock_ * postBufferNum;
        // 按输入和计算大小的比例折算成输入大小
        basicBlock_ = preBufSize / (preInputBufferNum + preRestBufferNum * ratio);
        basicBlock_ = FloorAlign(basicBlock_, compileInfo_->vRegSize);
    }
    if (basicBlock_ < compileInfo_->vRegSize) {
        std::string reasonMsg = "basic block: " + std::to_string(basicBlock_) + " is too small, not supported";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "basicBlock",
                                              std::to_string(basicBlock_).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context_, "preInputBufferNum:%lu, preRestBufferNum:%lu, postBufferNum:%lu, basicBlock:%ld, resBlock:%ld",
            preInputBufferNum, preRestBufferNum, postBufferNum, basicBlock_, resultBlock_);

    CalcUserBasicBlock(Pattern::ID == PATTERN_A);
    return ge::GRAPH_SUCCESS;
}

template <class Pattern>
ge::graphStatus ReduceOpTiling::ComputeEmptyTiling(uint64_t* shape)
{
    uint64_t outSize = 1;
    for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
        if (IsAxisA<Pattern>(dim)) {
            outSize *= shape[dim];
        }
    }
    tilingData_->outSize = outSize;
    context_->SetBlockDim(compileInfo_->vectorCoreNum);
    if (outSize == 0) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t ubAvilSize = compileInfo_->ubSize - CACHE_BUF_SIZE;
    basicBlock_ = FloorAlign(ubAvilSize / CONST2, compileInfo_->vRegSize); // double buffer
    uint64_t newshape[MAX_DIM] = {outSize};
    // 空tensor去除R轴后，作为全A的pattern计算切分
    ComputeCacheLineBlock<__reducePattern::A>(newshape);
    unitA_.outer *= cBlock_.cacheLineOuter;
    // 空tensor场景，按A轴合轴的结果重新赋值sliceShape和sliceNum
    sliceShape_[0] = outSize;
    sliceNum_[0] = 1;
    ComputeUnitA<__reducePattern::A>(newshape);
    SetTilingData<__reducePattern::A>(newshape);
    return ge::GRAPH_SUCCESS;
}

template <class Pattern>
ge::graphStatus ReduceOpTiling::ComputeTiling(uint64_t* shape)
{
    // Tiling计算分5步：
    // 1. 根据计算图计算搬入的BasicBlock大小
    // 2. 根据cacheline大小计算cacheline的切分轴，保证搬运性能
    // 3. 计算UB内A轴的切分大小，最大512B或者按A轴分核低于85%
    // 4. 根据BasicBlock大小和UB内A轴的切分大小，计算UB内R轴的切分大小
    // 5. 可选，针对R轴过小，UB内R轴全载，BasicBlock不能满载是，调整UB内A轴切分，上限为UB内二分缓存大小
    dimNum_ = Pattern::Dim;
    OP_CHECK_IF((CalcBasicBlock<Pattern>(shape) == ge::GRAPH_FAILED),
                OP_LOGE(context_, "calc basic block failed, maybe unsupported ubsize"), return ge::GRAPH_FAILED);
    if (IsEmtpyTensor<Pattern>(shape)) {
        return ComputeEmptyTiling<Pattern>(shape);
    }
    if (tilingKey_.batchInvariant && isContiguous_) {
        ComputeRFirst<Pattern>(shape);
        if (unitR_.idx == -1) {
            ComputeAWithRFullLoad<Pattern>(shape);
        } else {
            ComputeAWithRCannotFullLoad<Pattern>(shape);
        }
        SetTilingDataBatchInvariant<Pattern>(shape);
    } else {
        ComputeCacheLineBlockAndUnit<Pattern>(shape);
        ComputeUnitA<Pattern>(shape);
        ComputeUnitR<Pattern>(shape);
        ComputeProgressUnitA<Pattern>(shape);
        SetTilingData<Pattern>(shape);
    }

    OP_LOGI(context_,
            "tiling step outerA:%lu, innerA:%lu, stepA:%lu, idxA:%d, outerR:%lu, innerR:%lu, stepR:%lu, idxR:%d",
            unitA_.outer, unitA_.inner, unitA_.step, unitA_.idx, unitR_.outer, unitR_.inner, unitR_.step, unitR_.idx);

    SetTilingKey<Pattern>();
    return ge::GRAPH_SUCCESS;
}

template <class Pattern>
void ReduceOpTiling::SetTilingData(const uint64_t* shape)
{
    uint64_t perCoreNum = CeilDiv(unitA_.outer * unitR_.outer, compileInfo_->vectorCoreNum);
    uint64_t numBlocks = CeilDiv(unitA_.outer * unitR_.outer, perCoreNum);
    if (unitA_.outer < numBlocks) {
        auto tmpBlockDim = CeilAlign(numBlocks, unitA_.outer);
        if (tmpBlockDim <= compileInfo_->vectorCoreNum) {
            numBlocks = tmpBlockDim;
        } else {
            numBlocks = FloorAlign(numBlocks, unitA_.outer);
        }
    }

    tilingData_->ubFactorA = unitA_.step;
    uint64_t factorACntPerCore = CeilDiv(unitA_.outer, numBlocks);
    tilingData_->factorACntPerCore = factorACntPerCore;
    tilingData_->factorATotalCnt = unitA_.outer;

    tilingData_->ubFactorR = unitR_.step;
    uint64_t factorRCntPerCore = CeilDiv(unitR_.outer, CeilDiv(numBlocks, unitA_.outer));
    tilingData_->factorRCntPerCore = factorRCntPerCore;
    tilingData_->factorRTotalCnt = unitR_.outer;
    tilingData_->groupR = CeilDiv(unitR_.outer, factorRCntPerCore);
    if (tilingData_->groupR > 1) {
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to set ScheduleMode!"), return);
    }
    for (int32_t i = 0; i < MAX_DIM; i++) {
        tilingData_->shape[i] = shape[i];
        tilingData_->stride[i] = viewStride_[i];
        tilingData_->sliceNum[i] = sliceNum_[i];
        tilingData_->sliceShape[i] = sliceShape_[i];
        tilingData_->sliceStride[i] = sliceStride_[i];
    }
    tilingData_->basicBlock = basicBlock_;
    tilingData_->resultBlock = resultBlock_;
    tilingData_->coreNum = static_cast<int32_t>(compileInfo_->vectorCoreNum);
    tilingData_->useNddma = IsUseNddma<Pattern>(shape);
    ComputeStride<Pattern>(shape);
    uint32_t realCore = CeilDiv(unitA_.outer, factorACntPerCore) * CeilDiv(unitR_.outer, factorRCntPerCore);
    tilingData_->realCoreNum = realCore;
    context_->SetBlockDim(realCore);
}

template <class Pattern>
int32_t ReduceOpTiling::IsUseNddma(const uint64_t* shape)
{
    int32_t axis = cBlock_.axis;
    uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / dSize;
    if (viewStride_[Pattern::Dim - 1] != 1UL) {
        // 尾轴非连续, 做NDDMA
        return 1;
    }
    if (shape[Pattern::Dim - 1] >= ubBlockSize) {
        // last dim 大于ubblock, 不做NDDMA
        return 0;
    }
    if ((Pattern::Dim - 1 - axis > CONST2) || (Pattern::Dim - 1 - axis == CONST2 && cBlock_.cacheLineStep != 1UL)) {
        // cacheline切分超过3维，或者等于三维时，最高维不为1。即cacheline有效切分维度超过2维
        return 1;
    }
    if (Pattern::TailA) {
        uint64_t factorA = tilingData_->ubFactorA;
        for (auto iA = unitA_.idx + AXES_STEP; iA < Pattern::Dim; iA += AXES_STEP) {
            factorA = factorA * shape[iA];
        }
        if (factorA > ubBlockSize) {
            // 转置后A轴乘积大于ubblock, 不做NDDMA
            return 0;
        }
    } else {
        uint64_t factorR = tilingData_->ubFactorR;
        for (auto iR = unitR_.idx + AXES_STEP; iR < Pattern::Dim; iR += AXES_STEP) {
            factorR = factorR * shape[iR];
        }
        if (factorR > ubBlockSize) {
            // 转置后R轴乘积大于ubblock, 不做NDDMA
            return 0;
        }
    }
    return 1;
}

template <class Pattern>
void ReduceOpTiling::ComputeStride(const uint64_t* shape)
{
    uint64_t s = 1UL;
    uint64_t ds = 1UL;
    for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
        tilingData_->dstStride[dim] = ds;
        s *= shape[dim];
        if (IsAxisA<Pattern>(dim)) {
            ds *= shape[dim];
        }
    }
    double meanVar = static_cast<double>(1) / static_cast<double>(s / ds);
    tilingData_->outSize = ds;
    tilingData_->meanVar = static_cast<float>(meanVar);
}

bool ReduceOpTiling::isPrime(uint64_t num) const
{
    if (num <= CONST1)
        return false;
    if (num == CONST2 || num == CONST3)
        return true;
    if (num % CONST2 == 0 || num % CONST3 == 0)
        return false;

    for (uint64_t i = CONST5; i * i <= num; i += CONST6) {
        if (num % i == 0 || num % (i + CONST2) == 0) {
            return false;
        }
    }
    return true;
}

template <class Pattern>
void ReduceOpTiling::SetTilingKey()
{
    uint64_t groupR = tilingData_->groupR;
    int32_t aCount = 0;
    int32_t rCount = 0;
    int32_t innerACount = 0;
    int32_t innerRCount = 0;
    if (groupR == 1UL) {
        // normal case
        aCount = (unitA_.idx - (Pattern::FirstA ? 0 : 1)) / AXES_STEP + 1;
        innerRCount = (unitR_.idx - (Pattern::FirstA ? 1 : 0)) / AXES_STEP + 1;
    } else {
        // group case
        aCount = (unitA_.idx - (Pattern::FirstA ? 0 : 1)) / AXES_STEP + 1;
        rCount = (unitR_.idx - (Pattern::FirstA ? 1 : 0)) / AXES_STEP + 1;
        rCount = rCount + aCount;
    }

    int32_t innerID = Pattern::TailA ? 0 : 1;
    tilingKey_.patternID = Pattern::ID * CONST10 + innerID;
    tilingKey_.loopARCount = static_cast<uint32_t>(aCount * CONST10 + rCount);
    tilingKey_.loopInnerARCount = static_cast<uint32_t>(innerACount * CONST10 + innerRCount);
    tilingKey_.isContiguous = isContiguous_;
    OP_LOGI(context_, "patternID:%u, loopARCount:%u, loopInnerARCount:%u, isContiguous:%d, isBatchInvariant:%d",
            tilingKey_.patternID, tilingKey_.loopARCount, tilingKey_.loopInnerARCount, tilingKey_.isContiguous ? 1 : 0,
            tilingKey_.batchInvariant ? 1 : 0);
}

void ReduceOpTiling::GetTilingKey(ReduceTilingKey& key) { key = tilingKey_; }

void ReduceOpTiling::CalcUserWorkSpace()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    uint64_t groupR = tilingData_->groupR;
    uint64_t outSize = tilingData_->outSize;
    uint64_t size = opDag_.maxInputBytes;
    if (groupR > 1UL) {
        workSpaceSize_ = compileInfo_->vectorCoreNum * CeilAlign(outSize * size, compileInfo_->cacheLineSize);
    }
    workspaces[0] = WORKSPACE_SIZE + workSpaceSize_;
}

// dispatch tiling with different pattern
ge::graphStatus ReduceOpTiling::DoTilingMatchPattern(uint64_t* shape, int32_t shapeSize)
{
    switch (shapeSize) {
        case CONST1:
            return ComputeTiling<__reducePattern::A>(shape);
        case CONST2:
            return ComputeTiling<__reducePattern::AR>(shape);
        case CONST3:
            return ComputeTiling<__reducePattern::ARA>(shape);
        case CONST4:
            return ComputeTiling<__reducePattern::ARAR>(shape);
        case CONST5:
            PadDimOne<__reducePattern::ARARA>(shape);
            return ComputeTiling<__reducePattern::ARARARARA>(shape);
        case CONST6:
            PadDimOne<__reducePattern::ARARAR>(shape);
            return ComputeTiling<__reducePattern::ARARARAR>(shape);
        case CONST7:
            PadDimOne<__reducePattern::ARARARA>(shape);
            return ComputeTiling<__reducePattern::ARARARARA>(shape);
        case CONST8:
            return ComputeTiling<__reducePattern::ARARARAR>(shape);
        case CONST9:
            return ComputeTiling<__reducePattern::ARARARARA>(shape);
        default:
            OP_LOGE(context_, "unsupported pattern");
            return ge::GRAPH_FAILED;
    }
}

} // namespace Base
} // namespace Ops
