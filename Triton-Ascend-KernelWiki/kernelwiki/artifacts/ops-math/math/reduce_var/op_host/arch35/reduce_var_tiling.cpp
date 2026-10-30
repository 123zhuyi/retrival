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
 * \file reduce_var_tiling.cpp
 * \brief
 */

#include "reduce_var_tiling.h"
#include "op_api/op_util.h"
#include "log/log.h"
#include "math/reduce_var/op_kernel/arch35/reduce_var_tiling_key.h"

namespace optiling {
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INDEX_ATTR_DIM = 0;
constexpr size_t INDEX_ATTR_CORRECTION = 1;
constexpr size_t INDEX_ATTR_KEEPDIM = 2;
constexpr size_t INDEX_ATTR_MEANOUT = 3;

constexpr int32_t SIZE2 = 2;
constexpr int32_t SIZE3 = 3;
constexpr int32_t SIZE4 = 4;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t NDDMA_MAX_A_NUM = 4096;
constexpr static uint32_t MAX_INNER_A = 512; // 单位字节
constexpr static double THRES_HOLD = 0.95;
constexpr static double THRES_HOLD_PATTERN_A = 0.5;
constexpr static int32_t A_STEP_LEN = 4;
constexpr static int32_t AXES_STEP = 2; // A轴和R轴循环遍历的步长

constexpr uint32_t WELFORD_GROUP_NUM = 8;
constexpr uint32_t GROUP_CACHE_BUF_SIZE = (WELFORD_GROUP_NUM + 1) * MAX_INNER_A;
constexpr uint32_t MAX_RES_OUT_SIZE = 16 * 1024U;

static ge::DataType GetPromoteType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return ge::DT_FLOAT16;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
            return ge::DT_FLOAT;
        case ge::DT_FLOAT:
            return ge::DT_FLOAT;
        case ge::DT_INT32:
            return ge::DT_INT32;
        case ge::DT_INT64:
            return ge::DT_INT64;
        default:
            return ge::DT_UNDEFINED;
    }
}

void ReduceVarTiling::MakeWrapDim(const std::vector<int64_t>& shape, std::vector<int64_t>& axes)
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

void ReduceVarTiling::AssembleUnit(
    Ops::Base::ReduceTilingUnit& unit, int32_t idx, uint64_t inner, uint64_t outer, uint64_t step)
{
    unit.idx = idx;
    unit.inner = inner;
    unit.outer = outer;
    unit.step = step;
}

ge::graphStatus ReduceVarTiling::ReduceVarGetInputParams(Ops::Base::ReduceOpInputParam& inputParam)
{
    OP_CHECK_IF(
        (Ops::Base::ReduceOpTmpl::GetInputDtype(context_, INPUT_INDEX_X, inputParam.inputDtype) == ge::GRAPH_FAILED),
        OP_LOGE(context_->GetNodeName(), "ReduceOp get x input dtype failed"),
        return ge::GRAPH_FAILED);

    static const std::vector<ge::DataType> supDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
    OP_CHECK_IF((std::find(supDtypes.begin(), supDtypes.end(), inputParam.inputDtype) == supDtypes.end()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(inputParam.inputDtype),
            "The dtype of x must be within the range {DT_FLOAT16, DT_BF16, DT_FLOAT}"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (Ops::Base::ReduceOpTmpl::GetInputShape(context_, INPUT_INDEX_X, inputParam.shape) == ge::GRAPH_FAILED),
        OP_LOGE(context_->GetNodeName(), "ReduceOp get x input shape failed"), return ge::GRAPH_FAILED);

    inputParam.promoteDtpye = GetPromoteType(inputParam.inputDtype);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t* attrCorrection = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_CORRECTION);
    correction_ = (attrCorrection == nullptr) ? 1 : (*attrCorrection);

    const bool* attrMeantOut = attrs->GetAttrPointer<bool>(INDEX_ATTR_MEANOUT);
    isMeanOut_ = 1;
    if (attrMeantOut != nullptr && !(*attrMeantOut)) {
        isMeanOut_ = 0;
    }

    int64_t inputDimNum = inputParam.shape.size();
    auto dim = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dim);
    size_t dimSize = dim->GetSize();
    if (dimSize == 0u) {
        inputParam.axes.resize(inputDimNum);
        for (int64_t i = 0; i < inputDimNum; i++) {
            inputParam.axes[i] = i;
        }
    } else {
        auto dimData = reinterpret_cast<const int64_t*>(dim->GetData());
        inputParam.axes.resize(dimSize);
        for (size_t i = 0; i < dimSize; i++) {
            OP_CHECK_IF(!ops::IsDimValid(inputDimNum, dimData[i]),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dim",
                    std::to_string(dimData[i]).c_str(),
                    ("Each value of dim must be [-" + std::to_string(inputDimNum) + ", " +
                     std::to_string(inputDimNum) + ")").c_str()),
                return ge::GRAPH_FAILED);
            inputParam.axes[i] = dimData[i];
            if (dimData[i] < 0) {
                inputParam.axes[i] = dimData[i] + inputDimNum;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

void ReduceVarTiling::ReduceVarCalcInput(const Ops::Base::ReduceOpInputParam& inputParam)
{
    totalReduceSize_ = 1;
    std::stringstream ss;
    for (size_t i = 0; i < inputParam.axes.size(); i++) {
        totalReduceSize_ *= inputParam.shape[inputParam.axes[i]];
        ss << inputParam.axes[i] << " ";
    }

    varFactor_ = 1.0;
    meanFactor_ = totalReduceSize_ == 0 ? 0.0 : static_cast<double>(1.0) / static_cast<double>(totalReduceSize_);
    if (correction_ >= totalReduceSize_) {
        correctionInvalid_ = 1;
    } else {
        varFactor_ = static_cast<double>(1.0) / static_cast<double>(totalReduceSize_ - correction_);
    }

    OP_LOGI(context_->GetNodeName(),
        "correction:%ld isMeanOut:%ld correctionInvalid:%ld totalReduceSize:%ld "
        "inputParam.axes:%s inputParam.shape:%s varFactor:%f meanFactor:%f",
        correction_, isMeanOut_, correctionInvalid_, totalReduceSize_, ss.str().c_str(),
        Ops::Base::ReduceOpTmpl::VectorToString(inputParam.shape).c_str(), varFactor_, meanFactor_);
}

void ReduceVarTiling::SetReduceCntEachGroupR()
{
    uint64_t groupR = tilingData_->groupR;
    if (groupR <= 1) {
        return;
    }

    int64_t reduceCntEachGroupR[MAX_CORE_COUNT] = {0};

    uint64_t loopRStart;
    uint64_t loopREnd;
    uint64_t maxRCnt = tilingData_->factorRTotalCnt;
    uint64_t rCntPerCore = tilingData_->factorRCntPerCore;
    uint64_t* shape = tilingData_->shape;
    uint64_t factorR = tilingData_->ubFactorR;
    uint64_t rStepNum = unitR_.idx < 0 ? 1 : Ops::Base::CeilDiv(shape[unitR_.idx], factorR);
    uint64_t start = 0;
    uint64_t stride = 0;

    for (uint64_t groupIdx = 0; groupIdx < groupR; groupIdx++) {
        loopRStart = groupIdx % groupR * rCntPerCore;
        loopREnd = loopRStart + rCntPerCore;
        if (loopRStart > maxRCnt) {
            loopRStart = maxRCnt;
        }
        if (loopREnd > maxRCnt) {
            loopREnd = maxRCnt;
        }

        reduceCntEachGroupR[groupIdx] = 0;
        for (uint64_t i = loopRStart; i < loopREnd; i++) {
            auto cur = i % rStepNum;
            start = cur * factorR;
            stride = shape[unitR_.idx] - start;
            if (likely(stride >= factorR)) {
                stride = factorR;
            }
            reduceCntEachGroupR[groupIdx] += stride * innerUbRCnt_;
        }
    }
    OP_LOGI(
        context_->GetNodeName(), "reduceCntEachGroupR:%s innerUbRCnt_:%lu",
        Ops::Base::ReduceOpTmpl::VectorToString(reduceCntEachGroupR, groupR).c_str(), innerUbRCnt_);

    reduceVarTilingData_->set_reduceCntEachGroupR(reduceCntEachGroupR);
}

void ReduceVarTiling::SetUseNddma()
{
    reduceVarTilingData_->set_useNddma(0);
    if (tilingData_->groupR > 1U) {
        return;
    }
    if (dimNum_ != SIZE3) {
        return;
    }

    uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
    OP_CHECK_IF(dSize == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(opInput_.inputDtype).c_str(), "The dtype size of x must be greater than 0"),
        return);
    uint64_t dataBlockSize = compileInfo_.ubBlockSize / dSize;

    // 暂时只放开ARA且后两维都较小的场景。 尾轴reduce场景kernel实现较复杂，暂时先不做
    uint64_t* shape = tilingData_->shape;
    if (shape[dimNum_ - 1] >= dataBlockSize || shape[dimNum_ - SIZE2] >= dataBlockSize) {
        return;
    }

    // 4096: 64*64
    if (shape[0] < NDDMA_MAX_A_NUM) {
        return;
    }

    reduceVarTilingData_->set_useNddma(1);
    OP_LOGD(context_->GetNodeName(), "use nddma");
}

void ReduceVarTiling::ComputeInnerUbRCnt(const uint64_t* shape)
{
    if (unitR_.idx < -1 * Ops::Base::ReduceOpTmpl::CONST2) {
        return;
    }

    for (auto idx = unitR_.idx + Ops::Base::ReduceOpTmpl::CONST2; idx < dimNum_;
         idx += Ops::Base::ReduceOpTmpl::CONST2) {
        innerUbRCnt_ *= shape[idx];
    }
}

void ReduceVarTiling::ConvertReduceOpTilingData(ReduceVarTilingDataStru* dst, const Ops::Base::ReduceOpTilingData* src)
{
    dst->set_factorACntPerCore(src->factorACntPerCore);
    dst->set_factorATotalCnt(src->factorATotalCnt);
    dst->set_ubFactorA(src->ubFactorA);
    dst->set_factorRCntPerCore(src->factorRCntPerCore);
    dst->set_factorRTotalCnt(src->factorRTotalCnt);
    dst->set_ubFactorR(src->ubFactorR);
    dst->set_groupR(src->groupR);
    dst->set_outSize(src->outSize);
    dst->set_basicBlock(src->basicBlock);
    dst->set_resultBlock(src->resultBlock);
    dst->set_coreNum(src->coreNum);
    dst->set_useNddma(src->useNddma);
    dst->set_meanVar(src->meanVar);
    uint64_t shape[Ops::Base::ReduceOpTmpl::MAX_DIM] = {0};
    uint64_t stride[Ops::Base::ReduceOpTmpl::MAX_DIM] = {0};
    uint64_t dstStride[Ops::Base::ReduceOpTmpl::MAX_DIM] = {0};
    for (int32_t i = 0; i < Ops::Base::ReduceOpTmpl::MAX_DIM; i++) {
        shape[i] = src->shape[i];
        stride[i] = src->stride[i];
        dstStride[i] = src->dstStride[i];
    }
    dst->set_shape(shape);
    dst->set_stride(stride);
    dst->set_dstStride(dstStride);
}

void ReduceVarTiling::SetReduceVarTilingData()
{
    SetReduceCntEachGroupR();
    SetUseNddma();
    ConvertReduceOpTilingData(&reduceVarTilingData_->reduceOpTiling, tilingData_);
    reduceVarTilingData_->set_correction(correction_);
    reduceVarTilingData_->set_correctionInvalid(correctionInvalid_);
    reduceVarTilingData_->set_isMeanOut(isMeanOut_);
    reduceVarTilingData_->set_workSpaceSize(static_cast<int64_t>(workSpaceSize_));
    reduceVarTilingData_->set_varFactor(static_cast<float>(varFactor_));
    reduceVarTilingData_->set_meanFactor(static_cast<float>(meanFactor_));

    reduceVarTilingData_->SaveToBuffer(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(reduceVarTilingData_->GetDataSize());
}

void ReduceVarTiling::CalcUserBasicBlock(bool patternA)
{
    if (patternA) {
        basicBlock_ = Ops::Base::BASIC_BLOCK / SIZE2;
        resultBlock_ = basicBlock_;
        return;
    }

    uint64_t welfordCacheTimes = 1;
    if (ge::GetSizeByDataType(opInput_.inputDtype) == SIZE2) {
        welfordCacheTimes = SIZE2;
    }

    // 分组welford预留
    constexpr uint64_t groupWelfordCacheSize = GROUP_CACHE_BUF_SIZE * SIZE2;
    uint64_t ubAvilSize = compileInfo_.ubSize - groupWelfordCacheSize;

    // AR: x*2  +  (x*welfordCacheTimes)*2 + (x*welfordCacheTimes)/2
    // RA: x*2  +  (x*welfordCacheTimes)*2 + (x*welfordCacheTimes) + (x*welfordCacheTimes)/2
    // inputDb     mean/var cacheBuf          binarybuf                    tCountBuff
    // 二分累加buf（binarybuf）和 tCountBuff 当前都是按最大的预留的，后续可以优化

    // R全载: maxout = x / R, var + mean共2份out, 按fp32预留(ReduceSum高阶api可直接输出到out上), 输出开db
    double tmpIn = static_cast<double>(ubAvilSize * SIZE2) /
        (static_cast<double>(BUFFER_NUM * SIZE2 + SIZE4 * welfordCacheTimes + welfordCacheTimes * SIZE3) +
         static_cast<double>(BUFFER_NUM * SIZE2 * SIZE2 * welfordCacheTimes) / static_cast<double>(totalReduceSize_));
    // resultBlock_表示输出的大小, 按fp32预留
    resultBlock_ = static_cast<uint64_t>(
        static_cast<double>(tmpIn) * static_cast<double>(welfordCacheTimes) / static_cast<double>(totalReduceSize_));
    resultBlock_ = Ops::Base::FloorAlign(resultBlock_, compileInfo_.cacheLineSize);
    // 尾轴A=2，fp16场景, 计算cacheline切分时是会多切一点的，会导致尾轴pad后，ub内的A超过MAX_INNER_A
    // resultBlock_ 也作为groupreduce第一阶段的输出，要按fp32预留
    uint64_t innerASize = cBlock_.aSize * ge::GetSizeByDataType(opInput_.promoteDtpye);
    if (resultBlock_ < innerASize) {
        resultBlock_ = Ops::Base::CeilAlign(innerASize, compileInfo_.cacheLineSize);
    }
    if (resultBlock_ < MAX_INNER_A) {
        resultBlock_ = MAX_INNER_A;
    } else if (resultBlock_ > MAX_RES_OUT_SIZE) {
        resultBlock_ = MAX_RES_OUT_SIZE;
    }

    // basicBlock_表示输入的大小
    uint64_t preBufSize = ubAvilSize - resultBlock_ * BUFFER_NUM * SIZE2;
    basicBlock_ = (preBufSize * SIZE2) / (BUFFER_NUM * SIZE2 + SIZE4 * welfordCacheTimes + welfordCacheTimes * SIZE3);
    basicBlock_ = Ops::Base::FloorAlign(basicBlock_, compileInfo_.vRegSize);

    OP_LOGI(context_->GetNodeName(), "basicBlock_: %lu resultBlock_: %lu bytes", basicBlock_, resultBlock_);
}

void ReduceVarTiling::CalcUserWorkSpace()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    uint64_t groupR = tilingData_->groupR;
    uint64_t outSize = tilingData_->outSize;
    int32_t size = ge::GetSizeByDataType(opInput_.promoteDtpye);
    if (groupR > 1UL) {
        workSpaceSize_ = compileInfo_.vectorCoreNum * Ops::Base::CeilAlign(outSize * size, compileInfo_.cacheLineSize);
    }
    workspaces[0] = Ops::Base::WORKSPACE_SIZE + workSpaceSize_ * SIZE2;
}

ge::graphStatus ReduceVarTiling::PrepareCompileInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo_.vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo_.vectorCoreNum == 0UL),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "vectorCoreNum",
            std::to_string(compileInfo_.vectorCoreNum).c_str(),
            "The value of vectorCoreNum must be greater than 0"),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= Ops::Base::CACHE_BUF_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize",
            std::to_string(compileInfo_.ubSize).c_str(),
            ("The value of ubSize must be greater than " + std::to_string(Ops::Base::CACHE_BUF_SIZE)).c_str()),
        return ge::GRAPH_FAILED);
    compileInfo_.ubSize = ubSize;

    compileInfo_.cacheLineSize = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF(compileInfo_.cacheLineSize == 0UL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cacheLineSize",
            std::to_string(compileInfo_.cacheLineSize).c_str(),
            "The value of cacheLineSize must be greater than 0"),
        return ge::GRAPH_FAILED);

    compileInfo_.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF(compileInfo_.ubBlockSize == 0UL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubBlockSize",
            std::to_string(compileInfo_.ubBlockSize).c_str(),
            "The value of ubBlockSize must be greater than 0"),
        return ge::GRAPH_FAILED);

    compileInfo_.vRegSize = Ops::Base::GetVRegSize(context_);
    OP_CHECK_IF(compileInfo_.vRegSize == 0UL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "vRegSize",
            std::to_string(compileInfo_.vRegSize).c_str(),
            "The value of vRegSize must be greater than 0"),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        context_->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
        compileInfo_.vectorCoreNum, compileInfo_.ubSize, compileInfo_.cacheLineSize, compileInfo_.ubBlockSize,
        compileInfo_.vRegSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReduceVarTiling::PreProcessOptionalParam()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<Ops::Base::ReduceOpTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (memset_s(tilingData_, sizeof(Ops::Base::ReduceOpTilingData), 0, sizeof(Ops::Base::ReduceOpTilingData)) != EOK),
        OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);

    return PrepareCompileInfo();
}

// elimniate dim and axes where dim = 1
void ReduceVarTiling::EliminateOne(
    const std::vector<int64_t>& oriShape, std::vector<int64_t>& axes, uint64_t* shape, int32_t& shapeSize)
{
    int32_t dstIdx = 1; // shape中第一个数给了1, 跳过第一个数
    for (size_t i = 0; i < axes.size(); i++) {
        // 前面补了一维，所有的axes需要加1
        axes[i] = axes[i] + 1;
    }
    int32_t eraseNum = 0;
    for (size_t i = 0; i < oriShape.size(); i++) {
        auto iter = std::find(axes.begin(), axes.end(), i + 1);
        if (oriShape[i] != 1) {
            shape[dstIdx++] = oriShape[i];
            if (iter != axes.end()) {
                *iter = *iter - eraseNum;
            }
        } else {
            eraseNum++;
            if (iter != axes.end()) {
                axes.erase(iter);
            }
        }
    }
    shapeSize = dstIdx;
    OP_LOGD(context_->GetNodeName(), "after EliminateOne, shape is:%s, axes:%s",
        Ops::Base::ReduceOpTmpl::VectorToString(shape, shapeSize).c_str(),
        Ops::Base::ReduceOpTmpl::VectorToString(axes).c_str());
}

// merge continuous r axes and a axes
void ReduceVarTiling::MergeAxis(std::vector<int64_t>& axes, uint64_t* shape, int32_t& shapeSize)
{
    int32_t tmpSize = 0;
    for (int32_t i = 0; i < shapeSize;) {
        auto iter0 = std::find(axes.begin(), axes.end(), i);
        bool isRAxis0 = iter0 != axes.end();
        uint64_t s = shape[i];
        int32_t j = i + 1;
        for (; j < shapeSize; j++) {
            auto iter1 = std::find(axes.begin(), axes.end(), j);
            bool isRAxis1 = iter1 != axes.end();
            if (isRAxis0 != isRAxis1) {
                break;
            }
            s *= shape[j];
            if (isRAxis1) {
                // 连续的R轴, 需要擦除后续R轴的索引
                axes.erase(iter1);
            }
        }
        i = j;
        shape[tmpSize++] = s;
        if (isRAxis0) {
            *iter0 = tmpSize - 1;
        }
    }
    for (int32_t i = tmpSize; i < shapeSize; i++) {
        shape[i] = 0UL;
    }
    shapeSize = tmpSize;
    OP_LOGD(context_->GetNodeName(), "after MergeAxis, shape is:%s, axes:%s",
        Ops::Base::ReduceOpTmpl::VectorToString(shape, shapeSize).c_str(),
        Ops::Base::ReduceOpTmpl::VectorToString(axes).c_str());
}

void ReduceVarTiling::TransformShape(
    const std::vector<int64_t>& oriShape, std::vector<int64_t>& axes, uint64_t* shape, int32_t& shapeSize)
{
    shape[0] = 1UL;
    EliminateOne(oriShape, axes, shape, shapeSize);
    MergeAxis(axes, shape, shapeSize);
}

// pad dim = 1 in front of actual dim
template <class Pattern>
void ReduceVarTiling::PadDimOne(uint64_t* shape)
{
    // TailA pad to ARARARARA, TailR pad to ARARARAR
    int32_t padNum = Pattern::TailA ? Ops::Base::ReduceOpTmpl::CONST9 - Pattern::Dim :
                                      Ops::Base::ReduceOpTmpl::CONST8 - Pattern::Dim;
    int32_t maxDim = Pattern::TailA ? Ops::Base::ReduceOpTmpl::CONST9 : Ops::Base::ReduceOpTmpl::CONST8;
    if (padNum == 0) {
        return;
    }
    for (int32_t i = 0; i < Pattern::Dim; ++i) {
        shape[maxDim - 1 - i] = shape[static_cast<uint64_t>(Pattern::Dim - 1 - i)];
    }
    for (int32_t i = 0; i < padNum; ++i) {
        shape[i] = 1UL;
    }
}

// dispatch tiling with different pattern
ge::graphStatus ReduceVarTiling::DoTilingMatchPattern(uint64_t* shape, int32_t shapeSize)
{
    switch (shapeSize) {
        case Ops::Base::ReduceOpTmpl::CONST1:
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::A>(shape);
        case Ops::Base::ReduceOpTmpl::CONST2:
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::AR>(shape);
        case Ops::Base::ReduceOpTmpl::CONST3:
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::ARA>(shape);
        case Ops::Base::ReduceOpTmpl::CONST4:
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::ARAR>(shape);
        case Ops::Base::ReduceOpTmpl::CONST5:
            PadDimOne<Ops::Base::ReduceOpTmpl::__reducePattern::ARARA>(shape);
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::ARARARARA>(shape);
        case Ops::Base::ReduceOpTmpl::CONST6:
            PadDimOne<Ops::Base::ReduceOpTmpl::__reducePattern::ARARAR>(shape);
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::ARARARAR>(shape);
        case Ops::Base::ReduceOpTmpl::CONST7:
            PadDimOne<Ops::Base::ReduceOpTmpl::__reducePattern::ARARARA>(shape);
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::ARARARARA>(shape);
        case Ops::Base::ReduceOpTmpl::CONST8:
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::ARARARAR>(shape);
        case Ops::Base::ReduceOpTmpl::CONST9:
            return ComputeTiling<Ops::Base::ReduceOpTmpl::__reducePattern::ARARARARA>(shape);
        default:
            OP_LOGE(context_->GetNodeName(), "unsupport pattern");
            return ge::GRAPH_FAILED;
    }
}

template <class Pattern>
bool ReduceVarTiling::IsAxisA(int32_t idx)
{
    if (Pattern::FirstA) {
        return idx % Ops::Base::ReduceOpTmpl::CONST2 == 0;
    }
    return idx % Ops::Base::ReduceOpTmpl::CONST2 == 1;
}

/*
 * cacheLine切分找到硬件cacheLine大小的位置
 * 例如: float32, shape:(2, 35, 7), cacheLine:256B
 * cacheLine切分找到axis:1, step:10, outer:4
 */
template <class Pattern>
void ReduceVarTiling::ComputeCacheLineBlock(const uint64_t* shape) {
    uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
    OP_CHECK_IF(dSize == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(opInput_.inputDtype).c_str(), "The dtype size of x must be greater than 0"),
        return);
    uint64_t cacheSize = compileInfo_.cacheLineSize / dSize;
    uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
    uint64_t cacheLineShape = 1, cacheLineStep = 1, cacheLineOuter = 1;
    uint64_t aInCacheLine = 1, rInCacheLine = 1;
    for (int32_t i = Pattern::Dim - 1; i > -1; --i) {
        cacheLineShape *= shape[i];
        if (cacheLineShape > cacheSize) {
            cacheLineShape /= shape[i];
            cacheLineStep = Ops::Base::CeilDiv(cacheSize, cacheLineShape);
            cacheLineShape *= cacheLineStep;
            cacheLineOuter = Ops::Base::CeilDiv(shape[i], cacheLineStep);
            cBlock_.axis = i;
            break;
        }
        cacheLineStep = shape[i];
        cBlock_.axis = i;
    }
    for (int32_t i = Pattern::Dim - 1; i > cBlock_.axis; --i) {
        if (i == Pattern::Dim - 1) {
            if (IsAxisA<Pattern>(i)) {
                aInCacheLine = aInCacheLine * Ops::Base::CeilAlign(shape[i], ubBlockSize);
            } else {
                rInCacheLine = rInCacheLine * Ops::Base::CeilAlign(shape[i], ubBlockSize);
            }
        } else {
            if (IsAxisA<Pattern>(i)) {
                aInCacheLine = aInCacheLine * shape[i];
            } else {
                rInCacheLine = rInCacheLine * shape[i];
            }
        }
    }
    if (IsAxisA<Pattern>(cBlock_.axis)) {
        aInCacheLine *= cacheLineStep;
    } else {
        rInCacheLine *= cacheLineStep;
    }
    cBlock_.cacheLineStep = cacheLineStep;
    cBlock_.cacheLineOuter = cacheLineOuter;
    cBlock_.aSize = aInCacheLine;
    cBlock_.rSize = rInCacheLine;
    OP_LOGD(context_->GetNodeName(), "cacheLine Block axis:%d, cacheLineStep:%lu, cacheLineOuter:%lu, aSize:%lu, rSize:%lu",
        cBlock_.axis, cBlock_.cacheLineStep, cBlock_.cacheLineOuter, cBlock_.aSize, cBlock_.rSize);
}

template <class Pattern>
ge::graphStatus ReduceVarTiling::ComputeEmptyTiling(uint64_t* shape)
{
    uint64_t outSize = 1;
    for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
        if (IsAxisA<Pattern>(dim)) {
            outSize *= shape[dim];
        }
    }
    tilingData_->outSize = outSize;
    context_->SetBlockDim(compileInfo_.vectorCoreNum);
    if (outSize == 0UL) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t ubAvilSize = compileInfo_.ubSize - Ops::Base::CACHE_BUF_SIZE;
    basicBlock_ = Ops::Base::FloorAlign(ubAvilSize / Ops::Base::ReduceOpTmpl::CONST2, compileInfo_.vRegSize); // double buffer
    uint64_t newshape[Ops::Base::ReduceOpTmpl::MAX_DIM] = {outSize};
    // 空tensor去除R轴后，作为全A的pattern计算切分
    ComputeCacheLineBlock<Ops::Base::ReduceOpTmpl::__reducePattern::A>(newshape);
    unitA_.outer *= cBlock_.cacheLineOuter;
    ComputeUnitA<Ops::Base::ReduceOpTmpl::__reducePattern::A>(newshape);
    SetTilingData<Ops::Base::ReduceOpTmpl::__reducePattern::A>(newshape);
    return ge::GRAPH_SUCCESS;
}

template <class Pattern>
bool ReduceVarTiling::IsEmptyTensor(const uint64_t* shape)
{
    for (int32_t i = 0; i < Pattern::Dim; i++) {
        if (shape[i] == 0UL) {
            return true;
        }
    }
    return false;
}

template <class Pattern>
void ReduceVarTiling::InitUnit(const uint64_t* shape)
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
void ReduceVarTiling::ComputeCacheLineBlockAndUnit(const uint64_t* shape)
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
void ReduceVarTiling::ComputeUnitA(const uint64_t* shape)
{
    int32_t axisInCacheLine = cBlock_.axis;
    uint64_t outerA = unitA_.outer;
    uint64_t innerA = unitA_.inner;
    uint64_t maxCacheA = MAX_INNER_A / maxInputBytes_;
    uint64_t maxInnerA =
        (Pattern::ID == Ops::Base::ReduceOpTmpl::PATTERN_A) ? basicBlock_ * Ratio() / maxInputBytes_ : maxCacheA;
    uint64_t stepLen = (Pattern::ID == Ops::Base::ReduceOpTmpl::PATTERN_A) ? A_STEP_LEN : 1; // 纯A的步长为4, 减少循环次数
    bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
    uint64_t bBlockNum = basicBlock_ * Ratio() / maxInputBytes_;
    uint64_t step = 1;
    int32_t iA;
    for (iA = basicSplitA ? axisInCacheLine : axisInCacheLine - 1; iA > -1; iA -= AXES_STEP) {
        uint64_t axisLen = ((iA == axisInCacheLine) ? cBlock_.cacheLineOuter : shape[iA]);
        bool splitHere = false;
        uint64_t maxStep = 0;
        double maxRate = 0.0f;
        for (step = 1UL; step <= axisLen / stepLen; step++) { // 从2开始查找，1的切分没有意义
            uint64_t s = step * stepLen;
            uint64_t tmpInnerA = innerA * s;
            uint64_t tmpOuterA = outerA / axisLen * Ops::Base::CeilDiv(axisLen, s);
            uint64_t aSize = tmpInnerA * cBlock_.aSize;
            if (iA == axisInCacheLine) {
                aSize = (cBlock_.aSize / cBlock_.cacheLineStep) * std::min(cBlock_.cacheLineStep * s, shape[iA]);
            }
            if (aSize <= maxInnerA && aSize * cBlock_.rSize <= bBlockNum) {
                uint64_t tempCoreNum =
                    (tmpOuterA * unitR_.outer) / Ops::Base::CeilDiv(tmpOuterA * unitR_.outer, compileInfo_.vectorCoreNum);
                tempCoreNum = tempCoreNum > tmpOuterA ? Ops::Base::FloorAlign(tempCoreNum, tmpOuterA) : tempCoreNum;
                double rate = static_cast<double>(tempCoreNum) / static_cast<double>(compileInfo_.vectorCoreNum);
                maxStep = (Pattern::ID == Ops::Base::ReduceOpTmpl::PATTERN_A) ?
                              (rate > THRES_HOLD_PATTERN_A ? step : maxStep) :
                              ((rate > THRES_HOLD && rate > maxRate) ? step : maxStep);
                maxRate = rate > maxRate ? rate : maxRate;
            } else {
                splitHere = true;
                break;
            }
        }
        if (splitHere || maxStep != axisLen / stepLen || iA - AXES_STEP < 0) {
            // A轴最大、分核最大或者无法继续迭代
            step = maxStep == 0UL ? 1UL : maxStep * stepLen;
            innerA = innerA * step;
            outerA = outerA / axisLen * Ops::Base::CeilDiv(axisLen, step);
            break;
        }
        innerA *= axisLen;
        outerA /= axisLen;
    }
    AssembleUnit(unitA_, iA, innerA, outerA, step);
}

/*
 * 根据BasicBlock大小和UB内A轴的切分大小，计算UB内R轴的切分大小
 * 用BasicBlock / UB内A的切分大小，找到满BasicBlock的R轴切分位置
 */
template <class Pattern>
void ReduceVarTiling::ComputeUnitR(const uint64_t* shape)
{
    int32_t axisInCacheLine = cBlock_.axis;
    uint64_t outerR = unitR_.outer;
    uint64_t innerR = unitR_.inner;
    uint64_t outerA = unitA_.outer;
    uint64_t innerA = unitA_.inner;
    uint64_t step = 1UL;
    uint64_t bBlockNum = basicBlock_ * Ratio() / maxInputBytes_;
    bool basicSplitA = IsAxisA<Pattern>(axisInCacheLine);
    int32_t iR;
    for (iR = basicSplitA ? axisInCacheLine - 1 : axisInCacheLine; iR > -1; iR -= AXES_STEP) {
        uint64_t axisLen = ((iR == axisInCacheLine) ? cBlock_.cacheLineOuter : shape[iR]);
        innerR *= axisLen;
        if (innerA * innerR * cBlock_.aSize * cBlock_.rSize <= bBlockNum) {
            outerR = outerR / axisLen;
            continue;
        }

        innerR /= axisLen;
        // maybe bBlockNum not full
        step = std::min(bBlockNum / (innerA * innerR * cBlock_.aSize * cBlock_.rSize), axisLen);
        for (uint64_t s = step; s > 1UL; s--) {
            auto tmpOuterR = outerR / axisLen * Ops::Base::CeilDiv(axisLen, s);
            uint64_t tempCoreNum = (outerA * tmpOuterR) / Ops::Base::CeilDiv(outerA * tmpOuterR, compileInfo_.vectorCoreNum);
            tempCoreNum = tempCoreNum > outerA ? Ops::Base::FloorAlign(tempCoreNum, outerA) : tempCoreNum;
            double rate = static_cast<double>(tempCoreNum) / static_cast<double>(compileInfo_.vectorCoreNum);
            if (rate > THRES_HOLD) {
                // 找不到合适的rate，就不刷新step
                step = s;
                break;
            }
        }
        innerR *= step;
        outerR = outerR / axisLen * Ops::Base::CeilDiv(axisLen, step);
        break;
    }
    AssembleUnit(unitR_, iR, innerR, outerR, step);
}

/*
 * 针对R轴过小，UB内R轴全载，BasicBlock不能满载时，调整UB内A轴切分，上限Reduce计算后输出buffer大小
 */
template <class Pattern>
void ReduceVarTiling::ComputeProgressUnitA(const uint64_t* shape)
{
    // if RAxis is fully loaded in UB, and basicBlock is not enough, we need to calculate extra unitA
    if (unitR_.idx != -1 || Pattern::ID == Ops::Base::ReduceOpTmpl::PATTERN_A) {
        return;
    }
    uint64_t axisLen = (unitA_.idx == cBlock_.axis ? cBlock_.cacheLineOuter : shape[unitA_.idx]);
    uint64_t innerA = unitA_.inner / unitA_.step; // restore original innerA
    uint64_t outerA = unitA_.outer / Ops::Base::CeilDiv(axisLen, unitA_.step) * axisLen;
    uint64_t bBlockNum = basicBlock_ * Ratio() / maxInputBytes_;
    uint64_t maxInnerA = resultBlock_ / maxInputBytes_;
    uint64_t innerR = unitR_.inner;
    uint64_t step = 1;
    int32_t iA;
    for (iA = unitA_.idx; iA > -1; iA -= AXES_STEP) {
        axisLen = (iA == cBlock_.axis ? cBlock_.cacheLineOuter : shape[iA]);
        bool splitHere = false;
        step = (iA == unitA_.idx ? unitA_.step : 1UL);
        for (uint64_t s = step + 1UL; s <= axisLen; s += 1UL) {
            uint64_t tmpInnerA = innerA * s;
            uint64_t tmpOuterA = outerA / axisLen * Ops::Base::CeilDiv(axisLen, s);
            double rate = static_cast<double>(tmpOuterA) /
                          static_cast<double>(Ops::Base::CeilAlign(tmpOuterA, compileInfo_.vectorCoreNum));
            bool isContinue =
                (rate > THRES_HOLD && tmpInnerA * innerR * cBlock_.aSize * cBlock_.rSize <= bBlockNum &&
                 tmpInnerA * cBlock_.aSize <= maxInnerA);
            if (isContinue) {
                continue;
            } else {
                // update step
                step = s > 1UL ? s - 1UL : s;
                splitHere = true;
                break;
            }
        }
        if (splitHere || iA - AXES_STEP < 0) {
            innerA *= step;
            outerA = outerA / axisLen * Ops::Base::CeilDiv(axisLen, step);
            break;
        }
        innerA *= axisLen;
        outerA /= axisLen;
    }
    AssembleUnit(unitA_, iA, innerA, outerA, step);
}

template <class Pattern>
int32_t ReduceVarTiling::IsUseNddma(const uint64_t* shape)
{
    int32_t axis = cBlock_.axis;
    uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
    OP_CHECK_IF(dSize == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(opInput_.inputDtype).c_str(), "The dtype size of x must be greater than 0"),
        return 0);
    uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
    if (shape[static_cast<uint64_t>(Pattern::Dim - 1)] >= ubBlockSize) {
        // last dim 大于ubblock, 不做NDDMA
        return 0;
    }
    if (Pattern::Dim - 1 - axis >= Ops::Base::ReduceOpTmpl::CONST2) {
        // cacheline 切分最后两维之外，使用NDDMA
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
void ReduceVarTiling::ComputeStride(const uint64_t* shape)
{
    uint64_t s = 1UL;
    uint64_t ds = 1UL;
    for (int32_t dim = Pattern::Dim - 1; dim > -1; dim--) {
        tilingData_->stride[dim] = s;
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

template <class Pattern>
void ReduceVarTiling::SetTilingData(const uint64_t* shape)
{
    uint64_t cacheStep = cBlock_.cacheLineStep;
    int32_t axis = cBlock_.axis;
    uint64_t perCoreNum = Ops::Base::CeilDiv(unitA_.outer * unitR_.outer, compileInfo_.vectorCoreNum);
    uint64_t numBlocks = Ops::Base::CeilDiv(unitA_.outer * unitR_.outer, perCoreNum);
    uint64_t factorA = unitA_.idx == axis ? unitA_.step * cacheStep : unitA_.step;
    uint64_t factorR = unitR_.idx == axis ? unitR_.step * cacheStep : unitR_.step;

    if (unitA_.outer < numBlocks) {
        auto tmpBlockDim = Ops::Base::CeilAlign(numBlocks, unitA_.outer);
        if (tmpBlockDim <= compileInfo_.vectorCoreNum) {
            numBlocks = tmpBlockDim;
        } else {
            numBlocks = Ops::Base::FloorAlign(numBlocks, unitA_.outer);
        }
    }

    tilingData_->ubFactorA = factorA;
    uint64_t factorACntPerCore = Ops::Base::CeilDiv(unitA_.outer, numBlocks);
    tilingData_->factorACntPerCore = factorACntPerCore;
    tilingData_->factorATotalCnt = unitA_.outer;

    tilingData_->ubFactorR = factorR;
    uint64_t factorRCntPerCore = Ops::Base::CeilDiv(unitR_.outer, Ops::Base::CeilDiv(numBlocks, unitA_.outer));
    tilingData_->factorRCntPerCore = factorRCntPerCore;
    tilingData_->factorRTotalCnt = unitR_.outer;
    tilingData_->groupR = Ops::Base::CeilDiv(unitR_.outer, factorRCntPerCore);
    if (tilingData_->groupR > 1) {
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to set ScheduleMode!"),
                    return );
    }
    OP_CHECK_IF(
        (memcpy_s(tilingData_->shape, sizeof(tilingData_->shape), shape, sizeof(tilingData_->shape)) != EOK),
        OP_LOGE(context_->GetNodeName(), "memcpy shape failed"), return);
    tilingData_->basicBlock = basicBlock_;
    tilingData_->resultBlock = resultBlock_;
    tilingData_->coreNum = static_cast<int32_t>(compileInfo_.vectorCoreNum);
    tilingData_->useNddma = IsUseNddma<Pattern>(shape);
    ComputeStride<Pattern>(shape);

    uint32_t realCore = Ops::Base::CeilDiv(unitA_.outer, factorACntPerCore) * Ops::Base::CeilDiv(unitR_.outer, factorRCntPerCore);
    context_->SetBlockDim(realCore);
}

template <class Pattern>
void ReduceVarTiling::SetTilingKey()
{
    uint64_t groupR = tilingData_->groupR;
    int32_t aCount = 0;
    int32_t rCount = 0;
    int32_t innerACount = 0;
    int32_t innerRCount = 0;
    int32_t isPatternA0 = Pattern::FirstA ? 0 : 1;
    int32_t isPatternA1 = Pattern::FirstA ? 1 : 0;
    if (groupR == 1UL) {
        // normal case
        aCount = (unitA_.idx - isPatternA0) / AXES_STEP + 1;
        innerRCount = (unitR_.idx - isPatternA1) / AXES_STEP + 1;
    } else {
        // group case
        aCount = (unitA_.idx - isPatternA0) / AXES_STEP + 1;
        rCount = (unitR_.idx - isPatternA1) / AXES_STEP + 1;
        rCount = rCount + aCount;
    }

    int32_t innerID = Pattern::TailA ? 0 : 1;
    tilingKey_.patternID = Pattern::ID * Ops::Base::ReduceOpTmpl::CONST10 + innerID;
    tilingKey_.loopARCount = static_cast<uint32_t>(aCount * Ops::Base::ReduceOpTmpl::CONST10 + rCount);
    tilingKey_.loopInnerARCount = static_cast<uint32_t>(innerACount * Ops::Base::ReduceOpTmpl::CONST10 + innerRCount);
    OP_LOGI(context_->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u", tilingKey_.patternID,
        tilingKey_.loopARCount, tilingKey_.loopInnerARCount);
}

template <class Pattern>
uint64_t ReduceVarTiling::CaculateReduceSize(const uint64_t* shape)
{
    uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
    OP_CHECK_IF(dSize == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(opInput_.inputDtype).c_str(), "The dtype size of x must be greater than 0"),
        return 1);
    uint64_t ubBlockSize = compileInfo_.ubBlockSize / dSize;
    int32_t dim = Pattern::TailA ? Pattern::Dim - AXES_STEP : Pattern::Dim - Ops::Base::ReduceOpTmpl::CONST1;
    uint64_t r = 1;
    for (int32_t i = dim; i > -1; i = i - AXES_STEP) {
        if (i == Pattern::Dim - 1) {
            r = r * Ops::Base::CeilAlign(shape[i], ubBlockSize);
        } else {
            r = r * shape[i];
        }
    }
    return r;
}

uint64_t ReduceVarTiling::Ratio()
{
    return Ops::Base::CeilDiv(maxInputBytes_, static_cast<uint64_t>(ge::GetSizeByDataType(opInput_.inputDtype)));
}

template <class Pattern>
ge::graphStatus ReduceVarTiling::CalcBasicBlock()
{
    OP_CHECK_IF(compileInfo_.ubSize <= Ops::Base::CACHE_BUF_SIZE + opInput_.reservedSize,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize",
            std::to_string(compileInfo_.ubSize).c_str(),
            ("The ubSize must be greater than " +
             std::to_string(Ops::Base::CACHE_BUF_SIZE + opInput_.reservedSize)).c_str()),
        return ge::GRAPH_FAILED);

    CalcUserBasicBlock(Pattern::ID == Ops::Base::ReduceOpTmpl::PATTERN_A);
    return ge::GRAPH_SUCCESS;
}

// check axes value range in [-dimNum, dimNum)
ge::graphStatus ReduceVarTiling::AxesCheck(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes)
{
    int64_t shapeSize = static_cast<int64_t>(shape.size());
    int64_t axesSize = static_cast<int64_t>(axes.size());
    OP_CHECK_IF((axesSize > shapeSize),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dimSize",
            std::to_string(axesSize).c_str(),
            ("The value of dimSize must be less than or equal to shapeSize " + std::to_string(shapeSize)).c_str()),
        return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < axesSize; i++) {
        OP_CHECK_IF((axes[i] >= shapeSize || axes[i] < 0),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dim",
                std::to_string(axes[i]).c_str(),
                ("Each value of dim must be [0, " + std::to_string(shapeSize) + ")").c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReduceVarTiling::ParamCheck(Ops::Base::ReduceOpInputParam& opInput)
{
    int32_t dtypeSize = ge::GetSizeByDataType(opInput.inputDtype);
    OP_CHECK_IF(dtypeSize <= 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(opInput.inputDtype).c_str(), "The dtype size of x must be greater than 0"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "origin shape is:%s, axes:%s",
        Ops::Base::ReduceOpTmpl::VectorToString(opInput.shape).c_str(),
        Ops::Base::ReduceOpTmpl::VectorToString(opInput.axes).c_str());
    MakeWrapDim(opInput.shape, opInput.axes);
    OP_CHECK_IF((AxesCheck(opInput.shape, opInput.axes) == ge::GRAPH_FAILED),
        OP_LOGE(context_->GetNodeName(), "illegal axes"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReduceVarTiling::DoTiling(Ops::Base::ReduceOpInputParam& opInput, Ops::Base::ReduceTilingKey& key)
{
    OP_CHECK_IF((ParamCheck(opInput) != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "Do tiling param check failed"),
        return ge::GRAPH_FAILED);

    opInput_ = opInput;
    if (maxInputBytes_ == 0UL && opInput_.promoteDtpye != ge::DT_UNDEFINED) {
        maxInputBytes_ = ge::GetSizeByDataType(opInput_.promoteDtpye);
    }

    OP_CHECK_IF((PreProcessOptionalParam() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "Do tiling preprocess optional param failed"),
        return ge::GRAPH_FAILED);

    DoReduceTiling(key);

    return ge::GRAPH_SUCCESS;
}

template <class Pattern>
ge::graphStatus ReduceVarTiling::ComputeTiling(uint64_t* shape)
{
    // Tiling计算分5步：
    // 1. 根据计算图计算搬入的BasicBlock大小
    // 2. 根据cacheline大小计算cacheline的切分轴，保证搬运性能
    // 3. 计算UB内A轴的切分大小，最大512B或者按A轴分核低于85%
    // 4. 根据BasicBlock大小和UB内A轴的切分大小，计算UB内R轴的切分大小
    // 5. 可选，针对R轴过小，UB内R轴全载，BasicBlock不能满载是，调整UB内A轴切分，上限为UB内二分缓存大小
    dimNum_ = Pattern::Dim;
    if (IsEmptyTensor<Pattern>(shape)) {
        return ComputeEmptyTiling<Pattern>(shape);
    }

    // 先计算cacheline切分, 再进行basicBlock_和resultBlock_的计算
    ComputeCacheLineBlockAndUnit<Pattern>(shape);
    OP_CHECK_IF((CalcBasicBlock<Pattern>() == ge::GRAPH_FAILED),
        OP_LOGE(context_->GetNodeName(), "calc basic block failed, maybe unsupport ubsize"),
        return ge::GRAPH_FAILED);

    ComputeUnitA<Pattern>(shape);

    ComputeUnitR<Pattern>(shape);

    ComputeProgressUnitA<Pattern>(shape);

    OP_LOGI(context_->GetNodeName(),
        "tiling step outerA:%lu, innerA:%lu, stepA:%lu, idxA:%d, outerR:%lu, innerR:%lu, stepR:%lu, idxR:%d",
        unitA_.outer, unitA_.inner, unitA_.step, unitA_.idx, unitR_.outer, unitR_.inner, unitR_.step, unitR_.idx);

    SetTilingData<Pattern>(shape);
    SetTilingKey<Pattern>();
    return ge::GRAPH_SUCCESS;
}

void ReduceVarTiling::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "TilingData: factorACntPerCore:%lu, factorATotalCnt:%lu, ubFactorA:%lu, factorRCntPerCore:%lu, "
        "factorRTotalCnt:%lu, ubFactorR:%lu, groupR:%lu, outSize:%lu, basicBlock:%lu, resultBlock:%lu, "
        "meanVar:%lf, numBlocks:%u",
        tilingData_->factorACntPerCore, tilingData_->factorATotalCnt, tilingData_->ubFactorA,
        tilingData_->factorRCntPerCore, tilingData_->factorRTotalCnt, tilingData_->ubFactorR, tilingData_->groupR,
        tilingData_->outSize, tilingData_->basicBlock, tilingData_->resultBlock, tilingData_->meanVar,
        context_->GetBlockDim());
}

void ReduceVarTiling::GetTilingKey(Ops::Base::ReduceTilingKey& key)
{
    key = tilingKey_;
}

void ReduceVarTiling::DoReduceTiling(Ops::Base::ReduceTilingKey& key)
{
    uint64_t newShape[Ops::Base::ReduceOpTmpl::MAX_DIM] = {0};
    int32_t newShapeSize = 0;
    TransformShape(opInput_.shape, opInput_.axes, newShape, newShapeSize);

    DoTilingMatchPattern(newShape, newShapeSize);

    CalcUserWorkSpace();

    GetTilingKey(key);

    PrintTilingData();
}

ge::graphStatus ReduceVarTiling::RunTiling(Ops::Base::ReduceTilingKey& key)
{
    Ops::Base::ReduceOpInputParam inputParam;
    if (ReduceVarGetInputParams(inputParam) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ReduceVarCalcInput(inputParam);

    // 基类 ReduceOpTiling 的 Run
    OP_CHECK_IF((DoTiling(inputParam, key) == ge::GRAPH_FAILED),
        OP_LOGE(context_->GetNodeName(), "ReduceVarTiling Run failed"),
        return ge::GRAPH_FAILED);

    ComputeInnerUbRCnt(tilingData_->shape);
    SetReduceVarTilingData();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ReduceVar(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const ReduceVarCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    Ops::Base::ReduceTilingKey key;
    ReduceVarTilingData tilingData;
    Ops::Base::ReduceOpTilingData reduceTiling;
    ReduceVarTiling tiling(context, compileInfo, &tilingData, &reduceTiling);
    OP_CHECK_IF((tiling.RunTiling(key) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "RunTiling Failed for ReduceVar"),
        return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(
        context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu", key.patternID,
        key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReduceVar(gert::TilingParseContext* context)
{
    OP_CHECK_IF((context == nullptr), OP_LOGE(context->GetNodeName(), "context is nil"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReduceVar).Tiling(Tiling4ReduceVar).TilingParse<ReduceVarCompileInfo>(TilingPrepare4ReduceVar);
} // namespace optiling