/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "conversion/im2col/op_kernel/arch35/im2col_tilingdata.h"
#include "conversion/im2col/op_kernel/arch35/im2col_tilingkey.h"
#include "platform/platform_ascendc.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "op_host/input_util.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include <algorithm>
#include <cmath>

namespace optiling {
// 属性索引
static constexpr size_t ATTR_IDX_KSIZES = 0U;
static constexpr size_t ATTR_IDX_STRIDES = 1U;
static constexpr size_t ATTR_IDX_DILATIONS = 2U;
static constexpr size_t ATTR_IDX_PADDING_MODE = 3U;
static constexpr size_t ATTR_IDX_PADS = 4U;

// 公共常量
static constexpr int32_t PADDING_SPLIT_CNT = 2;
static constexpr size_t ATTR_CNT_KSIZES = 2;
static constexpr size_t ATTR_CNT_STRIDES = 2;
static constexpr size_t ATTR_CNT_DILATIONS = 2;
static constexpr int32_t VREG_ELEMENT_MIN_SIZE = 2;

// NCHW 常量
// BUFFER分割数量
static constexpr uint32_t NCHW_BUFFER_NUM = 2;
// 预留GATHER索引大小
static constexpr uint64_t NCHW_GATHER_INDEX_SIZE = 0;
// 最小输出大小
static constexpr int32_t NCHW_MIN_OUTPUT_BUFFER = 1024;
// 无效BUFFER大小
static constexpr int32_t NCHW_INVALID_BUFFER = -1;
// UB内 gather 操作的最大元素个数
static constexpr int32_t MAX_UB_GATHER_ELEMENT_NUM = std::numeric_limits<uint16_t>::max();
// 最大输入输出比率
static constexpr double NCHW_MAX_INPUT_OUTPUT_RATIO = 128;

// NHWC 常量
static constexpr uint64_t NHWC_BUFFER_NUM = 2;
// 最小WC大小
static constexpr int32_t NHWC_MIN_WC_SIZE = 128;
// 最小buffer大小
static constexpr uint64_t NHWC_MIN_BUFFER_SIZE = 64 * 1024;

// SIMT 常量
static constexpr int64_t MAX_SHAPE_SIZE_FOR_SIMT = 1024;
static constexpr int64_t MAX_UINT32_NUM = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t SIMT_BUFFER_NUM = 2;
static constexpr uint32_t SIMT_THREAD_FACTOR = 2;

class Im2ColTiling {
private:
    /* data */
    // soc info
    uint64_t ubSize_{0};
    uint64_t ubBlockSize_{0};
    uint32_t coreNum_{0};
    uint64_t cacheLineSize_{0};
    int32_t ubBlockElements_{0};
    int32_t cacheLineElements_{0};
    uint32_t vRegSize_{0};
    int32_t gatherVRegElements_{0};

    // tiling key param
    ge::Format inputFormat_{ge::FORMAT_ND};
    bool isSIMT_{false};
    bool isPadding_{false};
    bool isBigShape_{false};
    uint64_t ubAxis_{0};

    // 输入参数
    int32_t dSize_{0};
    Im2ColInputInfo input_{};

    // 中间计算结果
    // 卷积核影响HW
    int64_t effectH_{0};
    int64_t effectW_{0};
    // 填充后的HW
    int64_t paddedH_{0};
    int64_t paddedW_{0};
    // Lw，每个输入W上的核数
    int64_t convKernelNumInWidth_{1};
    // Lh，每个输入H上的核数
    int64_t convKernelNumInHeight_{1};
    // 总核数 = Lw * Lh = L = Ow
    int64_t convKernelNum_{1};
    // 核大小 = kw * kh = Oh
    int64_t convKernelSize_{1};
    // 实际核数
    uint32_t realCoreNum_{0};

    // tiling context
    gert::TilingContext* context_;

public:
    explicit Im2ColTiling(gert::TilingContext* context) : context_(context) {};
    ~Im2ColTiling();

    ge::graphStatus DoTiling();

private:
    // 参数检查，数据获取
    ge::graphStatus ParamCheck();
    ge::graphStatus CheckKSizes(const gert::RuntimeAttrs* attrs);
    ge::graphStatus CheckStrides(const gert::RuntimeAttrs* attrs);
    ge::graphStatus CheckDilations(const gert::RuntimeAttrs* attrs);
    ge::graphStatus CheckPadding(const gert::RuntimeAttrs* attrs);
    ge::graphStatus InferOut();
    ge::graphStatus GetSocInfo();

    // tiling 计算
    ge::graphStatus Tiling4Format();
    ge::graphStatus Tiling4NCHW();
    [[maybe_unused]] ge::graphStatus Tiling4NHWC();
    ge::graphStatus Tiling4SIMT();

    // 辅助函数
    template <typename T>
    inline T AlignBlock(T elementCount);
    // NCHW
    inline int64_t NCHWCalcBurstLen(int64_t rectW, int64_t rectH);
    std::tuple<int32_t, int32_t> NCHWCalcBufSize(int32_t validBufSize);
    bool NCHWTryFullLoad(int32_t validBufSize);
    bool NCHWTryUnFullLoad(int32_t validBufSize);
    // NHWC
    void NHWCSetTilingData(Im2ColNHWCTilingData* tilingData, const int64_t (&ubfactor)[4]);
    // SIMT

    // 打印
    void ShowBaseTilingData();
    void ShowNCHWTilingData();
    void ShowNHWCTilingData();
    void ShowSIMTTilingData();
};

Im2ColTiling::~Im2ColTiling()
{}

ge::graphStatus Im2ColTiling::DoTiling()
{
    // 校验属性
    auto ret = ParamCheck();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling failed"), return ge::GRAPH_FAILED);

    // 输出宽高推导
    ret = InferOut();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling failed"), return ge::GRAPH_FAILED);

    // soc信息获取
    ret = GetSocInfo();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling failed"), return ge::GRAPH_FAILED);

    ret = Tiling4Format();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling failed"), return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(inputFormat_, ubAxis_, isPadding_, isSIMT_, isBigShape_);
    OP_LOGI(
        context_->GetNodeName(), "tilingKey is %lu, inputFormat %d, ubAxis %d, isPadding %d, isSIMT %d, isBigShape %d",
        tilingKey, inputFormat_, ubAxis_, isPadding_, isSIMT_, isBigShape_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum_);
    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline T Im2ColTiling::AlignBlock(T elementCount)
{
    return Ops::Base::CeilAlign(elementCount, static_cast<T>(ubBlockElements_));
}

void Im2ColTiling::ShowBaseTilingData()
{
    OP_LOGI(
        context_,
        "input: N %ld, C %ld, H %ld, W %ld,"
        " kernel (%ld, %ld), stride (%ld, %ld), dilation (%ld, %ld), pad (%ld, %ld, %ld, %ld)",
        input_.N, input_.C, input_.H, input_.W, input_.hKernelSize, input_.wKernelSize, input_.hStride, input_.wStride,
        input_.hDilation, input_.wDilation, input_.hPaddingBefore, input_.hPaddingAfter, input_.wPaddingBefore,
        input_.wPaddingAfter);
    // soc 信息
    OP_LOGI(
        context_, "soc info: ubSize %lu, coreNum %u, cacheLineSize %lu, ubBlockSize %lu ", ubSize_, coreNum_,
        cacheLineSize_, ubBlockSize_);
    // 中间计算结果
    OP_LOGI(
        context_,
        "middle data: convKernelNumInWidth %ld, convKernelNumInHeight %ld, convKernelNum %ld, convKernelSize %ld",
        convKernelNumInWidth_, convKernelNumInHeight_, convKernelNum_, convKernelSize_);
}

void Im2ColTiling::ShowNCHWTilingData()
{
    ShowBaseTilingData();
    auto tilingData = context_->GetTilingData<Im2ColNCHWTilingData>();
    OP_LOGI(
        context_,
        "tiling data: ubFactorH %d, ubFactorW %d, ubFactorNC %d, w4ubFactorW % d,"
        " lines4ubFactorW % d, lines4ubFactorH % d",
        tilingData->ubFactorH, tilingData->ubFactorW, tilingData->ubFactorNC, tilingData->w4ubFactorW,
        tilingData->lines4ubFactorW, tilingData->lines4ubFactorH);
    OP_LOGI(
        context_, "\t: convKernelNumInWidth %ld, convKernelNumInHeight %ld", tilingData->convKernelNumInWidth,
        tilingData->convKernelNumInHeight);
    OP_LOGI(
        context_, "\t: totalRectAngles %ld, rectAnglesPerCore %d, outHWrectAngles %d", tilingData->totalRectAngles,
        tilingData->rectAnglesPerCore, tilingData->outHWrectAngles);
    OP_LOGI(
        context_, "\t: inputBufferSize %d, outputBufferSize %d", tilingData->inputBufferSize,
        tilingData->outputBufferSize);
}

void Im2ColTiling::ShowNHWCTilingData()
{
    ShowBaseTilingData();
    auto tilingData = context_->GetTilingData<Im2ColNHWCTilingData>();
    OP_LOGI(
        context_, "tiling data: ubFactorC %d, ubFactorW %d, ubFactorH %d, ubFactorN %d", tilingData->ubFactorC,
        tilingData->ubFactorW, tilingData->ubFactorH, tilingData->ubFactorN);
    OP_LOGI(
        context_, "\t: convKernelNumInWidth %ld, convKernelNumInHeight %ld", tilingData->convKernelNumInWidth,
        tilingData->convKernelNumInHeight);
    OP_LOGI(context_, "\t: totalLines %ld, linesPerCore %d", tilingData->totalLines, tilingData->linesPerCore);
    OP_LOGI(context_, "\t: outputBufferSize %d", tilingData->outputBufferSize);
}

void Im2ColTiling::ShowSIMTTilingData()
{
    ShowBaseTilingData();
    auto tilingData = context_->GetTilingData<Im2ColSIMTTilingData>();
    OP_LOGI(
        context_, "tiling data: convKernelNumInHeight %ld, convKernelNumInWidth %ld", tilingData->convKernelNumInHeight,
        tilingData->convKernelNumInWidth);
    OP_LOGI(
        context_, "\t: realCoreNum %ld, blockFactor %d, blockTailFactor: %u, mainCoreNum %d, threadNum %d",
        tilingData->realCoreNum, tilingData->blockFactor, tilingData->blockTailFactor, tilingData->mainCoreNum,
        tilingData->threadNum);
}

ge::graphStatus Im2ColTiling::CheckKSizes(const gert::RuntimeAttrs* attrs)
{
    auto ksizes = attrs->GetListInt(ATTR_IDX_KSIZES);
    return Ops::Math::UnpackFixedDimListIntAttr<ATTR_CNT_KSIZES>(
        context_, "ksizes", ksizes, [](int64_t v) { return v > 0; }, "The value of ksizes must be positive",
        input_.hKernelSize, input_.wKernelSize);
}

ge::graphStatus Im2ColTiling::CheckStrides(const gert::RuntimeAttrs* attrs)
{
    auto strides = attrs->GetListInt(ATTR_IDX_STRIDES);
    return Ops::Math::UnpackAdaptDimListIntAttr<ATTR_CNT_STRIDES>(
        context_, "strides", strides, [](int64_t v) { return v > 0; }, "The value of strides must be positive",
        input_.hStride, input_.wStride);
}

ge::graphStatus Im2ColTiling::CheckDilations(const gert::RuntimeAttrs* attrs)
{
    auto dilations = attrs->GetListInt(ATTR_IDX_DILATIONS);
    auto ret = Ops::Math::UnpackAdaptDimListIntAttr<ATTR_CNT_DILATIONS>(
        context_, "dilations", dilations, [](int64_t v) { return v > 0; }, "The value of dilations must be positive",
        input_.hDilation, input_.wDilation);
    if (unlikely(ret != ge::GRAPH_SUCCESS)) {
        return ret;
    }

    // 相对于KernelSize，受影响的宽、高
    effectH_ = (input_.hKernelSize - 1) * input_.hDilation + 1;
    effectW_ = (input_.wKernelSize - 1) * input_.wDilation + 1;
    return ge::GRAPH_SUCCESS;
}

static int64_t CalcNeedPadding(
    const int64_t inputSize, const int64_t effectSize, const int64_t stride, int64_t& paddingBefore,
    int64_t& paddingAfter)
{
    int64_t outputSize = Ops::Base::CeilDiv(inputSize, stride);
    int64_t needPadding = std::max(0L, (outputSize - 1) * stride + effectSize - inputSize);
    paddingBefore = needPadding / PADDING_SPLIT_CNT;
    paddingAfter = needPadding - paddingBefore;
    return outputSize;
}

ge::graphStatus Im2ColTiling::CheckPadding(const gert::RuntimeAttrs* attrs)
{
    auto paddingMode = attrs->GetStr(ATTR_IDX_PADDING_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingMode);
    std::string_view mode = std::string_view(paddingMode);
    if (mode == "CALCULATED") {
        auto pads = attrs->GetListInt(ATTR_IDX_PADS);
        auto ret = Ops::Math::UnpackAdaptDimListIntAttr<4>(
            context_, "pads", pads, [](int64_t v) { return v >= 0; }, "The value of pads cannot be negative",
            input_.hPaddingBefore, input_.hPaddingAfter, input_.wPaddingBefore, input_.wPaddingAfter);
        if (unlikely(ret != ge::GRAPH_SUCCESS)) {
            return ret;
        }
    } else if (mode == "VALID") {
        input_.hPaddingBefore = 0;
        input_.hPaddingAfter = 0;
        input_.wPaddingBefore = 0;
        input_.wPaddingAfter = 0;
    } else if (mode == "SAME") {
        CalcNeedPadding(input_.H, effectH_, input_.hStride, input_.hPaddingBefore, input_.hPaddingAfter);
        CalcNeedPadding(input_.W, effectW_, input_.wStride, input_.wPaddingBefore, input_.wPaddingAfter);
    } else {
        std::string reasonMsg = "The value of mode must be in [CALCULATED , SYMMETRIC and VALID].";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "mode", std::string(mode).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // padding 后的HW
    paddedH_ = input_.H + input_.hPaddingBefore + input_.hPaddingAfter;
    paddedW_ = input_.W + input_.wPaddingBefore + input_.wPaddingAfter;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Im2ColTiling::ParamCheck()
{
    auto inputValueDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);

    auto inputDataType = inputValueDesc->GetDataType();
    dSize_ = ge::GetSizeByDataType(inputDataType);
    if (dSize_ <= 0) {
        std::string reasonMsg = "The size of inputDataType must be greater than 0.";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "inputDataType", std::to_string(dSize_).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验输入shape
    auto inputShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);

    // 获取 N/C/H/W
    auto storageShape = inputShape->GetStorageShape();
    inputFormat_ = inputValueDesc->GetStorageFormat();
    auto ret = Ops::Math::GetImgDataDimsByNCHWOrder(
        context_, "x", storageShape, inputFormat_, input_.N, input_.C, input_.H, input_.W);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "Param check failed"), return ge::GRAPH_FAILED);

    // 校验属性值是否合法
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    ret = CheckKSizes(attrs);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "Param check failed"), return ge::GRAPH_FAILED);
    ret = CheckStrides(attrs);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "Param check failed"), return ge::GRAPH_FAILED);
    ret = CheckDilations(attrs);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "Param check failed"), return ge::GRAPH_FAILED);
    ret = CheckPadding(attrs);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "Param check failed"), return ge::GRAPH_FAILED);
    isPadding_ =
        input_.hPaddingBefore > 0 || input_.hPaddingAfter > 0 || input_.wPaddingBefore > 0 || input_.wPaddingAfter > 0;

    OP_LOGI(context_, "effect HW is (%ld, %ld), padded HW is (%ld, %ld)", effectH_, effectW_, paddedH_, paddedW_);
    return ge::GRAPH_SUCCESS;
}

inline int64_t Im2ColTiling::NCHWCalcBurstLen(int64_t rectW, int64_t rectH)
{
    // kernel影响大小
    int64_t inputW = (rectH - 1) * input_.wDilation + 1;
    // + stride 大小
    inputW += (rectW - 1) * input_.wStride;
    return inputW;
}

std::tuple<int32_t, int32_t> Im2ColTiling::NCHWCalcBufSize(int32_t validBufSize)
{
    // 按照输入输出比例划分UB空间
    int64_t groupH = input_.wKernelSize;
    int64_t groupW = convKernelNumInWidth_;

    // 为防止H过大，先假设最小输入buffer大小1K，保证输入大小
    int64_t tmpInBufSize = 1024;
    // 根据最小输入大小，折算出最大输出大小
    int64_t tmpOutBufSize = validBufSize - tmpInBufSize;
    // 使用最大输出大小，换算为 (?, vectorLength) 的矩阵，去切输出
    int64_t rectW = gatherVRegElements_;
    int64_t rectH = tmpOutBufSize / dSize_ / rectW;
    // 如果折算后的矩阵，H和W方向都小于一个分组的大小，此时应该是最大的比例关系，用这个矩阵的宽高，换算出一行输入的长度
    // 如果折算后的矩阵，W/H方向超过一个分组的大小，截断到分组的大小即可
    // 此时不需要考虑会有多个分组被切分进来的情况，多个分组对比例关系没有影响
    rectW = std::min(rectW, groupW);
    rectH = std::min(rectH, groupH);
    // rectW 对应的输入长度
    int64_t inputW = NCHWCalcBurstLen(rectW, rectH);
    if (unlikely(inputW <= 0)) {
        OP_LOGW(context_, "The input length(%ld) should be positive", inputW);
        return {NCHW_INVALID_BUFFER, NCHW_INVALID_BUFFER};
    }
    // 求对应最大输入长度
    // 当最后一个包含左pad的输入，超过左pad的长度时，说明存在跨左pad和非左pad的场景，则需要考虑左pad，否则不需要
    // 当没有需要考虑的左pad时，最大输入长度为 block对齐burstLen
    // 当有需要考虑的左pad时，最大输入长度为 burstLen 拆分为左pad部分和剩余部分，两部分均block对齐
    // 则最大为对齐block后再加一个block： (ceil(burstLen / block) + 1) * block
    int64_t inputWRectOffset = rectW * input_.wStride;
    int64_t lastLeftPadEnd =
        input_.wPaddingBefore == 0 ? 0 : Ops::Base::FloorAlign(input_.wPaddingBefore, inputWRectOffset) + inputW;
    int64_t alignBurstLen =
        lastLeftPadEnd <= input_.wPaddingBefore ? AlignBlock(inputW) : AlignBlock(inputW + ubBlockElements_);
    // 一行有几个分组
    int64_t groupCnt = groupW >= gatherVRegElements_ ?
                           1 :
                           std::min(convKernelNumInHeight_, static_cast<int64_t>(gatherVRegElements_) / groupW);
    // 计算输出大小
    tmpOutBufSize = AlignBlock(rectW * groupCnt) * rectH * dSize_;
    tmpInBufSize = alignBurstLen * dSize_ * groupCnt;
    // 计算比例
    double ratio = static_cast<double>(tmpOutBufSize) / (tmpInBufSize + tmpOutBufSize);
    OP_LOGD(context_, "The ratio of the output buffer size to total size is %f", ratio);
    // 分配 buffsize
    tmpOutBufSize = static_cast<int64_t>(validBufSize * ratio);
    // 向下对齐 vector length
    tmpOutBufSize = Ops::Base::FloorAlign(tmpOutBufSize, static_cast<int64_t>(vRegSize_));
    tmpInBufSize = validBufSize - tmpOutBufSize;
    // 估算元素个数，限制 buff 大小
    int32_t maxInOutBufSize = MAX_UB_GATHER_ELEMENT_NUM * dSize_;
    if (tmpInBufSize > tmpOutBufSize) {
        if (tmpInBufSize > maxInOutBufSize) {
            double scaleRatio = static_cast<double>(maxInOutBufSize) / tmpInBufSize;
            tmpInBufSize = maxInOutBufSize;
            tmpOutBufSize *= scaleRatio;
        }
    } else {
        if (tmpOutBufSize > maxInOutBufSize) {
            double scaleRatio = static_cast<double>(maxInOutBufSize) / tmpOutBufSize;
            tmpOutBufSize = maxInOutBufSize;
            tmpInBufSize *= scaleRatio;
        }
    }

    // 不会超过int64
    return {static_cast<int32_t>(tmpInBufSize), static_cast<int32_t>(tmpOutBufSize)};
}

bool Im2ColTiling::NCHWTryFullLoad(int32_t validBufSize)
{
    // 1个HW输入需要大小，按HW整体对齐
    int64_t inputHW = input_.H * input_.W;
    int64_t inHWNeedSize = AlignBlock(inputHW) * dSize_;
    // 1个HW输出需要大小，按HW整体对齐
    int64_t outputHW = convKernelNum_ * convKernelSize_;
    int64_t outHWNeedSize = AlignBlock(outputHW) * dSize_;
    double ratio = static_cast<double>(inputHW) / static_cast<double>(outputHW);
    if (ratio > NCHW_MAX_INPUT_OUTPUT_RATIO) {
        return false;
    }

    // 输入输出大小是否满足全载条件
    int64_t allNeedSize = inHWNeedSize + outHWNeedSize;
    if (allNeedSize <= 0 || allNeedSize > validBufSize) {
        return false;
    }
    // 输入1个补pad后的HW的元素个数不能超出gather索引大小
    int64_t inHWElements = paddedH_ * paddedW_;
    if (inHWElements > MAX_UB_GATHER_ELEMENT_NUM) {
        return false;
    }
    // 1个输入元素个数不能超出gather索引大小
    int64_t& outHWElements = outputHW;
    if (outHWElements > MAX_UB_GATHER_ELEMENT_NUM) {
        return false;
    }

    // 满足全载条件，则此时 HW 相关的参数较小，不会有溢出风险
    // 切 NC
    ubAxis_ = TPL_UB_AXIS_NCHW_NC;
    auto tilingData = context_->GetTilingData<Im2ColNCHWTilingData>();
    // rect_h = out_h
    tilingData->ubFactorH = static_cast<int32_t>(convKernelSize_);
    // rect_w = out_w = L
    tilingData->ubFactorW = static_cast<int32_t>(convKernelNum_);
    // 计算NC
    tilingData->ubFactorNC = validBufSize / allNeedSize;
    int64_t inputNC = input_.N * input_.C;
    tilingData->ubFactorNC = std::min(static_cast<int64_t>(tilingData->ubFactorNC), inputNC);
    // 计算buffer size
    tilingData->inputBufferSize = inHWNeedSize * tilingData->ubFactorNC;
    tilingData->outputBufferSize = outHWNeedSize * tilingData->ubFactorNC;
    // 矩阵数量 ceil(N * C / rect_nc)
    tilingData->totalRectAngles = Ops::Base::CeilDiv(inputNC, static_cast<int64_t>(tilingData->ubFactorNC));
    tilingData->outHWrectAngles = 0;
    // 其他参数
    tilingData->w4ubFactorW = inputHW;
    tilingData->lines4ubFactorW = input_.H;
    tilingData->lines4ubFactorH = input_.H;
    return true;
}

bool Im2ColTiling::NCHWTryUnFullLoad(int32_t validBufSize)
{
    auto [inputBufSize, outputBufSize] = NCHWCalcBufSize(validBufSize);
    // 稀疏率较大，回退给SIMT
    if (outputBufSize < NCHW_MIN_OUTPUT_BUFFER) {
        return false;
    }

    int64_t groupH = input_.wKernelSize;
    int64_t groupW = convKernelNumInWidth_;
    // 切HW
    ubAxis_ = TPL_UB_AXIS_NCHW_HW;
    auto tilingData = context_->GetTilingData<Im2ColNCHWTilingData>();
    tilingData->inputBufferSize = inputBufSize;
    tilingData->outputBufferSize = outputBufSize;
    // 用 (?, vectorLength) 的矩阵，去切输出
    tilingData->ubFactorW = static_cast<int32_t>(std::min(static_cast<int64_t>(gatherVRegElements_), convKernelNum_));
    // 对齐 group，防止产生跨行搬运
    int64_t rectCntW;
    if (tilingData->ubFactorW > groupW) {
        // ubw 对齐到 group_w 的倍数
        tilingData->ubFactorW = Ops::Base::FloorAlign(tilingData->ubFactorW, static_cast<int32_t>(groupW));
        // ceil(out_w / rect_w)
        rectCntW = Ops::Base::CeilDiv(convKernelNum_, static_cast<int64_t>(tilingData->ubFactorW));
    } else {
        // ceil(group_w / rect_w) * group_cnt_of_out_w
        rectCntW = Ops::Base::CeilDiv(groupW, static_cast<int64_t>(tilingData->ubFactorW)) * convKernelNumInHeight_;
    }
    // 计算输出行数，输出的每一行需要Block对齐
    tilingData->ubFactorH = tilingData->outputBufferSize / dSize_ / AlignBlock(tilingData->ubFactorW);
    // 对齐out_h，防止跨NC
    tilingData->ubFactorH =
        static_cast<int32_t>(std::min(static_cast<int64_t>(tilingData->ubFactorH), convKernelSize_));
    int64_t rectCntH;
    if (tilingData->ubFactorH > groupH) {
        tilingData->ubFactorH = Ops::Base::FloorAlign(tilingData->ubFactorH, static_cast<int32_t>(groupH));
        // ceil(out_h / rect_h)
        rectCntH = Ops::Base::CeilDiv(convKernelSize_, static_cast<int64_t>(tilingData->ubFactorH));
    } else {
        // ceil(group_h / rect_h) * group_cnt_of_out_h
        rectCntH = Ops::Base::CeilDiv(groupH, static_cast<int64_t>(tilingData->ubFactorH)) * input_.hKernelSize;
    }

    // ceil(rect_h / out_h)
    tilingData->ubFactorNC = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->ubFactorH), convKernelSize_);
    // 输入一行的长度，截取到group大小来算
    tilingData->w4ubFactorW = static_cast<int32_t>(NCHWCalcBurstLen(
        std::min(static_cast<int64_t>(tilingData->ubFactorW), groupW),
        std::min(static_cast<int64_t>(tilingData->ubFactorH), groupH)));
    // ceil(rect_w / group_w)，跨几个group
    tilingData->lines4ubFactorW = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->ubFactorW), groupW);
    // ceil(rect_h / group_h)，跨几个group
    tilingData->lines4ubFactorH = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->ubFactorH), groupH);
    tilingData->outHWrectAngles = rectCntW * rectCntH;
    tilingData->totalRectAngles = tilingData->outHWrectAngles * input_.N * input_.C;
    return true;
}

ge::graphStatus Im2ColTiling::Tiling4NCHW()
{
    // 可用UB空间, (UB - 索引) / 2
    int32_t validBufSize = static_cast<int32_t>(ubSize_ - NCHW_GATHER_INDEX_SIZE) / NCHW_BUFFER_NUM;

    do {
        // 尝试全载
        if (NCHWTryFullLoad(validBufSize)) {
            break;
        }
        // 不满足要求则走非全载
        if (NCHWTryUnFullLoad(validBufSize)) {
            break;
        }
        // SIMT兜底
        return Tiling4SIMT();
    } while (false);

    // 设置 tiling data 公共参数
    auto tilingData = context_->GetTilingData<Im2ColNCHWTilingData>();
    tilingData->input = input_;
    tilingData->convKernelNumInWidth = convKernelNumInWidth_;
    tilingData->convKernelNumInHeight = convKernelNumInHeight_;
    // 设置核数
    tilingData->rectAnglesPerCore =
        static_cast<int32_t>(Ops::Base::CeilDiv(tilingData->totalRectAngles, static_cast<int64_t>(coreNum_)));
    realCoreNum_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(tilingData->totalRectAngles, static_cast<int64_t>(tilingData->rectAnglesPerCore)));

    // 打印
    ShowNCHWTilingData();
    return ge::GRAPH_SUCCESS;
}

void Im2ColTiling::NHWCSetTilingData(Im2ColNHWCTilingData* tilingData, const int64_t (&ubfactor)[4])
{
    tilingData->input = input_;
    tilingData->convKernelNumInHeight = convKernelNumInHeight_;
    tilingData->convKernelNumInWidth = convKernelNumInWidth_;
    tilingData->ubFactorN = ubfactor[0];
    tilingData->ubFactorH = ubfactor[1];
    tilingData->ubFactorW = ubfactor[2];
    tilingData->ubFactorC = ubfactor[3];
    // 4. 计算每个核处理的行数（基于N×HW总维度）
    tilingData->linesPerCore = Ops::Base::CeilDiv(tilingData->totalLines, static_cast<int64_t>(coreNum_));
    realCoreNum_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(tilingData->totalLines, static_cast<int64_t>(tilingData->linesPerCore)));

    // 5. 计算输出缓冲区大小（适配新维度：N×HW×K×C）
    tilingData->outputBufferSize = static_cast<int64_t>(
        tilingData->ubFactorN * tilingData->ubFactorH * tilingData->ubFactorW * tilingData->ubFactorC * dSize_ *
        NHWC_BUFFER_NUM);
}

ge::graphStatus Im2ColTiling::Tiling4NHWC()
{
    auto tilingData = context_->GetTilingData<Im2ColNHWCTilingData>();
    uint64_t UB_SIZE_LIMIT = std::min(ubSize_ / NHWC_BUFFER_NUM, NHWC_MIN_BUFFER_SIZE); // 64KB
    auto remainingElem = static_cast<int64_t>(UB_SIZE_LIMIT / dSize_);                  // 剩余UB元素数，初始为最大值

    int64_t ubfactorAlign[4] = {1, convKernelNumInWidth_, input_.wKernelSize, ubBlockElements_}; // 0:N 1:W 2:Kw 3:C 32b
    int64_t ubfactor[4] = {1, 1, 1, 1}; // 对应索引：0=N 1=HW 2=Kw 3=C，初始全为1
    int64_t dimValuesAlign[4] = {input_.N, convKernelNum_, convKernelSize_, AlignBlock(input_.C)}; // 各维度判断条件
    size_t dim = std::size(dimValuesAlign);
    int64_t ubAxises[4] = {
        TPL_UB_AXIS_NHWC_N, TPL_UB_AXIS_NHWC_H, TPL_UB_AXIS_NHWC_W, TPL_UB_AXIS_NHWC_C}; // 各维度对应的ubAxis_值
    tilingData->totalLines = 1;

    for (int i = dim - 1; i >= 0; i--) { //    3=C→2=Kw→1=HW→0=N
        auto currDimValue = dimValuesAlign[i];
        auto currAlign = ubfactorAlign[i];
        auto currUbAxis = ubAxises[i]; // 当前维度ubAxis_标识

        // 核心切分逻辑：剩余空间是否大于当前维度值
        if (remainingElem >= currDimValue) {
            // 情况1：剩余空间充足，按维度值对齐后赋值
            ubfactor[i] = currDimValue;
        } else {
            if (remainingElem >= currAlign) {
                ubfactor[i] = Ops::Base::FloorAlign(remainingElem, currAlign);
                tilingData->totalLines = Ops::Base::CeilDiv(currDimValue, ubfactor[i]);
            } else {
                ubfactor[i] = remainingElem;
                tilingData->totalLines =
                    Ops::Base::CeilDiv(currAlign, ubfactor[i]) * Ops::Base::CeilDiv(currDimValue, currAlign);
            }
            ubAxis_ = currUbAxis; // 替换为ubAxis_
            for (int j = 0; j < i; j++) {
                tilingData->totalLines *= dimValuesAlign[j];
            }
            break;
        }
        remainingElem /= ubfactor[i];
    }
    NHWCSetTilingData(tilingData, ubfactor);
    if (static_cast<int64_t>(tilingData->ubFactorW * input_.C) * dSize_ < NHWC_MIN_WC_SIZE) {
        return Tiling4SIMT();
    }
    ShowNHWCTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Im2ColTiling::Tiling4SIMT()
{
    // 未使用 tiling key 参数置为默认值
    isPadding_ = false;
    ubAxis_ = 0;
    // 设置 tiling key 参数
    isSIMT_ = true;
    uint64_t inputTotalElement = static_cast<uint64_t>(input_.N) * input_.C * input_.H * input_.W;
    uint64_t outputTotalElement = static_cast<uint64_t>(input_.N) * input_.C * input_.hKernelSize * input_.wKernelSize *
                                  convKernelNumInHeight_ * convKernelNumInWidth_;
    isBigShape_ = inputTotalElement >= MAX_UINT32_NUM || outputTotalElement >= MAX_UINT32_NUM;

    // 输入信息
    auto tilingData = context_->GetTilingData<Im2ColSIMTTilingData>();
    tilingData->input = input_;
    tilingData->convKernelNumInWidth = convKernelNumInWidth_;
    tilingData->convKernelNumInHeight = convKernelNumInHeight_;

    // 分核信息
    // SIMT只关注element，所以分核简化为一维，和纯搬运一致
    uint32_t sideLengthFactor = Ops::Base::GetVRegSize(context_) / SIMT_BUFFER_NUM / dSize_;
    uint64_t alignEleBlockCount = Ops::Base::CeilDiv(outputTotalElement, static_cast<uint64_t>(sideLengthFactor));
    uint64_t cores = std::min(static_cast<uint64_t>(coreNum_), alignEleBlockCount);
    if (cores == 0) {
        std::string reasonMsg = "The value of realCoreNum cannot be 0.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "realCoreNum", std::to_string(cores).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    tilingData->realCoreNum = static_cast<uint64_t>(cores);
    tilingData->blockFactor = Ops::Base::CeilDiv(alignEleBlockCount, cores) * sideLengthFactor;
    tilingData->mainCoreNum = (alignEleBlockCount % cores == 0) ? cores : (alignEleBlockCount % cores);
    tilingData->blockTailFactor = tilingData->blockFactor - sideLengthFactor;
    OP_LOGI(context_->GetNodeName(), "Get block split alignEleBlockCount: %ld", alignEleBlockCount);
    tilingData->threadNum = Ops::Base::GetSimtMaxThreadNum(context_) / SIMT_THREAD_FACTOR;

    // 打印 tiling data
    ShowSIMTTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Im2ColTiling::Tiling4Format()
{
    cacheLineElements_ = cacheLineSize_ / dSize_;
    ubBlockElements_ = ubBlockSize_ / dSize_;
    // gather vector register元素数量
    gatherVRegElements_ = static_cast<int64_t>(vRegSize_) / std::max(dSize_, VREG_ELEMENT_MIN_SIZE);

    int64_t shapeSize = input_.N * input_.C * input_.H * input_.W;
    if (shapeSize <= MAX_SHAPE_SIZE_FOR_SIMT) {
        return Tiling4SIMT();
    }

    if (inputFormat_ == ge::FORMAT_NCHW) {
        return Tiling4NCHW();
    }
    if (inputFormat_ == ge::FORMAT_NHWC) {
        return Tiling4NHWC();
    }
    std::string reasonMsg =
        "When the shapeSize greater than MAX_SHAPE_SIZE_FOR_SIMT, the value of inputFormat must be in [NCHW and "
        "NHWC].";
    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
        context_->GetNodeName(), "inputFormat", Ops::Base::ToString(inputFormat_).c_str(), reasonMsg.c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus Im2ColTiling::GetSocInfo()
{
    // 获取soc信息, 如ub大小, core数等
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    realCoreNum_ = coreNum_;
    OP_CHECK_IF((coreNum_ == 0U), OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF((ubSize_ == 0U), OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF((cacheLineSize_ == 0U), OP_LOGE(context_, "Failed to get cache line size."), return ge::GRAPH_FAILED);
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ == 0U), OP_LOGE(context_, "Failed to get ub block size."), return ge::GRAPH_FAILED);
    vRegSize_ = Ops::Base::GetVRegSize(context_);
    OP_CHECK_IF((vRegSize_ == 0U), OP_LOGE(context_, "Failed to get vector register size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Im2ColTiling::InferOut()
{
    convKernelNumInWidth_ = (paddedW_ - effectW_) / input_.wStride + 1;
    convKernelNumInHeight_ = (paddedH_ - effectH_) / input_.hStride + 1;
    if (convKernelNumInWidth_ <= 0 || convKernelNumInHeight_ <= 0) {
        std::string shapeMsg =
            std::to_string(convKernelNumInWidth_) + "," + std::to_string(convKernelNumInHeight_);
        std::string reasonMsg =
            "The value of the calculated shape of the array of sliding blocks must be greater than 0.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "convKernelNumInWidth_ and convKernelNumInHeight_", shapeMsg.c_str(),
            reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    convKernelNum_ = convKernelNumInWidth_ * convKernelNumInHeight_; // 输出W
    convKernelSize_ = input_.wKernelSize * input_.hKernelSize;       // 输出H
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Im2Col(gert::TilingContext* context)
{
    // DoTiling
    Im2ColTiling tiling{context};
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForIm2Col([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Im2col).Tiling(Tiling4Im2Col).TilingParse<Im2ColCompileInfo>(TilingPrepareForIm2Col);
} // namespace optiling
