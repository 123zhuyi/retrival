/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cummin_tiling.h
 * \brief cummin tiling
 */

#include "cummin_tiling.h"

using namespace ge;

namespace optiling {
static constexpr uint64_t INPUT_X = 0;
static constexpr uint64_t INPUT_ARGMIN = 1;
static constexpr uint64_t DIM = 0;
static constexpr uint64_t RESERVED_UB = 4 * 1024;
static constexpr int64_t BLOCK_SIZE = 32;
static constexpr int64_t UB_SLICE_FOUR = 4;
static constexpr int64_t UB_SLICE_THREE = 3;
static constexpr int64_t TWO_ROWS = 2;
static constexpr int64_t MIN_N = 128;
static constexpr int64_t MAX_COUNT = 12800;

void CumminTiling::PrintCumminSplitInfo(const CumminSplitInfo& info)
{
    OP_LOGD(context_, "CumminSplitInfo 成员：");
    OP_LOGD(context_, "isNFullyLoad:      %ld", info.isNFullyLoad);
    OP_LOGD(context_, "splitR:            %ld", info.splitR);
    OP_LOGD(context_, "reservedR:         %ld", info.reservedR);
    OP_LOGD(context_, "computeR:          %ld", info.computeR);
    OP_LOGD(context_, "reservedN:         %ld", info.reservedN);
    OP_LOGD(context_, "splitN:            %ld", info.splitN);
    OP_LOGD(context_, "computeN:          %ld", info.computeN);
    OP_LOGD(context_, "computeLength:     %ld", info.computeLength);
    OP_LOGD(context_, "allocLength:       %ld", info.allocLength);
}

void CumminTiling::PrintCumminAll()
{
    OP_LOGD(context_, "==================== Cummin 所有变量 ====================");

    OP_LOGD(context_, "基础变量：");
    OP_LOGD(context_, "blockDim_:  %ld", blockDim_);
    OP_LOGD(context_, "M (合轴后): %ld", M);
    OP_LOGD(context_, "R (合轴后): %ld", R);
    OP_LOGD(context_, "N (合轴后): %ld", N);
    OP_LOGD(context_, "alignedN_:  %ld", alignedN_);
    OP_LOGD(context_, "dSize_:     %ld", dSize_);
    OP_LOGD(context_, "ubSize_:    %ld", ubSize_);
    OP_LOGD(context_, "tilingKey_: %ld", tilingKey_);

    OP_LOGD(context_, "CumminCompileInfo：");
    if (compileInfo_ == nullptr) {
        OP_LOGD(context_, "[compileInfo_ 为空指针]");
    } else {
        OP_LOGD(context_, "coreNum:    %ld", compileInfo_->coreNum);
        OP_LOGD(context_, "ubSize:     %ld", compileInfo_->ubSize);
        OP_LOGD(context_, "blockSize:  %ld", compileInfo_->blockSize);
        OP_LOGD(context_, "clSize:     %ld", compileInfo_->clSize);
        OP_LOGD(context_, "vRegSize:   %ld", compileInfo_->vRegSize);
    }

    OP_LOGD(context_, "CumminRegbaseTilingData：");
    OP_LOGD(context_, "M:               %ld", tilingData_.M);
    OP_LOGD(context_, "R:               %ld", tilingData_.R);
    OP_LOGD(context_, "N:               %ld", tilingData_.N);
    OP_LOGD(context_, "mRowsPerCore:    %ld", tilingData_.mRowsPerCore);
    OP_LOGD(context_, "formerCore:      %ld", tilingData_.formerCore);
    OP_LOGD(context_, "tailCore:        %ld", tilingData_.tailCore);
    OP_LOGD(context_, "coreNum:         %ld", tilingData_.coreNum);
    OP_LOGD(context_, "reservedRows:    %ld", tilingData_.reservedRows);
    OP_LOGD(context_, "splitM:          %ld", tilingData_.splitM);
    OP_LOGD(context_, "computeM:        %ld", tilingData_.computeM);
    OP_LOGD(context_, "ReservedM:       %ld", tilingData_.ReservedM);

    // 嵌套打印SplitInfo（lambda捕获this，使用类内PrintCumminSplitInfo）
    auto printSplitInfo = [this](const std::string& name, const CumminSplitInfo& info) {
        OP_LOGD(context_, "%s：", name.c_str());
        this->PrintCumminSplitInfo(info);
    };

    printSplitInfo("generalProcessInfo", tilingData_.generalProcessInfo);
    printSplitInfo("formerCoreProcessInfo", tilingData_.formerCoreProcessInfo);
    printSplitInfo("tailCoreProcessInfo", tilingData_.tailCoreProcessInfo);

    OP_LOGD(context_, "==========================================================");
}

bool CumminTiling::IsCapable()
{
    return true;
}

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus CumminTiling::GetPlatformInfo()
{
    OP_LOGD(context_, "CumminTiling GetPlatformInfo.");
    compileInfo_ = static_cast<const CumminCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);

    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus CumminTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_, "CumminTiling GetShapeAttrsInfo.");
    // 获取输入shape和dtype
    const gert::StorageShape* shape = context_->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
    auto xShape = shape->GetStorageShape();
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF((attrs == nullptr), OP_LOGE(context_, "Get attrs Failed."), return ge::GRAPH_FAILED);

    int64_t dim = *(attrs->GetAttrPointer<int64_t>(DIM));
    int64_t dimNum = xShape.GetDimNum();

    OP_CHECK_IF(
        (dim < -dimNum || dim >= dimNum),
        OP_LOGE(context_, " dim must be in [%ld, %ld], dim: [%ld].", -dimNum, dimNum - 1, dim),
        return ge::GRAPH_FAILED);
    if (dimNum == 0) {
        dimNum = 1;
    }
    dim = (dim + dimNum) % dimNum;

    for (int64_t i = 0; i < dim; i++) {
        M *= xShape.GetDim(i);
    }
    tilingData_.M = M;
    R = xShape.GetDim(dim);
    tilingData_.R = R;
    for (int64_t i = dim + 1; i < dimNum; i++) {
        N *= xShape.GetDim(i);
    }
    tilingData_.N = N;

    ge::DataType xDataType = context_->GetInputDesc(INPUT_X)->GetDataType();
    ge::DataType argminDataType = context_->GetOutputDesc(INPUT_ARGMIN)->GetDataType();
    dSize_ = ge::GetSizeByDataType(xDataType);
    argminDSize_ = ge::GetSizeByDataType(argminDataType);
    OP_CHECK_IF(dSize_ <= 0, OP_LOGE(context_, "X data size less or equal than 0"), return ge::GRAPH_FAILED);
    perBlockNum_ = Ops::Base::CeilDiv(BLOCK_SIZE, dSize_);
    alignedN_ = Ops::Base::CeilAlign(N, perBlockNum_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CumminTiling::DoUBSliceForM(int64_t m)
{
    tilingData_.computeM = Ops::Base::FloorDiv(ubSize_ / dSize_ - alignedN_, alignedN_ * R);
    tilingData_.splitM = Ops::Base::CeilDiv(m, tilingData_.computeM);
    tilingData_.ReservedM = m % tilingData_.computeM;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CumminTiling::DoUBSliceForR(CumminSplitInfo& info, int64_t length)
{
    auto alignedLength = Ops::Base::CeilAlign(length * dSize_, BLOCK_SIZE);
    info.computeR = Ops::Base::FloorDiv(ubSize_ - alignedLength, alignedLength);
    info.splitR = Ops::Base::CeilDiv(R, info.computeR);
    info.reservedR = R % info.computeR;
    info.computeLength = length;
    info.allocLength = Ops::Base::CeilAlign(length, perBlockNum_);
    PrintCumminSplitInfo(info);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CumminTiling::DoUBSliceForN(CumminSplitInfo& info, int64_t length)
{
    info.computeN =
        Ops::Base::FloorAlign(ubSize_ / TWO_ROWS, static_cast<int64_t>(compileInfo_->vRegSize)) / dSize_;
    info.splitN = Ops::Base::CeilDiv(length, info.computeN);
    info.reservedN = length % info.computeN;
    info.computeLength = info.computeN;
    info.allocLength = info.computeN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CumminTiling::DoUBSlice(CumminSplitInfo& info, int64_t length)
{
    auto alignedLength = Ops::Base::CeilAlign(length * dSize_, BLOCK_SIZE);
    int64_t ubLines = Ops::Base::FloorDiv(ubSize_ - alignedLength, alignedLength);
    // 一次处理多行N
    if (ubLines >= TWO_ROWS) {
        info.isNFullyLoad = 1;
        DoUBSliceForR(info, length);
    } else {
        info.isNFullyLoad = 0;
        DoUBSliceForN(info, length);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CumminTiling::DoComputeWithSimt()
{
    if (N < MIN_N && M > MAX_COUNT && R < MAX_COUNT) {
        tilingKey_ = COMPUTE_SIMT;
        blockDim_ = M > compileInfo_->coreNum ? compileInfo_->coreNum : M;
    }
    return ge::GRAPH_SUCCESS;
}

// 3、计算数据切分TilingData
ge::graphStatus CumminTiling::DoOpTiling()
{
    OP_LOGD(context_, "CumminTiling DoOpTiling.");
    // ub内存需要分配给y及argmin
    ubSize_ = (compileInfo_->ubSize - RESERVED_UB) / (1 + argminDSize_ / dSize_) / BLOCK_SIZE * BLOCK_SIZE;

    tilingData_.mRowsPerCore = M / compileInfo_->coreNum;

    // RN
    if (Ops::Base::FloorDiv(ubSize_ / dSize_ - alignedN_, alignedN_ * R) >= 1) {
        tilingKey_ = 0;
        DoUBSliceForM(tilingData_.mRowsPerCore);
    } else {
        DoUBSlice(tilingData_.generalProcessInfo, N);
        tilingKey_ = 2 - tilingData_.generalProcessInfo.isNFullyLoad;
    }

    tilingData_.reservedRows = M % compileInfo_->coreNum;

    // 对于R，N轴进行切分
    if (tilingData_.reservedRows > 0) {
        tilingData_.coreNum = alignedN_ * dSize_ <= compileInfo_->vRegSize ?
                                  tilingData_.reservedRows :
                                  compileInfo_->coreNum / tilingData_.reservedRows * tilingData_.reservedRows;
        // 根据 cacheline 切分N，不足2个cacheline 不值得切分。
        tilingData_.perGroupCoreNum = tilingData_.coreNum / tilingData_.reservedRows;
        tilingData_.formerCore = N % tilingData_.perGroupCoreNum;
        tilingData_.tailCore = tilingData_.coreNum - tilingData_.formerCore;
        tilingData_.formerCoreComputeLength = N / tilingData_.perGroupCoreNum + 1;
        tilingData_.tailCoreComputeLength = N / tilingData_.perGroupCoreNum;
        DoUBSlice(tilingData_.formerCoreProcessInfo, tilingData_.formerCoreComputeLength);
        DoUBSlice(tilingData_.tailCoreProcessInfo, tilingData_.tailCoreComputeLength);
    }
    blockDim_ = tilingData_.mRowsPerCore > 0 ? compileInfo_->coreNum : tilingData_.coreNum;
    DoComputeWithSimt();
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus CumminTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t CumminTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus CumminTiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus CumminTiling::PostTiling()
{
    OP_LOGD(context_, "CumminTiling PostTiling.");

    // 设置workspace大小
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context_, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);

    PrintCumminAll();
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), &tilingData_,
        sizeof(CumminRegbaseTilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(CumminRegbaseTilingData));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Cummin(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4Cummin start.");

    CumminTiling cumminTiling(context);
    auto ret = cumminTiling.DoTiling();
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGD(context, "Tiling4Cummin  failed!"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "Tiling4Cummin end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4CumminAscendc(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4CumminAscendc.");

    auto compileInfo = context->GetCompiledInfo<CumminCompileInfo>();

    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "core num is negative."), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "fail to get ub size."), return ge::GRAPH_FAILED);

    compileInfo->clSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        (compileInfo->clSize <= 0), OP_LOGE(context->GetNodeName(), "fail to get cache line size."),
        return ge::GRAPH_FAILED);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        (compileInfo->blockSize <= 0), OP_LOGE(context->GetNodeName(), "fail to get block size."),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        (compileInfo->vRegSize <= 0), OP_LOGE(context->GetNodeName(), "fail to get vReg size."),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4CumminAscendc.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Cummin(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<CumminCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD("TilingPrepare4Cummin", "Ascend C TilingPrepare4Cummin success.");
    return TilingPrepare4CumminAscendc(context);
}

// register tiling interface of the Cummin op.
IMPL_OP_OPTILING(Cummin).Tiling(Tiling4Cummin).TilingParse<CumminCompileInfo>(TilingPrepare4Cummin);
} // namespace optiling
