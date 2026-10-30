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
 * \file transdata_tiling_arch35.cpp
 * \brief calc tiling data for transdata AscendC kernel
 */

#include "transdata_tiling_arch35.h"

#include <algorithm>

#include "log/log.h"
#include "op_host/tiling_base_util.h"
#include "util/math_util.h"
#include "util/platform_util.h"

using namespace Ops::Base;

namespace optiling {
namespace transdata_asc {

ge::graphStatus TransDataTilingAscendC::GetHardwareInfo()
{
    auto compileInfo = context_->GetCompileInfo<TransDataCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = static_cast<uint32_t>(compileInfo->coreNum);
    ubSize_ = compileInfo->ubSize;
    blockSize_ = compileInfo->blockSize;
    OP_CHECK_IF(
        (coreNum_ <= 0U || ubSize_ <= 0L),
        OP_LOGE(context_->GetNodeName(), "TransData GetHardwareInfo failed, core num: %u, ub size: %ld", coreNum_,
                ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void TransDataTilingAscendC::ReshapeInShape()
{
    auto dimCnt = inShape_.GetDimNum();
    if (dimCnt > 1) {
        return;
    }
    auto bakDim = inShape_.GetDim(0);
    inShape_.SetDim(0, 1);
    inShape_.AppendDim(1);
    inShape_.AppendDim(bakDim);
}

bool TransDataTilingAscendC::GetShapeInfo()
{
    auto xStorage = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    inShape_ = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorage);
    outShape_ = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());

    OP_CHECK_IF(
        inShape_.GetShapeSize() == 0 || outShape_.GetShapeSize() == 0,
        OP_LOGE(context_->GetNodeName(), "The input or output shape is empty!"),
        return false);

    ReshapeInShape();
    return true;
}

bool TransDataTilingAscendC::GetTransFormatAndDType()
{
    auto srcTd = context_->GetInputDesc(0);
    auto dstTd = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, srcTd);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstTd);
    auto srcFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(srcTd->GetStorageFormat()));
    dstFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(dstTd->GetStorageFormat()));
    srcDtype_ = srcTd->GetDataType();
    dtypeSize_ = ge::GetSizeByDataType(srcDtype_);

    std::vector<ge::Format> supportSrcFormat = {ge::FORMAT_ND, ge::FORMAT_NCL, ge::FORMAT_NCHW, ge::FORMAT_NHWC};
    std::vector<ge::Format> supportDstFormat = {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_16,
                                                ge::FORMAT_FRACTAL_NZ_C0_32};
    OP_CHECK_IF(
        (std::find(supportSrcFormat.begin(), supportSrcFormat.end(), srcFormat) == supportSrcFormat.end() ||
         std::find(supportDstFormat.begin(), supportDstFormat.end(), dstFormat_) == supportDstFormat.end()),
        OP_LOGE(context_->GetNodeName(), "The input or output format is invalid!"),
        return false);
    return true;
}

bool TransDataTilingAscendC::GetTransNz2NdFormatAndDType()
{
    auto srcTd = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, srcTd);
    auto srcFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(srcTd->GetStorageFormat()));
    srcDtype_ = srcTd->GetDataType();
    dtypeSize_ = ge::GetSizeByDataType(srcDtype_);
    if (dtypeSize_ == ge::kDataTypeSizeBitOffset + B4_BIT_SIZE) {
        dtypeSize_ = 1;
    }
    OP_CHECK_IF(
        (dtypeSize_ <= 0), OP_LOGE(context_->GetNodeName(), "The dtypeSize is invalid!"),
        return false);

    std::vector<ge::Format> supportSrcFormat = {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_2,
                                                ge::FORMAT_FRACTAL_NZ_C0_4, ge::FORMAT_FRACTAL_NZ_C0_16,
                                                ge::FORMAT_FRACTAL_NZ_C0_32};
    std::map<ge::Format, int64_t> formatMap = {
        {ge::FORMAT_FRACTAL_NZ, blockSize_ / dtypeSize_},
        {ge::FORMAT_FRACTAL_NZ_C0_2, C0_2},
        {ge::FORMAT_FRACTAL_NZ_C0_4, C0_4},
        {ge::FORMAT_FRACTAL_NZ_C0_16, C0_16},
        {ge::FORMAT_FRACTAL_NZ_C0_32, C0_32}
    };
    auto it = formatMap.find(srcFormat);
    OP_CHECK_IF(
        (it == formatMap.end()), OP_LOGE(context_->GetNodeName(), "The input format is invalid!"),
        return false);
    // NZ数据类型，fp4 dtypeSize_为1 (实际应该为0.5)，expectC0 = 32，b4当做b8处理，c0理论上需要除2，expectC0不做特殊处理
    // NZ_CX类型，，expectC0固定为CX，expectC0需要除2
    expectC0_ = it->second;
    if (srcDtype_ == ge::DT_FLOAT4_E2M1 && srcFormat != ge::FORMAT_FRACTAL_NZ) {
        expectC0_ = expectC0_ >> 1;
    }
    return true;
}

bool TransDataTilingAscendC::CalcNzToNdShapeSize()
{
    auto xStorage = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    inShape_ = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorage);
    outShape_ = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());

    OP_CHECK_IF(
        inShape_.GetDimNum() < nFour, OP_LOGE(context_->GetNodeName(), "The input shape dim is less than 4!"),
        return false);
    OP_CHECK_IF(
        inShape_.GetDimNum() != outShape_.GetDimNum() + nTwo,
        OP_LOGE(context_->GetNodeName(), "The input dim must be 2 higher than the output dim."), return false);

    auto inputDimCnt = inShape_.GetDimNum();
    c1_ = inShape_.GetDim(inputDimCnt - nFour);
    n1_ = inShape_.GetDim(inputDimCnt - nThree);
    n0_ = inShape_.GetDim(inputDimCnt - nTwo);
    c0_ = inShape_.GetDim(inputDimCnt - 1);
    OP_CHECK_IF(
        (c1_ * n1_ * n0_ * c0_ == 0),
        OP_LOGE(context_->GetNodeName(), "All input dim must not be zero"), return false);

    auto outputDimCnt = outShape_.GetDimNum();
    n_ = outShape_.GetDim(outputDimCnt - nTwo);
    c_ = outShape_.GetDim(outputDimCnt - 1);

    // 输入shape的H维必须和输出shape的H维完全相同
    for (size_t i = 0; i < outputDimCnt - nTwo; i++) {
        OP_CHECK_IF(
            (inShape_.GetDim(i) != outShape_.GetDim(i) || inShape_.GetDim(i) == 0),
            OP_LOGE(context_->GetNodeName(), "The input H dim must be same with output dim and not be zero."),
            return false);
    }

    OP_CHECK_IF(
        (n1_ != CeilDiv(n_, n0_)),
        OP_LOGE(context_->GetNodeName(), "The dim N1 should be equal CeilDiv(N, N0)"), return false);
    OP_CHECK_IF(
        (c1_ != CeilDiv(c_, c0_)),
        OP_LOGE(context_->GetNodeName(), "The dim C1 should be equal CeilDiv(C, C0)"), return false);

    if (srcDtype_ == ge::DT_FLOAT4_E2M1 || srcDtype_ == ge::DT_FLOAT4_E1M2) {
        // 当作u8处理，c缩小一半
        c0_ = c0_ >> 1;
        c_ = c_ >> 1;
    }

    OP_CHECK_IF(
        (c0_ != expectC0_ || n0_ != N0_16),
        OP_LOGE(context_->GetNodeName(), "The n0 should be 16 and c0 should be  %ld", expectC0_), return false);

    int64_t res = 1;
    for (size_t i = 0; i < inputDimCnt - nFour; i++) {
        res *= inShape_.GetDim(i);
    }
    h_ = res;
    return true;
}

bool TransDataTilingAscendC::CalcC0Size()
{
    auto dimCnt = outShape_.GetDimNum();
    auto tmpC0 = outShape_.GetDim(dimCnt - 1);

    OP_CHECK_IF(
        dstFormat_ == ge::FORMAT_FRACTAL_NZ_C0_16 && tmpC0 != C0_16,
        OP_LOGE(context_->GetNodeName(), "The c0 should be 16 when dst format is FRACTAL_NZ_C0_16!"),
        return false);
    OP_CHECK_IF(
        dstFormat_ == ge::FORMAT_FRACTAL_NZ_C0_32 && tmpC0 != C0_32,
        OP_LOGE(context_->GetNodeName(), "The c0 should be 32 when dst format is FRACTAL_NZ_C0_32!"),
        return false);

    int64_t expectC0 = tmpC0;
    if (dtypeSize_ == 1U) {
        expectC0 = C0_32;
    } else if (dtypeSize_ == nTwo) {
        expectC0 = C0_16;
    }
    OP_CHECK_IF(
        dstFormat_ == ge::FORMAT_FRACTAL_NZ && dtypeSize_ == nTwo * nTwo && C0_8 != expectC0 && C0_16 != expectC0,
        OP_LOGE(context_->GetNodeName(), "The c0 should be 8 or 16 when dst format is FRACTAL_NZ and dtype size is %zu!",
                dtypeSize_),
        return false);
    OP_CHECK_IF(
        dstFormat_ == ge::FORMAT_FRACTAL_NZ && tmpC0 != expectC0,
        OP_LOGE(context_->GetNodeName(), "The c0 should be %ld when dst format is FRACTAL_NZ and dtype size is %zu!",
                expectC0, dtypeSize_),
        return false);

    c0_ = tmpC0;
    if (srcDtype_ == ge::DT_FLOAT4_E2M1 || srcDtype_ == ge::DT_FLOAT4_E1M2) {
        // 当作u8处理，c0缩小一半
        c0_ = c0_ >> 1;
    }
    return true;
}

void TransDataTilingAscendC::CalcHSize()
{
    int64_t res = 1;
    auto dimCnt = inShape_.GetDimNum();
    if (dimCnt <= nTwo) {
        h_ = res;
        return;
    }

    for (size_t i = 0; i < dimCnt - nTwo; i++) {
        res *= inShape_.GetDim(i);
    }
    h_ = res;
}

void TransDataTilingAscendC::CalcNCSize()
{
    auto dimCnt = inShape_.GetDimNum();
    n_ = inShape_.GetDim(dimCnt - nTwo);
    c_ = inShape_.GetDim(dimCnt - 1);
    if (srcDtype_ == ge::DT_FLOAT4_E2M1 || srcDtype_ == ge::DT_FLOAT4_E1M2) {
        // 当作u8处理，c缩小一半
        c_ = c_ >> 1;
    }
}

void TransDataTilingAscendC::CalcTilingKey()
{
    int64_t ni = 16;
    int64_t shapeSize = h_ * CeilAlign(n_, ni) * CeilAlign(c_, c0_);
    tilingKey_ = (shapeSize > MAX_INT32_SIZE) ? TILING_MODE_SIMT_LARGE_SHAPE : TILING_MODE_SIMT;
}

void TransDataTilingAscendC::CalcBlockAndThreadNum()
{
    bNum_ = coreNum_;
    tNum_ = (tilingKey_ == TILING_MODE_SIMT) ? tNum512 : tNum256;
}

ge::graphStatus TransDataTilingAscendC::CalcTilingData()
{
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "Failed to get shape info!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !GetTransFormatAndDType(), OP_LOGE(context_->GetNodeName(), "Failed to get format and dtype info!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CalcC0Size(), OP_LOGE(context_->GetNodeName(), "Failed to get c0 size!"), return ge::GRAPH_FAILED);
    CalcHSize();
    CalcNCSize();
    CalcTilingKey();
    CalcBlockAndThreadNum();

    return ge::GRAPH_SUCCESS;
}

void TransDataTilingAscendC::WriteNzToNdTilingData()
{
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(TILING_MODE_SIMT_NZ_TO_ND);
    context_->SetLocalMemorySize(ubSize_ - SIMT_RSV_SIZE);

    tilingNzToNdData_.set_h(h_);
    tilingNzToNdData_.set_n(n_);
    tilingNzToNdData_.set_c(c_);
    tilingNzToNdData_.set_c1(c1_);
    tilingNzToNdData_.set_n1(n1_);
    tilingNzToNdData_.set_n0(n0_);
    tilingNzToNdData_.set_c0(c0_);
    tilingNzToNdData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingNzToNdData_.GetDataSize());    
}

void TransDataTilingAscendC::WriteTilingData()
{
    context_->SetBlockDim(bNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetLocalMemorySize(ubSize_ - SIMT_RSV_SIZE);

    tilingData_.set_c0(c0_);
    tilingData_.set_h(h_);
    tilingData_.set_n(n_);
    tilingData_.set_c(c_);
    tilingData_.set_tNum(tNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

std::string TransDataTilingAscendC::PrintTilingData()
{
    std::string tilingStr;
    tilingStr += std::to_string(c0_) + ",";
    tilingStr += std::to_string(h_) + ",";
    tilingStr += std::to_string(n_) + ",";
    tilingStr += std::to_string(c_) + ",";
    tilingStr += std::to_string(tNum_);
    return tilingStr;
}

std::string TransDataTilingAscendC::PrintNz2NdTilingData()
{
    std::string tilingStr;
    tilingStr += std::to_string(h_) + ",";
    tilingStr += std::to_string(n_) + ",";
    tilingStr += std::to_string(c_) + ",";
    tilingStr += std::to_string(c1_) + ",";
    tilingStr += std::to_string(n1_) + ",";
    tilingStr += std::to_string(n0_) + ",";
    tilingStr += std::to_string(c0_);
    return tilingStr;
}

ge::graphStatus TransDataTilingAscendC::DoNz2NdTiling()
{
    OP_CHECK_IF(
            !GetTransNz2NdFormatAndDType(), OP_LOGE(context_->GetNodeName(), "Failed to get format and dtype info!"),
            return ge::GRAPH_FAILED);

    // 设置h n1 n0 c1 c0
    OP_CHECK_IF(
        !CalcNzToNdShapeSize(), OP_LOGE(context_->GetNodeName(), "The input shape is invalid!"),
        return ge::GRAPH_FAILED);

    WriteNzToNdTilingData();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = kSyncWorkSpaceSize;
    OP_LOGI(context_->GetNodeName(), "TransData tiling data: %s", PrintNz2NdTilingData().c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransDataTilingAscendC::DoTiling()
{
    // NZ2ND的tiling分支
    auto dstTd = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstTd);
    dstFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(dstTd->GetStorageFormat()));
    if (dstFormat_ == ge::FORMAT_ND) {
        return DoNz2NdTiling();
    }

    OP_CHECK_IF(
        (CalcTilingData() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "TransDataTilingAscendC failed to calc tiling data."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= SIMT_RSV_SIZE, OP_LOGE(context_->GetNodeName(), "UB size too small for SIMT reserved size."),
                return ge::GRAPH_FAILED);
    WriteTilingData();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = kSyncWorkSpaceSize;
    OP_LOGI(context_->GetNodeName(), "TransData tiling data: %s", PrintTilingData().c_str());
    return ge::GRAPH_SUCCESS;
}

}  // namespace transdata_asc

ge::graphStatus Tiling4TransDataAscendC(gert::TilingContext* context)
{
    transdata_asc::TransDataTilingAscendC tdTiling(context);
    OP_CHECK_IF(
        (tdTiling.GetHardwareInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "TransDataTilingAscendC failed to get hardware info."),
        return ge::GRAPH_FAILED);
    return tdTiling.DoTiling();
}

static ge::graphStatus TilingPrepare4TransData(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<TransDataCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "The core num is invalid."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF((compileInfo->blockSize < 1),
                OP_LOGE(context->GetNodeName(), "The block size is invalid, %ld.",
                                                compileInfo->blockSize),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransData).Tiling(Tiling4TransDataAscendC).TilingParse<TransDataCompileInfo>(TilingPrepare4TransData);

}  // namespace optiling
