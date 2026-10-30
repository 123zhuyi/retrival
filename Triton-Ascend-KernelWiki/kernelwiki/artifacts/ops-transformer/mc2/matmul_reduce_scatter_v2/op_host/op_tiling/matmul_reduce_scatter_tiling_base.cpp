/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_tiling_base.cpp
 * \brief
 */
#include <queue>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "reduce_scatter_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "ops_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "matmul_reduce_scatter_tiling_base.h"
#include "arch35/reduce_scatter_fit_balance_tiling.h"
#include "../../op_kernel/matmul_reduce_scatter_v2_apt_tiling_key.h"
#include "util/math_util.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optiling {
// 完成公式化tiling切分相关功能
constexpr uint32_t COMMTURN_INDEX = 4;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t RECV_ARG_INDEX = 3;
constexpr uint32_t BLOCKSIZE_INDEX = 6;
constexpr uint32_t IS_TRANS_A = 2;
constexpr uint32_t IS_TRANS_B = 3;
constexpr uint32_t COMM_TURN = 4;
const std::set<int> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64};

uint32_t MatmulReduceScatterTilingBase::ReduceScatterSpliteM(mc2tiling::TilingArgs& args, uint32_t maxTileCnt) const
{
    // 检查允许通信的最大次数
    if (args.commTurn >= maxTileCnt) {
        args.commTurn = maxTileCnt;
    }

    uint64_t tileLen = 1;
    if (args.mValue > args.commTurn) {
        tileLen = args.mValue / args.commTurn;
    }
    if (args.outputDtypeSize == 2) { // 数据长度为2, 则向 2*64 = 128，则向128对齐
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 64);
    } else if (args.outputDtypeSize == 4) {                  // 4 is float32 tpye size
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 32); // 32 is used to align to 128
    }
    return args.mValue > tileLen ? tileLen : args.mValue;
}

CutResult MatmulReduceScatterTilingBase::GetTilingResult()
{
    if (mc2tiling::IsStandardCard4P(args_.rankDim, npuArch_)) {
        MMReduceScatterFitBalanceTiling scatterTiling(args_, KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL);
        return scatterTiling.GetTiling();
    } else {
        SocVersion inputSocVersion = (npuArch_ == NpuArch::DAV_3510) ? SocVersion::SOC950 : SocVersion::SOC910_B;
        MMPlusReduceScatter scatterTiling(args_, args_.rankDim, KernelType::REDUCE_SCATTER, inputSocVersion);
        scatterTiling.GetTiling();
        return scatterTiling.tilingM_.cutRes;
    }
}

void MatmulReduceScatterTilingBase::DoFormulaticTiling(Mc2Tiling::RCSTiling &rcsCfg)
{
    CutResult mCutScatter = GetTilingResult();
    rcsCfg.tailCnt = 0;
    if (mCutScatter.shortTileAtBack || (mCutScatter.numShortTile == 0)) {
        rcsCfg.tileCnt = mCutScatter.numLongTile;
        rcsCfg.tailM = mCutScatter.shortTileLen;
        if (mCutScatter.numShortTile > 0) {
            rcsCfg.tailCnt = mCutScatter.numShortTile;
        }
        tileMValue_ = mCutScatter.longTileLen;
        tailMValue_ = mCutScatter.shortTileLen;
    } else {
        rcsCfg.tileCnt = mCutScatter.numShortTile;
        rcsCfg.tailM = mCutScatter.longTileLen;
        if (mCutScatter.numLongTile > 0) {
            rcsCfg.tailCnt = mCutScatter.numLongTile;
        }
        tileMValue_ = mCutScatter.shortTileLen;
        tailMValue_ = mCutScatter.longTileLen;
    }
    longTileLen_ = mCutScatter.longTileLen;
}

/**
 * Due to communication constraints: 
 * 1. The maximum number of communication attempts is limited to 16
 * 2. The data volume of a single communication shall not exceed 256MB;
 * Thus, it is required to pre-intercept the x1 that still exceeds the limit after being evenly split into 16 parts
 */
ge::graphStatus MatmulReduceScatterTilingBase::CheckHCCLSize()
{
    uint64_t sizeOfSingleM = args_.nValue * ge::GetSizeByDataType(args_.geCType);
    OP_TILING_CHECK(sizeOfSingleM > mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT,
        OP_LOGE(opName_, "Unsupported matmul output size. Even after splitting data matmul output into (1, n), the size %lu still exceeds 256MB.", sizeOfSingleM),
                return ge::GRAPH_FAILED);
    
    uint64_t sizeOfSplitM = Ops::Base::CeilDiv(args_.mValue, mc2tiling::ALL_GATHER_HCCL_NUM_LIMIT) * sizeOfSingleM;
    OP_TILING_CHECK(sizeOfSplitM > mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT,
        OP_LOGE(opName_, "Unsupported x1 size. Even after splitting data M into 16 parts (rounded up), the size %lu still exceeds 256MB.", sizeOfSplitM),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterTilingBase::AdjustHCCLLimit(Mc2Tiling::RCSTiling &rcfCfg, mc2tiling::Mc2QuantMode quantMmMode)
{
    if (tileMValue_ * args_.mValue * ge::GetSizeByDataType(args_.geCType) <= mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT) {
        return ge::GRAPH_SUCCESS;
    }
    OPS_LOG_I(opName_, "The result of formulaic tiling result does not meet the hccl restriction,"
     " current splitting: tileM [%ld], tileCnt [%ld], tailM [%ld], tailCnt [%ld].",
        tileMValue_, rcfCfg.tileCnt, tailMValue_, rcfCfg.tailCnt);
    
    OP_TILING_CHECK((quantMmMode == mc2tiling::Mc2QuantMode::PERBLOCK_MODE),
        OP_LOGE(opName_, "Unsupported x1 size. Even after formulaic splitting, the size still exceeds 256MB."), 
        return ge::GRAPH_FAILED);
    
    uint64_t minSplitPart = Ops::Base::CeilDiv(args_.mValue * args_.nValue * ge::GetSizeByDataType(args_.geCType), mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT);
    tileMValue_ = Ops::Base::CeilDiv(args_.mValue, minSplitPart);
    rcfCfg.tileCnt = Ops::Base::FloorDiv(args_.mValue, tileMValue_);
    rcfCfg.tailM = args_.mValue - rcfCfg.tileCnt * tileMValue_;
    tailMValue_ = rcfCfg.tailM;
    if (tailMValue_ == 0) {
        rcfCfg.tailCnt = 0;
    } else {
        rcfCfg.tailCnt = 1;
    }
    longTileLen_ = tileMValue_;
    OPS_LOG_I(opName_, "Because the result of formulaic tiling result does not meet the hccl restriction,"
     " the re-splitM result: tileM [%ld], tileCnt [%ld], tailM [%ld], tailCnt [%ld]. end re-splitM.",
        tileMValue_, rcfCfg.tileCnt, tailMValue_, rcfCfg.tailCnt);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterTilingBase::DoSplitMTiling(Mc2Tiling::RCSTiling &rcsCfg)
{
    args_.mValue = args_.orgMValue / args_.rankDim; // 必须能够整数切分, 并且不能切K
    args_.rankTileNum = args_.rankDim;
    // cmdType = HCCL_CMD_ALLGATHER, 是允许切K
    if (args_.enableSplitK) { // 只有1份
        rcsCfg.tileCnt = 1;
        rcsCfg.tailCnt = 0;
        rcsCfg.tailM = 0;
    } else if (args_.commTurn != 0) {
        uint64_t splite = ReduceScatterSpliteM(args_);
        // 现在找到1个合适的切分
        auto tileCnt = args_.mValue / splite;  // 切的份数
        auto tileTail = args_.mValue % splite; // 尾巴
        tileMValue_ = splite;
        tailMValue_ = tileTail;
        rcsCfg.tileCnt = tileCnt;
        rcsCfg.tailCnt = 0;
        rcsCfg.tailM = tileTail;
        if (tileTail != 0) {
            rcsCfg.tailCnt = 1;
        }
        longTileLen_ = splite;
    } else {
        DoFormulaticTiling(rcsCfg);
    }
    return ge::GRAPH_SUCCESS;
}

void MatmulReduceScatterTilingBase::SetStorageAWorkSpaceSize(Mc2Tiling::RCSTiling &rcsCfg)
{
    if (args_.cmdType == mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER) {
        // A*B 的数据，需要找地方存储，因为 C 的长度 = A*B的长度/ RankDim
        mmResultLen_ = static_cast<uint64_t>(rcsCfg.rankM) * static_cast<uint64_t>(rcsCfg.rankN) *
                                             static_cast<uint64_t>(args_.outputDtypeSize);
        mmResultLen_ = mc2tiling::AlignUp<uint64_t>(mmResultLen_, 512); // 512 is used to get gm
        OP_LOGD("MatmulReduceScatter", " reduce scatter gmcFloat size %lu", mmResultLen_);
        rcsCfg.nd2NzWorkLen = 0;
        rcsCfg.cToFloatLen = mmResultLen_;
        OP_LOGD("MatmulReduceScatter", " reduce scatter storage_a size %lu", mmResultLen_);
    }

    rcsCfg.biasLen = 0;
    return;
}

void MatmulReduceScatterTilingBase::SetRcsTilingData(Mc2Tiling::RCSTiling &rcsCfg)
{
    rcsCfg.rankDim = args_.rankDim;
    rcsCfg.isTransposeA = args_.isATrans;
    rcsCfg.isTransposeB = args_.isBTrans;
    OP_LOGD(opName_,
        "MatmulReduceScatter SetRcsTilingData, args_.orgMValue: %lu, args_.orgNValue: %lu, args_.orgKValue: %lu.",
        args_.orgMValue, args_.orgNValue, args_.orgKValue);

    auto reduceOpType = context_->GetAttrs()->GetAttrPointer<char>(1);
    if (strncmp(reduceOpType, "sum", 3) == 0) { // 3 is index
        OP_LOGD("MatmulReduceScatter", "strncmp(reduceOpType, sum, 3) == 0");
        rcsCfg.subtype = (static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM));
    } else {
        OP_LOGD("MatmulReduceScatter", "strncmp(reduceOpType, sum, 3) != 0");
        rcsCfg.subtype = (static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_RESERVED));
    }

    rcsCfg.commtype = (static_cast<uint8_t>(mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER));
    rcsCfg.rankM = args_.orgMValue; // 存放用户原始输入的mValue
    rcsCfg.rankN = args_.orgNValue; // 存放用户原始输入的nValue
    rcsCfg.rankK = args_.orgKValue; // 存放用户原始输入的kValue
    rcsCfg.aicCoreNum = args_.aicCoreNum;

    SetStorageAWorkSpaceSize(rcsCfg);
}

ge::graphStatus MatmulReduceScatterTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(
        workspaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "get workspace failed"),
        return ge::GRAPH_FAILED);

    // 如果是 A2A 路径，需要额外的 recvBuf (mmResultLen_)，否则只需要 senBuf
    workspaceSize_ = libApiWorkSpaceSize_ + mmResultLen_ + (isA2APath_ ? mmResultLen_ : 0);

    workspaces[0] = workspaceSize_;
    OP_LOGD(opName_, "workspaces[0] size %ld.", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulReduceScatterTilingBase::GetTilingKey() const
{
    uint8_t inputIsBf16Fp16 = INPUT_TYPE_IS_FP8;
    if ((args_.geAType == ge::DT_BF16) || (args_.geAType == ge::DT_FLOAT16)) {
        inputIsBf16Fp16 = INPUT_TYPE_IS_FP16_BF16;
    }
    uint8_t commAlg = isA2APath_ ? TPL_CCU_ALL2ALL_VEC_REDUCE : TPL_CCU_REDUCESUM;
    uint64_t tilingKey = GET_TPL_TILING_KEY(    \
        false, args_.isATrans, args_.isBTrans, inputIsBf16Fp16, OUTPUT_TYPE_IS_FP8, TPL_X1_X2_DTYPE_IS_OTHER, commAlg);
    OP_LOGD(opName_, "args_.isATrans, args_.isBTrans, inputIsBf16Fp16 is: [%d, %d, %d]", \
            args_.isATrans, args_.isBTrans, inputIsBf16Fp16);
    return tilingKey;
}

ge::graphStatus MatmulReduceScatterTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(
        platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    isA2APath_ = mc2tiling::IsStandardCard4P(args_.rankDim, npuArch_); // 判断是否走标卡4p路径： (All2All + Vec Reduce)
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
};

bool MatmulReduceScatterTilingBase::CheckBias() const
{
    const gert::StorageShape* aShape = context_->GetInputShape(INPUT_X1);
    const gert::StorageShape* bShape = context_->GetInputShape(INPUT_X2);
    const gert::StorageShape* biasShape = context_->GetOptionalInputShape(BIAS);
    if (biasShape != nullptr) {
        uint64_t biasShapeDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            (biasShapeDimNum != 1), VECTOR_INNER_ERR_REPORT_TILING(opName_, "the bias dimNum should be one"),
            return false);
        ge::DataType aType = context_->GetInputDesc(INPUT_X1)->GetDataType();
        ge::DataType bType = context_->GetInputDesc(INPUT_X2)->GetDataType();
        if (((aType == ge::DT_BF16) && (bType == ge::DT_BF16)) ||
            ((aType == ge::DT_FLOAT16) && (bType == ge::DT_FLOAT16))) {
                uint64_t x2dim1 = bShape->GetStorageShape().GetDim(1);
                uint64_t x2dim0 = bShape->GetStorageShape().GetDim(0);
                uint64_t x1dim1 = aShape->GetStorageShape().GetDim(1);
                auto nAxis = (x1dim1 == x2dim0) ? x2dim1 : x2dim0;
                uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0);
                OP_TILING_CHECK(
                    (biasDim0 != nAxis),
                    VECTOR_INNER_ERR_REPORT_TILING(
                        opName_, "the bias dimNum0 should be %lu, but actual value is %lu.", nAxis, biasDim0),
                    return false);
        }
    }
    return true;
}

bool MatmulReduceScatterTilingBase::CheckGroupSize() const
{
    ge::DataType aType = context_->GetInputDesc(INPUT_X1)->GetDataType();
    ge::DataType bType = context_->GetInputDesc(INPUT_X2)->GetDataType();
    auto attrsPtr = context_->GetAttrs();
    auto groupSizePtr = attrsPtr->GetAttrPointer<uint64_t>(GROUPSIZE_IDX);
    if (((aType == ge::DT_BF16) && (bType == ge::DT_BF16)) || ((aType == ge::DT_FLOAT16) && (bType == ge::DT_FLOAT16))) {
        if (groupSizePtr != nullptr) {
             OP_TILING_CHECK((*groupSizePtr != 0), 
                CUBE_INNER_ERR_REPORT(opName_, "when the datatype of x1 and x2 are fp16 or bf16,"
                    "groupSizePtr should be nullptr or 0, but got %lu.", *groupSizePtr),
                return false);
        }
    }
    return CheckBias();
}

bool MatmulReduceScatterTilingBase::CheckInputScale() const
{
    auto quantscaleShape = context_->GetOptionalInputShape(QUANT_SCALE);
    OP_TILING_CHECK((quantscaleShape != nullptr),
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "the quantscale tensor should be nullptr"), return false);
    
    return CheckGroupSize();
}

bool MatmulReduceScatterTilingBase::CheckAttrInfoValid(uint64_t kValue)
{
    if (context_->GetAttrs() == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "get attrs failed");
    } else {
        auto reduceOpType = context_->GetAttrs()->GetAttrPointer<char>(1);
        OP_TILING_CHECK(
            strcmp(reduceOpType, "sum") != 0,
            VECTOR_INNER_ERR_REPORT_TILING(
                opName_, "the reduceOpType should be sum, but real value is %s", reduceOpType),
            return false);

        auto isTransA = context_->GetAttrs()->GetAttrPointer<bool>(2);
        OP_TILING_CHECK(
            *isTransA != false,
            VECTOR_INNER_ERR_REPORT_TILING(opName_, "the isTransA should be false, but real value is 1"),
            return false);
        OP_TILING_CHECK(
            (kValue < KVALUE_MIN) || (kValue >= KVALUE_MAX),
            VECTOR_INNER_ERR_REPORT_TILING(
                opName_, "The k-axis should be in range[256, 65535), but it is: %lu.", kValue),
            return false);
    }
    auto group = context_->GetAttrs()->GetAttrPointer<char>(0);
    OP_TILING_CHECK(
        !mc2tiling::GetRankSize(opName_, group, rankSize_),
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "GetRankSize failed."), return false);
    OP_TILING_CHECK(
        SUPPORT_RANK_SIZE.find(rankSize_) == SUPPORT_RANK_SIZE.end(),
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "world_size should be 2 or 4 or 8 or 16 or 32 or 64, but the actual value is %ld.", rankSize_),
        return false);
    auto commTurn = *context_->GetAttrs()->GetAttrPointer<int64_t>(COMMTURN_INDEX);
    OP_TILING_CHECK(
        commTurn != 0,
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "commTurn should be 0, but the actual value is %ld.", commTurn),
        return false);
    auto blockSize = *context_->GetAttrs()->GetAttrPointer<int64_t>(BLOCKSIZE_INDEX);
    OP_TILING_CHECK(
        blockSize != 0,
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "blockSize should be 0, but the actual value is %ld.", blockSize),
        return false);
    return CheckInputScale();
}

bool MatmulReduceScatterTilingBase::ReduceScatterCheckShapeInfo()
{
    const gert::StorageShape* aShape = context_->GetInputShape(0);
    const gert::StorageShape* bShape = context_->GetInputShape(1);
    OP_TILING_CHECK(
        (aShape == nullptr) || (bShape == nullptr),
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "the shape of a or shape of b is invalid"), return false);

    uint64_t aShapeDimNum = aShape->GetStorageShape().GetDimNum();
    uint64_t bShapeDimNum = bShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(
        (aShapeDimNum != 2) || (bShapeDimNum != 2), VECTOR_INNER_ERR_REPORT_TILING(opName_, "the dimNum is not two"),
        return false);
    auto aTensor = context_->GetInputDesc(0);
    auto bTensor = context_->GetInputDesc(1);
    auto output = context_->GetOutputDesc(0);
    OP_TILING_CHECK(
        (aTensor == nullptr) || (bTensor == nullptr) || (output == nullptr),
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "the tensor is invalid"), return false);

    auto aShapeFormat = aTensor->GetStorageFormat();
    auto bShapeFormat = bTensor->GetStorageFormat();
    auto outputFormat = output->GetStorageFormat();
    OP_TILING_CHECK(
        aShapeFormat != outputFormat,
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "a shape Format, output Format are not same, should be ND/ND"),
        return false);
    OP_TILING_CHECK(
        (!mc2tiling::CheckSuppportedFormat(aShapeFormat)) || (!mc2tiling::CheckSuppportedFormat(bShapeFormat)),
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "a shape Format, b shape Format only support ND, format of a is %s, format of b is %s",
            TypeUtils::FormatToSerialString(aShapeFormat).c_str(),
            TypeUtils::FormatToSerialString(bShapeFormat).c_str()),
        return false);
    uint64_t x1Dim0 = aShape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = aShape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = bShape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = bShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        x1Dim0 == 0, VECTOR_INNER_ERR_REPORT_TILING(opName_, "got 0 while parse Dim0 of x1's shape"), return false);
    OP_TILING_CHECK(
        x1Dim1 == 0, VECTOR_INNER_ERR_REPORT_TILING(opName_, "got 0 while parse Dim1 of x1's shape"), return false);
    OP_TILING_CHECK(
        x2Dim0 == 0, VECTOR_INNER_ERR_REPORT_TILING(opName_, "got 0 while parse Dim0 of x2's shape"), return false);
    OP_TILING_CHECK(
        x2Dim1 == 0, VECTOR_INNER_ERR_REPORT_TILING(opName_, "got 0 while parse Dim1 of x2's shape"), return false);

    return CheckAttrInfoValid(x1Dim1);
}

void MatmulReduceScatterTilingBase::Reset()
{
    tileMValue_ = 0UL;  // mc2 切块后主块M的大小；
    tailMValue_ = 0UL;  // mc2 切块后尾块M的大小；
    longTileLen_ = 0UL; // mc2 切块后长块的大小；
    mmResultLen_ = 0UL; // mc2 matmul计算后结果的大小
}

void MatmulReduceScatterTilingBase::SetReduceScatterTilingArgsDataType()
{
    ge::DataType biasType;
    bool isBias = true;
    auto aType = context_->GetInputDesc(0)->GetDataType();
    auto bType = context_->GetInputDesc(1)->GetDataType();
    auto cType = context_->GetOutputDesc(0)->GetDataType();
    const gert::StorageShape* matrix_bias = context_->GetOptionalInputShape(2);
    if (matrix_bias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = context_->GetInputDesc(BIAS_INDEX)->GetDataType();
    }

    uint64_t inputDtypeSize = mc2tiling::GetDataTypeSize(opName_, aType);
    uint64_t outputDtypeSize = mc2tiling::GetDataTypeSize(opName_, cType);

    args_.isBias = isBias;
    args_.geAType = aType;
    args_.geBType = bType;
    args_.geCType = cType;
    args_.geBiasType = biasType;
    args_.aType = mc2tiling::ConvertGeTypeToMmType(opName_, aType);
    args_.bType = mc2tiling::ConvertGeTypeToMmType(opName_, bType);
    args_.cType = mc2tiling::ConvertGeTypeToMmType(opName_, cType);
    args_.biasType = mc2tiling::ConvertGeTypeToMmType(opName_, biasType); // 因为bias可能不存在，先采用biasType规避
    args_.inputDtypeSize = inputDtypeSize;
    args_.outputDtypeSize = outputDtypeSize;
}

void MatmulReduceScatterTilingBase::SetReduceScatterTilingArgsShapeInfo()
{
    const gert::StorageShape* aShape = context_->GetInputShape(0);
    const gert::StorageShape* bShape = context_->GetInputShape(1);
    uint64_t mValue = aShape->GetStorageShape().GetDim(0);
    uint64_t kValue = aShape->GetStorageShape().GetDim(1);
    uint64_t nValue = bShape->GetStorageShape().GetDim(1);
    if (aShape->GetStorageShape().GetDim(1) != bShape->GetStorageShape().GetDim(0)) {
        OP_LOGD(
            context_->GetNodeName(), "A.shape(1) %lu B.shape(0) %lu, istransB = %d",
            aShape->GetStorageShape().GetDim(1), bShape->GetStorageShape().GetDim(0), args_.isBTrans);
        nValue = bShape->GetStorageShape().GetDim(0);
    }
    args_.orgMValue = mValue;
    args_.orgNValue = nValue;
    args_.orgKValue = kValue;
    args_.mValue = mValue;
    if (args_.commAlg == mc2tiling::COMM_ALG_DOUBLE_RING) {
        args_.mValue /= DOUBLE_RING_FACTOR;
        OP_LOGD(
            context_->GetNodeName(), " args_.mValue is set to be %lu under double ring communication algorithm.",
            args_.mValue);
    }
    args_.nValue = nValue;
    args_.kValue = kValue;
}

void MatmulReduceScatterTilingBase::SetReduceScatterTilingArgsBasicInfo()
{
    auto isTransA = context_->GetAttrs()->GetAttrPointer<bool>(IS_TRANS_A);
    auto isTransB = context_->GetAttrs()->GetAttrPointer<bool>(IS_TRANS_B);
    auto commTurn = *context_->GetAttrs()->GetAttrPointer<int64_t>(COMM_TURN);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAic();

    args_.isATrans = isTransA ? *isTransA : 0;
    args_.isBTrans = isTransB ? *isTransB : 0;
    args_.cmdType = mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER;
    args_.rankDim = static_cast<uint32_t>(rankSize_);
    args_.commTurn = mc2tiling::IsDeterministic() ? 1 : commTurn;
    args_.aicCoreNum = coreNum;
    args_.enablePad = false;
    args_.enableSplitK = false;
}

void MatmulReduceScatterTilingBase::SetReduceScatterTilingArgs()
{
    SetReduceScatterTilingArgsBasicInfo();
    SetReduceScatterTilingArgsShapeInfo();
    SetReduceScatterTilingArgsDataType();
}

ge::graphStatus MatmulReduceScatterTilingBase::SetReduceScatterTilingArgsCommAlgo()
{
    args_.commAlg = mc2tiling::COMM_ALG_FULL_MESH;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    OP_TILING_CHECK(
        (!ReduceScatterCheckShapeInfo()), VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to check context info"),
        return ge::GRAPH_FAILED);
    SetReduceScatterTilingArgs();
    OP_TILING_CHECK(
        (SetReduceScatterTilingArgsCommAlgo() != ge::GRAPH_SUCCESS),
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to set comm algo"), return ge::GRAPH_FAILED);
    // 为通信而进行调整搬运
    OP_TILING_CHECK(
        (args_.rankDim <= 0) || (args_.orgMValue % args_.rankDim != 0),
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "rankDim error : %u, mValue=%lu", args_.rankDim, args_.orgMValue),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
};

void MatmulReduceScatterTilingBase::SetMsgDataInfo(Mc2Tiling::RCSTiling &rcsCfg, 
                                                   ::TCubeTiling &mmTiling, ::TCubeTiling &tailTiling, 
                                                   uint32_t debugMode)
{
    // 只通信不计算模式下，如果没有gatherOut且K > N, recvOff和sendCnt需要根据N计算
    auto columnNum = args_.orgKValue;
    OP_LOGD(opName_, "Debug mode is %u, gather out flag is %d, K is %lu, N is %lu.", debugMode,
        (rcsCfg.gatherLen == 0), args_.orgKValue, args_.orgNValue);
    if ((debugMode == mc2tiling::MC2_DEBUG_ONLY_AICPU) && (rcsCfg.gatherLen != 0) && (args_.orgKValue > args_.orgNValue)) {
        OP_LOGW("MatmulReduceScatter",
            "K [%lu] is greater than N [%lu], cut recvOff and sendCnt according to N under "
            "debugMode 4 (i.e. communication only).",
            args_.orgKValue, args_.orgNValue);
        columnNum = args_.orgNValue;
    }
}

// tiling

void MatmulReduceScatterTilingBase::SetTilingResult(Mc2Tiling::RCSTiling &rcsCfg, 
                                                    ::TCubeTiling &mmTiling, ::TCubeTiling &tailTiling, 
                                                    uint32_t& debugMode, uint32_t& dataType)
{
    auto debugMode_ = mc2tiling::Mc2TilingUtils::GetDebugMode();

    debugMode = debugMode_;

    SetMsgDataInfo(rcsCfg, mmTiling, tailTiling, debugMode_);

    dataType = (static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType))); // hccl数据类型
    return;
}
} // namespace optiling
