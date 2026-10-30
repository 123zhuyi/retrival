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
 * \file allto_all_kc_quant_matmul_tiling_base.cpp
 * \brief
 */
#include "op_mc2.h"
#include "mc2_log.h"
#include "allto_all_kc_quant_matmul_tiling_base.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {
gert::StorageShape alltoallKcQuantStorageShape = gert::StorageShape();
/**
 * @brief AlltoAllMatmul KC量化的准入条件
 * 后续支持更多量化再进行修改
 *
 * @return true
 */
bool AllToAllKcQuantMatmulTilingBase::IsCapable()
{
    int x1QuantMode = 0;
    int x2QuantMode = 0;
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    if (const int *ptr = attrs->GetAttrPointer<int>(ATTR_X1_QUANTMODE_INDEX)) {
        x1QuantMode = *ptr;
    }
    if (const int *ptr = attrs->GetAttrPointer<int>(ATTR_X2_QUANTMODE_INDEX)) {
        x2QuantMode = *ptr;
    }
    if (x1QuantMode == X1_QUANTMODE_VALUES && x2QuantMode == X2_QUANTMODE_VALUES) {
        OP_LOGI(opName_, "Start with AlltoAllKcQuantMatmul tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip KcQuantMatmulAllToAll tiling when not KC_QUANT.");
    return false;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllKcQuantMatmulTilingBase::CheckOpInputInfo()
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(AllToAllMatmulTilingBase::CheckKcQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(AllToAllMatmulTilingBase::CheckKcQuantShapeInfo(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape input and output shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAlltoAllOut(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check allToAllOut failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus AllToAllKcQuantMatmulTilingBase::SetKcDataTypeInfo(const gert::TilingContext *context,
                                                                   const char *opName, TilingContextInfo &contextInfo)
{
    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(INPUT_BIAS_INDEX);
    int aDTypeNum = *context_->GetAttrs()->GetAttrPointer<uint64_t>(ALLTOALLMATMUL_ATTR_X1_QUANTDTYPE_INDEX);
    ge::DataType biasType;
    // 这是针对matmul的数据类型
    ge::DataType aType = static_cast<ge::DataType>(aDTypeNum);
    ge::DataType bType = context->GetInputDesc(INPUT_X2_INDEX)->GetDataType();
    ge::DataType cType = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    contextInfo.hcclGeType = context->GetInputDesc(INPUT_X1_INDEX)->GetDataType();
    bool isBias = true;
    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = context->GetOptionalInputDesc(INPUT_BIAS_INDEX)->GetDataType();
    }

    OP_TILING_CHECK(aDTypeNum != FP8_E5M2_VALUES && aDTypeNum != FP8_E4M3_VALUES,
    OP_LOGE(opName, "aDTypeNum %d is invalid, only 35(fp8e5m2) or 36(fp8e4m3) is supported.", aDTypeNum),
    return ge::GRAPH_FAILED);
    contextInfo.x1KcDynQuantDTypeVal = aDTypeNum;

    contextInfo.args_.outputDtypeSize = mc2tiling::GetDataTypeSize(opName, cType);
    // 设置为x1的数据类型
    contextInfo.args_.inputDtypeSize = mc2tiling::GetDataTypeSize(opName, contextInfo.hcclGeType);
    contextInfo.args_.isBias = isBias;
    contextInfo.args_.geCType = cType;
    contextInfo.args_.geBiasType = biasType;
    contextInfo.args_.geAType = aType;
    contextInfo.args_.geBType = bType;
    contextInfo.args_.cType = mc2tiling::ConvertGeTypeToMmType(opName, cType);
    contextInfo.args_.bType = mc2tiling::ConvertGeTypeToMmType(opName, bType);
    contextInfo.args_.aType = mc2tiling::ConvertGeTypeToMmType(opName, aType);
    contextInfo.args_.biasType = mc2tiling::ConvertGeTypeToMmType(opName, biasType);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllKcQuantMatmulTilingBase::InitTilingContextParameters()
{
    GE_ASSERT_GRAPH_SUCCESS(
        MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo, ALLTOALL_MATMUL_INDEX_SCHEMA));
    GE_ASSERT_GRAPH_SUCCESS(SetKcDataTypeInfo(context_, opName_, contextInfo));
    GE_ASSERT_GRAPH_SUCCESS(SetAlltoAllMatmulShapeInfo(context_, contextInfo));
    contextInfo.quantMode = QuantMode::KC_QUANT;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 主要处理逻辑，设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllKcQuantMatmulTilingBase::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    GE_ASSERT_GRAPH_SUCCESS(CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    GE_ASSERT_GRAPH_SUCCESS(InitTilingContextParameters());
    // 进行通算切分
    GE_ASSERT_GRAPH_SUCCESS(TileCommAndCompute());
    // 调用量化Matmul的tiling方法进行切分
    GE_ASSERT_GRAPH_SUCCESS(DoKcQuantMMTiling());
    // hccl的tiling参数赋值处理
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 进行通算切分之后单个块的MM Tiling
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllKcQuantMatmulTilingBase::DoKcQuantMMTiling()
{
    // 在切m时已经考虑除了rankDim
    mm_mvalue_len = inferredInfo.tileM;
    AlltoAllKcQuantMatmulHelper mmTile(*this, localTilingData_.mc2QuantMmTileTilingData, mm_mvalue_len);
    GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
    if (inferredInfo.tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }
    mm_mvalue_len = inferredInfo.tailM;
    AlltoAllKcQuantMatmulHelper mmTail(*this, localTilingData_.mc2QuantMmTailTilingData, mm_mvalue_len);
    GE_ASSERT_GRAPH_SUCCESS(mmTail.DoTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllKcQuantMatmulTilingBase::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                                                   static_cast<int32_t>(contextInfo.args_.geCType)),
                    return ge::GRAPH_FAILED;);
    Mc2CcTilingConfigBuilder allToAllBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);
    // reducetype接口附带的数据类型优先于调用通信接口传入的数据类型，因此这里需要设置
    AscendC::Mc2CcTilingConfig allToAllTilingConfig =
        allToAllBuilder
            .withReduceType(opName_, AscendC::HcclReduceOp::HCCL_REDUCE_SUM, contextInfo.hcclGeType,
                            contextInfo.hcclGeType)
            .withCommEngine(mc2tiling::A5_CCU_ENGINE)
            .build();
    if (!allToAllBuilder.isSuccess()) {
        OP_LOGE(opName_, "Build hccl tiling config failed: %s", allToAllBuilder.errorMsg().c_str());
        return ge::GRAPH_FAILED;
    }
    allToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
    allToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写获取MM index的信息
 * 由于本算子的context和MM不一样，需要重写获取MM
 * index的一些信息，把我们的context传给Matmul，来达到可以调用MM策略的目的。
 * @return ge::graphStatus
 */
const gert::Shape AlltoAllKcQuantMatmulHelper::GetX1Shape(const size_t index)
{
    (void)index;
    return gert::Shape({static_cast<int64_t>(mm_len), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
}
const gert::Shape AlltoAllKcQuantMatmulHelper::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.contextInfo.args_.isBTrans) {
        return gert::Shape({static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue),
                            static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
    }
    return gert::Shape({static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue),
                        static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue)});
}

const gert::Shape &AlltoAllKcQuantMatmulHelper::GetScaleShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_X2_SCALE_INDEX))->GetStorageShape();
}

const gert::StorageShape *AlltoAllKcQuantMatmulHelper::GetPertokenShape(const size_t index)
{
    (void)index;
    // 为了适配左矩阵的pertoken量化，需要构造pertoken量化的x1scale shape传给MM
    alltoallKcQuantStorageShape = gert::StorageShape({static_cast<int64_t>(mm_len)}, {static_cast<int64_t>(mm_len)});
    return &alltoallKcQuantStorageShape;
}

const gert::StorageShape *AlltoAllKcQuantMatmulHelper::GetBiasShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_BIAS_INDEX));
}

ge::graphStatus AlltoAllKcQuantMatmulHelper::GetShapeAttrsInfo()
{
    OP_LOGD(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto &&tilingArgs = tilingProcesser_.contextInfo.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transA = false;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    int yDType = *context_->GetAttrs()->GetAttrPointer<uint64_t>(ATTR_Y_DTYPE_INDEX);
    auto scaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((scaleTensorDesc == nullptr),
                    VECTOR_INNER_ERR_REPORT_TILING(tilingProcesser_.opName_, "the scale tensor is invalid"),
                    return ge::GRAPH_FAILED);
    inputParams_.scaleDtype = scaleTensorDesc->GetDataType();
    inputParams_.cDtype = static_cast<ge::DataType>(yDType);
    inputParams_.outDtype = static_cast<int64_t>(yDType);
    OP_LOGD(tilingProcesser_.opName_, "yDType is %ld", inputParams_.outDtype);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    if (inputParams_.isPerChannel) {
        inputParams_.groupSizeM = 1;
        inputParams_.groupSizeN = 1;
    }
    GE_ASSERT_TRUE(AnalyzeInputs());
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

void AlltoAllKcQuantMatmulHelper::PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld libApiWorkSpaceSize %u", quantMatmulInfo.mSize,
            quantMatmulInfo.kSize, quantMatmulInfo.nSize, quantMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(tilingProcesser_.opName_,
            "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d outDtype %ld"
            " scaleDtype %d",
            static_cast<int32_t>(quantMatmulInfo.aDtype), static_cast<int32_t>(quantMatmulInfo.bDtype),
            static_cast<int32_t>(quantMatmulInfo.cDtype), static_cast<int32_t>(quantMatmulInfo.biasDtype),
            quantMatmulInfo.outDtype, static_cast<int32_t>(quantMatmulInfo.scaleDtype));
    OP_LOGD(tilingProcesser_.opName_, "Check isPertoken=%d.", static_cast<int32_t>(quantMatmulInfo.isPerChannel));
}

ge::graphStatus AlltoAllKcQuantMatmulHelper::DoLibApiTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(Mc2AdaptiveSlidingWindowTiling::DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写友元类PostTiling方法
 * PostTiling主要做的是拷贝tilingdata的活，但是本算子拷贝tilingdata是在大结构体中拷贝，不需要在此处拷贝。
 * @return ge::graphStatus
 */
ge::graphStatus AlltoAllKcQuantMatmulHelper::PostTiling()
{
    tilingProcesser_.workspaceSize_ = std::max(tilingProcesser_.workspaceSize_, workspaceSize_);
    OP_LOGD(tilingProcesser_.opName_, "set mm workspace size %lu to mc2", tilingProcesser_.workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 构造函数，创建一个AlltoAllKcQuantMatmulHelper对象
 *
 * @param context
 */
AlltoAllKcQuantMatmulHelper::AlltoAllKcQuantMatmulHelper(
    AllToAllKcQuantMatmulTilingBase &allToAllKcQuantMatmulTilingBase,
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &data, uint64_t &mm_mvalue_len)
    : Mc2AdaptiveSlidingWindowTiling(allToAllKcQuantMatmulTilingBase.context_, &data),
      tilingProcesser_(allToAllKcQuantMatmulTilingBase), mm_len(mm_mvalue_len)
{
}

/**
 * @brief 打印量化matmul tiling的信息
 *
 * @param opName
 * @param tiling
 */
void AllToAllKcQuantMatmulTilingBase::PrintKcQuantMMV3TilingData(
    const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
{
    PrintTCubeTilingData(opName, tiling.matmulTiling);
    PrintExtendMatmulTiling(opName, tiling);
}

/**
 * @brief 打印执行过程中的matmul tiling信息
 *
 * @param opName
 * @param tiling
 */
void AllToAllKcQuantMatmulTilingBase::PrintExtendMatmulTiling(const std::string &opName,
                                                              DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
{
    OP_LOGD(opName, "QuantBmmV3Params.batchA=%u.", tiling.params.batchA);
    OP_LOGD(opName, "QuantBmmV3Params.batchB=%u.", tiling.params.batchB);
    OP_LOGD(opName, "QuantBmmV3Params.batchC=%u.", tiling.params.batchC);
    OP_LOGD(opName, "QuantBmmV3Params.batchA1=%u.", tiling.params.batchA1);
    OP_LOGD(opName, "QuantBmmV3Params.batchA2=%u.", tiling.params.batchA2);
    OP_LOGD(opName, "QuantBmmV3Params.batchA3=%u.", tiling.params.batchA3);
    OP_LOGD(opName, "QuantBmmV3Params.batchA4=%u.", tiling.params.batchA4);
    OP_LOGD(opName, "QuantBmmV3Params.batchB1=%u.", tiling.params.batchB1);
    OP_LOGD(opName, "QuantBmmV3Params.batchB2=%u.", tiling.params.batchB2);
    OP_LOGD(opName, "QuantBmmV3Params.batchB3=%u.", tiling.params.batchB3);
    OP_LOGD(opName, "QuantBmmV3Params.batchB4=%u.", tiling.params.batchB4);
    OP_LOGD(opName, "QuantBmmV3Params.batchC1=%u.", tiling.params.batchC1);
    OP_LOGD(opName, "QuantBmmV3Params.batchC2=%u.", tiling.params.batchC2);
    OP_LOGD(opName, "QuantBmmV3Params.batchC3=%u.", tiling.params.batchC3);
    OP_LOGD(opName, "QuantBmmV3Params.batchC4=%u.", tiling.params.batchC4);
    OP_LOGD(opName, "QuantBmmV3Params.singleCoreBatch=%u.", tiling.params.singleCoreBatch);
    OP_LOGD(opName, "QuantBmmV3Params.isPerTensor=%u.", tiling.params.isPerTensor);
    OP_LOGD(opName, "QuantBmmV3Params.isPertoken=%u.", tiling.params.isPertoken);
    OP_LOGD(opName, "QuantBmmV3Params.isDoubleScale=%u.", tiling.params.isDoubleScale);
    OP_LOGD(opName, "QuantBmmV3Params.biasThreeDim=%u.", tiling.params.biasThreeDim);
    OP_LOGD(opName, "QuantBmmV3Params.ubCalcM=%u.", tiling.params.ubCalcM);
    OP_LOGD(opName, "QuantBmmV3Params.ubCalcN=%u.", tiling.params.ubCalcN);
    OP_LOGD(opName, "QuantBmmV3Params.needUbBuffer=%u.", tiling.params.needUbBuffer);
    OP_LOGD(opName, "QuantBmmV3Params.realSingleCoreM=%u.", tiling.params.realSingleCoreM);
    OP_LOGD(opName, "QuantBmmV3Params.realSingleCoreN=%u.", tiling.params.realSingleCoreN);
    OP_LOGD(opName, "QuantBmmV3Params.biasDtype=%u.", tiling.params.biasDtype);
    OP_LOGD(opName, "QuantBmmV3Params.ubSize=%u.", tiling.params.ubSize);
    OP_LOGD(opName, "QuantBmmV3Params.isMClash=%u.", tiling.params.isMClash);
    OP_LOGD(opName, "QuantBmmV3Params.isNClash=%u.", tiling.params.isNClash);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeM=%u.", tiling.params.groupSizeM);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeK=%u.", tiling.params.groupSizeK);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeN=%u.", tiling.params.groupSizeN);
    OP_LOGD(opName, "TileL2cacheTiling.mTileCntL2=%u.", tiling.tileL2cacheTiling.mTileCntL2);
    OP_LOGD(opName, "TileL2cacheTiling.nTileCntL2=%u.", tiling.tileL2cacheTiling.nTileCntL2);
    OP_LOGD(opName, "TileL2cacheTiling.mTileBlock=%u.", tiling.tileL2cacheTiling.mTileBlock);
    OP_LOGD(opName, "TileL2cacheTiling.nTileBlock=%u.", tiling.tileL2cacheTiling.nTileBlock);
    OP_LOGD(opName, "TileL2cacheTiling.calOrder=%u.", tiling.tileL2cacheTiling.calOrder);
    OP_LOGD(opName, "TileL2cacheTiling.isBasicTiling=%u.", tiling.tileL2cacheTiling.isBasicTiling);
    OP_LOGD(opName, "AdaptiveSlidingWin.mTailTile=%u.", tiling.adaptiveSlidingWin.mTailTile);
    OP_LOGD(opName, "AdaptiveSlidingWin.nTailTile=%u.", tiling.adaptiveSlidingWin.nTailTile);
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void AllToAllKcQuantMatmulTilingBase::PrintAlltoAllKcQuantMatmulTilingInfo(const std::string &opName,
                                                                           AlltoAllMatmulTilingInfo &tilingInfo)
{
    OP_LOGD(opName, "TilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "TilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "TilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "TilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "TilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "TilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "TilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "TilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "TilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "TilingInfo.commLen: %u", tilingInfo.commLen);
    OP_LOGD(opName, "TilingInfo.permuteLen: %u", tilingInfo.permuteLen);
    OP_LOGD(opName, "tilingInfo.x1ScaleOptionalLen: %u", tilingInfo.x1ScaleOptionalLen);
    OP_LOGD(opName, "TilingInfo.hcclDataType: %u", tilingInfo.hcclDataType);
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void AllToAllKcQuantMatmulTilingBase::PrintAlltoAllKcQuantMatmulTilingData(
    AlltoAllQuantMatmulTilingData &outTilingData)
{
    PrintAlltoAllKcQuantMatmulTilingInfo(opName_, outTilingData.alltoAllQuantMatmulTilingInfo);
    PrintKcQuantMMV3TilingData(opName_, outTilingData.mc2QuantMmTileTilingData);
    if (outTilingData.alltoAllQuantMatmulTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "AlltoallKcQuantMatmul has tail");
    PrintKcQuantMMV3TilingData(opName_, outTilingData.mc2QuantMmTailTilingData);
}

/**
 * @brief 保存量化tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllKcQuantMatmulTilingBase::PostTiling()
{
    context_->SetScheduleMode(1);
    SetTilingInfo(localTilingData_.alltoAllQuantMatmulTilingInfo);
    AlltoAllQuantMatmulTilingData *outTilingData = context_->GetTilingData<AlltoAllQuantMatmulTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "AlltoAllMatmul postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.",
            sizeof(AlltoAllQuantMatmulTilingData), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(AlltoAllQuantMatmulTilingData));
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);
    PrintAlltoAllKcQuantMatmulTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void AllToAllKcQuantMatmulTilingBase::SetTilingInfo(AlltoAllMatmulTilingInfo &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.tileM = inferredInfo.tileM;
    tilingInfo.tailM = inferredInfo.tailM;
    tilingInfo.tileCnt = inferredInfo.tileCnt;
    tilingInfo.tailCnt = inferredInfo.tailCnt;
    tilingInfo.rankN = contextInfo.args_.nValue;
    tilingInfo.rankM = contextInfo.args_.orgMValue;
    tilingInfo.rankK = contextInfo.args_.orgKValue;
    tilingInfo.biasLen = inferredInfo.biasLen;
    tilingInfo.commLen = inferredInfo.commLen;
    tilingInfo.permuteLen = inferredInfo.permuteLen;
    tilingInfo.x1ScaleOptionalLen = inferredInfo.x1ScaleOptionalLen;
    tilingInfo.rankDim = contextInfo.args_.rankDim;
    tilingInfo.hcclDataType =
        (static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.hcclGeType))); // hccl数据类型
    tilingInfo.x1QuantDtype = contextInfo.x1KcDynQuantDTypeVal;
    tilingInfo.dynamicExtraSpace = 0UL;  
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pertoken量化(K)，x2为perchannel量化(C)
 * 为例子，K-C量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t AllToAllKcQuantMatmulTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    uint32_t biasDType = DTYPE_BIAS_FP32;
    uint32_t x1QuantDtype = static_cast<int>(contextInfo.args_.geAType);
    // 35代表float8_e5m2,36代表float8e4m3
    uint32_t QUANT_MODE = (x1QuantDtype == FP8_E5M2_VALUES) ? KC_QUANT_FP8E5M2_MODE : KC_QUANT_FP8E4M3_MODE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(QUANT_MODE, x2TransposeFlag, biasDType);
    OP_LOGD(opName_, "QUANTMODE,X2TRANSPOSE,DTYPEBIAS: [%d,%d,%d], TilingKey is [%lu].", QUANT_MODE, x2TransposeFlag,
            biasDType, tilingKey);
    return tilingKey;
}

ge::graphStatus AllToAllKcQuantMatmulTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    SetUserWorkSpace();
    uint64_t workspaceSize = libApiWorkSpaceSize_ + inferredInfo.commLen + inferredInfo.permuteLen +
                             inferredInfo.biasLen + +inferredInfo.x1ScaleOptionalLen + inferredInfo.quantOutLen;
    workspaces[0] = workspaceSize;
    OP_LOGD(
        opName_,
        "Workspaces[0] size=%zu, commlen=%zu, permuteLen=%zu, biasLen=%zu, x1ScaleOptionalLen=%zu, quantOutLen=%zu",
        workspaces[0], inferredInfo.commLen, inferredInfo.permuteLen, inferredInfo.biasLen,
        inferredInfo.x1ScaleOptionalLen, inferredInfo.quantOutLen);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置额外需要的空间，包括通信结果地址，重排地址，偏移地址等
 *
 */
void AllToAllKcQuantMatmulTilingBase::SetUserWorkSpace()
{
    constexpr uint64_t alignAddrLen = 512;
    // AlltoAllMatmul先进行通信，需要有对应的空间先存放结果，假设x1(m,k),假设原始rank上X1的第0维为M，这里的m就是M/ranksize,
    // m已经在前面获取输入参数的时候进行过处理
    inferredInfo.commLen = mc2tiling::AlignUp(
        contextInfo.args_.mValue * contextInfo.args_.kValue * contextInfo.args_.inputDtypeSize, alignAddrLen);
    // 重排空间等于通信结果结果空间,如果存在alltoallout空间的话，不需要申请这块
    if (!contextInfo.allToAllOutFlag) {
        inferredInfo.permuteLen = inferredInfo.commLen;
    }
    if (contextInfo.args_.isBias) {
        inferredInfo.biasLen =
            mc2tiling::AlignUp(contextInfo.args_.nValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }

    inferredInfo.x1ScaleOptionalLen = mc2tiling::AlignUp(contextInfo.args_.mValue * sizeof(float), alignAddrLen);
    // 量化后的结果为fp8
    inferredInfo.quantOutLen = mc2tiling::AlignUp(contextInfo.args_.mValue * contextInfo.args_.kValue, alignAddrLen);
}

/**
 * @brief 构造函数，创建一个AllToAllKcQuantMatmulTilingBase对象
 *
 * @param context
 */
AllToAllKcQuantMatmulTilingBase::AllToAllKcQuantMatmulTilingBase(gert::TilingContext *context)
    : AllToAllMatmulTilingBase(context)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(AlltoAllMatmul, AllToAllKcQuantMatmulTilingBase, \
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

} // namespace MC2Tiling