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
 * \file allto_all_fp_matmul_tiling_base.cpp
 * \brief
 */
#include "op_mc2.h"
#include "mc2_log.h"
#include "allto_all_fp_matmul_tiling_base.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief AlltoAllMatmul非量化的准入条件，当前仅实现非量化，所以直接return true
 * 后续支持更多量化再进行修改
 *
 * @return true
 */
bool AllToAllFpMatmulTilingBase::IsCapable()
{
    QuantMode mode = MatmulAlltoAllTilingUtil::GetQuantMode(context_, opName_);
    if (mode == QuantMode::NON_QUANT) {
        OP_LOGI(opName_, "Start with FpMatmulAllToAll tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip FpMatmulAllToAll tiling when not NON_QUANT.");
    return false;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBase::CheckOpInputInfo()
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);                
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckNonQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckShapeInfo(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape input and output shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAlltoAllOut(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check allToAllOut failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBase::InitTilingContextParameters()
{
    GE_ASSERT_GRAPH_SUCCESS(
        MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo, ALLTOALL_MATMUL_INDEX_SCHEMA));
    GE_ASSERT_GRAPH_SUCCESS(MatmulAlltoAllTilingUtil::SetDataTypeInfo(context_, opName_, contextInfo));
    GE_ASSERT_GRAPH_SUCCESS(SetAlltoAllMatmulShapeInfo(context_, contextInfo));
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 主要处理逻辑，设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBase::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    GE_ASSERT_GRAPH_SUCCESS(CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    GE_ASSERT_GRAPH_SUCCESS(InitTilingContextParameters());
    // 进行通算切分
    GE_ASSERT_GRAPH_SUCCESS(TileCommAndCompute());
    // 调用非量化Matmul的tiling方法进行切分
    GE_ASSERT_GRAPH_SUCCESS(DoMMTiling());
    // hccl的tiling参数赋值处理
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 非量化MatmulTiling进行通算切分之后执行MM Tiling
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBase::DoMMTiling()
{
    // platform非空校验已在GetPlatformInfo校验过
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    if (mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &mmV3compileInfo_) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Fail to Init CompileInfo!");
        return ge::GRAPH_FAILED;
    }

    std::vector<int32_t> priorities;
    GE_ASSERT_GRAPH_SUCCESS(mc2tiling::NewGetMatmulV3PriorityPolicy(npuArch_, priorities, opName_));
    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", npuArch_, priorities};

    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, contextInfo.args_, opName_);

    //  tile  tiling
    mmV3Args_.mValue = inferredInfo.tileM;
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void *>(&mmV3compileInfo_),
                                                      reinterpret_cast<const void *>(&mmV3Args_));
    GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, localTilingData_.mc2MmV3TileTilingData));
    if (inferredInfo.tailM > 0) {
        //  tail  tiling
        mmV3Args_.mValue = inferredInfo.tailM;
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void *>(&mmV3compileInfo_),
                                                          reinterpret_cast<const void *>(&mmV3Args_));
        GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tailTilingCfg, registerCfg, localTilingData_.mc2MmV3TailTilingData));
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 调用MM的tiling
 *
 * @param tilingCfg MM的tiling的编译与参数信息
 * @param mmRegisterCfg 实际MM的注册信息
 * @param tilingData 对应首块或尾块的tilingData
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBase::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg,
                                                             Mc2MMRegisterCfg &mmRegisterCfg,
                                                             Mc2MatMulV3TilingData &tilingData)
{
    tilingCfg.SetRankDim(contextInfo.args_.rankDim);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, mmRegisterCfg) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "DoMatmulV3Tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBase::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    OP_LOGE(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                            static_cast<int32_t>(contextInfo.args_.geCType)),
                    return ge::GRAPH_FAILED;);

    Mc2CcTilingConfigBuilder allToAllBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);
    
    //reducetype接口附带的数据类型优先于调用通信接口传入的数据类型，因此这里需要设置
    AscendC::Mc2CcTilingConfig allToAllTilingConfig = allToAllBuilder.withCommEngine(mc2tiling::A5_CCU_ENGINE).
        withReduceType(opName_, AscendC::HcclReduceOp::HCCL_REDUCE_SUM, contextInfo.args_.geAType, contextInfo.args_.geAType).build();
    if (!allToAllBuilder.isSuccess()) {
        return ge::GRAPH_FAILED;
    }
    allToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
    allToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的也是量化组合，对于非量化的场景，QUANT_MODE=0
 *
 * @return uint64_t tilingKey结果
 */
uint64_t AllToAllFpMatmulTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    // 0代表数据类型和x一致(FP16 OR BF16)，1代表FP32
    uint32_t biasDType = DTYPE_BIAS_SAME_WITH_X;
    if (contextInfo.args_.geBiasType != contextInfo.args_.geAType) {
        biasDType = DTYPE_BIAS_FP32;
    }
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(NON_QUANT_MODE, x2TransposeFlag, biasDType);
    OP_LOGD(opName_, "QUANTMODE,X2TRANSPOSE,DTYPEBIAS is: [%d,%d,%d], and tilingKey is [%lu].", NON_QUANT_MODE,
            x2TransposeFlag, biasDType, tilingKey);
    return tilingKey;
}

/**
 * @brief 保存tiling数据到context,并打印相关信息
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBase::PostTiling()
{
    SetTilingInfo(localTilingData_.alltoAllMatmulTilingInfo);
    AlltoAllMatmulTilingData *outTilingData = context_->GetTilingData<AlltoAllMatmulTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "PostTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(AlltoAllMatmulTilingData),
            context_->GetRawTilingData()->GetCapacity());

    context_->GetRawTilingData()->SetDataSize(sizeof(AlltoAllMatmulTilingData));
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);
    PrintAlltoAllMatmulTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 将runInfo的信息拷贝到tilingInfo结构体
 * @param tilingInfo 目标结构体
 */
void AllToAllFpMatmulTilingBase::SetTilingInfo(AlltoAllMatmulTilingInfo &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.tileM = inferredInfo.tileM;
    tilingInfo.tileCnt = inferredInfo.tileCnt;
    tilingInfo.tailM = inferredInfo.tailM;
    tilingInfo.tailCnt = inferredInfo.tailCnt;
    tilingInfo.rankM = contextInfo.args_.orgMValue;
    tilingInfo.rankN = contextInfo.args_.nValue;
    tilingInfo.rankK = contextInfo.args_.orgKValue;
    tilingInfo.commLen = inferredInfo.commLen;
    tilingInfo.permuteLen = inferredInfo.permuteLen;
    tilingInfo.biasLen = inferredInfo.biasLen;
    tilingInfo.rankDim = contextInfo.args_.rankDim;
    tilingInfo.hcclDataType =
        (static_cast<uint64_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geAType))); // hccl数据类型
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void AllToAllFpMatmulTilingBase::PrintAlltoAllMatmulTilingData(AlltoAllMatmulTilingData &alltoAllMatmulTilingData)
{
    PrintAlltoAllMatmulTilingInfo(opName_, alltoAllMatmulTilingData.alltoAllMatmulTilingInfo);
    PrintMMV3TilingData(opName_, alltoAllMatmulTilingData.mc2MmV3TileTilingData);
    if (alltoAllMatmulTilingData.alltoAllMatmulTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "Matmulalltoall has tail");
    PrintMMV3TilingData(opName_, alltoAllMatmulTilingData.mc2MmV3TailTilingData);
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void AllToAllFpMatmulTilingBase::PrintAlltoAllMatmulTilingInfo(const std::string &opName,
                                                               AlltoAllMatmulTilingInfo &tilingInfo)
{
    OP_LOGD(opName, "TilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "TilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "TilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "TilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "TilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "TilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "TilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "TilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "TilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "TilingInfo.commLen: %u", tilingInfo.commLen);
    OP_LOGD(opName, "TilingInfo.permuteLen: %u", tilingInfo.permuteLen);
    OP_LOGD(opName, "TilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "TilingInfo.hcclDataType: %u", tilingInfo.hcclDataType);
}

/**
 * @brief 打印matmul tiling的信息,注：当前蓝区冒烟找不到mc2_log.h的对应方法，暂时自己实现
 *
 * @param opName
 * @param tiling
 */
void AllToAllFpMatmulTilingBase::PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling)
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, " MMtiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, " MMtiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, " MMtiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, " MMtiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, " MMtiling.mBaseTailSpiltCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, " MMtiling.nBaseTailSpiltCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, " MMtiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, " MMtiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, " MMtiling.aswWindowLen %d", tiling.aswWindowLen);
}


AllToAllFpMatmulTilingBase::AllToAllFpMatmulTilingBase(gert::TilingContext *context) : AllToAllMatmulTilingBase(context)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(AlltoAllMatmul, AllToAllFpMatmulTilingBase, \
                                   static_cast<int32_t>(NpuArch::DAV_3510), 0);
} // namespace MC2Tiling

