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
 * \file all_gather_matmul_tiling_v2.cpp
 * \brief
 */
#include <queue>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mc2_hcom_topo_info.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "mc2_log.h"
#include "tiling/new_mc2_tiling_utils.h"
#include "all_gather_matmul_tiling_v2.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace optiling
{
bool AllGatherMatmulTilingV2::IsCapable()
{
    if ((npuArch_ == NpuArch::DAV_3510) && inputIsBf16Fp16_) {
        OP_LOGI(opName_, "Start with AllGatherMatmulTilingV2 tiling.");
        return true;
    }

    OP_LOGI(opName_, "Skip AllGatherMatmulTilingV2 tiling when inutDatatype is not fp16 or bf16.");
    return false;
}

ge::graphStatus AllGatherMatmulTilingV2::SetRawTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_TILING_CHECK((rawTilingData == nullptr), VECTOR_INNER_ERR_REPORT_TILING(opName_, "Fail to get rawTilingData."),
                    return ge::GRAPH_FAILED);
    allGatherMatmulTilingDataV2_ = context_->GetTilingData<AllGatherMatmulTilingDataV2>();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckHCCLSize());
    GE_ASSERT_GRAPH_SUCCESS(CheckInput());
    GE_ASSERT_GRAPH_SUCCESS(SetRawTilingData());
    OP_TILING_CHECK(SetMc2Hcomm(MutableRCSTilingData()) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "Fail to set Mc2Hcomm."),
                    return ge::GRAPH_FAILED);
    SetRcsTilingData(MutableRCSTilingData());
    DoSplitMTiling(MutableRCSTilingData());
    GE_ASSERT_GRAPH_SUCCESS(AdjustHCCLLimit(MutableRCSTilingData(), mc2tiling::Mc2QuantMode::DEFAULT));
    GE_ASSERT_GRAPH_SUCCESS(DoVersion2Tiling());
    DoAllGatherTiling(MutableRCSTilingData(), MutableMC2MatmulV3TileTilingData().tCubeTiling,
                      MutableMC2MatmulV3TailTilingData().tCubeTiling, allGatherMatmulTilingDataV2_->debugMode, 
                      allGatherMatmulTilingDataV2_->dataType);
    return ge::GRAPH_SUCCESS;
}

void Mc2PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling) 
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, " tiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, " tiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, " tiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, " tiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, " tiling.mBaseTailSpiltCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.nBaseTailSpiltCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, " tiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, " tiling.aswWindowLen %d", tiling.aswWindowLen);
}

void AllGatherMatmulTilingV2::PrintAllTilingData()
{
    if (MutableRCSTilingData().rankID != 0) {
        return;
    }
    PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingData());
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3LocalTilingData());
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3TileTilingData());
    if (MutableRCSTilingData().tailM <= 0) {
        return;
    }
    OP_LOGD(opName_, "Matmul has tail.");
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3TailTilingData());
}

ge::graphStatus AllGatherMatmulTilingV2::PostTiling()
{
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.",
            sizeof(AllGatherMatmulTilingDataV2), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(AllGatherMatmulTilingDataV2));

    OP_TILING_CHECK(sizeof(AllGatherMatmulTilingDataV2) % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "Tiling data size[%zu] not aligned to 8.",
                                                    sizeof(AllGatherMatmulTilingDataV2)),
                    return ge::GRAPH_FAILED);
    PrintAllTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg& tilingCfg, Mc2MMRegisterCfg& registerCfg,
                                                          Mc2MatMulV3TilingData& tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim - 1);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (args_.nValue != 0)
        if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS) {
            OP_LOGE(opName_, "DoMatmulV3Tiling failed.");
            return ge::GRAPH_FAILED;
        }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::DoVersion2Tiling()
{
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "Fail to get platformInfo."),
                    return ge::GRAPH_FAILED);

    // 获取 compileInfo
    if (mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Fail to Init CompileInfo!");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatForm = platform_ascendc::PlatformAscendC(platformInfo);

    // 根据芯片型号获取策略模板
    platform_ascendc::SocVersion socVersion = ascendcPlatForm.GetSocVersion();
    NpuArch npuArch = ascendcPlatForm.GetCurNpuArch();

    std::vector<int32_t> priorities;
    GE_ASSERT_GRAPH_SUCCESS(mc2tiling::NewGetMatmulV3PriorityPolicy(npuArch, priorities, opName_));

    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", npuArch, priorities};

    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);

    // 计算 local 块 tiling
    Mc2MatmulHelper::Mc2MatmulTilingCfg localTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                      reinterpret_cast<const void*>(&mmV3Args_));
    GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(localTilingCfg, registerCfg, MutableMC2MatmulV3LocalTilingData()));

    // 计算 tile 块 tiling
    mmV3Args_.mValue = tileMValue_ * (args_.rankDim - 1) * (MutableRCSTilingData().tileCnt);
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                     reinterpret_cast<const void*>(&mmV3Args_), tileMValue_);
    tileTilingCfg.SetCommCnt(MutableRCSTilingData().tileCnt);
    GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MatmulV3TileTilingData()));
    MutableMC2MatmulV3TileTilingData().tCubeTiling.M = (tileMValue_ * (args_.rankDim - 1));

    if (tailMValue_ > 0) {
        // 计算 tail 块 tiling
        mmV3Args_.mValue = tailMValue_ * (args_.rankDim - 1) * (MutableRCSTilingData().tailCnt);
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                         reinterpret_cast<const void*>(&mmV3Args_), tailMValue_);
        tailTilingCfg.SetCommCnt(MutableRCSTilingData().tailCnt);
        GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tailTilingCfg, registerCfg, MutableMC2MatmulV3TailTilingData()));
        MutableMC2MatmulV3TailTilingData().tCubeTiling.M = (tailMValue_ * (args_.rankDim - 1));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::SetMc2Hcomm(Mc2Tiling::RCSTiling& rcsCfg)
{
    int index = 0;
    auto group = context_->GetAttrs()->GetAttrPointer<char>(index++);
    std::string algConfig = "AllGather=level0:fullmesh";
    Mc2CcTilingConfig mc2CcTilingConfig(group, static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER), 
                                        algConfig, 0, 
                                        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)), 
                                        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)));
    uint8_t skipBufferWindowCopy = (allGatherMatmulTilingDataV2_->param.gatherLen == 0) ? 
                                    static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT) :
                                    static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
    mc2CcTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingDataV2_->mc2InitTiling) != 0,
        OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingDataV2_->mc2CcTiling) != 0,
        OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus AllGatherMatmulTilingV2::CheckInput()
{
    auto x1ScaleShape = context_->GetOptionalInputShape(SCALE_INV1);
    OP_TILING_CHECK(x1ScaleShape != nullptr, CUBE_INNER_ERR_REPORT(opName_,
                    "x1scale should be nullptr when x1 and x2 dtype is fp16 or bf16"),
                    return ge::GRAPH_FAILED);
    auto x2ScaleShape = context_->GetOptionalInputShape(SCALE_INV2);
    OP_TILING_CHECK(x2ScaleShape != nullptr, CUBE_INNER_ERR_REPORT(opName_,
                    "x2scale should be nullptr when x1 and x2 dtype is fp16 or bf16"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

AllGatherMatmulTilingV2::AllGatherMatmulTilingV2(gert::TilingContext* context)
    : AllGatherMatmulTilingBase(context), allGatherMatmulTilingDataV2_(&allGatherMatmulTilingDataV2Self_)
{
}
//注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(AllGatherMatmulV2, AllGatherMatmulTilingV2, \
                                   static_cast<int32_t>(NpuArch::DAV_3510), 0);
}  // namespace optiling