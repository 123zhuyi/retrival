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
 * \file matmul_reduce_scatter_v2_tiling.cpp
 * \brief
 */
#include "matmul_reduce_scatter_v2_tiling.h"
#include <queue>
#include <stdlib.h>
#include <sys/types.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "tiling/new_mc2_tiling_utils.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optiling {
constexpr uint32_t X1SCALE_INDEX = 3;
constexpr uint32_t X2SCALE_INDEX = 4;
// 新功能从这里开始
bool MatmulReduceScatterV2Tiling::IsCapable()
{
    if ((npuArch_ == NpuArch::DAV_3510) &&
        ((args_.geAType == ge::DT_BF16) || (args_.geAType == ge::DT_FLOAT16))) {
        OP_LOGI(opName_, "start with MatmulReduceScatterV2Tiling tiling.");
        return true;
    }

    OP_LOGI(opName_, "skip MatmulReduceScatterV2Tiling tiling when inutDatatype is not fp16 or bf16.");
    return false;
}

void PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling) 
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, " tiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, " tiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, " tiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, " tiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, " tiling.mBaseTailSpiltCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.nBaseTailSpiltCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, " tiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, " tiling.aswWindowLen %d", tiling.aswWindowLen);
}

void MatmulReduceScatterV2Tiling::PrintAllTilingData() const
{
    if (matmulReduceScatterV2TilingData_->param.rankID == 0) {
        PrintRCSTilingData(context_->GetNodeName(), matmulReduceScatterV2TilingData_->param);
        OP_LOGD(opName_, "MutableMC2MmV3TileTilingData matmulTiling");
        PrintMMV3TilingData(context_->GetNodeName(), matmulReduceScatterV2TilingData_->mC2Mmv3TileTilingData);
        if (matmulReduceScatterV2TilingData_->param.tailM > 0) {
            OP_LOGD(opName_, "MutableMC2MmV3TileTilingData matmulTiling");
            PrintMMV3TilingData(context_->GetNodeName(), matmulReduceScatterV2TilingData_->mC2Mmv3TailTilingData);
        }
    }
}

ge::graphStatus MatmulReduceScatterV2Tiling::CheckInput()
{
    auto x1ScaleShape = context_->GetOptionalInputShape(X1SCALE_INDEX);
    OP_TILING_CHECK(x1ScaleShape != nullptr, CUBE_INNER_ERR_REPORT(opName_,
                    "x1scale should be nullptr when x1 and x2 dtype is fp16 or bf16"),
                    return ge::GRAPH_FAILED);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2SCALE_INDEX);
    OP_TILING_CHECK(x2ScaleShape != nullptr, CUBE_INNER_ERR_REPORT(opName_,
                    "x2scale should be nullptr when x1 and x2 dtype is fp16 or bf16"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::SetMc2Hcomm() 
{
    const uint32_t reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    const uint32_t opType = isA2APath_ 
        ? static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL)
        : static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER);

    const std::string rsConfig = isA2APath_ 
        ? "AlltoAll=level0:fullmesh" 
        : "ReduceScatter=level0:fullmesh";

    int index = 0;
    auto group = context_->GetAttrs()->GetAttrPointer<char>(index++);
    uint32_t dataType = static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType));

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(
        group, opType, rsConfig, reduceType, dataType, dataType
    );

    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulReduceScatterV2TilingData_->mc2InitTiling) != 0,
        OP_LOGE(opName_, "mc2CcTilingConfig GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
        
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulReduceScatterV2TilingData_->mc2CcTiling) != 0,
        OP_LOGE(opName_, "mc2CcTilingConfig GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg, Mc2MMRegisterCfg &registerCfg,
                                                              Mc2MatMulV3TilingData &tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim);
    OP_LOGD(opName_, "execte DoMatmulV3Tiling!");
    tilingCfg.SetMatMulV3TilingData(tilingData);
    OP_TILING_CHECK(Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "do tiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::DoAllMatmulTiling()
{
    OP_LOGD(opName_, "excute DoAllMatmulTiling!");
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "get platform info failed"),
        return ge::GRAPH_FAILED);
    // 获取compileInfo
    OP_TILING_CHECK(mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "init compile info failed"), return ge::GRAPH_FAILED);

    // 根据芯片型号获取策略模板
    std::vector<int32_t> priorities;
    OP_TILING_CHECK(mc2tiling::NewGetMatmulV3PriorityPolicy(npuArch_, priorities, opName_) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "get mmv3 priority policy failed"), return ge::GRAPH_FAILED);
    Mc2MMRegisterCfg registerCfg {"Mc2MatMulV3", npuArch_, priorities};
    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);

    // 获取tileTiling
    mmV3Args_.mValue = tileMValue_ * args_.rankDim;
    OP_LOGD(opName_, "Do Mc2MatMulV3 tile tiling!");
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(static_cast<const void*>(&compileInfo_),
                                     static_cast<const void*>(&mmV3Args_), tileMValue_);
    GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TileTilingData()));
    if (tailMValue_ != 0UL) {
        mmV3Args_.mValue = tailMValue_ * args_.rankDim;
        OP_LOGD(opName_, "Do Mc2MatMulV3 tail tiling!");
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(static_cast<const void*>(&compileInfo_),
                                         static_cast<const void*>(&mmV3Args_), tailMValue_);
        GE_ASSERT_GRAPH_SUCCESS(DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TailTilingData()));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckHCCLSize());
    GE_ASSERT_GRAPH_SUCCESS(CheckInput());
    OP_TILING_CHECK(SetMc2Hcomm() != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "Tiling SetHcommCfg failed."), return ge::GRAPH_FAILED);
    SetRcsTilingData(matmulReduceScatterV2TilingData_->param);
    DoSplitMTiling(matmulReduceScatterV2TilingData_->param);
    GE_ASSERT_GRAPH_SUCCESS(AdjustHCCLLimit(matmulReduceScatterV2TilingData_->param, mc2tiling::Mc2QuantMode::DEFAULT));
    GE_ASSERT_GRAPH_SUCCESS(DoAllMatmulTiling());
    SetTilingResult(matmulReduceScatterV2TilingData_->param, MutableMC2MmV3TileTilingData().tCubeTiling,
                    MutableMC2MmV3TailTilingData().tCubeTiling, matmulReduceScatterV2TilingData_->debugMode,
                    matmulReduceScatterV2TilingData_->dataType);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::PostTiling()
{
    auto rawTilingDataPtr = context_->GetRawTilingData();
    OP_TILING_CHECK((rawTilingDataPtr == nullptr),
                    CUBE_INNER_ERR_REPORT(opName_, "rawTilingDataPtr is nullptr"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(opName_, "final tiling data size: %zu and context capacity size: %zu ",
        sizeof(MatmulReduceScatterV2TilingData), rawTilingDataPtr->GetCapacity());
    rawTilingDataPtr->SetDataSize(sizeof(MatmulReduceScatterV2TilingData));

    OP_TILING_CHECK(sizeof(MatmulReduceScatterV2TilingData) % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "tiling data size[%zu] not aligned to 8",
        sizeof(MatmulReduceScatterV2TilingData)),
        return ge::GRAPH_FAILED);
    PrintAllTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}
//注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulReduceScatterV2, MatmulReduceScatterV2Tiling, \
                                   static_cast<int32_t>(NpuArch::DAV_3510), 0);

}