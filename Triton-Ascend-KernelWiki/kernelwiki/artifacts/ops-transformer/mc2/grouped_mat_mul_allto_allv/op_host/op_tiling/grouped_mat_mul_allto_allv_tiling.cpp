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
 * \file grouped_mat_mul_allto_allv_tiling.cc
 * \brief
 */
#include "grouped_mat_mul_allto_allv_tiling.h"
#include <string>
#include <numeric>
#include <vector>
#include "grouped_mat_mul_allto_allv_tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/mc2_tiling_common_var.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "tiling/mc2_calc_num_blocks.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/hccl_formulaic_tiling.h"
#include "tiling/mc2_tiling_utils.h"
#include "../../op_kernel/grouped_mat_mul_allto_allv_tiling.h"
#include "../../op_kernel/grouped_mat_mul_allto_allv_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
constexpr uint32_t GMM_X_INDEX = 0;
constexpr uint32_t GMM_WEIGHT_INDEX = 1;
constexpr uint32_t SEND_COUNTS_TENSOR_OPTIONAL_INDEX = 2;
constexpr uint32_t RECV_COUNTS_TENSOR_OPTIONAL_INDEX = 3;
constexpr uint32_t MM_X_OPTIONAL_INDEX = 4;
constexpr uint32_t MM_WEIGHT_OPTIONAL_INDEX = 5;
constexpr uint32_t OUTPUT_Y_INDEX = 0;
constexpr uint32_t OUTPUT_MM_Y_OPTIONAL_INDEX = 1;

constexpr uint32_t DIM_TWO = 2;
constexpr uint32_t DIM_ONE = 1;
constexpr uint32_t DIM_THREE = 3;

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2;
constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3;
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5;

constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8;

constexpr uint32_t INDEX_TWO = 2U;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_EIGHT = 8;

constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;
constexpr int64_t BEST_BASEN = 256;
constexpr uint32_t UB_DIVIDE_NUM = 2;
constexpr uint32_t UB_CALSIZE_PER_BLOCK = 16 * 1024;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t MAX_TURN_NUM = 24;
constexpr int32_t MAX_BASE_K = 128;
constexpr uint64_t COMM_TILE = 8; // 每卡数据分配几次计算
constexpr uint64_t MAX_EXPERT_NUM = 256;
constexpr int64_t MAX_EXPERT_NUM_PER_RANK = 32;
constexpr int64_t MAX_DIM_VALUE = 65536;
constexpr uint32_t MAX_SHARED_H_SHAPE_SIZE = 12288;
constexpr int64_t MAX_BSK_VALUE = 52428800;
constexpr int64_t RECV_SEND_MIN = static_cast<int64_t>((2 * 1024 * 1024) / 2);         // 2M / sizeof(gmmX)

const char* C_INNER_DEBUG = "GroupedMatMulAlltoAllv Tiling Debug";
const char* C_INNER_PRINT = "GroupedMatMulAlltoAllv Tiling Print";

static int32_t maxM = 0;
static int32_t maxN = 0;
static int32_t maxK = 0;
static int32_t baseM_ = 0;
static int32_t baseN_ = 0;
static int32_t baseK_ = 0;

static uint64_t GMMGetSizePlatForm(
    const platform_ascendc::CoreMemType memType, platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size = 0;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

struct PlatFormMemSize {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;

    explicit PlatFormMemSize(platform_ascendc::PlatformAscendC ascendcPlatform)
        : ubSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::UB, ascendcPlatform)),
          l1Size(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L1, ascendcPlatform)),
          l0CSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_C, ascendcPlatform)),
          l0ASize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_A, ascendcPlatform)),
          l0BSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_B, ascendcPlatform))
    {}
};

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15U; // 15: 16 bytes up-align
    }
    return a & ~15U; // ~15: 16 bytes down-align
}

static void PrintCommonTilingInfo(GmmAlltoAllvCommonTilingInfo &commonTilingInfo)
{
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.A %lu.", commonTilingInfo.A);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.H %lu.", commonTilingInfo.H);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.sharedMatmulH %lu.", commonTilingInfo.sharedMatmulH);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.E_ep %lu.", commonTilingInfo.E_ep);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.N1 %lu.", commonTilingInfo.N1);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.Bs %lu.", commonTilingInfo.Bs);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.N2 %lu.", commonTilingInfo.N2);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.BsK %lu.", commonTilingInfo.BsK);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.epWorldSize %lu.", commonTilingInfo.epWorldSize);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.aivCoreNum %lu.", commonTilingInfo.aivCoreNum);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.aicCoreNum %lu.", commonTilingInfo.aicCoreNum);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.isGmmWeightTrans %d.", commonTilingInfo.isGmmWeightTrans);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.isMmWeightTrans %d.", commonTilingInfo.isMmWeightTrans);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.isOptionalMatmul %d.", commonTilingInfo.isOptionalMatmul);
    OP_LOGD(C_INNER_DEBUG, " commonTilingInfo.isOptionalSendRecvCountTensors %d.",
        commonTilingInfo.isOptionalSendRecvCountTensors);
}

static bool CheckDimNum(
    const GroupedMatMulAlltoAllvTilingData* tilingData, const gert::StorageShape* gmmX,
    const gert::StorageShape* gmmWeight, const gert::StorageShape* sendCountsTensorStorageShape,
    const gert::StorageShape* recvCountsTensorStorageShape, const gert::StorageShape* mmX,
    const gert::StorageShape* mmWeight, const gert::StorageShape* y, const gert::StorageShape* mmY)
{
    OP_TILING_CHECK(
        gmmX->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE(C_INNER_DEBUG, "GmmX's dim is %lu but should be 2!", gmmX->GetStorageShape().GetDimNum()),
        return false);
    OP_TILING_CHECK(
        gmmWeight->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE(C_INNER_DEBUG, "GmmWeight's dim is %lu but should be 3!", gmmWeight->GetStorageShape().GetDimNum()),
        return false);
    OP_TILING_CHECK(
        y->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE(C_INNER_DEBUG, "Y's dim is %lu but should be 2!", y->GetStorageShape().GetDimNum()), return false);

    if (tilingData->commonTilingInfo.isOptionalMatmul) {
        OP_TILING_CHECK(
            mmX->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(C_INNER_DEBUG, "mmX's dim is %lu but should be 2!", mmX->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            mmWeight->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(C_INNER_DEBUG, "mmWeight's dim is %lu but should be 2!", mmWeight->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            mmY->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(C_INNER_DEBUG, "mmY's dim is %lu but should be 2!", mmY->GetStorageShape().GetDimNum()),
            return false);
    }
    if (tilingData->commonTilingInfo.isOptionalSendRecvCountTensors) {
        OP_TILING_CHECK(
            sendCountsTensorStorageShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE(
                C_INNER_DEBUG, "sendCountsTensor's dim is %lu but should be 1!",
                sendCountsTensorStorageShape->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            recvCountsTensorStorageShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE(
                C_INNER_DEBUG, "recvCountsTensor's dim is %lu but should be 1!",
                recvCountsTensorStorageShape->GetStorageShape().GetDimNum()),
            return false);
    }

    return true;
}

static bool CheckDimRelationship(
    const GroupedMatMulAlltoAllvTilingData* tilingData, const gert::StorageShape* gmmX,
    const gert::StorageShape* gmmWeight, const gert::StorageShape* mmX, const gert::StorageShape* mmWeight,
    const gert::StorageShape* y, const gert::StorageShape* mmY, const gert::RuntimeAttrs* attrs)
{
    (void)attrs; // Unused
    int64_t gmmWeightH = tilingData->commonTilingInfo.isGmmWeightTrans ?
                             gmmWeight->GetStorageShape().GetDim(INDEX_TWO) :
                             gmmWeight->GetStorageShape().GetDim(1);
    int64_t gmmWeightN = tilingData->commonTilingInfo.isGmmWeightTrans ? gmmWeight->GetStorageShape().GetDim(1) :
                                                                         gmmWeight->GetStorageShape().GetDim(INDEX_TWO);
    OP_TILING_CHECK(
        gmmX->GetStorageShape().GetDim(1) != gmmWeightH,
        OP_LOGE(
            C_INNER_DEBUG, "The H1 of gmmX %ld is not equal to H1 of gmmWeight %ld!", gmmX->GetStorageShape().GetDim(1),
            gmmWeightH),
        return false);
    OP_TILING_CHECK(
        gmmWeightN != y->GetStorageShape().GetDim(1),
        OP_LOGE(
            C_INNER_DEBUG, "The N1 of gmmWeight %ld is not equal to N1 of gmmY %ld!", gmmWeightN,
            y->GetStorageShape().GetDim(1)),
        return false);
    if (tilingData->commonTilingInfo.isOptionalMatmul) {
        int64_t mmWeightH = tilingData->commonTilingInfo.isMmWeightTrans ? mmWeight->GetStorageShape().GetDim(1) :
                                                                           mmWeight->GetStorageShape().GetDim(0);
        int64_t mmWeightN = tilingData->commonTilingInfo.isMmWeightTrans ? mmWeight->GetStorageShape().GetDim(0) :
                                                                           mmWeight->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            mmX->GetStorageShape().GetDim(1) != mmWeightH,
            OP_LOGE(
                C_INNER_DEBUG, "The H2 of mmX %ld is not equal to H2 of mmWeight %ld!",
                mmX->GetStorageShape().GetDim(1), mmWeightH),
            return false);
        OP_TILING_CHECK(
            mmX->GetStorageShape().GetDim(0) != mmY->GetStorageShape().GetDim(0),
            OP_LOGE(
                C_INNER_DEBUG, "The BS of mmX %ld is not equal to BS of mmY %ld!", mmX->GetStorageShape().GetDim(0),
                mmY->GetStorageShape().GetDim(0)),
            return false);
        OP_TILING_CHECK(
            mmWeightN != mmY->GetStorageShape().GetDim(1),
            OP_LOGE(
                C_INNER_DEBUG, "The N2 of mmWeight %ld is not equal to N2 of mmY %ld!", mmWeightN,
                mmY->GetStorageShape().GetDim(1)),
            return false);
    }
    return true;
}

static bool CheckRecvCnt(
    const gert::RuntimeAttrs* attrs, 
    int64_t BsK, int64_t H, int64_t eExpert, int64_t epWorldSize,
    gert::TilingContext* context)
{
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    size_t recvSize = recvCountsPtr->GetSize();
    const int64_t* recvArray = static_cast<const int64_t*>(recvCountsPtr->GetData());
    OP_TILING_CHECK(
        static_cast<int64_t>(recvSize) != epWorldSize * eExpert,
        OP_LOGE(
            C_INNER_DEBUG, 
            "The length of recvCnts[%lu] should be equal to eExpert * epworldSize[%ld]"
            "Got: epWorldSize = %ld, eExpert = %ld", 
            recvSize, epWorldSize * eExpert, epWorldSize, eExpert),
        return false);
    int64_t recvSum = 0;
    for (uint64_t i = 0; i < recvSize; i++) {
        recvSum += recvArray[i];
    }
    OP_TILING_CHECK(
        BsK != recvSum, OP_LOGE(C_INNER_DEBUG, "BsK[%ld] should be equal to the sum of recvCounts[%ld]!", BsK, recvSum),
        return false);

    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_93) {
        for (int64_t i = 1; i <= epWorldSize; i++) {
            recvSum = 0;
            for (int64_t j = (i - 1) * eExpert; j <= i * eExpert - 1; j++) {
                OP_TILING_CHECK(
                    (recvArray[j] < NUM_ZERO) || (recvArray[j] > BsK),
                    OP_LOGE(C_INNER_DEBUG, "recvCounts[%ld] should be in [0, bsK[%ld]], but get %ld",j, BsK, recvArray[j]),
                    return false);
                recvSum += recvArray[j] * H;
            }
        }
    }
    return true;
}

static bool CheckSendCnt(
    const gert::RuntimeAttrs* attrs, 
    int64_t A,  int64_t H, int64_t eExpert, int64_t epWorldSize,
    gert::TilingContext* context)
{
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    size_t sendSize = sendCountsPtr->GetSize();
    const int64_t* sendArray = static_cast<const int64_t*>(sendCountsPtr->GetData());
    OP_TILING_CHECK(
        static_cast<int64_t>(sendSize) != epWorldSize * eExpert,
        OP_LOGE(
            C_INNER_DEBUG,
            "The length of sendCnts[%lu] should be equal to eExpert * epworldSize[%ld]"
            "Got: epWorldSize = %ld, eExpert = %ld", 
            sendSize, epWorldSize * eExpert, epWorldSize, eExpert),
        return false);
    int64_t sendSum = 0;
    for (uint64_t i = 0; i < sendSize; i++) {
        sendSum += sendArray[i];
    }
    OP_TILING_CHECK(
        A != sendSum, OP_LOGE(C_INNER_DEBUG, "A[%ld] should be equal to the sum of sendCounts[%ld]!", A, sendSum),
        return false);
    
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_93) {
        for (int64_t i = 1; i <= epWorldSize; i++) {
            sendSum = 0;
            for (int64_t j = (i - 1) * eExpert; j <= i * eExpert - 1; j++) {
                OP_TILING_CHECK(
                    (sendArray[j] < NUM_ZERO) || (sendArray[j] > A),
                    OP_LOGE(C_INNER_DEBUG, "sendCounts[%ld] should be in [0, a[%ld]], but get %ld",j, A, sendArray[j]),
                    return false);
                sendSum += sendArray[j] * H;
            }
        }
    }
    return true;
}

static bool CheckSendCntAndRecvCnt(
    const gert::RuntimeAttrs* attrs, int64_t BsK, int64_t A, int64_t H, int64_t eExpert, int64_t epWorldSize,
    gert::TilingContext* context)
{
    if (!CheckRecvCnt(attrs, BsK, H, eExpert, epWorldSize, context)) {
        return false;
    }
    if (!CheckSendCnt(attrs, A, H, eExpert, epWorldSize, context)) {
        return false;
    }
    return true;
}

static bool CheckCommunicationConfig(
    const gert::ContinuousVector* recvCountsPtr,
    const gert::ContinuousVector* sendCountsPtr)
{
    size_t recvSize = recvCountsPtr->GetSize();
    size_t sendSize = sendCountsPtr->GetSize();
    OP_TILING_CHECK(
        recvSize <= 0 || recvSize > MAX_EXPERT_NUM,
        OP_LOGE(C_INNER_DEBUG, "The length of recvCnts[%lu] should be in (0, 256]", recvSize), return false);
    OP_TILING_CHECK(
        sendSize <= 0 || sendSize > MAX_EXPERT_NUM,
        OP_LOGE(C_INNER_DEBUG, "The length of sendCnts[%lu] should be in (0, 256]", sendSize), return false);
    return true;
}

static bool CheckCoreDimensionsAndCommunication(
    GroupedMatMulAlltoAllvTilingData* tilingData,
    const gert::StorageShape* gmmX, const gert::StorageShape* gmmWeight,
    const gert::StorageShape* y,
    const gert::RuntimeAttrs* attrs, gert::TilingContext* context)
{
    int64_t a = gmmX->GetStorageShape().GetDim(0);
    int64_t h = gmmX->GetStorageShape().GetDim(1);
    int64_t eEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t n1 = tilingData->commonTilingInfo.isGmmWeightTrans ? gmmWeight->GetStorageShape().GetDim(1) :
                                                                 gmmWeight->GetStorageShape().GetDim(INDEX_TWO);

    int64_t bsk = y->GetStorageShape().GetDim(0);
    int64_t epWorldSize = static_cast<int64_t>(tilingData->commonTilingInfo.epWorldSize);

    OP_TILING_CHECK(
        (bsk <= NUM_ZERO || bsk >= MAX_BSK_VALUE), OP_LOGE(C_INNER_DEBUG, "BsK[%ld] should be in (0, 52428800)!", bsk),
        return false);
    OP_TILING_CHECK(
        (h <= NUM_ZERO || h >= MAX_DIM_VALUE), OP_LOGE(C_INNER_DEBUG, "H1[%ld] should be in (0, 65536)!", h),
        return false);
    OP_TILING_CHECK(
        (n1 <= NUM_ZERO || n1 >= MAX_DIM_VALUE), OP_LOGE(C_INNER_DEBUG, "N1[%ld] should be in (0, 65536)!", n1),
        return false);

    OP_TILING_CHECK(
        (eEp <= NUM_ZERO || eEp > MAX_EXPERT_NUM_PER_RANK),
        OP_LOGE(C_INNER_DEBUG, "E_ep[%ld] should be in (0, 32]!", eEp), return false);

    OP_TILING_CHECK(
        !CheckSendCntAndRecvCnt(attrs, bsk, a, h, eEp, epWorldSize, context),
        OP_LOGE(C_INNER_DEBUG, "CheckSendCntAndRecvCnt failed!"), return false);
    return true;
}

static bool CheckEpWorldSizeConstraints(
    GroupedMatMulAlltoAllvTilingData* tilingData, gert::TilingContext* context)
{
    std::vector<int64_t> epWorldSizeOptional;
    int64_t epWorldSize = static_cast<int64_t>(tilingData->commonTilingInfo.epWorldSize);
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        epWorldSizeOptional = {2, 4, 8, 16, 32, 64}; //A5限制epWorldSize为{2，4，8，16，32，64}
    } else {
        epWorldSizeOptional = {8, 16, 32, 64, 128}; //A3限制epWorldSize为{8，16，32，64, 128}
    }
    std::string epWorldSizeNum;
    for (size_t i = 0; i < epWorldSizeOptional.size(); i++) {
        epWorldSizeNum += (std::to_string(epWorldSizeOptional[i]) + " ");
    }
    OP_TILING_CHECK(
        std::find(epWorldSizeOptional.begin(), epWorldSizeOptional.end(), epWorldSize) == epWorldSizeOptional.end(),
        OP_LOGE(C_INNER_DEBUG, "epWorldSize[%ld] should be %s!", epWorldSize, epWorldSizeNum.c_str()), return false);
    return true;
}

static bool SetupTilingDataFromValidatedDims(
    GroupedMatMulAlltoAllvTilingData* tilingData, const gert::StorageShape* gmmX, const gert::StorageShape* gmmWeight,
    const gert::StorageShape* mmX, const gert::StorageShape* mmWeight, const gert::StorageShape* y)
{
    int64_t a = gmmX->GetStorageShape().GetDim(0);
    int64_t h = gmmX->GetStorageShape().GetDim(1);
    int64_t eEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t n1 = tilingData->commonTilingInfo.isGmmWeightTrans ? gmmWeight->GetStorageShape().GetDim(1) :
                                                                 gmmWeight->GetStorageShape().GetDim(INDEX_TWO);

    int64_t bsk = y->GetStorageShape().GetDim(0);

    tilingData->commonTilingInfo.BsK = static_cast<uint64_t>(bsk);
    tilingData->commonTilingInfo.H = static_cast<uint64_t>(h);
    tilingData->commonTilingInfo.A = static_cast<uint64_t>(a);
    tilingData->commonTilingInfo.N1 = static_cast<uint64_t>(n1);
    tilingData->commonTilingInfo.E_ep = static_cast<uint64_t>(eEp);
    if (tilingData->commonTilingInfo.isOptionalMatmul) {
        int64_t bs = mmX->GetStorageShape().GetDim(0);
        int64_t sharedH = mmX->GetStorageShape().GetDim(1);
        int64_t n2 = tilingData->commonTilingInfo.isMmWeightTrans ? mmWeight->GetStorageShape().GetDim(0) :
                                                                    mmWeight->GetStorageShape().GetDim(1);
        int64_t k = (bs != 0) ? bsk / bs : 0;
        OP_TILING_CHECK(
            (bs <= NUM_ZERO), OP_LOGE(C_INNER_DEBUG, "bs[%ld] should be larger than 0!", bs),
            return false);
        OP_TILING_CHECK(
            (sharedH <= NUM_ZERO || sharedH > MAX_SHARED_H_SHAPE_SIZE),
            OP_LOGE(C_INNER_DEBUG, "H2[%ld] should be in (0, 12288]!", sharedH), return false);
        OP_TILING_CHECK(
            (n2 <= NUM_ZERO || n2 >= MAX_DIM_VALUE), OP_LOGE(C_INNER_DEBUG, "N2[%ld] should be in (0, 65536)!", n2),
            return false);
        OP_TILING_CHECK(
            (k < NUM_TWO || k > NUM_EIGHT), OP_LOGE(C_INNER_DEBUG, "K[%ld] should be in [2, 8]!", k), return false);
        tilingData->commonTilingInfo.Bs = static_cast<uint64_t>(bs);
        tilingData->commonTilingInfo.sharedMatmulH = static_cast<uint64_t>(sharedH);
        tilingData->commonTilingInfo.N2 = static_cast<uint64_t>(n2);
    }
    return true;
}

static bool CheckDimValue(
    GroupedMatMulAlltoAllvTilingData* tilingData, const gert::StorageShape* gmmX, const gert::StorageShape* gmmWeight,
    const gert::StorageShape* mmX, const gert::StorageShape* mmWeight, const gert::StorageShape* y,
    const gert::StorageShape* mmY, const gert::RuntimeAttrs* attrs, gert::TilingContext* context)
{
    (void)mmY; // Unused
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    
    if (!CheckCommunicationConfig(recvCountsPtr, sendCountsPtr)) {
        return false;
    }
    if (!CheckCoreDimensionsAndCommunication(tilingData, gmmX, gmmWeight, y, attrs, context)) {
        return false;
    }
    if (!CheckEpWorldSizeConstraints(tilingData, context)) {
        return false;
    }
    if (!SetupTilingDataFromValidatedDims(tilingData, gmmX, gmmWeight, mmX, mmWeight, y)) {
        return false;
    }

    std::copy_n(
        static_cast<const int64_t*>(recvCountsPtr->GetData()), recvCountsPtr->GetSize(),
        tilingData->aicpuTilingInfo.recvCnt);
    std::copy_n(
        static_cast<const int64_t*>(sendCountsPtr->GetData()), sendCountsPtr->GetSize(),
        tilingData->aicpuTilingInfo.sendCnt);
    return true;
}

static bool CheckMmDtype(const gert::TilingContext* context)
{
    auto mmXDex = context->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXDex == nullptr, OP_LOGE(C_INNER_DEBUG, "MM_X_OPTIONAL_INDEX is null."), return false);
    auto mmWeightDesc = context->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX);
    OP_TILING_CHECK(
        mmWeightDesc == nullptr, OP_LOGE(C_INNER_DEBUG, "MM_WEIGHT_OPTIONAL_INDEX is null."), return false);
    auto mmYDesc = context->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmYDesc == nullptr, OP_LOGE(C_INNER_DEBUG, "GetOutputDesc mmY returned null."), return false);

    auto mmXDtype = mmXDex->GetDataType();
    auto gmmXDtype = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    auto mmWeightDtype = mmWeightDesc->GetDataType();

    OP_TILING_CHECK(
        (mmXDtype != ge::DT_FLOAT16) &&
            (mmXDtype != ge::DT_BF16),
        OP_LOGE(C_INNER_DEBUG, "Unsupported dataType, mmx only support float16 and bfloat16!"), return false);
    OP_TILING_CHECK(
        (mmXDtype !=
            mmWeightDtype) ||
            (mmXDtype != mmYDesc->GetDataType()),
        OP_LOGE(C_INNER_DEBUG, "The dataType of mmWeight and mmY should be the same with mmX."), return false);

    // 校验mmdataType和gmmdataType一致

    OP_TILING_CHECK(mmXDtype != gmmXDtype,
            OP_LOGE(context->GetNodeName(),
                "mmX data type (%s) must be the same as gmmX data type (%s) when shared expert is enabled.",
                ge::TypeUtils::DataTypeToSerialString(mmXDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(gmmXDtype).c_str()),
            return false);
    OP_TILING_CHECK(mmWeightDtype != gmmXDtype,
        OP_LOGE(context->GetNodeName(),
            "mmWeight data type (%s) must be the same as gmmX data type (%s) when shared expert is enabled.",
            ge::TypeUtils::DataTypeToSerialString(mmWeightDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(gmmXDtype).c_str()),
        return false);
    return true;
}

static bool CheckDtype(const gert::TilingContext* context, const GroupedMatMulAlltoAllvTilingData* tilingData)
{
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX) == nullptr) || (context->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(C_INNER_DEBUG, "GetInputDesc gmmX or gmmWeight returned null."), return false);
    OP_TILING_CHECK(
        context->GetOutputDesc(OUTPUT_Y_INDEX) == nullptr, OP_LOGE(C_INNER_DEBUG, "GetOutputDesc y returned null."),
        return false);
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
            (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_BF16),
        OP_LOGE(C_INNER_DEBUG, "Unsupported dataType, gmmx only support float16 and bfloat16!"), return false);
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType()) ||
            (context->GetInputDesc(GMM_X_INDEX)->GetDataType() !=
             context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType()),
        OP_LOGE(C_INNER_DEBUG, "The dataType of gmmWeight and gmmY should be the same with gmmX."), return false);
    if (tilingData->commonTilingInfo.isOptionalMatmul) {
        OP_TILING_CHECK(!CheckMmDtype(context), OP_LOGE(C_INNER_DEBUG, "CheckMmDtype failed!"), return false);
    }
    if (tilingData->commonTilingInfo.isOptionalSendRecvCountTensors) {
        auto sendCount = context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_OPTIONAL_INDEX);
        OP_TILING_CHECK(
            sendCount == nullptr, OP_LOGE(C_INNER_DEBUG, "SEND_COUNTS_TENSOR_OPTIONAL_INDEX is null."), return false);
        auto recvCount = context->GetOptionalInputDesc(RECV_COUNTS_TENSOR_OPTIONAL_INDEX);
        OP_TILING_CHECK(
            recvCount == nullptr, OP_LOGE(C_INNER_DEBUG, "RECV_COUNTS_TENSOR_OPTIONAL_INDEX is null."), return false);
        OP_TILING_CHECK(
            (context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_OPTIONAL_INDEX)->GetDataType() != ge::DT_INT64) &&
                (context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_OPTIONAL_INDEX)->GetDataType() != ge::DT_INT32),
            OP_LOGE(C_INNER_DEBUG, "Unsupported dataType, sendCounts only support int64 and int32."), return false);
        OP_TILING_CHECK(
            (context->GetOptionalInputDesc(RECV_COUNTS_TENSOR_OPTIONAL_INDEX)->GetDataType() != ge::DT_INT64) &&
                (context->GetOptionalInputDesc(RECV_COUNTS_TENSOR_OPTIONAL_INDEX)->GetDataType() != ge::DT_INT32),
            OP_LOGE(C_INNER_DEBUG, "Unsupported dataType, recvCounts only support int64 and int32."), return false);
    }

    return true;
}

static bool CheckAndSetAttrs(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingData* tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(C_INNER_DEBUG, "GetAttrs returned nullptr!"), return false);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(C_INNER_DEBUG, "groupEpPtr is null!"), return false);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(C_INNER_DEBUG, "epWorldSizePtr is null!"), return false);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(C_INNER_DEBUG, "sendCountsPtr is null!"), return false);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE(C_INNER_DEBUG, "recvCountsPtr is null!"), return false);
    OP_TILING_CHECK(transMmWeightPtr == nullptr, OP_LOGE(C_INNER_DEBUG, "transMmWeightPtr is null!"), return false);
    OP_TILING_CHECK(transGmmWeightPtr == nullptr, OP_LOGE(C_INNER_DEBUG, "transGmmWeightPtr is null!"), return false);
    if (*transMmWeightPtr == true && context->GetOptionalInputShape(MM_X_OPTIONAL_INDEX) == nullptr) {
        OP_LOGE(C_INNER_DEBUG, "transMmWeightPtr should not be true when mmX is null!");
        return ge::GRAPH_FAILED;
    }
    tilingData->commonTilingInfo.epWorldSize = *epWorldSizePtr;
    tilingData->commonTilingInfo.isGmmWeightTrans = *transGmmWeightPtr;
    tilingData->commonTilingInfo.isMmWeightTrans = *transMmWeightPtr;

    return true;
}

static bool CkeckInput(
    GroupedMatMulAlltoAllvTilingData* tilingData,
    const gert::StorageShape* gmmXStorageShape,
    const gert::StorageShape* gmmWeightStorageShape,
    const gert::StorageShape* sendCountsTensorStorageShape,
    const gert::StorageShape* recvCountsTensorStorageShape)
{
    // 在aclnn侧有拦截
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE(C_INNER_DEBUG, "gmmXStorageShape is null!"), return false);
    OP_TILING_CHECK(
        gmmWeightStorageShape == nullptr, OP_LOGE(C_INNER_DEBUG, "gmmWeightStorageShape is null!"), return false);

    // 暂时拦截
    if (sendCountsTensorStorageShape != nullptr || recvCountsTensorStorageShape != nullptr) {
        OP_LOGE(C_INNER_DEBUG, "sendCountsTensor and recvCountsTensor should all be nullptr now!");
        return false;
    }
    if (sendCountsTensorStorageShape != nullptr && recvCountsTensorStorageShape != nullptr) {
        tilingData->commonTilingInfo.isOptionalSendRecvCountTensors = true;
    } else {
        tilingData->commonTilingInfo.isOptionalSendRecvCountTensors = false;
    }

    return true;
}

static bool CkeckOutput(
    GroupedMatMulAlltoAllvTilingData* tilingData,
    const gert::StorageShape* mmXStorageShape,
    const gert::StorageShape* mmWeightStorageShape,
    const gert::StorageShape* outputYStorageShape,
    const gert::StorageShape* outputMmYStorageShape)
{
    OP_TILING_CHECK(
        outputYStorageShape == nullptr, OP_LOGE(C_INNER_DEBUG, "outputYStorageShape is null!"), return false);

    bool isMmXStorageShapeNull = (mmXStorageShape == nullptr);
    bool isMmWeightStorageShapeNull = (mmWeightStorageShape == nullptr);
    bool isOutputMmYStorageShapeNull = (outputMmYStorageShape == nullptr);
    bool isOutputMmYStorageShapeValidDim = ((!isOutputMmYStorageShapeNull) && 
                                            (outputMmYStorageShape->GetStorageShape().GetDimNum() != NUM_ZERO));

    // 拦截条件一：三者都为nullptr，或者 outputMmYStorageShape 不为 nullptr 但维度为 0
    bool allNull = isMmXStorageShapeNull && isMmWeightStorageShapeNull && 
                   (isOutputMmYStorageShapeNull || (!isOutputMmYStorageShapeValidDim));
    // 拦截条件二：三者都不为nullptr，且 outputMmYStorageShape 的维度不为 0
    bool allNotNull = !isMmXStorageShapeNull && !isMmWeightStorageShapeNull && 
                      !isOutputMmYStorageShapeNull && isOutputMmYStorageShapeValidDim;

    if (!(allNull || allNotNull)) {
        OP_LOGE(C_INNER_DEBUG, 
                "mmX, mmWeight and mmY should all be nullptr or all be not nullptr!"
                "mmXStorageShape is %s, mmWeightStorageShape is %s, outputMmYStorageShape is %s"
                "outputMmY dim num = %s",
                isMmXStorageShapeNull ? "null" : "not null",
                isMmWeightStorageShapeNull ? "null" : "not null",
                isOutputMmYStorageShapeNull ? "null" : "not null",
                isOutputMmYStorageShapeNull ? "N/A" : (isOutputMmYStorageShapeValidDim ? "valid" : "zero dim"));
        return false;
    }

    tilingData->commonTilingInfo.isOptionalMatmul = (mmXStorageShape != nullptr);

    return true;
}

static bool CheckInputAndOutput(gert::TilingContext* context, GroupedMatMulAlltoAllvTilingData* tilingData)
{
    auto attrs = context->GetAttrs();

    const gert::StorageShape* gmmXStorageShape = context->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape* gmmWeightStorageShape = context->GetInputShape(GMM_WEIGHT_INDEX);
    const gert::StorageShape* sendCountsTensorStorageShape =
        context->GetOptionalInputShape(SEND_COUNTS_TENSOR_OPTIONAL_INDEX);
    const gert::StorageShape* recvCountsTensorStorageShape =
        context->GetOptionalInputShape(RECV_COUNTS_TENSOR_OPTIONAL_INDEX);
    const gert::StorageShape* mmXStorageShape = context->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    const gert::StorageShape* mmWeightStorageShape = context->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    const gert::StorageShape* outputYStorageShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    const gert::StorageShape* outputMmYStorageShape = context->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);

    if (!CkeckInput(tilingData, gmmXStorageShape, gmmWeightStorageShape,
                    sendCountsTensorStorageShape, recvCountsTensorStorageShape)) {
        return false;
    }
    if (!CkeckOutput(tilingData, mmXStorageShape, mmWeightStorageShape, outputYStorageShape, outputMmYStorageShape)) {
        return false;
    }

    OP_TILING_CHECK(!CheckDtype(context, tilingData), OP_LOGE(C_INNER_DEBUG, "CheckDtype failed!"), return false);

    OP_TILING_CHECK(
        !CheckDimNum(
            tilingData, gmmXStorageShape, gmmWeightStorageShape, sendCountsTensorStorageShape,
            recvCountsTensorStorageShape, mmXStorageShape, mmWeightStorageShape, outputYStorageShape,
            outputMmYStorageShape),
        OP_LOGE(C_INNER_DEBUG, "CheckDimNum failed!"), return false);
    OP_TILING_CHECK(
        !CheckDimRelationship(
            tilingData, gmmXStorageShape, gmmWeightStorageShape, mmXStorageShape, mmWeightStorageShape,
            outputYStorageShape, outputMmYStorageShape, attrs),
        OP_LOGE(C_INNER_DEBUG, "CheckDimValue failed!"), return false);
    OP_TILING_CHECK(
        !CheckDimValue(
            tilingData, gmmXStorageShape, gmmWeightStorageShape, mmXStorageShape, mmWeightStorageShape,
            outputYStorageShape, outputMmYStorageShape, attrs, context),
        OP_LOGE(C_INNER_DEBUG, "CheckDimValue failed!"), return false);

    return true;
}

static ge::graphStatus SetHcclTiling(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingData* tilingData)
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(C_INNER_DEBUG, "GetAttrs returned nullptr!"), return ge::GRAPH_FAILED);
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);

    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    auto inputDataType = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(C_INNER_DEBUG, "%s is Unsupported outputdata type!", Ops::Base::ToString(outputDataType).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(inputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(C_INNER_DEBUG, "%s is Unsupported inputdata type!", Ops::Base::ToString(inputDataType).c_str()),
        return ge::GRAPH_FAILED);   

    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(inputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(groupEpPtr, alltoAllvCmd, alltoAllvConfig, 
                                         alltoAllvReduceType, alltoAllvDstDataType, alltoAllvSrcDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclInitTiling) != 0,
        OP_LOGE(C_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling hcclInitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->alltoAllvCcTiling) != 0,
        OP_LOGE(C_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling alltoAllvCcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeBaseMNK(GroupedMatMulAlltoAllvTilingData* tilingData, const PlatFormMemSize PLATFORM_SIZE)
{
    uint32_t baseN = BEST_BASEN;
    maxM = static_cast<int32_t>(tilingData->commonTilingInfo.A);
    maxN = static_cast<int32_t>(tilingData->commonTilingInfo.N1);
    maxK = static_cast<int32_t>(tilingData->commonTilingInfo.H);
    while (baseN > static_cast<uint32_t>(maxN)) {
        baseN = baseN >> 1;
    }
    if (baseN < static_cast<uint32_t>(maxN)) {
        baseN = baseN << 1;
    }
    baseN_ = std::min<int32_t>(BEST_BASEN, baseN);

    // 基于使能double buffer的L0B内存计算baseK
    baseK_ = (PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B) / (baseN_ * FP16_DATASIZE);
    baseK_ = SixteenAlign(baseK_);
    if (baseK_ > MAX_BASE_K) {
        baseK_ = MAX_BASE_K;
        int32_t maxBaseN = SixteenAlign(PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B / (baseK_ * FP16_DATASIZE));
        baseN_ = std::min<int32_t>(baseN_, maxBaseN);
        baseN_ = std::max<int32_t>(16, SixteenAlign(baseN_, true)); // 16: minimum value for baseN
    }
    if (baseK_ > maxK) {
        baseK_ = std::min<int32_t>(baseK_, SixteenAlign(maxK, true));
    }
    OP_TILING_CHECK(baseK_ == 0, OP_LOGE(C_INNER_DEBUG, "baseK_ should not be 0."), return ge::GRAPH_FAILED);
    // 基于使能double buffer的L0A内存和L0B内存计算baseM(cube)
    uint32_t maxBaseM = PLATFORM_SIZE.l0CSize / (baseN_ * sizeof(float));
    baseM_ = std::min<uint32_t>((PLATFORM_SIZE.l0ASize / DOUBLE_BUFFER_L0A_L0B) / (baseK_ * FP16_DATASIZE), maxBaseM);
    baseM_ = SixteenAlign(baseM_);
    if (maxM != 0 && baseM_ > maxM) {
        baseM_ = SixteenAlign(maxM, true);
    }
    OP_TILING_CHECK(baseM_ == 0, OP_LOGE(C_INNER_DEBUG, "baseM_ should not be 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeSharedBaseMNK(
    GroupedMatMulAlltoAllvTilingData* tilingData, const PlatFormMemSize PLATFORM_SIZE)
{
    uint32_t baseN = BEST_BASEN;
    maxM = static_cast<int32_t>(tilingData->commonTilingInfo.Bs);
    maxN = static_cast<int32_t>(tilingData->commonTilingInfo.N2);
    maxK = static_cast<int32_t>(tilingData->commonTilingInfo.sharedMatmulH);
    while (baseN > static_cast<uint32_t>(maxN)) {
        baseN = baseN >> 1;
    }
    if (baseN < static_cast<uint32_t>(maxN)) {
        baseN = baseN << 1;
    }
    baseN_ = std::min<int32_t>(BEST_BASEN, baseN);

    // 基于使能double buffer的L0B内存计算baseK
    baseK_ = (PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B) / (baseN_ * FP16_DATASIZE);
    baseK_ = SixteenAlign(baseK_);
    if (baseK_ > MAX_BASE_K) {
        baseK_ = MAX_BASE_K;
        int32_t maxBaseN = SixteenAlign(PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B / (baseK_ * FP16_DATASIZE));
        baseN_ = std::min<int32_t>(baseN_, maxBaseN);
        baseN_ = std::max<int32_t>(16, SixteenAlign(baseN_, true)); // 16: minimum value for baseN
    }
    if (baseK_ > maxK) {
        baseK_ = std::min<int32_t>(baseK_, SixteenAlign(maxK, true));
    }
    OP_TILING_CHECK(baseK_ == 0, OP_LOGE(C_INNER_DEBUG, "baseK_ should not be 0."), return ge::GRAPH_FAILED);
    // 基于使能double buffer的L0A内存和L0B内存计算baseM(cube)
    uint32_t maxBaseM = PLATFORM_SIZE.l0CSize / (baseN_ * sizeof(float));
    baseM_ = std::min<uint32_t>((PLATFORM_SIZE.l0ASize / DOUBLE_BUFFER_L0A_L0B) / (baseK_ * FP16_DATASIZE), maxBaseM);
    baseM_ = SixteenAlign(baseM_);
    if (baseM_ > maxM) {
        baseM_ = SixteenAlign(maxM, true);
    }
    OP_TILING_CHECK(baseM_ == 0, OP_LOGE(C_INNER_DEBUG, "baseM_ should not be 0."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoMatmulApiTiling(
    GroupedMatMulAlltoAllvTilingData* tilingData, const PlatFormMemSize PLATFORM_SIZE, matmul_tiling::DataType mmDtype, 
    const gert::TilingContext* context)
{
    bool isBTrans = tilingData->commonTilingInfo.isGmmWeightTrans;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype, isBTrans);
    mm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, mmDtype);
    mm.SetOrgShape(maxM, maxN, maxK);
    mm.SetShape(maxM, baseN_, maxK);
    mm.SetFixSplit(std::min(baseM_, maxM), baseN_);
    mm.SetBufferSpace(PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize, PLATFORM_SIZE.ubSize);

    OP_TILING_CHECK(
        mm.GetTiling(tilingData->matmulTiling) == -1, OP_LOGE(C_INNER_DEBUG, "matmul api GetTiling failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoSharedMatmulApiTiling(
    GroupedMatMulAlltoAllvTilingData* tilingData, const PlatFormMemSize PLATFORM_SIZE, matmul_tiling::DataType mmDtype, 
    const gert::TilingContext* context)
{
    bool isBTrans = tilingData->commonTilingInfo.isMmWeightTrans;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    matmul_tiling::MatmulApiTiling sharedmm(ascendcPlatform);
    sharedmm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype, false);
    sharedmm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype, isBTrans);
    sharedmm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, mmDtype);
    sharedmm.SetOrgShape(maxM, maxN, maxK);
    sharedmm.SetShape(maxM, baseN_, maxK);
    sharedmm.SetFixSplit(std::min(baseM_, maxM), baseN_);
    sharedmm.SetBufferSpace(PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize, PLATFORM_SIZE.ubSize);

    OP_TILING_CHECK(
        sharedmm.GetTiling(tilingData->sharedExpMatmulTiling) == -1,
        OP_LOGE(C_INNER_DEBUG, "sharedExpMatmul api getTiling failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetMatmulTiling(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingData* tilingData,
    const PlatFormMemSize PLATFORM_SIZE)
{
    auto gmmXdec = context->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXdec == nullptr, OP_LOGE(C_INNER_DEBUG, "GMM_X_INDEX returned nullptr!"), return false);
    auto inputDtype = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    matmul_tiling::DataType mmDtype =
        (inputDtype == ge::DT_FLOAT16) ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_BFLOAT16;

    OP_TILING_CHECK(
        ComputeBaseMNK(tilingData, PLATFORM_SIZE) != ge::GRAPH_SUCCESS,
        OP_LOGE(C_INNER_DEBUG, "GMM Tiling compute baseMNK failed."), return ge::GRAPH_FAILED);
    if (tilingData->commonTilingInfo.A != 0) {
        OP_TILING_CHECK(
            DoMatmulApiTiling(tilingData, PLATFORM_SIZE, mmDtype, context) != ge::GRAPH_SUCCESS,
            OP_LOGE(C_INNER_DEBUG, "GMM Tiling matmul api do tiling failed."), return ge::GRAPH_FAILED);
    }
    if (tilingData->commonTilingInfo.isOptionalMatmul) {
        OP_TILING_CHECK(
            ComputeSharedBaseMNK(tilingData, PLATFORM_SIZE) != ge::GRAPH_SUCCESS,
            OP_LOGE(C_INNER_DEBUG, "GMM shared expert Tiling compute baseMNK failed."), return ge::GRAPH_FAILED);

        OP_TILING_CHECK(
            DoSharedMatmulApiTiling(tilingData, PLATFORM_SIZE, mmDtype, context) != ge::GRAPH_SUCCESS,
            OP_LOGE(C_INNER_DEBUG, "GMM shared expert Tiling matmul api do tiling failed."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context, const GroupedMatMulAlltoAllvTilingData* tilingData)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(C_INNER_DEBUG, "get workspace failed"), return ge::GRAPH_FAILED);
    uint64_t gmmOut = tilingData->commonTilingInfo.A * tilingData->commonTilingInfo.N1 * FP16_DATASIZE;
    workspaces[0] = gmmOut + SYS_WORKSPACE_SIZE;
    OP_LOGD(C_INNER_DEBUG, "workspaces[0] size is %ld. gmmOut is %lu.", workspaces[0], gmmOut);

    return ge::GRAPH_SUCCESS;
}

static void UpdateTilingKey(uint64_t& tilingKey, const GroupedMatMulAlltoAllvTilingData* tilingData, gert::TilingContext* context)
{
    bool tilingkeyComputeMm = false;
    bool tilingkeyGmmTrans = false;
    bool tilingkeyMmTrans = false;

    if (tilingData->commonTilingInfo.isOptionalMatmul) {
        tilingkeyComputeMm = true;
    } else {
        tilingkeyComputeMm = false;
    }
    if (tilingData->commonTilingInfo.isGmmWeightTrans) {
        tilingkeyGmmTrans = true;
    } else {
        tilingkeyGmmTrans = false;
    }
    if (tilingData->commonTilingInfo.isMmWeightTrans) {
        tilingkeyMmTrans = true;
    } else {
        tilingkeyMmTrans = false;
    }

    tilingKey = GET_TPL_TILING_KEY(tilingkeyComputeMm,
                                    tilingkeyGmmTrans, tilingkeyMmTrans);
    return;
}


static ge::graphStatus GroupedMatMulAlltoAllvTilingFuncA3(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    GroupedMatMulAlltoAllvTilingData* tilingData = context->GetTilingData<GroupedMatMulAlltoAllvTilingData>();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // Function that get check and set Attrs
    OP_TILING_CHECK(
        !CheckAndSetAttrs(context, tilingData), OP_LOGE(C_INNER_DEBUG, "Check and set attributes failed!"),
        return ge::GRAPH_FAILED);

    // Function that check input/output dims
    OP_TILING_CHECK(
        !CheckInputAndOutput(context, tilingData), OP_LOGE(C_INNER_DEBUG, "Check Inputs and Outputs failed!"),
        return ge::GRAPH_FAILED);

    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t numBlocks = mc2tiling::GetNumBlocks(aicNum, aivNum, C_INNER_DEBUG);
    uint64_t ubSize = 0LU;
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    tilingData->commonTilingInfo.aicCoreNum = numBlocks;
    tilingData->commonTilingInfo.aivCoreNum = numBlocks * NUM_TWO;    // aic:aiv按照1：2配比
    context->SetBlockDim(static_cast<uint32_t>(numBlocks));           // 通算融合场景 AIC_NUM:AIV_NUM = 1:2 默认启动
    
    // Set HCCL tiling
    OP_TILING_CHECK(SetHcclTiling(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(C_INNER_DEBUG, "SetHcclTiling Failed!"), return ge::GRAPH_FAILED);

    // Set matmul tiling
    OP_TILING_CHECK(
        SetMatmulTiling(context, tilingData, PLATFORM_SIZE) != ge::GRAPH_SUCCESS,
        OP_LOGE(C_INNER_DEBUG, "Set matmul tiling Failed!"), return ge::GRAPH_FAILED);

    // Compute and set workspace
    // allGatherOut, gmmOut, 16kb
    OP_TILING_CHECK(
        SetWorkspace(context, tilingData) != ge::GRAPH_SUCCESS, OP_LOGE(C_INNER_DEBUG, "Set workspace Failed!"),
        return ge::GRAPH_FAILED);

    // Calculate and Set TilingKey
    // 个位表示是否执行可选的matmul
    // 百位表示mm的weight是否转置
    // 千位表示gmm的weight是否转置
    uint64_t tilingKey = 0;
    UpdateTilingKey(tilingKey, tilingData, context);
    OP_LOGD(nodeName, "Computed tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    PrintCommonTilingInfo(tilingData->commonTilingInfo);
    OP_LOGD("GroupedMatMulAlltoAllv", "tiling process finished successfully!!!");

    return ge::GRAPH_SUCCESS;
}

bool GmmAlltoAllvTilingStruct::IsCapable()
{
    return true;
}

ge::graphStatus GmmAlltoAllvTilingStruct::DoOpTiling()
{
    return GroupedMatMulAlltoAllvTilingFuncA3(context_);
}

uint64_t GmmAlltoAllvTilingStruct::GetTilingKey() const
{
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(C_INNER_DEBUG, "GmmAlltoAllvTiling get tiling key %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus GmmAlltoAllvTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(
        platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(C_INNER_DEBUG, "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    return ge::GRAPH_SUCCESS;
}

// Every thing is done by DoOptiling.
ge::graphStatus GmmAlltoAllvTilingBase::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatMulAlltoAllv, GmmAlltoAllvTilingStruct, 0);


static ge::graphStatus GroupedMatMulAlltoAllvTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct GroupedMatMulAlltoAllvCompileInfo {
};
static ge::graphStatus TilingParseForGroupedMatMulAlltoAllv(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatMulAlltoAllv)
    .Tiling(GroupedMatMulAlltoAllvTilingFunc)
    .TilingParse<GroupedMatMulAlltoAllvCompileInfo>(TilingParseForGroupedMatMulAlltoAllv);
} // end of namespace optiling