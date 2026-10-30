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
 * \file moe_update_expert_tiling.cpp
 * \brief
 */

#include <string>
#include <numeric>
#include <climits>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/moe_update_expert_tiling.h"
#include "../../op_kernel/moe_update_expert_tiling_key.h"

using namespace ge;
using namespace AscendC;
using namespace MoeUpdateExpertNamespace;
using namespace Mc2Tiling;

namespace optiling {
constexpr uint32_t EXPERT_IDS_INDEX = 0U;
constexpr uint32_t EPLB_TABLE_INDEX = 1U;
constexpr uint32_t EXPERT_SCALES_INDEX = 2U;
constexpr uint32_t PRUNING_THRESHOLD_INDEX = 3U;
constexpr uint32_t ACTIVE_MASK_INDEX = 4U;
constexpr uint32_t OUTPUT_BALANCED_EXPERT_IDS = 0U;
constexpr uint32_t OUTPUT_ACTIVE_MASK_IDS = 1U;

constexpr uint32_t ATTR_LOCAL_RANK_ID_INDEX = 0U;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_BALANCE_MODE_INDEX = 2U;

constexpr uint32_t NUM_ONE = 1U;
constexpr uint32_t NUM_TWO = 2U;
constexpr int32_t K_MAX = 16;
constexpr int32_t BS_MAX = 512;
constexpr int32_t MAX_MOE_EXPERT_NUM = 1024;
constexpr int64_t MAX_WORLD_SIZE = 768LL;
constexpr int64_t MIN_WORLD_SIZE = 2LL;

constexpr uint32_t TAILOR_NONE = 0x0U;
constexpr uint32_t TAILOR_EXPERT_SCALES = 0x04U;
constexpr uint32_t TAILOR_PRUNING_THRESHOLD = 0x08U;
constexpr uint32_t TAILOR_PRUNING_THRESHOLD_EXPERT_SCALES = 0x0CU;
constexpr uint32_t TAILOR_ACTIVE_MASK = 0x10U;
constexpr uint32_t TAILOR_ACTIVE_MASK_EXPERT_SCALES = 0x14U;
constexpr uint32_t TAILOR_ACTIVE_MASK_PRUNING_THRESHOLD = 0x18U;
constexpr uint32_t TAILOR_TAILOR_ACTIVE_MASK_PRUNING_THRESHOLD_EXPERT_SCALES = 0x1CU;
constexpr uint64_t KEY_SCALES_FP32 = 0ULL;
constexpr uint64_t KEY_SCALES_FP16 = 10ULL;
constexpr uint64_t KEY_SCALES_BF16 = 20ULL;

const char* MOE_UPDATE_EXPERT_DEBUG = "MoeUpdateExpert Tiling";

class MoeUpdateExpertTiling
{
public:
    MoeUpdateExpertTilingData* tilingData;

    ge::graphStatus Init(gert::TilingContext* context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext* context);

protected:
    ge::graphStatus CheckAttrs(const gert::TilingContext* context) const;
    ge::graphStatus CheckInputDataType(const gert::TilingContext* context) const;
    ge::graphStatus CheckOptionalInputDataType(const gert::TilingContext* context);
    ge::graphStatus CheckOutputDataType(const gert::TilingContext* context) const;
    ge::graphStatus CheckDataType(const gert::TilingContext* context);
    ge::graphStatus CheckInputShape(const gert::TilingContext* context) const;
    ge::graphStatus CheckExpertScalesShape(const gert::TilingContext* context);
    ge::graphStatus CheckPruningThresholdShape(const gert::TilingContext* context);
    ge::graphStatus CheckActiveMaskShape(const gert::TilingContext* context);
    ge::graphStatus CheckOptionalInputShape(const gert::TilingContext* context);
    ge::graphStatus CheckOutputShape(const gert::TilingContext* context) const;
    ge::graphStatus CheckShape(const gert::TilingContext* context);
    uint64_t GetTilingKey() const;

private:
    uint32_t libApiWorkSpaceSize_{0U};
    uint32_t tailorCfg_{0U};
    uint64_t keyScales_{0ULL};
};

ge::graphStatus MoeUpdateExpertTiling::CheckAttrs(const gert::TilingContext* context) const
{
    int64_t localRankId = -1LL;
    int64_t worldSize = -1LL;

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "GetAttrs returned nullptr!"), return ge::GRAPH_FAILED);

    auto balanceModePtr = attrs->GetAttrPointer<int64_t>(ATTR_BALANCE_MODE_INDEX);
    if (balanceModePtr == nullptr) {
        tilingData->balanceMode = 0;
    } else {
        OP_TILING_CHECK(((*balanceModePtr < 0) || (*balanceModePtr > 1)),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
                "balanceMode is invalid, which should be in [0, 1] but got %ld.", *balanceModePtr),
            return ge::GRAPH_FAILED);
        tilingData->balanceMode = *balanceModePtr;
    }

    if (tilingData->balanceMode == 0) {
        auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
        OP_TILING_CHECK(worldSizePtr == nullptr,
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "worldSizePtr is null!"), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((*worldSizePtr < MIN_WORLD_SIZE) || (*worldSizePtr > MAX_WORLD_SIZE),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "worldSize is invalid, only support [%ld, %ld], but got %ld.",
                MIN_WORLD_SIZE, MAX_WORLD_SIZE, *worldSizePtr),
            return ge::GRAPH_FAILED);
        worldSize = *worldSizePtr;

        auto localRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_ID_INDEX);
        OP_TILING_CHECK(localRankIdPtr == nullptr,
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "localRankIdPtr is null!"), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((*localRankIdPtr < 0) || (*localRankIdPtr >= worldSize),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "localRankId is invalid, should in [0, %ld), but got %ld.",
                worldSize, *localRankIdPtr),
            return ge::GRAPH_FAILED);
        localRankId = *localRankIdPtr;
    }

    tilingData->localRankId = localRankId;
    tilingData->worldSize = worldSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckInputDataType(const gert::TilingContext* context) const
{
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "expertIdsDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((expertIdsDesc->GetDataType() != ge::DT_INT32) &&
        (expertIdsDesc->GetDataType() != ge::DT_INT64)),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
            "expertIds datatype is invalid, which should be int32 or int64 but is %s.",
            Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    auto eplbTableDesc = context->GetInputDesc(EPLB_TABLE_INDEX);
    OP_TILING_CHECK(eplbTableDesc == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "eplbTableDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(eplbTableDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "eplbTable datatype is invalid, which should be int32 but is %s.",
            Ops::Base::ToString(eplbTableDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckOptionalInputDataType(const gert::TilingContext* context)
{
    auto expertScalesDesc = context->GetOptionalInputDesc(EXPERT_SCALES_INDEX);
    if(expertScalesDesc != nullptr) {
        auto dtScales = expertScalesDesc->GetDataType();
        if (dtScales == ge::DT_FLOAT) {
            keyScales_ = KEY_SCALES_FP32;
        } else if (dtScales == ge::DT_FLOAT16) {
            keyScales_ = KEY_SCALES_FP16;
        } else if (dtScales == ge::DT_BF16) {
            keyScales_ = KEY_SCALES_BF16;
        } else {
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
                "expertScales datatype is invalid, which should be fp16/bf16/float but is %s.",
                Ops::Base::ToString(dtScales).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    auto pruningThresholdDesc = context->GetOptionalInputDesc(PRUNING_THRESHOLD_INDEX);
    if(pruningThresholdDesc != nullptr) {
        OP_TILING_CHECK((pruningThresholdDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
                "pruningThreshold datatype is invalid, which should be float but is %s.",
                Ops::Base::ToString(pruningThresholdDesc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }

    auto activeMaskDesc = context->GetOptionalInputDesc(ACTIVE_MASK_INDEX);
    if(activeMaskDesc != nullptr) {
        OP_TILING_CHECK((activeMaskDesc->GetDataType() != ge::DT_BOOL),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
                "activeMask datatype is invalid, which should be bool but is %s.",
                Ops::Base::ToString(activeMaskDesc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckOutputDataType(const gert::TilingContext* context) const
{
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "expertIdsDesc is null."), return ge::GRAPH_FAILED);

    auto balancedExpertIdsDesc = context->GetOutputDesc(OUTPUT_BALANCED_EXPERT_IDS);
    OP_TILING_CHECK(balancedExpertIdsDesc == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "balancedExpertIdsDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(balancedExpertIdsDesc->GetDataType() != expertIdsDesc->GetDataType(),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
            "datatype of balancedExpertIds[%s] should be the same with datatype of expertIds[%s]!",
            Ops::Base::ToString(balancedExpertIdsDesc->GetDataType()).c_str(),
            Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    auto balancedActiveMaskDesc = context->GetOutputDesc(OUTPUT_ACTIVE_MASK_IDS);
    OP_TILING_CHECK(balancedActiveMaskDesc == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "balancedActiveMaskDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(balancedActiveMaskDesc->GetDataType() != ge::DT_BOOL,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "datatype of balancedActiveMask[%s] should be bool!",
            Ops::Base::ToString(balancedActiveMaskDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckDataType(const gert::TilingContext* context)
{
     OP_TILING_CHECK(CheckInputDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check input data type failed!"), return ge::GRAPH_FAILED);
     OP_TILING_CHECK(CheckOptionalInputDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check optional input data type failed!"), return ge::GRAPH_FAILED);
     OP_TILING_CHECK(CheckOutputDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check input data type failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckInputShape(const gert::TilingContext* context) const
{
    int64_t worldSize = static_cast<int64_t>(MAX_WORLD_SIZE);
    if (tilingData->worldSize != -1LL) {
        worldSize = tilingData->worldSize;
    }

    const gert::StorageShape* expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "expert_ids shape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsStorageShape->GetStorageShape().GetDimNum() != NUM_TWO),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "The dim of expert_ids(BS, K) should be 2, but got %lu!",
            expertIdsStorageShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    tilingData->bs = expertIdsStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(((tilingData->bs <= 0) || (tilingData->bs > BS_MAX)),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "expert_ids's dim0(bs) should be in (0, %d], but got %d!", BS_MAX,
            tilingData->bs),
        return ge::GRAPH_FAILED);
    tilingData->k = expertIdsStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(((tilingData->k <= 0) || (tilingData->k > K_MAX)),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "expert_ids's dim1(k) should be in (0, %d], but got %d!", K_MAX,
            tilingData->k),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* eplbTableStorageShape = context->GetInputShape(EPLB_TABLE_INDEX);
    OP_TILING_CHECK(eplbTableStorageShape == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "eplb_table shape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((eplbTableStorageShape->GetStorageShape().GetDimNum() != NUM_TWO),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "The dim of eplb_table(moeExpertNum, F) should be 2, but got %lu!",
            eplbTableStorageShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    tilingData->moeExpertNum = eplbTableStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(((tilingData->moeExpertNum <= 0) || (tilingData->moeExpertNum > MAX_MOE_EXPERT_NUM) ||
        (tilingData->moeExpertNum < tilingData->k)),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
            "eplb_table's dim0(moeExpertNum) should be in (0, %d], and not less than K[%d], but got %d!",
            MAX_MOE_EXPERT_NUM, tilingData->k, tilingData->moeExpertNum),
        return ge::GRAPH_FAILED);
    tilingData->f = eplbTableStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(((tilingData->f <= 1) || (static_cast<int64_t>(tilingData->f) > worldSize + 1)),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "eplb_table's dim1(f) should be in (1, %ld], but got %d!",
            worldSize + 1, tilingData->f),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckExpertScalesShape(const gert::TilingContext* context)
{
    const gert::StorageShape* expertScalesShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);

    if (expertScalesShape != nullptr) {
        OP_TILING_CHECK((expertScalesShape->GetStorageShape().GetDimNum() != NUM_TWO),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "The dim of expert_scales(BS, K) should be 2, but got %lu!",
                expertScalesShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(((expertScalesShape->GetStorageShape().GetDim(0) != tilingData->bs) ||
            (expertScalesShape->GetStorageShape().GetDim(1) != tilingData->k)),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
                "expert_scales dim(bs, k) should be equals to dim[%d, %d], but got [%ld, %ld]!",
                tilingData->bs, tilingData->k, expertScalesShape->GetStorageShape().GetDim(0),
                expertScalesShape->GetStorageShape().GetDim(1)),
            return ge::GRAPH_FAILED);
        tailorCfg_ += (1U << EXPERT_SCALES_INDEX);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckPruningThresholdShape(const gert::TilingContext* context)
{
    const gert::StorageShape* pruningThresholdShape = context->GetOptionalInputShape(PRUNING_THRESHOLD_INDEX);

    if (pruningThresholdShape != nullptr) {
        if (pruningThresholdShape->GetStorageShape().GetDimNum() == NUM_ONE) {
            OP_TILING_CHECK((pruningThresholdShape->GetStorageShape().GetDim(0) != tilingData->k),
                OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
                    "pruning_threshold dim(k,) should be equals to dim[%d,], but got [%ld,]!", tilingData->k,
                    pruningThresholdShape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
        } else if (pruningThresholdShape->GetStorageShape().GetDimNum() == NUM_TWO) {
            OP_TILING_CHECK(((pruningThresholdShape->GetStorageShape().GetDim(0) != 1) ||
                (pruningThresholdShape->GetStorageShape().GetDim(1) != tilingData->k)),
                OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
                    "pruning_threshold dim(1, k) should be equals to dim[1, %d], but got [%ld, %ld]!",
                    tilingData->k, pruningThresholdShape->GetStorageShape().GetDim(0),
                    pruningThresholdShape->GetStorageShape().GetDim(1)),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
                "The dim of pruning_threshold[(k,) or (1, K)] should be 1 or 2, but got %lu!",
                pruningThresholdShape->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }

        tailorCfg_ += (1U << PRUNING_THRESHOLD_INDEX);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckActiveMaskShape(const gert::TilingContext* context)
{
    const gert::StorageShape* activeMaskShape = context->GetOptionalInputShape(ACTIVE_MASK_INDEX);

    if (activeMaskShape != nullptr) {
         OP_TILING_CHECK((activeMaskShape->GetStorageShape().GetDimNum() != NUM_ONE),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "The dim of active_mask(BS,) should be 1, but got %lu!",
                activeMaskShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK((activeMaskShape->GetStorageShape().GetDim(0) != tilingData->bs),
            OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "active_mask dim(bs,) should be equals to [%d,], but got [%ld,]!",
                tilingData->bs, activeMaskShape->GetStorageShape().GetDim(0)),
            return ge::GRAPH_FAILED);
        tailorCfg_ += (1U << ACTIVE_MASK_INDEX);
        tilingData->isActiveMask = 1;
    } else {
        tilingData->isActiveMask = 0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckOptionalInputShape(const gert::TilingContext* context)
{
    OP_TILING_CHECK(CheckExpertScalesShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "CheckExpertScalesShape failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckPruningThresholdShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "CheckPruningThresholdShape failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckActiveMaskShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "CheckActiveMaskShape failed!"),
        return ge::GRAPH_FAILED);

    // active_mask, expert_scales, pruning_threshold, 
    OP_TILING_CHECK(tailorCfg_ == TAILOR_EXPERT_SCALES,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "expert_scales has been set, pruning_threshold must be set"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tailorCfg_ == TAILOR_PRUNING_THRESHOLD,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "pruning_threshold has been set, expert_scales must be set"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tailorCfg_ == TAILOR_ACTIVE_MASK,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "active_mask has been set, pruning_threshold and expert_scales must be set"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tailorCfg_ == TAILOR_ACTIVE_MASK_EXPERT_SCALES,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "active_mask and expert_scales have been set, pruning_threshold must be set"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tailorCfg_ == TAILOR_ACTIVE_MASK_PRUNING_THRESHOLD,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "active_mask and pruning_threshold have been set, expert_scales must be set"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckOutputShape(const gert::TilingContext* context) const
{
    const gert::StorageShape* balancedExpertIdsStorageShape = context->GetOutputShape(OUTPUT_BALANCED_EXPERT_IDS);
    OP_TILING_CHECK(balancedExpertIdsStorageShape == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "balanced_expert_ids shape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((balancedExpertIdsStorageShape->GetStorageShape().GetDimNum() != NUM_TWO),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "The dim of balanced_expert_ids(BS, K) should be 2, but got %lu!",
            balancedExpertIdsStorageShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((balancedExpertIdsStorageShape->GetStorageShape().GetDim(0) != tilingData->bs) ||
        (balancedExpertIdsStorageShape->GetStorageShape().GetDim(1) != tilingData->k)),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
            "balanced_expert_ids dim(bs, k) should be equals to expert_ids's dim[%d, %d], but got [%ld, %ld]!",
            tilingData->bs, tilingData->k, balancedExpertIdsStorageShape->GetStorageShape().GetDim(0),
            balancedExpertIdsStorageShape->GetStorageShape().GetDim(1)),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* balancedActiveMaskShape = context->GetOutputShape(OUTPUT_ACTIVE_MASK_IDS);
    OP_TILING_CHECK(balancedActiveMaskShape == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "balanced_active_mask shape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((balancedActiveMaskShape->GetStorageShape().GetDimNum() != NUM_TWO),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "The dim of balanced_active_mask(BS, K) should be 2, but got %lu!",
            balancedActiveMaskShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((balancedActiveMaskShape->GetStorageShape().GetDim(0) != tilingData->bs) ||
        (balancedActiveMaskShape->GetStorageShape().GetDim(1) != tilingData->k)),
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG,
            "balanced_active_mask dim(bs, k) should be equals to expert_ids's dim[%d, %d], but got [%ld, %ld]!",
            tilingData->bs, tilingData->k, balancedActiveMaskShape->GetStorageShape().GetDim(0),
            balancedActiveMaskShape->GetStorageShape().GetDim(1)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckShape(const gert::TilingContext* context)
{
     OP_TILING_CHECK(CheckInputShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check input shape failed!"), return ge::GRAPH_FAILED);
     OP_TILING_CHECK(CheckOptionalInputShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check optional input shape failed!"), return ge::GRAPH_FAILED);
     OP_TILING_CHECK(CheckOutputShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check output shape failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::Init(gert::TilingContext* context)
{
    tilingData = context->GetTilingData<MoeUpdateExpertTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check attrs failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check data type failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckShape(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check shape failed!"), return ge::GRAPH_FAILED);

    OP_LOGD(MOE_UPDATE_EXPERT_DEBUG, "end tiling init, bs=%d, k=%d, moeExpertNum=%d, f=%d",
        tilingData->bs, tilingData->k, tilingData->moeExpertNum, tilingData->f);
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeUpdateExpertTiling::GetTilingKey() const
{
    uint32_t tilingKeyBalanceMode = (tailorCfg_ == TAILOR_NONE) ? RANK_ID_BALANCING_MODE : TOKEN_ID_BALANCING_MODE;
    uint32_t tilingKeyDataType = TILINGKEY_FLOAT;
    if (keyScales_ == KEY_SCALES_FP16) {
        tilingKeyDataType = TILINGKEY_HALF;
    } else if (keyScales_ == KEY_SCALES_BF16) {
        tilingKeyDataType = TILINGKEY_BFLOAT16;
    }
    
    uint64_t tilingKey = GET_TPL_TILING_KEY(tilingKeyBalanceMode, tilingKeyDataType);
    return tilingKey;
}

ge::graphStatus MoeUpdateExpertTiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    OP_LOGD(MOE_UPDATE_EXPERT_DEBUG, "begin RunFusionKernelTiling.");

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->aivCoreNum = aivNum;

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "get workspace failed"), return ge::GRAPH_FAILED);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaces[0] = libApiWorkSpaceSize_;

    uint64_t tilingKey = GetTilingKey();
    context->SetTilingKey(tilingKey);

    OP_LOGD(MOE_UPDATE_EXPERT_DEBUG, "end RunFusionKernelTiling, tilingKey=%lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeUpdateExpertTilingFunc(gert::TilingContext* context)
{
    MoeUpdateExpertTiling tiling;
    OP_TILING_CHECK(tiling.Init(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "MoeUpdateExpert tiling init failed."), return ge::GRAPH_FAILED);
    return tiling.RunFusionKernelTiling(context);
}

struct MoeUpdateExpertCompileInfo {
};
static ge::graphStatus TilingPrepareForMoeUpdateExpert(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<MoeUpdateExpertCompileInfo>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeUpdateExpert)
    .Tiling(MoeUpdateExpertTilingFunc)
    .TilingParse<MoeUpdateExpertCompileInfo>(TilingPrepareForMoeUpdateExpert); // 向框架注册入口函数
} // namespace optiling
