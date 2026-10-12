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
 * \file grouped_matmul_torch.cpp
 * \brief
 */
#include "grouped_matmul_tiling_common.h"
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"

#include "kernel_operator.h"

#include "op_kernel/grouped_matmul_kernel.h"


namespace ascend_ops {

namespace GroupedMatmul {


__global__ __aicore__ void groupedmatmul_kernel(__gm__ uint8_t *x, __gm__ uint8_t *weight, __gm__ uint8_t *bias,
                                                __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                                __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                                __gm__ uint8_t *groupList, __gm__ uint8_t *perTokenScale,
                                                __gm__ uint8_t *y, __gm__ uint8_t *workspace,
                                                GroupedMatmulTilingData tilingData)
{
    GroupedMatmulKernelImpl<GMM_TPL_BF16, GMM_TPL_BF16, GMM_TPL_BF16, 0, 0, 1, 0, 0, 0, 0, 0>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y, workspace,
        &tilingData);
}

void groupedmatmul_api(const TypeCombo &matchedCombo, int comboIndex, aclrtStream stream, const at::TensorList &x,
                       const at::TensorList &weight, const c10::optional<at::TensorList> &bias,
                       const c10::optional<at::TensorList> &scale, const c10::optional<at::TensorList> &offset,
                       const c10::optional<at::TensorList> &antiquantScale,
                       const c10::optional<at::TensorList> &antiquantOffset,
                       const c10::optional<torch::Tensor> &groupList,
                       const c10::optional<at::TensorList> &perTokenScale, const at::TensorList &y,
                       const int64_t splitItem, const int64_t groupType, const int64_t groupListType,
                       const int64_t actType, const vector<int64_t> *tuningConfigOptional)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);
    GroupedMatmulTilingData tilingData;
    GroupedMatmulNs::GroupedMatmulTiling::GroupedMatmulCommonTiling<at::TensorList, c10::optional<at::TensorList>,
                                                                    c10::optional<torch::Tensor>>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, tilingData,
        ascendcPlatform->GetCoreNumAic(), ubSizePlatFrom);
    uint32_t blockDim = tilingData.gmmBaseParams.get_coreNum();
    auto x_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.x, x, false);
    auto weight_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.weight, weight, false);
    auto y_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.output, y, false);
    auto bias_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.bias, bias, true);
    auto scale_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.scale, scale, true);
    auto offset_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.offset, offset, true);
    auto antiquantScale_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.antiquantScale, antiquantScale, true);
    auto antiquantOffset_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.antiquantOffset, antiquantOffset, true);
    auto groupList_ptr =
        get_first_tensor_address<c10::optional<torch::Tensor>>(matchedCombo.groupList, groupList, true);
    auto perTokenScale_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.perTokenScale, perTokenScale, true);
    uint64_t workspaceSize = 16U * 1024U * 1024U;
    void *workspace_ptr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);
    }
    groupedmatmul_kernel<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)x_ptr, (__gm__ uint8_t *)weight_ptr, (__gm__ uint8_t *)bias_ptr, (__gm__ uint8_t *)scale_ptr,
        (__gm__ uint8_t *)offset_ptr, (__gm__ uint8_t *)antiquantScale_ptr, (__gm__ uint8_t *)antiquantOffset_ptr,
        (__gm__ uint8_t *)groupList_ptr, (__gm__ uint8_t *)perTokenScale_ptr, (__gm__ uint8_t *)y_ptr,
        (__gm__ uint8_t *)workspace_ptr, tilingData);
}

namespace {
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_X = {at::kBFloat16};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_WEIGHT = {at::kBFloat16};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_BIAS = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_SCALE = {at::kLong};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OFFSET = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ANTIQUANTSCALE = {at::kHalf};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ANTIQUANTOFFSET = {at::kHalf};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_GROUPLIST = {at::kLong};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_PERTOKENSCALE = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OUTPUT = {at::kBFloat16};

const std::vector<TypeCombo> &getSupportedCombos()
{
    static const auto combos = TypeComboManager::createCombosFromLists(
        SUPPORTED_DTYPE_X, SUPPORTED_DTYPE_BIAS, SUPPORTED_DTYPE_SCALE, SUPPORTED_DTYPE_OFFSET,
        SUPPORTED_DTYPE_ANTIQUANTSCALE, SUPPORTED_DTYPE_ANTIQUANTOFFSET, SUPPORTED_DTYPE_GROUPLIST,
        SUPPORTED_DTYPE_PERTOKENSCALE, SUPPORTED_DTYPE_WEIGHT, SUPPORTED_DTYPE_OUTPUT);
    return combos;
}
} // namespace

torch::Tensor groupedmatmul_npu(
    const torch::TensorList &x, const torch::TensorList &weight, const c10::optional<torch::TensorList> &bias,
    const c10::optional<torch::TensorList> &scale, const c10::optional<torch::TensorList> &offset,
    const c10::optional<torch::TensorList> &antiquantScale, const c10::optional<torch::TensorList> &antiquantOffset,
    const c10::optional<torch::Tensor> &groupList, const c10::optional<torch::TensorList> &perTokenScale,
    const int64_t splitItem, const int64_t groupType, const int64_t groupListType, const int64_t actType,
    const c10::optional<c10::IntArrayRef> &tuningConfigOptional)
{
    checkTensorOnNPU(x, "x", false);
    checkTensorOnNPU(weight, "weight", false);
    checkTensorOnNPU(bias, "bias", true);
    checkTensorOnNPU(scale, "scale", true);
    checkTensorOnNPU(offset, "offset", true);
    checkTensorOnNPU(antiquantScale, "antiquantScale", true);
    checkTensorOnNPU(antiquantOffset, "antiquantOffset", true);
    checkTensorOnNPU(perTokenScale, "perTokenScale", true);
    checkTensorOnNPU(groupList, "groupList", true);

    const auto &SUPPORTED_COMBOS = getSupportedCombos();

    int matched_index = TypeComboManager::findMatchingCombo(SUPPORTED_COMBOS, x, weight, bias, scale, offset,
                                                            antiquantScale, antiquantOffset, groupList, perTokenScale);

    if (matched_index == -1) {
        TORCH_CHECK(false, "no match dtype combo");
    }

    const auto &matched_combo = SUPPORTED_COMBOS[matched_index];
    const c10::ScalarType output_type = matched_combo.output;
    auto shapeX = getTensorListFirstShape(x);
    int64_t m = shapeX[0];
    auto shapeWeight = getTensorListFirstShape(weight);
    int64_t n = shapeWeight[1];

    at::Tensor y = at::empty({m, n},                    // 指定目标shape [M, N]
                             at::dtype(output_type)     // 数据类型
                                 .device(x[0].device()) // 对齐x[0]的设备（CPU/GPU/Ascend）
                                 .layout(x[0].layout()) // 对齐x[0]的内存布局
    );

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto acl_call = [=, &matched_combo]() -> int {
        vector<int64_t> tuning_config_vec;
        vector<int64_t> *tuning_config_ptr = nullptr;

        if (tuningConfigOptional.has_value()) {
            tuning_config_vec =
                vector<int64_t>(tuningConfigOptional.value().begin(), tuningConfigOptional.value().end());
            tuning_config_ptr = &tuning_config_vec;
        }
        groupedmatmul_api(matched_combo, matched_index, stream, x, weight, bias, scale, offset, antiquantScale,
                          antiquantOffset, groupList, perTokenScale, y, splitItem, groupType, groupListType, actType,
                          tuning_config_ptr);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("GroupedMatmul", acl_call);
    return y;
}

// Register Ascend implementations for groupedmatmul
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("groupedmatmul", groupedmatmul_npu);
}

} // namespace GroupedMatmul
} // namespace ascend_ops
