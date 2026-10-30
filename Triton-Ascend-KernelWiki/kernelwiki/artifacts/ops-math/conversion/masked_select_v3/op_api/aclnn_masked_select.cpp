/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_masked_select.h"
#include "masked_select.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* MaskedSelect 算子的完整计算流程如下:
 * self                               mask
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_1)
 *      \                             /
 *          \                 Cast(workspace_2)
 *             \                 /
 *             MaskedSelect(workspace_3)
 *                    |
 *               Cast(workspace_4)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */
namespace ACLNN_MASKED_SELECT {
constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST_NOT_SUPPORT_BF16 = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST_SUPPORT_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST_SUPPORT_REGBASE = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,   op::DataType::DT_UINT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT64, op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_UINT16,
    op::DataType::DT_INT8,   op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8, op::DataType::DT_BOOL};
} // namespace ACLNN_MASKED_SELECT
using namespace ACLNN_MASKED_SELECT;
inline static bool CheckNotNull(const aclTensor* self, const aclTensor* mask, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const aclTensor* ResetFormatForRegBase(const aclTensor* x, const aclIntArray* shape)
{
    if (!IsRegBase()) {
        return x;
    }
    size_t tensorXSize = x->GetViewShape().GetDimNum();
    if (tensorXSize == (*shape).Size()) {
        return x;
    }
    auto constX = const_cast<aclTensor*>(x);
    constX->SetViewFormat(op::Format::FORMAT_ND);
    constX->SetStorageFormat(op::Format::FORMAT_ND);
    constX->SetOriginalFormat(op::Format::FORMAT_ND);
    return constX;
}

static const std::initializer_list<op::DataType> CheckSocVersionIsSupportBf16(void)
{
    if (IsRegBase()) {
        return SELF_DTYPE_SUPPORT_LIST_SUPPORT_REGBASE;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return SELF_DTYPE_SUPPORT_LIST_SUPPORT_BF16;
    }
    return SELF_DTYPE_SUPPORT_LIST_NOT_SUPPORT_BF16;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* mask, const aclTensor* out)
{
    auto SELF_DTYPE_SUPPORT_LIST = CheckSocVersionIsSupportBf16();
    // 检查self的数据类型是否在maskedSelect算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    // 检查mask的数据类型是否在maskedSelect算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(mask, MASK_DTYPE_SUPPORT_LIST, return false);
    // 检查out的数据类型是否在maskedSelect算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, SELF_DTYPE_SUPPORT_LIST, return false);

    return true;
}

inline static bool isOutSizeSameWithBroadcastShapeSize(const aclTensor* y, op::Shape broadcastShape)
{
    int64_t broadcastShapeSize = broadcastShape.GetShapeSize();
    if (y->GetViewShape().GetShapeSize() == broadcastShapeSize) {
        return true;
    }
    return false;
}

static bool CheckShape(const aclTensor* self, const aclTensor* mask, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(mask, MAX_DIM_LEN, return false);

    Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, mask, broadcastShape, return false);
    OP_CHECK_WRONG_DIMENSION(y, 1, return false);

    if (!isOutSizeSameWithBroadcastShapeSize(y, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The out shape size  is not same with broadcastShapeSize.");
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "y.shape: %ld, broadcastShape.shape: %ld.", y->GetViewShape().GetShapeSize(),
            broadcastShape.GetShapeSize());
        return false;
    }
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* mask, const aclTensor* y)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mask, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, mask, y), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入形状是否满足
    CHECK_RET(CheckShape(self, mask, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (IsRegBase()) {
        return CheckType(self->GetDataType(), SELF_DTYPE_SUPPORT_LIST_SUPPORT_REGBASE);
    } else if (
        GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return CheckType(self->GetDataType(), SELF_DTYPE_SUPPORT_LIST_SUPPORT_BF16);
    }
    return false;
}

static void CheckFormat(const aclTensor* self, const aclTensor* target){
  ge::Format selfStorageFormat = self->GetStorageFormat();
  ge::Format targetStorageFormat = target->GetStorageFormat();
  if (selfStorageFormat != ge::Format::FORMAT_ND || targetStorageFormat != ge::Format::FORMAT_ND){
    OP_LOGW("aclnnMaskedSelect only support format ND.");
  }
}

aclnnStatus aclnnMaskedSelectGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mask, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnMaskedSelect, DFX_IN(self, mask), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, mask, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CheckFormat(self, mask);

    if (self->IsEmpty() || mask->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入mask转换成连续的tensor
    auto maskContiguous = l0op::Contiguous(mask, uniqueExecutor.get());
    CHECK_RET(maskContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入mask的数据类型转换成bool数据类型
    auto maskCasted = l0op::Cast(maskContiguous, DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(maskCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成out数据类型
    auto selfCasted = l0op::Cast(selfContiguous, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* selfBroadcast;
    const aclTensor* maskBroadcast;
    selfBroadcast = selfCasted;
    maskBroadcast = maskCasted;
    if (IsAiCoreSupport(self)) {
        // 判断输入shape不相等需要调用BroadcastTo
        if (self->GetViewShape() != mask->GetViewShape()) {
            op::Shape broadcastShape;
            if (BroadcastInferShape(self->GetViewShape(), mask->GetViewShape(), broadcastShape)) {
                op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
                auto broadcastShapeArray =
                    uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
                CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto selfCastedAfterFormat = ResetFormatForRegBase(selfCasted, broadcastShapeArray);
                selfBroadcast = l0op::BroadcastTo(selfCastedAfterFormat, broadcastShapeArray, uniqueExecutor.get());
                CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto maskCastedAfterFormat = ResetFormatForRegBase(maskCasted, broadcastShapeArray);
                maskBroadcast = l0op::BroadcastTo(maskCastedAfterFormat, broadcastShapeArray, uniqueExecutor.get());
                CHECK_RET(maskBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
        // 调用MaskedSelect算子
        l0op::MaskedSelectV3(selfBroadcast, maskBroadcast, out, uniqueExecutor.get());
    } else {
        // 调用MaskedSelect算子
        l0op::MaskedSelect(selfBroadcast, maskBroadcast, out, uniqueExecutor.get());
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaskedSelect(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaskedSelect);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
