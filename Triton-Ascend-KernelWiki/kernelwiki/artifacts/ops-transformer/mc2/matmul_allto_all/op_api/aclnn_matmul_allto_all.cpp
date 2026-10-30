/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_matmul_allto_all.h"
#include "matmul_allto_all_util.h"
#include "securec.h"
#include "op_mc2.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "hccl_util.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"

namespace {

using namespace op;
using namespace matmul_allto_all_check;

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* output) {
    if (x1 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x1 should not be null.");
        return false;
    }
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2 should not be null.");
        return false;
    }
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Output should not be null.");
        return false;
    }
    return true;
}

// 检查是否有空tensor
// 950 非量化场景支持x1的m轴为0，即token提示词为空
static bool CheckNotEmptyTensor(const aclTensor* x1, const aclTensor* x2, bool transposeX2) {
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        auto mVal = x1->GetViewShape().GetDim(0);
        OP_API_CHECK((mVal == ZERO), {
          OP_LOGE(ACLNN_ERR_PARAM_INVALID,
          "X1 is empty tensor with zero dimM, which is unsupported.");
          return false;
        });
    }
    auto kVal1 = x1->GetViewShape().GetDim(1);
    auto kVal2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    OP_API_CHECK((kVal1 == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "X1 is empty tensor with zero dimK, which is unsupported.");
      return false;
    });
    OP_API_CHECK((kVal2 == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "X2 is empty tensor with zero dimK, which is unsupported.");
      return false;
    });
    OP_API_CHECK((nVal == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "X2 is empty tensor with zero dimN, which is unsupported.");
      return false;
    });
    return true;
}

// 非量化模式下X支持的FP16数据类型
static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
// 非量化模式下Output支持的FP16数据类型
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
// 非量化模式下910B对Bias数据类型有特殊规范
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

// 非量化场景下，校验所有输入的参数类型是否正确（A5）
static bool CheckAllDtypesValid(const aclTensor* x1, const aclTensor* x2,
                                const aclTensor* biasOptional, const aclTensor* output) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    // biasDtype可以为输入xDtype，也可以为fp32
    if (biasOptional != nullptr) {
        if (biasOptional->GetDataType() != op::DataType::DT_FLOAT && biasOptional->GetDataType() != x1->GetDataType()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MatmulAlltoAll, biasOptional dtype should be x1Dtype or float32 , but it is %s.",
                op::ToString(biasOptional->GetDataType()).GetString());
            return false;
        }
    }
    return true;
}

// 910B在输入为bfloat16时，bias不支持bfloat16，只能为float32
static bool CheckAllDtypesValid910B(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* output) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    if (biasOptional != nullptr) {
        auto biasDtype = biasOptional->GetDataType();
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, BIAS_DTYPE_SUPPORT_LIST, return false);
        if (x1->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(x1, biasOptional, return false);
        } else if (biasDtype != op::DataType::DT_FLOAT){
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The dtype of bias must be [DT_FLOAT] when x's dtype is [DT_BFLOAT16], but get: [%s].",
            op::ToString(biasDtype).GetString());
            return false;
        }
    }
    return true;
}

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor* x1, const aclTensor* x2,
                        const aclTensor* biasOptional, const aclTensor* output) {
    // 输入格式不支持私有格式
    if (IsPrivateFormat(x1->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, x1 format %s does not support private format.",
                op::ToString(x1->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(x2->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, x2 format %s does not support private format.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return false;
    }
    if (biasOptional != nullptr) {
        if (IsPrivateFormat(biasOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, biasOptional format %s does not support private format.",
                op::ToString(biasOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, output format %s does not support private format.",
                op::ToString(output->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor* x1, const aclTensor* x2,
                          const aclTensor* biasOptional, const aclTensor* output) {
    // 内部只处理ND格式，这里做reformat操作
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1 origin format is %s.", op::ToString(x1->GetStorageFormat()).GetString());
        x1 = l0op::ReFormat(x1, op::Format::FORMAT_ND);
        CHECK_RET(x1 != nullptr, false);
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2 origin format is %s.", op::ToString(x2->GetStorageFormat()).GetString());
        x2 = l0op::ReFormat(x2, op::Format::FORMAT_ND);
        CHECK_RET(x2 != nullptr, false);
    }
    if (biasOptional != nullptr) {
        if (biasOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("bias origin format is %s.", op::ToString(biasOptional->GetStorageFormat()).GetString());
            biasOptional = l0op::ReFormat(biasOptional, op::Format::FORMAT_ND);
            CHECK_RET(biasOptional != nullptr, false);
        }
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = l0op::ReFormat(output, op::Format::FORMAT_ND);
        CHECK_RET(output != nullptr, false);
    }
    return true;
}

static aclnnStatus CheckAndHandleParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                        const aclIntArray* alltoAllAxesOptional, const char *group,
                                        bool transposeX1, bool transposeX2, const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查空tensor
    CHECK_RET(CheckNotEmptyTensor(x1, x2, transposeX2), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape
    CHECK_RET(CheckShapeMMAA(x1, x2, biasOptional, transposeX2, output), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    // bias的数据类型限制在950和910B上有所区别，这里根据芯片版本做区分
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        CHECK_RET(CheckAllDtypesValid(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckAllDtypesValid910B(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    }
    // 5. 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    // 6. 兼容性处理非ND格式
    CHECK_RET(ReFormatNotND(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    // 7. 检查alltoallAxes是否为空或者[-1,-2]
    CHECK_RET(CheckAlltoAllAxes(alltoAllAxesOptional, true), ACLNN_ERR_PARAM_INVALID);
    // 8. 检查transposeX1是否合法, 目前不能为true
    CHECK_RET(CheckTransposeX1(transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 9. 检查group长度是否小于等于128
    CHECK_RET(CheckGroupLength(group), ACLNN_ERR_PARAM_INVALID);
    // 如果所有检查都通过，且reformat也通过，输出参数检查成功
    OP_LOGD("aclnnMatmulAlltoAll checkParams success");
    return ACLNN_SUCCESS;
}

static aclnnStatus DealWithEmptyTensor(uint64_t *workspaceSize, aclOpExecutor **executor) {
    OP_LOGD("MatmulAlltoAll, dealing with empty tensor.");
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

// L0层两段式接口Inner，根据算子原型op_graph/matmul_allto_all_proto.h，由模板自动生成。非量化L2层接口和量化L2层接口共用一套L0层接口。
// worldSize为硬件方参数，在aclnn侧不感知。yDtype在aclnn侧不感知。这两个参数需要在Inner接口处声明，在aclnn侧通过默认值传参。
extern "C" aclnnStatus aclnnInnerMatmulAlltoAllGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                                                                const aclTensor* x1ScaleOptional, const aclTensor* x2ScaleOptional,
                                                                const aclTensor* commScaleOptional,
                                                                const aclTensor* x1OffsetOptional, const aclTensor* x2OffsetOptional,
                                                                char* group, int64_t worldSize, const aclIntArray* all2allAxesOptional,
                                                                int64_t yDtype, int64_t x1QuantMode, int64_t x2QuantMode,
                                                                int64_t commQuantMode, int64_t commQuantDtype,
                                                                bool transposeX1, bool transposeX2, int64_t groupSize,
                                                                const aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnInnerMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 非量化L2接口调用L0时需要设置较多默认值，通过InnerMatmulAlltoAllGetWorkspaceSize完成默认值传参和调用L0层接口
extern "C" aclnnStatus InnerMatmulAlltoAllGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                                           const aclIntArray* alltoAllAxesOptional, const char *group,
                                                           bool transposeX1, bool transposeX2, const aclTensor *output,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // ACL和GE的datatype枚举值对undefined定义不同，inner接口进入到算子内部，需要使用GE枚举值，因此此处使用的枚举值为28
    const int64_t GE_UNDEFINED = 28;
    // 根据算子原型定义默认值
    aclTensor* x1ScaleOptional = nullptr;
    aclTensor* x2ScaleOptional = nullptr;
    aclTensor* commScaleOptional = nullptr;
    aclTensor* x1OffsetOptional = nullptr;
    aclTensor* x2OffsetOptional = nullptr;
    const aclTensor* out = output;
    char* str_group = const_cast<char*>(group);
    int64_t worldSize = -1;
    int64_t yDtype = GE_UNDEFINED;
    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    int64_t commQuantMode = 0;
    int64_t commQuantDtype = GE_UNDEFINED;
    int64_t groupSize = 0;
    aclnnStatus ret = aclnnInnerMatmulAlltoAllGetWorkspaceSize(
        x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        str_group, worldSize, alltoAllAxesOptional, yDtype, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype,
        transposeX1, transposeX2, groupSize, out, workspaceSize, executor);
    OP_LOGD("MatmulAlltoAll, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

// 两段式接口
extern "C" aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                                           const aclIntArray* alltoAllAxesOptional, const char *group,
                                                           bool transposeX1, bool transposeX2, const aclTensor *output,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前只有支持转置的x2涉及该处理
    CHECK_RET(CheckX2Valid(x2), ACLNN_ERR_PARAM_INVALID);	// 先检查x2是否合法，避免访问空指针等等非法操作
    bool notContiguous = IsTransposeLastTwoDims(x2);    // notContiguous标识x2是否是非连续的，通常在pytorch经过.t()会导致x2非连续
    auto transX2 = x2;    // 复制一个x2
    if (notContiguous && transposeX2) {    // 当非连续和转置同时生效时，判断为错误用法，直接报错
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 not contiguous, and set x2 transpose, it is error!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {    // 只有当非连续时，才会涉及到转连续等情况
        transposeX2 = !transposeX2;
        // 把非连续x2转成连续
        transX2 = TransX2Tensor(x2);
        CHECK_RET(transX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("X2 is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. After processing, transX2 dim0 is %ld, and dim1 is %ld.",
            x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1), transX2->GetViewShape().GetDim(0), transX2->GetViewShape().GetDim(1));
    }
    aclnnStatus retParam = CheckAndHandleParams(x1, transX2, biasOptional, alltoAllAxesOptional, group, transposeX1, transposeX2, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 处理空tensor，目前非量化matmulalltoall只支持x1第一维度bs为0，空tensor作异常处理
    if (x1->GetViewShape().GetDim(0) == 0 && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return DealWithEmptyTensor(workspaceSize, executor);
    }
    aclnnStatus ret = InnerMatmulAlltoAllGetWorkspaceSize(
        x1, transX2, biasOptional, alltoAllAxesOptional, group, transposeX1, transposeX2, output, workspaceSize, executor);
    OP_LOGD("MatmulAlltoAll, end ret %d", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnMatmulAlltoAllGetWorkspaceSize interface call failed.");
    }
    return ret;
}

extern "C" aclnnStatus aclnnMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else if (GetCurrentPlatformInfo().GetSocVersion()== SocVersion::ASCEND910_93) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B){
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAlltoAll(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnInnerMatmulAlltoAll interface call failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
