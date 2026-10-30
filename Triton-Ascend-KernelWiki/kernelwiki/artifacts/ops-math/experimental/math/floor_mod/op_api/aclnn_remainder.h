/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_REMAINDER_H_
#define OP_API_INC_LEVEL2_ACLNN_REMAINDER_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

// tensor self, tensor other
/**
 * @brief aclnnRemainderTensorTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] self: npu
 * device侧的aclTensor，self的数据类型与other的数据类型需满足[数据类型推导规则](#)，并且推导出的数据类型
 * 必须属于INT32、INT64、FLOAT16、FLOAT、DOUBLE类型中的一种。shape需要与other满足[broadcast关系](#)。支持[非连续的Tensor](#)，
 * 数据格式支持ND([参考](#))。
 * @param [in] other: npu
 * device侧的aclTensor。self的数据类型与other的数据类型需满足[数据类型推导规则](#)，并且推导出的数据
 * 类型必须属于INT32、INT64、FLOAT16、FLOAT、DOUBLE类型中的一种。shape需要与self满足[broadcast关系](#)。支持[非连续的Tensor]
 * (#)，数据格式支持ND([参考](#))。
 * @param [in] out: npu device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、
 * COMPLEX64、COMPLEX128类型，且数据类型需要是self与other推导之后可转换的数据类型([参考](#))。shape需要是self与other
 * broadcast之后的shape。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRemainderTensorTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnRemainderTensorTensor的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRemainderTensorTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRemainderTensorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

// tensor self, scalar other
/**
 * @brief aclnnRemainderTensorScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] self: npu device侧的aclTensor，self的数据类型必须属于INT32、INT64、FLOAT16、FLOAT、DOUBLE类型中的一种。
 * 支持[非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [in] other:
 * host侧的aclScalar。self的数据类型与other的数据类型需满足[数据类型推导规则](#)，并且推导出的数据类型必须
 * 属于INT32、INT64、FLOAT16、FLOAT、DOUBLE类型中的一种。shape需要与self满足[broadcast关系](#)。支持[非连续的Tensor](#)，数据
 * 格式支持ND([参考](#))。
 * @param [in] out: npu device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE、
 * COMPLEX64、COMPLEX128类型，且数据类型需要是self可转换的数据类型（[参考](#)）。shape需要与self一致。支持[非连续的Tensor](#)，
 * 数据格式支持ND（[参考](#)。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRemainderTensorScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnRemainderTensorScalar的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRemainderTensorScalarGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRemainderTensorScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

// scalar self, tensor other
/**
 * @brief aclnnRemainderScalarTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] self: host侧的aclScalar，self的数据类型需要能转换成other的数据类型。
 * @param [in] other: npu
 * device侧的aclTensor。other的数据类型必须属于INT32、INT64、FLOAT16、FLOAT、DOUBLE类型中的一种。支持
 * [非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE类型，且数据类型需要与other的数
 * 据类型一致（[参考](#)）。shape需要与other一致。支持[非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(
    const aclScalar* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnRemainderScalarTensor的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRemainderScalarTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRemainderScalarTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

// inplace
// tensor self, tensor other
/**
 * @brief aclnnInplaceRemainderTensorTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] selfRef: npu
 * device侧的aclTensor。selfRef的数据类型与other的数据类型需满足[数据类型推导规则](#)，并且推导出的
 * 数据类型必须属于INT32、INT64、FLOAT16、FLOAT、DOUBLE类型中的一种，且需要是推导之后可转换的数据类型（[参考](#)）。shape需要与
 * other满足[broadcast关系](#)，并且shape与最终broadcast关系的结果一致。支持[非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [in] other: npu
 * device侧的aclTensor。selfRef的数据类型与other的数据类型需满足[数据类型推导规则](#)，并且推导出的数据
 * 类型必须属于INT32、INT64、FLOAT16、FLOAT、DOUBLE类型中的一种。shape需要与selfRef满足[broadcast关系](#)。支持
 * [非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceRemainderTensorTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceRemainderTensorTensor的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口
 * aclnnInplaceRemainderTensorTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceRemainderTensorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

// tensor self, scalar other
/**
 * @brief aclnnInplaceRemainderTensorScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] selfRef: npu
 * device侧的aclTensor。selfRef的数据类型与other的数据类型需满足[数据类型推导规则](#)，并且推导出的数
 * 据类型必须属于INT32、INT64、FLOAT16、FLOAT、DOUBLE类型中的一种，且需要是推导之后可转换的数据类型（[参考](#)）。支持
 * [非连续的Tensor](#)，数据格式支持ND([参考](#))。
 * @param [in] other: host侧的aclScalar。other的数据类型与selfRef的数据类型需满足[数据类型推导规则](#)，并且推导出的数据
 * 类型必须能转换为selfRef的数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceRemainderTensorScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceRemainderTensorScalar的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成remainder操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口
 * aclnnInplaceRemainderTensorScalarGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceRemainderTensorScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_REMAINDER_H_