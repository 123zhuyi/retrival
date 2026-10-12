/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_H
#define CHECKER_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

// 量化与非量化共用的方法和常量、枚举值
namespace matmul_allto_all_check {

// 需要使用的常量定义
constexpr int64_t NEG_ONE = -1;
constexpr int64_t NEG_TWO = -2;
constexpr int64_t ZERO = 0;
constexpr int64_t TWO = 2;
constexpr size_t ONE_DIM = 1;
constexpr size_t TWO_DIMS = 2U;
constexpr size_t MAX_GROUP_LEN = 128U;
constexpr int64_t KVALUE_MIN = 1;
constexpr int64_t KVALUE_MAX = 65535;
constexpr int64_t MX_QUANT_GROUP_SIZE = 4295032864;

// 通信域相关枚举值
enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
// 量化相关枚举值
enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
    DYN_PERTOKEN_QUANT = 7
};

// 校验AlltoAll和Permute数据交换的方向参数, 在alltoallmatmul中可以为空和{-1,-2}, 在matmulalltoall中可以为空和{-2,-1}, 不允许为其他值
bool CheckAlltoAllAxes(const aclIntArray* alltoAllAxesOptional, bool isMatmulAlltoAll);

// 校验输入的转置配置，x1不允许转置
bool CheckTransposeX1(bool transposeX1);

// 校验通信域名的字符串长度是否符合要求
bool CheckGroupLength(const char *group);

// 校验MatmulAlltoAll和QuantMatmulAlltoAll输入属性shape
bool CheckShapeMMAA(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, bool transposeX2, const aclTensor* output);
// 校验AlltoAllMatmul和AlltoAllQuantMatmul输入属性shape
bool CheckShapeAAMM(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, bool transposeX2, const aclTensor* output, const aclTensor* alltoAllOutOptional);

// 校验groupSize是否合法，仅在组量化（MX）场景下需要取值，其它场景默认为0
bool CheckGroupSizeValid(int64_t groupSize, int64_t x1QuantMode, int64_t x2QuantMode);

// 检查tensor是否连续
bool IsTransposeLastTwoDims(const aclTensor *tensor);

// 检查x2是否合法，空指针、空tensor和维度
bool CheckX2Valid(const aclTensor* x2);

// 检查是否有alltoallout输出
bool IsAll2AllOut(const aclTensor *alltoAllOut);

// 检查预留参数是否为合法值，若为预设之外的值会提示Warning
bool CheckReservedParams(const aclTensor *commScaleOptional, const aclTensor* x1OffsetOptional,
                         const aclTensor* x2OffsetOptional, int64_t commQuantMode, int64_t commQuantDtype);

// 处理支持转置的tensor物理排布不连续问题
aclTensor *TransX2Tensor(const aclTensor *x2);
} // namespace matmul_allto_all_check

#endif //CHECKER_H
