/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SET_PYTORCH_RANDOM_H_
#define OP_API_INC_LEVEL2_ACLNN_SET_PYTORCH_RANDOM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 设置随机数精度对标模式，供PyTorch Ascend等上层框架调用。
 *        0: 生成随机数兼容A2/A3A4；1: 对标pytorch（默认，对齐H20）。
 * @param pytorchRandom 精度对标模式，取值0或1。
 * @return aclnnStatus: 成功返回ACLNN_SUCCESS，失败返回对应错误码。
 */
ACLNN_API aclnnStatus aclnnSetPytorchRandom(int32_t pytorchRandom);

/**
 * @brief 获取随机数精度对标模式，供其他aclnn接口查询当前配置。
 * @return 当前精度对标模式（0或1）。
 */
ACLNN_API int32_t aclnnGetPytorchRandom();

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_SET_PYTORCH_RANDOM_H_
