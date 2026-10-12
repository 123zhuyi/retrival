/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QUANT_ALL_REDUCE_H
#define QUANT_ALL_REDUCE_H
// <<<>>>调用函数声明
void quant_all_reduce_demo(int8_t type, uint32_t blockDim, void* stream, uint8_t* x, uint8_t* scales, uint8_t* output, 
                           uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM);

#endif //QUANT_ALL_REDUCE_H