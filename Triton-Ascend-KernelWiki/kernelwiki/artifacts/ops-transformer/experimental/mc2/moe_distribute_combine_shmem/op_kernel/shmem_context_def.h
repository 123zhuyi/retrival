/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "moe_distribute_dispatch_shmem_tiling.h"
#include "shmem.h"



__aicore__ inline int64_t GetShmemDataAddr(__gm__ uint8_t *shmemSpace, int32_t pe) {
    return (int64_t) aclshmem_ptr(shmemSpace, pe);
}

__aicore__ inline int64_t GetShmemSignalAddr(__gm__ uint8_t *shmemSpace, int32_t pe) {
    return (int64_t) aclshmem_ptr(shmemSpace, pe) + 1022 * 1024 * 1024;
}