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
 * \file reduce_mean_with_count_tiling_key.h
 * \brief reduce mean with count tiling key
 */

#ifndef _REDUCE_MEAN_WITH_COUNT_TILING_KEY_H_
#define _REDUCE_MEAN_WITH_COUNT_TILING_KEY_H_

#include "atvoss/reduce/reduce_tiling_key_decl.h"
#include "atvoss/reduce/reduce_tiling_key_decl_non_contiguous.h"
#include "atvoss/reduce/reduce_tiling_key_sel_non_contiguous.h"

ASCENDC_TPL_ARGS_DECL(ReduceMeanWithCount, REDUCE_TPL_KEY_DECL_NON_CONTIGUOUS());

#endif  // _REDUCE_MEAN_WITH_COUNT_TILING_KEY_H_
