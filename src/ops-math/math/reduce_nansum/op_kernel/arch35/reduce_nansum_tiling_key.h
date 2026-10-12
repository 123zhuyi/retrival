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
 * \file reduce_nansum_tiling_key.h
 * \brief ReduceNansum tiling key for ascend950 (arch35).
 */

#ifndef _REDUCE_NANSUM_TILING_KEY_H_
#define _REDUCE_NANSUM_TILING_KEY_H_

#include "atvoss/reduce/reduce_tiling_key_decl.h"
#include "atvoss/reduce/reduce_tiling_key_sel.h"

ASCENDC_TPL_ARGS_DECL(ReduceNansum, REDUCE_TPL_KEY_DECL());

#endif  // _REDUCE_NANSUM_TILING_KEY_H_