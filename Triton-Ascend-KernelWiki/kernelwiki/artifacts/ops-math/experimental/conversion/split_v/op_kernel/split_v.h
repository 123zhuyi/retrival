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
 * \file split_v.h
 * \brief
 */
#ifndef OP_KERNEL_SPLIT_V_H_
#define OP_KERNEL_SPLIT_V_H_

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "split_v_pure_copy.h"
#include "split_v_one_row_pure_copy.h"
#include "split_v_one_outer.h"
#include "split_v_general.h"
#include "split_v_same_len.h"
#include "split_v_same_len_compact.h"
#include "split_v_same_len_compact_double_buffer.h"
#include "split_v_same_len_compact_large_outer.h"
#include "split_v_uneven_compact.h"
#include "split_v_same_len_compact_8bit.h"
#include "split_v_same_len_pure_copy_8bit.h"
#include "split_v_uneven_compact_8bit.h"
#include "split_v_uneven_pure_copy_16bit.h"
#include "split_v_same_len_inner_copy.h"
#include "split_v_uneven_len_inner_copy.h"
#include "split_v_tiling_key.h"

#endif // OP_KERNEL_SPLIT_V_H_
