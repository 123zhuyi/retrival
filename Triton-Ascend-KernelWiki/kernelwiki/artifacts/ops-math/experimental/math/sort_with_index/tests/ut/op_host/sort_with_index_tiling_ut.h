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
 * \file sort_with_index_tiling_ut.h
 * \brief Local UT copy of SortWithIndex tiling compile info (mirrors
 *        op_host/sort_with_index_tiling_compile_info.h). The UT include path does not expose
 *        each op's op_host directory, so the SortWithIndexCompileInfo declaration
 *        is provided next to the test (same pattern as other experimental ops).
 */

#ifndef SORT_WITH_INDEX_TILING_UT_H
#define SORT_WITH_INDEX_TILING_UT_H

namespace optiling {
struct SortWithIndexCompileInfo {};
} // namespace optiling

#endif // SORT_WITH_INDEX_TILING_UT_H
