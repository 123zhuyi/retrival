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
 * \file truncate_div_tiling_data.h
 * \brief TruncateDiv tiling data struct + schMode (tiling key) 声明。
 *
 * schMode 编码 (x1, x2, y) dtype 组合，索引与 op_host def 的 DataType 列表一致：
 *   0: bf16 / bf16 / bf16      1: half / half / half
 *   2: half / float / float    3: float / half / float
 *   4: float / float / float   5: float / int32 / float
 *   6: int32 / int32 / int32   7: int32 / float / float
 *   8: uint8 / uint8 / uint8   9: int8 / int8 / int8
 *  10: int64 / int64 / int64  11: int16 / int16 / int16
 */
#ifndef _TRUNCATEDIV_TILING_DATA_H_
#define _TRUNCATEDIV_TILING_DATA_H_

#include <cstdint>

#ifndef TRUNCATEDIV_TPL_SCH_MODE_0
#define TRUNCATEDIV_TPL_SCH_MODE_0 0
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_1
#define TRUNCATEDIV_TPL_SCH_MODE_1 1
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_2
#define TRUNCATEDIV_TPL_SCH_MODE_2 2
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_3
#define TRUNCATEDIV_TPL_SCH_MODE_3 3
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_4
#define TRUNCATEDIV_TPL_SCH_MODE_4 4
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_5
#define TRUNCATEDIV_TPL_SCH_MODE_5 5
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_6
#define TRUNCATEDIV_TPL_SCH_MODE_6 6
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_7
#define TRUNCATEDIV_TPL_SCH_MODE_7 7
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_8
#define TRUNCATEDIV_TPL_SCH_MODE_8 8
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_9
#define TRUNCATEDIV_TPL_SCH_MODE_9 9
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_10
#define TRUNCATEDIV_TPL_SCH_MODE_10 10
#endif
#ifndef TRUNCATEDIV_TPL_SCH_MODE_11
#define TRUNCATEDIV_TPL_SCH_MODE_11 11
#endif

// element-wise 二元算子：无 reduction / 无 workspace。所有 *Length 均为元素个数。
struct TruncateDivTilingData {
    uint64_t coreNum = 1;
    uint64_t totalLength = 0;
    uint64_t coreLength = 0; // 每核处理的元素数（末核 = totalLength - (coreNum-1)*coreLength）
    uint64_t tileLength = 0; // 单 tile 最大元素数（64 的倍数）
};
#endif
