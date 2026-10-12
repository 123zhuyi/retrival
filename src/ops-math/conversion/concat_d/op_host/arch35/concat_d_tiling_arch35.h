/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_d_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_D_TILING_AECH35_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_D_TILING_AECH35_H_

#include "register/tilingdata_base.h"
#include "conversion/concat/op_host/arch35/concat_tiling_arch35.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(ConcatD, ConcatTilingData)

REGISTER_TILING_DATA_CLASS(ConcatD_12111, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12112, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12114, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12118, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12121, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12122, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12124, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12128, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12211, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12212, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12214, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12311, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12312, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12314, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12221, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12222, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12224, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_12228, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatD_20001, ConcatTilingDataNoArray)

// compact 版本注册: 万位 2=compact_no_array, 3=compact_array
REGISTER_TILING_DATA_CLASS(ConcatD_22111, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22112, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22114, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22118, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22121, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22122, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22124, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22128, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22211, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22212, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22214, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22311, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22312, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22314, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22221, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22222, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22224, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_22228, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_20003, ConcatTilingDataNoArrayCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32111, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32112, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32114, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32118, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32121, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32122, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32124, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32128, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32211, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32212, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32214, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32221, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32222, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32224, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32228, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32311, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32312, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_32314, ConcatTilingDataCompact)
REGISTER_TILING_DATA_CLASS(ConcatD_20004, ConcatTilingDataCompact)

REGISTER_TILING_DATA_CLASS(ConcatD_30001, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatD_30002, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatD_30004, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatD_30008, ConcatTilingDataForSimt)
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_D_TILING_AECH35_H_
