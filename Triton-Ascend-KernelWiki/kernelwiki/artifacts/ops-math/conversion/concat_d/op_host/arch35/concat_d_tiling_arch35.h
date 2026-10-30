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

REGISTER_TILING_DATA_CLASS(ConcatD_30001, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatD_30002, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatD_30004, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatD_30008, ConcatTilingDataForSimt)
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_D_TILING_AECH35_H_
