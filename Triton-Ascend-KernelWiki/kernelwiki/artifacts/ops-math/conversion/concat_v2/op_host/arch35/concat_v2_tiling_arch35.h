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
 * \file concat_v2_tiling_arch35.h
 * \brief concat_v2_tiling_arch35
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_CONCAT_V2_TILING_ARCH35_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_CONCAT_V2_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "conversion/concat/op_host/arch35/concat_tiling_arch35.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(ConcatV2, ConcatTilingData)

REGISTER_TILING_DATA_CLASS(ConcatV2_12111, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12112, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12114, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12118, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12121, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12122, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12124, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12128, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12211, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12212, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12214, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12311, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12312, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12314, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12221, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12222, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12224, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_12228, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(ConcatV2_20001, ConcatTilingDataNoArray)

REGISTER_TILING_DATA_CLASS(ConcatV2_30001, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatV2_30002, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatV2_30004, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(ConcatV2_30008, ConcatTilingDataForSimt)
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_CONCAT_V2_TILING_ARCH35_H