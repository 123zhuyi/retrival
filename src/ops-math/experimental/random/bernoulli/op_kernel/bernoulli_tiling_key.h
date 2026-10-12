/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef OPS_RANDOM_BERNOULLI_TILING_KEY_H_
#define OPS_RANDOM_BERNOULLI_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define BERNOULLI_TPL_SCH_MODE_0 0

ASCENDC_TPL_ARGS_DECL(Bernoulli, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, BERNOULLI_TPL_SCH_MODE_0));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, BERNOULLI_TPL_SCH_MODE_0)));

#endif // OPS_RANDOM_BERNOULLI_TILING_KEY_H_
