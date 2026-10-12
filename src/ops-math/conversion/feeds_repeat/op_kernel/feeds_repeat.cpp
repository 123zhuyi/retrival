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
 * \file feeds_repeat.cpp
 * \brief
 */
#include "feeds_repeat.h"
using namespace AscendC;
using namespace FeedsRepeat;
extern "C" __global__ __aicore__ void feeds_repeat(GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if(TILING_KEY_IS(1)){ //<fp32, int32>
        FeedsRepeatND<float, int32_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(2)){ //<fp16, int32>
        FeedsRepeatND<half, int32_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    else if(TILING_KEY_IS(3)){ //<bf16, int32>
        FeedsRepeatND<bfloat16_t, int32_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
#endif
    else if(TILING_KEY_IS(101)){ //<fp32, int64>
        FeedsRepeatND<float, int64_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(102)){ //<fp16, int64>
        FeedsRepeatND<half, int64_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    else if(TILING_KEY_IS(103)){ //<bf16, int64>
        FeedsRepeatND<bfloat16_t, int64_t> op;
        op.Init(feeds, feeds_repeat_times, y, &tiling_data);
        op.Process();
    }
#endif
}

