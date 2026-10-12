# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .moe_distribute_dispatch_v2 import npu_moe_distribute_dispatch_v2
from .moe_distribute_combine_v2 import npu_moe_distribute_combine_v2
from .moe_distribute_dispatch_v3 import npu_moe_distribute_dispatch_v3
from .moe_distribute_combine_v3 import npu_moe_distribute_combine_v3
from .deep_ep import MoeDistributeBuffer
from .graph_convert.graph_convert_moe_distribute_dispatch_v3 import converter_moe_distribute_dispatch_v3
from .graph_convert.graph_convert_moe_distribute_combine_v3 import convert_npu_moe_distribute_combine_v3