# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import sys

def main():
    tiling_data = np.array(
        [0, 0, 9, 0, 0, 9, 124, 1, 3, 1, 63488, 16384, 63488, 32, 9, 53, 19, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1024, 64, 8, 124, 64, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6572, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 124, 64, 8, 1, 1, 32, 64, 8, 0, 0, 0, 0, 0, 53, 0, 19, 0, 32, 0, 1, 1, 512, 8, 1, 0],
                           dtype=np.int32)

    tiling_file = open("tiling.bin", "wb")
    tiling_data.tofile(tiling_file)


if __name__ == '__main__':
    main()