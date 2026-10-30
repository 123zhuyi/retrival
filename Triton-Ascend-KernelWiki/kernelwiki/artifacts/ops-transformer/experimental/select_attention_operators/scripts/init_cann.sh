#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# This script sets up the required paths in order to use CANN libraries
# Usage: source init_cann.sh [SOC_VERSION]

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# number of davinci cores in a single Ascend NPU device
source $script_dir/num_cores_map.sh
if [ $# -ne 1 ]; then
    echo "Usage: source $0 [SOC_VERSION]"
    echo "Valid SOC_VERSION values: ${!NumCoresMap[@]}"
    return 1 2>/dev/null || exit 1
fi

SOC_VERSION=$1

# Validate SOC_VERSION input
if [[ ! -v NumCoresMap["$SOC_VERSION"] ]]; then
    echo "Error: Invalid SOC_VERSION '$SOC_VERSION'"
    echo "Valid values: ${!NumCoresMap[@]}"
    return 1 2>/dev/null || exit 1
fi

conda activate sa
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# Export SOC_VERSION and NUM_CORES environment variables
export SOC_VERSION="$SOC_VERSION"
export NUM_CORES="${NumCoresMap[$SOC_VERSION]}"

# Verify that everything is correctly set:
source $script_dir/check_cann.sh
check_cann_environment true

