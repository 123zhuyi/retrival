#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

check_cann_environment() {
    # Check if first argument is "true" for 
    local verbose=false
    if [[ "$1" == "true" ]]; then
        verbose=true
    fi

    # Check if NumCoresMap is declared (indicates init_cann.sh was sourced)
    if ! declare -p NumCoresMap &>/dev/null; then
        echo "Error: NumCoresMap is not declared"
        echo "Please run: source init_cann.sh [SOC_VERSION] first"
        return 1
    fi

    # Check if SOC_VERSION is set and valid
    if [ -z "$SOC_VERSION" ]; then
        echo "Error: SOC_VERSION environment variable is not set"
        echo "Please run: source init_cann.sh [SOC_VERSION]"
        echo "Valid SOC_VERSION values: ${!NumCoresMap[@]}"
        return 1
    fi

    if [[ ! -v NumCoresMap["$SOC_VERSION"] ]]; then
        echo "Error: Invalid SOC_VERSION '$SOC_VERSION'"
        echo "Valid values: ${!NumCoresMap[@]}"
        return 1
    fi

    # Check if NUM_CORES is set
    if [ -z "$NUM_CORES" ]; then
        echo "Error: NUM_CORES environment variable is not set"
        echo "Please run: source init_cann.sh [SOC_VERSION] first"
        return 1
    fi

    if [[ "$verbose" == "true" ]]; then
        echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
        echo "SOC_VERSION=$SOC_VERSION"
        echo "NUM_CORES=$NUM_CORES"
    fi

    return 0
}