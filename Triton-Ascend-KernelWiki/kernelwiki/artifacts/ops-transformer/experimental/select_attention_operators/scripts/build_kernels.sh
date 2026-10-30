#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Iterate over all subdirectories in kernels/ and build each
for dir in kernels/*/; do
    echo "Building kernel in: $dir"
    pushd "$dir"
    # Check if build.sh exists and is executable, then run it
    if [ -f "./build.sh" ]; then
        chmod +x ./build.sh  # Ensure it's executable
        ./build.sh
    else
        echo "Warning: build.sh not found in $dir"
    fi
    popd
done