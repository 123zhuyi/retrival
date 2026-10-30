#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -euo pipefail

current_dir=$(pwd)
variables_cmake="cmake/variables.cmake"
op_category_list=$(perl -0777 -ne 'if (/set\(OPS_CATEGORY_LIST\s*(.*?)\)/s) { print $1 }' "$current_dir/$variables_cmake" | grep -oP '"[^"]+"' | sed 's/"//g' | xargs)
IFS=' ' read -r -a op_categories <<< "$op_category_list"
builtin_dirs=()
experimental_dirs=()
pr_file=$(realpath "${1:-pr_filelist.txt}")

for category in "${op_categories[@]}"
do
    builtin_category_path="$current_dir/$category"
    if [ -d "$builtin_category_path" ]; then
        for dir in "$builtin_category_path"/*/
        do
            dir_name=$(basename "$dir")
            if [[ "$dir_name" != *"common"* ]]; then
                builtin_dirs+=("$dir_name")
            fi
        done
    fi
    exper_category_path="$current_dir/experimental/$category"
    if [ -d "$exper_category_path" ]; then
        for dir in "$exper_category_path"/*/
        do
            dir_name=$(basename "$dir")
            if [[ "$dir_name" != *"common"* ]]; then
                experimental_dirs+=("$dir_name")
            fi
        done
    fi
done

builtin_ops_name=()
experimental_ops_name=()
build_all=${build_all:-0}

mapfile -t lines < ${pr_file}
for file_path in "${lines[@]}"
do
    file_path=$(echo "$file_path" | xargs)
    if [ -z "$file_path" ]; then
        continue
    fi
    if [[ "$file_path" == *.md ]]; then
        continue
    fi

    if [ -f "$current_dir/$file_path" ]; then
        if [[ ! "$file_path" == "experimental/"* ]]; then
            for dir in "${builtin_dirs[@]}"
            do
                if [[ "$file_path" == *"/$dir/"*"/arch35/"* ]]; then
                    if [[ ! " ${builtin_ops_name[@]} " =~ " $dir " ]]; then
                        builtin_ops_name+=("$dir")
                    fi
                    break
                fi
            done
        fi

    if [[ ! "$file_path" == "experimental/"* ]]; then
        for dir in "${builtin_dirs[@]}"
        do
            if [[ "$file_path" == *"/$dir/"*"/arch35/"* ]]; then
                if [[ ! " ${builtin_ops_name[@]} " =~ " $dir " ]]; then
                    builtin_ops_name+=("$dir")
                fi
                break
            fi
        done
    fi

        for dir in "${experimental_dirs[@]}"
        do
            if [[ "$file_path" == "experimental/"*"/$dir/"*"/arch35/"* ]]; then
                if [[ ! " ${experimental_ops_name[@]} " =~ " $dir " ]]; then
                    experimental_ops_name+=("$dir")
                fi
                break
            fi
        done
    fi
    if [[ "$file_path" == "common/"* || "$file_path" == "cmake/"* ]]; then
        build_all=1
    fi
done

echo "related op: ${builtin_ops_name[*]}"
echo "related experimental op: ${experimental_ops_name[*]}"
echo "need build all: ${build_all}"
run_build_command() {
    local cmd=$1
    echo "$cmd"
    if ! eval "$cmd"; then
        echo "build pkg error"
        exit 1
    fi
}

execute_run_file() {
    local pkg_type=$1
    local run_files=(./build_out/*.run)
    if [ ! -f "${run_files[0]}" ];then
        echo "no run pkg found"
        return 1
    fi
    chmod +x "${run_files[0]}"
    cmd=""${run_files[0]}" --install-path="/tmp" --force"
    if [ $pkg_type == "builtin" ];then
        cmd+=" --full"
    fi
    echo $cmd
    if ! eval "$cmd"; then
        echo "execute pkg error"
        exit 1
    fi
}

if [ ${#builtin_ops_name[@]} -gt 0 ]; then
    builtin_ops_str=$(IFS=,; echo "${builtin_ops_name[*]}")
    build_cmd="bash build.sh --pkg --ops=${builtin_ops_str} --soc=ascend950 -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}"
    run_build_command "$build_cmd"
    execute_run_file "custom"
fi

if [ ${#experimental_ops_name[@]} -gt 0 ]; then
    experimental_ops_str=$(IFS=,; echo "${experimental_ops_name[*]}")
    build_cmd="bash build.sh --pkg --experimental --ops=${experimental_ops_str} --soc=ascend950 -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}"
    run_build_command "$build_cmd"
    execute_run_file "custom"
fi

if [ ${build_all} -eq 1 ]; then
    build_cmd="bash build.sh --pkg --jit --soc=ascend950 -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}"
    run_build_command "$build_cmd"
    execute_run_file "builtin"
fi

exit 0
