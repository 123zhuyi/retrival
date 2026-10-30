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

current_dir=$(pwd)
variables_cmake="cmake/variables.cmake"
op_category_list=$(perl -0777 -ne 'if (/set\(OPS_CATEGORY_LIST\s*(.*?)\)/s) { print $1 }' "$current_dir/$variables_cmake" | grep -oP '"[^"]+"' | sed 's/"//g' | xargs)
IFS=' ' read -r -a op_categories <<< "$op_category_list"
valid_dirs=()
SOC_VERSION="ascend910b"
for arg in "$@"; do
    case "$arg" in
        --soc=*) SOC_VERSION="${arg#*=}" ;;
    esac
done
pr_file=$(realpath "${1:-pr_filelist.txt}")

for category in "${op_categories[@]}"
do
    category_path="$current_dir/$category"
    if [ -d "$category_path" ]; then
        for dir in "$category_path"/*/
        do
            dir_name=$(basename "$dir")
            if [[ "$dir_name" != *"common"* ]]; then
                valid_dirs+=("$dir_name")
            fi
        done
    fi 
done
valid_example_dirs=("add_example")
for first_level in "$current_dir"/*/
do
    for second_level in "$first_level"*/
    do
        examples_dir="$second_level""examples"
        if [ -d "$examples_dir" ]; then
            if ls "$examples_dir"/test_aclnn_* 1> /dev/null 2> /dev/null; then
                dir_name=$(basename "$second_level")
                valid_example_dirs+=("$dir_name")
            fi
        fi
    done
done
ops_name=("add_example")
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
    if [[ "$file_path" == "experimental/"* ]]; then
        continue
    fi
    for dir in "${valid_dirs[@]}"
    do
        if [[ "$file_path" == *"/$dir/"* ]]; then
            if [[ ! " ${ops_name[@]} " =~ " $dir " ]]; then
                ops_name+=("$dir")
            fi
            break
        fi
    done
    # #如果file_path为scripts/ci/mirror_update_time.txt，则将准备好的需验证算子列表加到ops_name与ops_name_mirror中
    # 暂无950设备，注释掉，待支持后放开
    # if [[ "$file_path" == "scripts/ci/mirror_update_time.txt" ]]; then

    #     calc_ops_950=("add" "sub" "mul" "div")
    #     operator_list_950=("${calc_ops_950[@]}")

    #     for op in "${operator_list_950[@]}"; do
    #         op_exists=0
    #         for existing_op in "${ops_name[@]}"; do
    #             if [ "$existing_op" = "$op" ]; then
    #                 op_exists=1
    #                 break
    #             fi
    #         done
    #         if [ "$op_exists" -eq 0 ]; then
    #             ops_name+=("$op")
    #         fi
    #     done
    # fi
done

for name in "${ops_name[@]}"
do
    rm -rf build
    rm -rf build_out
    if [[ "${REPOSITORY_NAME}" == "ops-math-dev" ]]; then
        echo "bash build.sh --pkg --vendor_name=$name --ops=$name --soc=${SOC_VERSION} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16"
        bash build.sh --pkg --vendor_name=$name --ops=$name --soc=${SOC_VERSION} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        status=$?
    elif [[ "${REPOSITORY_NAME}" == "ops-math" ]]; then
        echo "bash build.sh --pkg --vendor_name=$name --ops=$name --soc=${SOC_VERSION} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16"
        bash build.sh --pkg --vendor_name=$name --ops=$name --soc=${SOC_VERSION} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        status=$?       
    fi
    if [[ $status -ne 0 ]]; then
        echo "check pkg fail"
        exit 1
    fi
    for dir in "${valid_example_dirs[@]}"
    do
        if [[ "$dir" == "$name" ]]; then
        mkdir -p ${WORKSPACE}/single
        cp build_out/cann-ops-math-${name}_linux*.run ${WORKSPACE}/single/
        fi
    done
done

cd ${WORKSPACE}
tar -zcf single.tar.gz single/
ls