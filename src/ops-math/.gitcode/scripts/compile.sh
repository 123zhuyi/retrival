#!/bin/bash

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------s.

set -e
set -o pipefail

echo "ge_st_rt2: ${ge_st_rt2}"
echo "GIT_TARGET_BRANCH: ${GIT_TARGET_BRANCH}"
echo "OS_TYPE: ${OS_TYPE}"
echo "task_name: ${task_name}"

########
# Init #
########

function LOG_HEAD() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[INFO] ${date_time} ${assert_msg}"
}

function LOG_DO() {
   local cmd="$*"
   date_time=$(date +%Y%m%d-%H%M%S)
   echo -e "[Command] ${date_time} ${cmd}$"
   ${cmd}
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    if [ "${actual_value}" != "${expect_value}" ]; then
        echo "${assert_msg} is failed."
        exit 1
    else
        echo "${assert_msg} is success."
    fi
}

export REPOSITORY_NAME="ops-math"
rm -rf /home/jenkins/opensource/json
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
if [[ "${task_name}" == *ubuntu24* ]]; then
    if [ "${GIT_TARGET_BRANCH}" == "master" ]; then
        sudo update-alternatives --set gcc /usr/bin/gcc-15
    else
        sudo update-alternatives --set gcc /usr/bin/gcc-14
    fi
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi
if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d /home/jenkins/opensource/gcc15 ]; then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
    elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
    fi
elif gcc --version | head -n1 | grep -q "14\."; then
    gcc --version
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi
gcc --version

if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi

if [ -z "${OS_TYPE}" ]; then
    OS_TYPE=$(uname -m)
fi

if [ "${ge_st_rt2}X" != "kirinx90X" ];then
    whoami
    su - jenkins -c "sh /home/jenkins/Ascend/cann/share/info/ops_math/script/uninstall.sh"
fi

if [ -f /home/jenkins/Ascend/cann/bin/setenv.bash ]; then
    source /home/jenkins/Ascend/cann/bin/setenv.bash
fi

#########
# Build #
#########
cd ${WORKSPACE}/ || exit

if [[ "${task_name}" =~ Compile_Ascend_X86_ubuntu24 ]]; then
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi

if  [ "${ge_st_rt2}X" == "experimentalX" ];then
    if [ "${GIT_TARGET_BRANCH}" = "master" ];then
        LOG_DO bash build.sh --experimental --pkg -f "pr_filelist.txt" --cann_3rd_lib_path="/home/jenkins/opensource" -j16
        DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
    else
        echo "not need build A5"
        mkdir build_out
        touch build_out/cann-ops-math-experimental_linux-${OS_TYPE}.run
        exit 0
    fi
elif [ "${ge_st_rt2}X" == "kirinx90X" ];then
    if [ "${GIT_TARGET_BRANCH}" = "master" ];then
        LOG_DO bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
    else
        echo "not need build mobile_station"
        mkdir build_out
        touch build_out/cann-ops-math-kirinx90_linux-x86_64.run
        exit 0
    fi
elif [ "${ge_st_rt2}X" == "kirin9030X" ];then
    if [ "${GIT_TARGET_BRANCH}" = "master" ];then
        LOG_DO bash build.sh --pkg --soc=kirin9030 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
    else
        echo "not need build mobile_station"
        mkdir build_out
        touch build_out/cann-ops-math-kirin9030_linux-x86_64.run
        exit 0
    fi
elif [ "${ge_st_rt2}X" == "singleX" ];then
    if [ "${GIT_TARGET_BRANCH}" = "master" ];then
        LOG_DO bash scripts/ci/check_pkg.sh "pr_filelist.txt" "${REPOSITORY_NAME}" "-j16"
        DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        exit 0
    else
        echo "not need build single"
        touch single.tar.gz
        exit 0
    fi
elif [ "${ge_st_rt2}X" == "A5X" ];then
    if [ "${GIT_TARGET_BRANCH}" = "master" ];then
        LOG_DO bash scripts/ci/compile_a5_pkg.sh "pr_filelist.txt" "-j16"
        DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        # 目录不存在 或者 目录内无 .run 文件时，生成空占位文件
        compile_package_name=""
        build_dir="${WORKSPACE}/build_out"

        # 目录存在才执行ls检索文件
        if [[ -d "${build_dir}" ]]; then
            compile_package_name=$(ls "${build_dir}/" 2>/dev/null | grep -E "\.run$" | head -n1)
        fi

        if [[ -z "${compile_package_name}" ]]; then
            echo "not need build 950"
            mkdir -p build_out
            touch build_out/cann-ops-math-950_linux-x86_64.run
        fi
        exit 0
    else
        echo "not need build A5"
        mkdir -p ${WORKSPACE}/build_out
        touch ${WORKSPACE}/build_out/cann-ops-math-custom_linux-x86_64.run
        exit 0
    fi
elif [[ "${task_name}" =~ monitor ]];then
    if [ "${GIT_TARGET_BRANCH}" = "master" ];then
        if [[ "${task_name}" =~ "910c" ]];then
            LOG_DO bash build.sh --pkg --jit --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend910_93
            DP_ASSERT_EQUAL "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910_93]"
        elif [[ "${task_name}" =~ "950" ]];then
            LOG_DO bash build.sh --pkg --jit --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend950
            DP_ASSERT_EQUAL "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend950]"
        else
            LOG_DO bash build.sh --pkg --jit --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend910b
            DP_ASSERT_EQUAL "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910b]"
        fi
    else
        echo "not need build monitor"
        mkdir build_out
        touch build_out/cann-ops-math_linux-x86_64.run
        exit 0
    fi
else
    LOG_DO bash build.sh --pkg --jit --cann_3rd_lib_path="/home/jenkins/opensource" -j16
    DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
fi
exit 0
