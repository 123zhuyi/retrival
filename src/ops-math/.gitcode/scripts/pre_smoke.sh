#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e
set -o pipefail

echo "single_tar_url=${single_tar_url}"
echo "smoke_run_file_url=${smoke_run_file_url}"
echo "obs_smoke_path=${obs_smoke_path}"

shopt -s extglob

cd ${WORKSPACE}

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 下载并解压 single.tar.gz
# ==============================
log "start run test case, please wait ..."

DOWNLOAD_FILE=$(basename "${single_tar_url}")
echo "Starting to download file: ${DOWNLOAD_FILE}"
wget -nv --no-clobber "${single_tar_url}"

if [ ! -f "${DOWNLOAD_FILE}" ]; then
    echo "File ${DOWNLOAD_FILE} does not exist, no need to execute smoke test task"
    exit 0
fi

FILE_SIZE=$(stat -c%s "${DOWNLOAD_FILE}" 2>/dev/null || echo 0)
if [ "${FILE_SIZE}" -eq 0 ]; then
    echo "No compiled operators, no need to execute smoke test task"
    rm -f "${DOWNLOAD_FILE}"
    exit 0
fi
echo "File download completed, size ${FILE_SIZE}, starting decompression."

tar -ztf "${DOWNLOAD_FILE}"
tar -zxf "${DOWNLOAD_FILE}"

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

rm -rf /home/jenkins/opensource/json
source /usr/local/Ascend/cann/set_env.sh
# ==============================
# 运行测试主循环
# ==============================

bash ${WORKSPACE}/scripts/ci/check_example.sh ${WORKSPACE}/pr_filelist.txt  2>&1 | tee -a ./run_test.log
arm_package=$(basename "${smoke_run_file_url}")
# "cann-ops-math_experimental_linux-aarch64_ubuntu24.run"
wget -nv ${smoke_run_file_url}
chmod +x ${arm_package}
echo 'y' | bash ${arm_package} --quiet
source /usr/local/Ascend/cann/set_env.sh
bash build.sh -f ${WORKSPACE}/pr_filelist.txt --experimental --run_example 2>&1 | tee -a ./run_test.log

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log
obs_key=${obs_smoke_path}/plog/${slog_name}
# upload plog
if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${obs_key}"; then
  echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_key}"
fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log
if grep "dcmi module initialize failed" "./npu_log/npu_info.log";then
  date_time=$(date '+%Y%m%d.%H%M%S')
  echo "$date_time : ${repo_name}_${pr_id} dcmi module initialize failed" >> "./npu_log/$(date +%Y%m%d).log"
fi

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=$(date '+%Y%m%d.%H%M%S')
if grep -w -e "example fail" -e "execute samples failed" "./run_test.log"; then
  echo "$date_time : run test case failed"
  exit 1
else
  echo "$date_time : run test case success"
fi
