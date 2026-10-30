#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import urllib.request
import subprocess
import os


def down_files_native(url_list):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for url in url_list:
        file_name = url.split('/')[-1]
        
        if not file_name:
            file_name = "downloaded_file"
        
        # 将下载的文件保存到脚本所在目录
        file_path = os.path.join(current_dir, file_name)
        
        urllib.request.urlretrieve(url, file_path)

def git_download(url_list):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for url in url_list:
        file_name = url.split('/')[-1]
        file_name = file_name.rsplit(".git", 1)[0]
        if not file_name:
            file_name = "downloaded_file"
        file_path = os.path.join(current_dir, file_name)

        result = subprocess.run(
            ["git", "clone", url, file_path],
            capture_output = True
        )
        if result.returncode != 0:
            print(f"Failed to clone {url}, {result.stderr}")


if __name__ == "__main__":
    my_urls = [
        "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.11.3.tar.gz",
        ("https://gitcode.com/cann-src-third-party/makeself/releases/download/"
        "release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz"),
        "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz",
        "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz",
        ("https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/"
        "20230802.1/abseil-cpp-20230802.1.tar.gz"),
        "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-016.tar.gz"
    ]
    
    down_files_native(my_urls)

    my_git_urls = [
        "https://gitcode.com/cann/opbase.git"
    ]

    git_download(my_git_urls)