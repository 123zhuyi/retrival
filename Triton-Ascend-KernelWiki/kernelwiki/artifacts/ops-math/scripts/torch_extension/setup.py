# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import shutil
import subprocess
from setuptools import setup, find_packages, Distribution, Command
from wheel.bdist_wheel import bdist_wheel

# package_name and version
PACKAGE_NAME = "npu_math_extension"
VERSION = "1.0.0"


class CleanCommand(Command):
    """
    usage: python3 setup.py clean
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # 定义要删除的目录列表
        dirs_to_remove = ['build', 'dist']

        # 安全查找 .egg-info 目录
        try:
            for directory_path in os.listdir('.'):
                if (directory_path.endswith('.egg-info') and
                    os.path.isdir(directory_path) and
                    directory_path not in dirs_to_remove):
                    dirs_to_remove.append(directory_path)
        except Exception:
            print("Warning: Error scanning directory for .egg-info folders")

        # 安全删除 .so 文件
        so_file = os.path.join(PACKAGE_NAME, "_C.abi3.so")
        if os.path.exists(so_file):
            try:
                print(f"Removing generated library: {so_file}")
                os.remove(so_file)
            except Exception:
                print(f"Warning: Failed to remove {so_file}")

        # 安全删除目录
        for directory_path in dirs_to_remove:
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                try:
                    print(f"Removing directory: {directory_path}")
                    shutil.rmtree(directory_path)
                except Exception:
                    print(f"Warning: Failed to remove directory {directory_path}")


class BinaryDistribution(Distribution):
    """
    Make this wheel not a pure python package
    """
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


class CMakeBuild(Command):
    """
    CMake config and build
    """
    description = "Build the C++ extension with CMake"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        current_working_directory = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(current_working_directory, "../../"))
        build_dir = os.path.join(project_root, "build")
        print(f"-- Project Root: {project_root}")
        print(f"-- Build Dir:    {build_dir}")
        experimental_option = os.getenv("ENABLE_EXPERIMENTAL")
        parallel_option = os.getenv("THREAD_NUM", 16)

        cmake_config_cmd = [
            "cmake",
            "-S", project_root,
            "-B", build_dir,
            "-DENABLE_TORCH_EXTENSION=ON",
            f"-DENABLE_EXPERIMENTAL={experimental_option}",
            "-G", "Ninja",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            "-DCMAKE_BUILD_TYPE=Release"
        ]

        cmake_build_cmd = [
            "cmake",
            "--build", build_dir,
            "--config", "Release",
            "--target", "_C",
            "--", "-v", f"-j{parallel_option}"
        ]

        print(f"Running CMake Config: {' '.join(cmake_config_cmd)}")
        subprocess.check_call(cmake_config_cmd, cwd=current_working_directory)

        print(f"Running CMake Build: {' '.join(cmake_build_cmd)}")
        subprocess.check_call(cmake_build_cmd, cwd=current_working_directory)


class ABI3Wheel(bdist_wheel):
    """
    Force to use abi3 tag
    """
    def run(self):
        self.run_command('cmake_build')
        super().run()

    def get_tag(self):
        python, abi, plat = super().get_tag()
        if python.startswith("cp"):
            return "cp38", "abi3", plat
        return python, abi, plat


cmdclass = {
    "clean": CleanCommand,
    "bdist_wheel": ABI3Wheel,
    "cmake_build": CMakeBuild
}


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="NPU Math Extension",
    packages=find_packages(),
    package_data={
        PACKAGE_NAME: ["_C.abi3.so"],
    },
    distclass=BinaryDistribution,
    cmdclass=cmdclass,
    zip_safe=False,
    install_requires=[
        "torch",
    ],
    # meta-data
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
)
