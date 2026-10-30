# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import os
import shutil
import sys
import re
import logging


class OpGenerator:
    """算子工程生成器"""

    def __init__(self, op_type, op_name, output_path, template_variant):
        self.op_type = op_type
        self.op_name = op_name
        self.output_path = output_path
        self.template_name = "add_example"

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        if template_variant == "aicpu":
            self.template_dir = os.path.abspath(
                os.path.join(self.script_dir, "template", "add_example_aicpu")
            )
        else:
            self.template_dir = os.path.abspath(
                os.path.join(self.script_dir, "template", "add_example")
            )

        self.category_dir = os.path.abspath(
            os.path.join(self.output_path, self.op_type)
        )
        self.category_existed = os.path.isdir(self.category_dir)
        self.dest_dir = os.path.abspath(
            os.path.join(self.output_path, self.op_type, self.op_name)
        )

    def run(self):
        """执行生成流程"""
        self._validate_inputs()
        self._copy_template()
        self._rename_files()
        self._replace_content()
        self._update_cmake_chain()
        logging.info(f"成功为 {self.op_type}/{self.op_name} 创建算子工程！")
        logging.info(f"工程路径: {self.dest_dir}")

    def _validate_inputs(self):
        """校验输入参数的有效性和安全性"""
        if not self.op_type or not self.op_name:
            raise ValueError("算子类型和算子名称均不能为空。")

        if not re.match(r"^[a-zA-Z0-9_]+$", self.op_type):
            raise ValueError(
                f"算子类型 '{self.op_type}' 包含无效字符。只允许字母、数字和下划线。"
            )

        if not re.match(r"^[a-zA-Z0-9_]+$", self.op_name):
            raise ValueError(
                f"算子名称 '{self.op_name}' 包含无效字符。只允许字母、数字和下划线。"
            )

        if os.path.exists(self.dest_dir):
            raise FileExistsError(f"目标目录 '{self.dest_dir}' 已存在。")

    def _copy_template(self):
        """复制模板文件到目标目录"""
        logging.info(f"使用模板在 '{self.dest_dir}' 创建算子工程...")
        if not os.path.exists(self.template_dir):
            raise FileNotFoundError(
                f"找不到模板目录 '{self.template_dir}'。请确保模板目录存在。"
            )

        try:
            shutil.copytree(self.template_dir, self.dest_dir)
        except OSError as e:
            raise OSError(f"复制模板文件失败: {e}") from e

    def _rename_files(self):
        """重命名文件和目录中的占位符"""
        for root, dirs, files in os.walk(self.dest_dir, topdown=False):
            for name in files + dirs:
                if self.template_name not in name:
                    continue

                old_path = os.path.join(root, name)
                new_name = name.replace(self.template_name, self.op_name)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                except OSError as e:
                    raise OSError(
                        f"重命名 '{old_path}' 到 '{new_path}' 失败: {e}"
                    ) from e

    @staticmethod
    def _create_category_cmake(dir_path):
        """为新的分类目录创建CMakeLists.txt，使用与math/conversion/random相同的glob模式"""
        cmake_lines = [
            "file(GLOB SUBDIRECTORIES LIST_DIRECTORIES true",
            "    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)",
            "foreach(SUBDIR ${SUBDIRECTORIES})",
            "  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)",
            "    add_subdirectory(${SUBDIR})",
            "  endif()",
            "endforeach()",
        ]
        cmake_file = os.path.join(dir_path, "CMakeLists.txt")
        if not os.path.exists(cmake_file):
            with open(cmake_file, "w", encoding="utf-8") as f:
                f.write("\n".join(cmake_lines) + "\n")
            logging.info(f"Created CMakeLists.txt in {dir_path}")

    @staticmethod
    def _add_to_ops_category_list(category_list_file, child):
        """将新的分类添加到OPS_CATEGORY_LIST中"""
        with open(category_list_file, "r", encoding="utf-8") as f:
            cat_content = f.read()
        updated = re.sub(
            r"(set\s*\(\s*OPS_CATEGORY_LIST\s+[^)]*)", rf'\1\n  "{child}"', cat_content
        )
        with open(category_list_file, "w", encoding="utf-8") as f:
            f.write(updated)
        logging.info(f"Added '{child}' to OPS_CATEGORY_LIST in {category_list_file}")

    @staticmethod
    def _check_included_in_cmake(content, cmake_file, child):
        """检查child是否已被CMakeLists.txt包含，返回(already_included, ops_category_file_path)"""
        already_included = bool(
            re.search(rf"add_subdirectory\s*\(\s*{re.escape(child)}\s*[\s\)]", content)
        )
        ops_category_file_path = None
        if already_included:
            return True, None

        cmake_dir = os.path.dirname(cmake_file)
        files_to_check = [(content, None)]
        for inc_match in re.finditer(r"include\s*\(\s*([^)\s]+)\s*\)", content):
            inc_path = inc_match.group(1)
            if not os.path.isabs(inc_path):
                inc_path = os.path.join(cmake_dir, inc_path)
            if os.path.exists(inc_path):
                try:
                    with open(inc_path, "r", encoding="utf-8") as inc_f:
                        files_to_check.append((inc_f.read(), inc_path))
                except (IOError, OSError):
                    pass

        for check_content, file_path in files_to_check:
            match = re.search(
                r"set\s*\(\s*OPS_CATEGORY_LIST\s+([^)]+)\)", check_content
            )
            if not match:
                continue
            categories = [c.strip('"') for c in match.group(1).split()]
            if child in categories:
                return True, None
            if file_path is not None:
                ops_category_file_path = file_path

        return False, ops_category_file_path

    @staticmethod
    def _append_subdirectory(cmake_file, child):
        """在CMakeLists.txt末尾追加add_subdirectory(child)"""
        with open(cmake_file, "a", encoding="utf-8") as f:
            f.write(f"\nadd_subdirectory({child})\n")
        logging.info(f"Added add_subdirectory({child}) to {cmake_file}")

    def _update_cmake_chain(self):
        """在父目录的CMakeLists.txt中添加add_subdirectory，确保构建系统能找到新目录"""
        current = self.dest_dir
        project_root = os.path.abspath(self.output_path)

        while True:
            parent = os.path.dirname(current)
            child = os.path.basename(current)
            parent_abs = os.path.abspath(parent)

            cmake_file = os.path.join(parent, "CMakeLists.txt")
            if os.path.exists(cmake_file):
                with open(cmake_file, "r", encoding="utf-8") as f:
                    content = f.read()

                already_included, ops_category_file = self._check_included_in_cmake(
                    content, cmake_file, child
                )
                uses_glob = bool(re.search(r"file\s*\(\s*GLOB", content))

                if not already_included and not uses_glob:
                    if ops_category_file:
                        self._add_to_ops_category_list(ops_category_file, child)
                    else:
                        self._append_subdirectory(cmake_file, child)
            else:
                if parent_abs != project_root and os.path.isdir(parent):
                    self._create_category_cmake(parent)

            # An existing category was already connected to its ancestors before
            # this operator was generated. Only its own CMake file may need an
            # update; walking farther would risk adding the category twice when
            # an ancestor includes it indirectly (for example via a CMake list).
            if self.category_existed and parent_abs == self.category_dir:
                break

            if parent_abs == project_root:
                break

            current = parent

    def _replace_content_in_file(self, file_path, replacements):
        """Helper to replace content in a single file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (IOError, OSError) as e:
            logging.warning(f"读取文件 '{file_path}' 失败: {e}")
            return

        original_content = content
        for old, new in replacements.items():
            content = content.replace(old, new)

        if content == original_content:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except (IOError, OSError) as e:
            logging.warning(f"写入文件 '{file_path}' 失败: {e}")

    def _replace_content(self):
        """替换文件内容中的占位符"""
        op_name_capitalized = "".join(
            word.capitalize() for word in self.op_name.split("_")
        )
        template_name_capitalized = "".join(
            word.capitalize() for word in self.template_name.split("_")
        )

        replacements = {
            self.template_name: self.op_name,
            self.template_name.upper(): self.op_name.upper(),
            template_name_capitalized: op_name_capitalized,
            "add_example": self.op_name,
        }
        for root, _, files in os.walk(self.dest_dir):
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    continue

                file_path = os.path.join(root, file)
                self._replace_content_in_file(file_path, replacements)


def execute(args):
    """根据命令行参数执行算子生成"""
    generator = OpGenerator(
        op_type=args.op_type,
        op_name=args.op_name,
        output_path=args.output_path,
        template_variant=args.template_variant,
    )
    generator.run()


def register_parser(subparsers):
    """为 opgen 命令注册解析器。"""
    parser_opgen = subparsers.add_parser("opgen", help="生成项目骨架")
    parser_opgen.add_argument(
        "--op_type", "-t", required=True, help="算子分类，例如 math"
    )
    parser_opgen.add_argument(
        "--op_name", "-n", required=True, help="新算子的名称，例如 asinh"
    )
    parser_opgen.add_argument(
        "--output_path", "-p", default=".", help="生成工程的根路径"
    )
    parser_opgen.add_argument(
        "--template_variant",
        "-v",
        choices=["default", "aicpu"],
        default="default",
        help="选择模板变种",
    )
    parser_opgen.set_defaults(func=execute)


def main():
    """主函数，用于独立执行"""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    parser = argparse.ArgumentParser(description="生成项目骨架")

    parser.add_argument("--op_type", "-t", required=True, help="算子分类，例如 math")
    parser.add_argument(
        "--op_name", "-n", required=True, help="新算子的名称，例如 asinh"
    )
    parser.add_argument("--output_path", "-p", default=".", help="生成工程的根路径")
    parser.add_argument(
        "--template_variant",
        "-v",
        choices=["default", "aicpu"],
        default="default",
        help="选择模板变种",
    )

    args = parser.parse_args()

    try:
        execute(args)
    except Exception as e:
        logging.error(f"发生非预期的错误，退出。错误信息: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
