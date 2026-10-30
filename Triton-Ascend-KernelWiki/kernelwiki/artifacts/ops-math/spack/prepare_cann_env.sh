#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -e  # 如果启用，脚本会在任何命令返回非零状态时立即退出
original_umask=$(umask)
umask 0022 # cann-toolkit安装时会检验父目录权限，需要权限为755才可成功安装
restore_umask() {
    echo "$original_umask"
    umask $original_umask
}
# 记录当前脚本的路径，用于后续的相对路径计算
script_path="${BASH_SOURCE[0]}"
script_dir="$(cd "$(dirname "$script_path")" && pwd)"
develop_path="$script_dir/.."

# 包名定义，遵循规范：cann-组件名@版本号 +变体
# 默认值（无参数时使用）
DEFAULT_PACKAGE_NAME="cann-ops-math@master +pkg"
# 最终使用的包名，会在参数解析后确定
PACKAGE_NAME="$DEFAULT_PACKAGE_NAME"

# Spack 安装目录，默认安装在用户家目录下
SPACK_INSTALL_DIR="$HOME"
# Spack 环境名称，可以自定义
ENV_NAME="cann-dev-env"

# repos.yaml 配置文件的路径
REPOS_YAML_DOWNLOAD_PATH="$script_dir/repos.yaml"

# CANN Spack 包仓库地址（请根据实际情况修改为开源仓库地址）
CANN_PACKAGES_REPO_URL="https://gitcode.com/cann/cann-spack-package.git"

# 检查 Git 是否已安装，Spack 依赖于 Git 进行源码管理
check_git() {
    if ! git --version > /dev/null 2>&1; then
        echo "❌ 错误：未检测到 Git 安装"
        echo "   在使用 Spack 之前，请先安装 Git"
        echo "   例如在 Ubuntu/Debian 系统上可以使用：sudo apt-get install git"
        echo "false"
        return
    fi
    echo "✅ Git 已安装"
}

# 检查、下载并激活 Spack
check_and_install_and_activate_spack() {
    local spack_dir="$SPACK_INSTALL_DIR/"
    local setup_script="$spack_dir/spack/share/spack/setup-env.sh"
    
    echo "🔍 检查 Spack 安装状态..."
    
    if [ ! -f "$setup_script" ]; then
        echo "📥 未检测到 Spack，开始下载 Spack v1.1.0..."
        cd "$SPACK_INSTALL_DIR"
        if [ ! -d "$SPACK_INSTALL_DIR/spack" ]; then
            git clone https://gitcode.com/GitHub_Trending/sp/spack.git -b v1.1.0 --depth=2
            if [ $? -ne 0 ]; then
                echo "❌ 错误：Spack 下载失败，请检查网络连接"
                return 1 2>/dev/null || exit 1
            fi
            echo "✅ Spack 下载完成"
        fi
    else
        echo "✅ Spack 已存在，跳过下载"
    fi

    echo "🔧 激活 Spack 环境..."
    . "$setup_script"
    
    if ! spack --version > /dev/null 2>&1; then
        echo "❌ 错误：Spack 激活失败，请检查安装"
        return 1 2>/dev/null || exit 1
    fi
    
    echo "✅ Spack 环境激活成功"
    echo "   Spack 版本：$(spack --version | head -n1)"
}

# 创建 Spack 环境并激活
create_spack_env_and_activate() {
    local spack_dir="$SPACK_INSTALL_DIR/spack"
    local env_name="$ENV_NAME"
    local dir_environment="$spack_dir/var/spack/environments/$env_name"
    local spack_buitin_repo="$SPACK_INSTALL_DIR/.spack/package_repos/ughgqqd/spack-repo-index.yaml"

    echo "🔧 正在创建 Spack 环境：$env_name"
    
    # 如果因网络问题下载仓库失败会导致生成空目录，导致第二次git clone失败，检测仓库目录是否为空，如果为空则删除
    if [ ! -d "$spack_buitin_repo" ]; then
        rm -rf "$SPACK_INSTALL_DIR/.spack/package_repos"
    fi

    # 使用 spack.yaml 配置文件创建环境
    if [ ! -d "$dir_environment" ]; then
        spack env create "$env_name" "$script_dir/spack.yaml"
        if [ $? -ne 0 ]; then
            echo "❌ 错误：创建 Spack 环境失败"
            return 1 2>/dev/null || exit 1
        fi
        echo "✅ Spack 环境创建成功"
    else
        echo "ℹ️  Spack 环境 '$env_name' 已存在，跳过创建"
    fi

    # 激活 Spack 环境
    echo "🔧 正在激活 Spack 环境：$env_name"
    spack env activate "$env_name"
        if [ $? -ne 0 ]; then
            echo "❌ 错误：激活 Spack 环境失败"
            return 1 2>/dev/null || exit 1
        fi
    
    # 查找并配置编译器
    echo "🔧 正在查找并配置可用编译器..."
    spack compiler find
    
    #配置并信任官方证书
    #echo "🔧 正在设置并信任官方二进制包（Buildcache）的 GPG 签名密钥..."   
    #spack buildcache keys --install --trust

    
    echo "✅ Spack 环境 '$env_name' 已激活并准备就绪"
}

# 配置 Spack 的软件包安装目录到用户目录
add_user_spack_config() {
    echo "⚙️  配置 Spack 用户级安装路径..."
    
    CURRENT_USER=$(whoami)
    if [ "$CURRENT_USER" = "root" ]; then
        # 管理员路径（全局共享目录）
        local SPACK_PACKAGES_INSTALL_DIR="/opt/spack"
        echo "⚠️  检测到 root 用户，将使用全局安装目录：$SPACK_PACKAGES_INSTALL_DIR"
    else
        # 普通用户路径（用户家目录）
        local SPACK_PACKAGES_INSTALL_DIR="$HOME/.spack"
        echo "   将使用用户级安装目录：$SPACK_PACKAGES_INSTALL_DIR"
    fi
    
    spack config --scope user add "config:install_tree:root:$SPACK_PACKAGES_INSTALL_DIR"
    echo "✅ Spack 安装路径配置完成"
}

# 配置 Spack 仓库（repos.yaml）
copy_spack_repos_yaml() {
    echo "⚙️  配置 Spack 仓库..."
    
    # 确保 ~/.spack 目录存在
    mkdir -p ~/.spack
    cd ~/.spack

    if [ -f repos.yaml ]; then
        echo "ℹ️  检测到 repos.yaml 已存在，跳过配置"
        return
    fi

    if [ ! -f "$script_dir/repos.yaml" ]; then
        echo "⚠️  警告：未找到 repos.yaml 配置文件"
        echo "   请确保 $script_dir/repos.yaml 文件存在"
        return
    fi

    echo "📄 复制 repos.yaml 到 ~/.spack 目录"
    cp "$script_dir/repos.yaml" .
    
    if ! spack repo list > /dev/null 2>&1; then
        echo "❌ 错误：Spack 仓库配置失败"
        return 1 2>/dev/null || exit 1
    fi
    
    echo "✅ Spack 仓库配置完成"
}

# 下载 CANN Spack 包仓库
download_cann_repo() {
    echo "📥 正在下载 CANN Spack 包仓库..."
    
    cd $SPACK_INSTALL_DIR
    if [ ! -d cann-spack-package ]; then
        echo "   从 $CANN_PACKAGES_REPO_URL 下载仓库..."
        git clone --depth=1 $CANN_PACKAGES_REPO_URL
        if [ $? -ne 0 ]; then
            echo "❌ 错误：CANN Spack 包仓库下载失败"
            echo "   请检查："
            echo "   1. 网络连接"
            echo "   2. 仓库地址是否正确"
            echo "   3. 是否有访问权限"
            return 1 2>/dev/null || exit 1
        fi
        echo "✅ CANN Spack 包仓库下载完成"
    else
        echo "ℹ️  CANN Spack 包仓库已存在，跳过下载"
    fi

    # 将仓库添加到 Spack
    if spack repo list | grep -q "cann-spack-package"; then
        echo "ℹ️  CANN Spack 包仓库已添加，跳过添加"
        return
    fi
    
    echo "🔗 将 CANN 仓库添加到 Spack..."
    spack repo add "$SPACK_INSTALL_DIR/cann-spack-package"
    if [ $? -ne 0 ]; then
        echo "❌ 错误：添加仓库失败"
        return 1 2>/dev/null || exit 1
    fi
    echo "✅ CANN Spack 包仓库添加完成"
}

# 将 Spack 环境配置添加到 .bashrc，实现终端启动时自动加载
add_spack_env_to_bashrc() {
    echo "⚙️  配置终端自动加载 Spack 环境..."
    
    target_line="source $SPACK_INSTALL_DIR/spack/share/spack/setup-env.sh"
    
    # 检查是否已配置
    if ! grep -qF "$target_line" ~/.bashrc; then
        # 如果不存在则追加
        echo "$target_line" >> ~/.bashrc
        echo "✅ 已添加 Spack 环境配置到 ~/.bashrc"
        echo "   下次打开终端时会自动加载 Spack 环境"
    else
        echo "ℹ️  Spack 环境配置已存在，无需添加"
    fi
    
    # 立即生效（仅当前会话）
    echo "🔄 立即应用配置..."
    source ~/.bashrc
    echo "✅ 配置已完成"
    echo "   Spack 安装目录：$SPACK_INSTALL_DIR/spack"
}

# 将本地代码仓库设置为开发分支，并安装所有依赖
pull_develop_branch_local() {
    echo "🔧 设置 $PACKAGE_NAME 为开发模式..."
    
    # 将本地路径设置为开发分支
    spack develop -p $develop_path "$PACKAGE_NAME"
    if [ $? -ne 0 ]; then
        echo "❌ 错误：设置开发模式失败"
        return 1 2>/dev/null || exit 1
    fi
    
    echo "✅ $PACKAGE_NAME 已设置为开发模式"
    echo "   源代码路径：$develop_path"
    
    # 添加包到环境
    echo "📦 添加 $PACKAGE_NAME 到当前环境..."
    spack add "$PACKAGE_NAME"
    
    # 解决依赖关系
    echo "🔍 解析依赖关系..."
    spack concretize -f
    if [ $? -ne 0 ]; then
        echo "❌ 错误：解析依赖关系失败，请确认环境中是否存在依赖版本冲突"
        return 1 2>/dev/null || exit 1
    fi
    # 开始安装
    echo "📥 正在安装（这可能需要一些时间）..."
    spack install
    if [ $? -ne 0 ]; then
        echo "❌ 错误：安装失败，请查看日志确认失败原因"
        return 1 2>/dev/null || exit 1
    fi

    echo "✅ 安装完成！"
    echo ""
    echo "📝 常用命令："
    echo "   查看环境状态：spack find"
    echo "   重新激活环境：spack env activate $ENV_NAME"
    echo "   退出当前环境：spack env deactivate"
}

# 清理 Spack 环境（用于卸载）
environment_clean() {
    echo "🧹 开始清理 Spack 环境..."
    echo "⚠️  警告：此操作将删除所有 Spack 相关文件和配置"
    echo "   包括："
    echo "   - 所有已安装的软件包"
    echo "   - Spack 本身"
    echo "   - 所有配置文件"
    echo ""
    read -p "   确定要继续吗？(y/N): " confirm
    
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "❌ 操作已取消"
        return 0 2>/dev/null || exit 0
    fi
    
    echo "正在清理..."
    
    # 删除 Spack 相关目录
    rm -rf ~/.spack
    rm -rf "$SPACK_INSTALL_DIR/spack"
    rm -rf "$SPACK_INSTALL_DIR/spack-repo"
    rm -rf "$SPACK_INSTALL_DIR/spack-packages-develop"
    rm -rf "$SPACK_INSTALL_DIR/cann-spack-package"
    
    # 显示清理后的目录
    echo "✅ 清理完成，剩余文件："
    ls -al $SPACK_INSTALL_DIR
    
    # 从 .bashrc 中移除 Spack 配置
    sed -i "\|^source $SPACK_INSTALL_DIR/spack/share/spack/setup-env.sh$|d" ~/.bashrc
    echo "✅ 已从 ~/.bashrc 中移除 Spack 配置"
    
    echo ""
    echo "🎉 所有 Spack 相关文件已清理完成！"
    echo "   注意：可能存在环境变量残留，需要重启终端以清理环境变量"
}

# 显示帮助信息
show_help() {
    echo "🔧 CANN 开发环境准备脚本"
    echo ""
    echo "使用方法："
    echo "  $0 [选项] [包名]"
    echo ""
    echo "选项："
    echo "  clean    - 清理并卸载所有 Spack 相关文件和配置"
    echo "  help     - 显示此帮助信息"
    echo ""
    echo "参数："
    echo "  包名     - 指定要使用的 CANN 包名（包含版本和变体）"
    echo "            例如：cann-ops-math@master +pkg +jit"
    echo "            如果不指定，默认使用：$DEFAULT_PACKAGE_NAME"
    echo ""
    echo "功能说明："
    echo "  本脚本将自动完成以下操作："
    echo "  1. 检查并安装 Git（如果未安装）"
    echo "  2. 下载并安装 Spack"
    echo "  3. 配置 Spack 仓库"
    echo "  4. 下载 CANN Spack 包仓库"
    echo "  5. 创建并激活 CANN 开发环境"
    echo "  6. 将 Spack 添加到终端自动加载"
    echo "  7. 设置当前目录为开发分支并安装所有依赖"
    echo ""
    echo "环境变量："
    echo "  ENV_NAME             - Spack 环境名称（默认：cann-dev-env）"
    echo "  SPACK_INSTALL_DIR    - Spack 安装目录（默认：$HOME）"
    echo ""
    echo "示例："
    echo "  # 安装开发环境"
    echo "  $0"
    echo ""
    echo "  # 清理环境"
    echo "  $0 clean"
}

# 主函数
main() {
    echo "========================================="
    echo "🔧 CANN 开发环境配置工具"
    echo "========================================="
    
    # 显示环境信息
    echo "📋 环境配置："
    echo "   - 环境名称：$ENV_NAME"
    echo "   - Spack 目录：$SPACK_INSTALL_DIR"
    echo "   - 包名称：$PACKAGE_NAME"
    echo "========================================="
    
    # 检查 Git
    check_git
    
    # 安装并激活 Spack
    check_and_install_and_activate_spack
    
    # 配置仓库
    copy_spack_repos_yaml
    download_cann_repo
    
    # 配置用户安装路径
    add_user_spack_config
    
    # 创建并激活环境
    create_spack_env_and_activate
    
    # 配置终端自动加载
    add_spack_env_to_bashrc
    
    # 设置开发分支并安装依赖
    pull_develop_branch_local
    
    echo ""
    echo "========================================="
    echo "🎉 CANN 开发环境配置完成！"
    echo "========================================="
    echo ""
    echo "💡 提示：Spack 环境已添加到 .bashrc，下次打开终端会自动加载"
    echo ""
}

# 命令行参数处理
case "$1" in
    clean)
        environment_clean
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        # 无参数：使用默认包名
        PACKAGE_NAME="$DEFAULT_PACKAGE_NAME"
        main
        ;;
    *)
        # 其他参数：视为自定义包名
        PACKAGE_NAME="$1"
        main
        ;;
esac
restore_umask
cd $develop_path