#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
# 

# !
#  \file dump_analysis.sh
#  \brief


export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python3
chmod 777 *

# 文件运行使用方法 #####################
# 使用方法
# bash dump_analysis.sh TARGET_DIR=xxx TOOL_PATH=xxx SOC_VERSION=xxx SP_MOE_NUM=x TP_WORLDSIZE=x SHARE_EXPERT_CARD_COUNT=x SHARE_EXPERT_NUM=x ---正常调用"
# bash dump_analysis.sh -h -----查看参数列表"

function help {
    echo "执行方法"
    echo "bash dump_analysis.sh TARGET_DIR=xxx TOOL_PATH=xxx SOC_VERSION=xxx SP_MOE_NUM=x TP_WORLDSIZE=x SHARE_EXPERT_CARD_COUNT=x SHARE_EXPERT_NUM=x ---正常调用"
    echo "bash dump_analysis.sh -h -----查看参数列表"
    echo "参数列表"
    echo "TARGET_DIR(必填):指向dump数据的路径,例:xxx/xxx/data-dump/"
    echo "TOOL_PATH(必填):指向装包路径下tools目录所在的路径,例:xxx/xxx/pkg/8cann-8.x.0/"
    echo "SOC_VERSION(必填):所需要分析的数据的芯片版本,910_93 or 950"
    echo "SP_MOE_NUM(选填):所需要分析的数据使用的特殊专家数(SP_MOE_NUM>0),不填默认为0"
    echo "TP_WORLDSIZE(选填):所需要分析的数据使用的TP_WORLDSIZE(TP_WORLDSIZE>1),不填默认为1"
    echo "SHARE_EXPERT_CARD_COUNT(选填):所需要分析的数据输入的共享专家卡数(SHARE_EXPERT_CARD_COUNT>0),不填默认为0"
    echo "SHARE_EXPERT_NUM(选填):所需要分析的数据输入的共享专家数(SHARE_EXPERT_NUM>0),不填默认为0"
    exit 0
}
#获取sh脚本的文件路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOC_VERSION_950="950"
SOC_VERSION_910_93="910_93"

#入参
for arg in "$@"; do
    if [[ "$arg" = "-h" || "$arg" = "-help" ]]; then
        help
    fi
    if [[ "$arg" == TARGET_DIR=* ]]; then
        TARGET_DIR="${arg#*=}"
    fi
    if [[ "$arg" == TOOL_PATH=* ]]; then
        TOOL_PATH="${arg#*=}"
    fi
    if [[ "$arg" == SOC_VERSION=* ]]; then
        SOC_VERSION="${arg#*=}"
    fi
    if [[ "$arg" == SP_MOE_NUM=* ]]; then
        SP_MOE_NUM="${arg#*=}"
    fi
    if [[ "$arg" == TP_WORLDSIZE=* ]]; then
        TP_WORLDSIZE="${arg#*=}"
    fi
    if [[ "$arg" == SHARE_EXPERT_CARD_COUNT=* ]]; then
        SHARE_EXPERT_CARD_COUNT="${arg#*=}"
    fi
    if [[ "$arg" == SHARE_EXPERT_NUM=* ]]; then
        SHARE_EXPERT_NUM="${arg#*=}"
    fi
    if ! [[ "$arg" =~ ^(-h|-help|TARGET_DIR=|TOOL_PATH=|SP_MOE_NUM=|TP_WORLDSIZE=|SOC_VERSION=|SHARE_EXPERT_CARD_COUNT=|SHARE_EXPERT_NUM=) ]]; then
        echo "warning: 未知参数 $arg ,使用 -h or -help 查看帮助"
    fi
done
#判断入参
judge=0
#判断TARGET_DIR
if [ ! -n "$TARGET_DIR" ]; then
    echo "error:TARGET_DIR undefind"
    judge=1
fi
if [ ! -d "$TARGET_DIR" ]; then
    echo "error: unfind $TARGET_DIR"
    judge=1
fi
#判断TOOL_PATH
if [ ! -n "$TOOL_PATH" ]; then
    echo "error:TOOL_PATH undefind"
    judge=1
fi
if [ ! -e "$TOOL_PATH/tools/msaicerr/msaicerr.py" ]; then
    echo "error: unfind $TOOL_PATH/tools/msaicerr/msaicerr.py"
    judge=1
fi
#判断SOC_VERSION
if [ ! -n "$SOC_VERSION" ]; then
    echo "error:SOC_VERSION undefind"
    judge=1
fi

if [ "$SOC_VERSION" != "$SOC_VERSION_910_93" ] && [ "$SOC_VERSION" != "$SOC_VERSION_950" ]; then
    echo "error:SOC_VERSION:$SOC_VERSION 为非法输入"
    judge=1
fi
#判断SP_MOE_NUM,TP_WORLDSIZE
if [ ! -n "$SP_MOE_NUM" ]; then
    SP_MOE_NUM=0
    echo "warning:SP_MOE_NUM undefind,使用默认值 SP_MOE_NUM  = 0"
fi
if [ "$SP_MOE_NUM" -lt 0 ]; then
    echo "error:SP_MOE_NUM:$SP_MOE_NUM should > 0"
    judge=1
fi
if [ ! -n "$TP_WORLDSIZE" ]; then
    TP_WORLDSIZE=1
    echo "warning:TP_WORLDSIZE undefind,使用默认值 TP_WORLDSIZE  = 0"
fi
if [ "$TP_WORLDSIZE" -lt 1 ]; then
    echo "error:TP_WORLDSIZE:$TP_WORLDSIZE should > 1"
    judge=1
fi
#判断共享专家卡数,共享专家数
if [ ! -n "$SHARE_EXPERT_CARD_COUNT" ]; then
    SHARE_EXPERT_CARD_COUNT=0
    echo "warning:SHARE_EXPERT_CARD_COUNT undefind,使用默认值 SHARE_EXPERT_CARD_COUNT  = 0"
fi
if [ "$SHARE_EXPERT_CARD_COUNT" -lt 0 ]; then
    echo "error:SHARE_EXPERT_CARD_COUNT:$SHARE_EXPERT_CARD_COUNT should > 0"
    judge=1
fi
if [ ! -n "$SHARE_EXPERT_NUM" ]; then
    echo "warning:SHARE_EXPERT_NUM undefind,使用默认值 SHARE_EXPERT_NUM  = 0"
    SHARE_EXPERT_NUM=0
fi
if [ "$SHARE_EXPERT_NUM" -lt 0 ]; then
    echo "error:SHARE_EXPERT_NUM:$SHARE_EXPERT_NUM should > 0"
    judge=1
fi

if [ "$judge" = "1" ]; then
    help
fi

echo "-----------------------------"
echo "-----------------------------"
echo "dump_path = $TARGET_DIR"
echo "tool_path = $TOOL_PATH/tools/msaicerr/msaicerr.py"
echo "SOC_VERSION = $SOC_VERSION"
echo "SP_MOE_NUM = $SP_MOE_NUM"
echo "TP_WORLDSIZE = $TPWORLDSIZE"
echo "SHARE_EXPERT_CARD_COUNT = $SHARE_EXPERT_CARD_COUNT"
echo "SHARE_EXPERT_NUM = $SHARE_EXPERT_NUM"
echo "-----------------------------"
echo "-----------------------------"

#开始解析
file_num=$(ls $TARGET_DIR | wc -l)
#判断输入的共享专家卡数是否超出dump数据对应的卡数
if ls "$TARGET_DIR/1/exception_info."* >/dev/null 2>&1; then
    if [ $SHARE_EXPERT_CARD_COUNT -gt $file_num ]; then
        echo "error:SHARE_EXPERT_CARD_COUNT($SHARE_EXPERT_CARD_COUNT) should <= all_care_num($file_num)"
        exit 1
    fi
else
    if [ $SHARE_EXPERT_CARD_COUNT -gt 1 ]; then
        echo "error:SHARE_EXPERT_CARD_COUNT($SHARE_EXPERT_CARD_COUNT) should <= all_care_num(1)"
        exit 1
    fi
fi

if [ "$SOC_VERSION" = "$SOC_VERSION_910_93" ]; then
    echo "进入 A3 处理流程"
    if ls "$TARGET_DIR/exception_info."* >/dev/null 2>&1; then
        echo "开始解析:单卡dump数据"
        if ls "$TARGET_DIR/exception_info."*.workspace.* >/dev/null 2>&1; then
            python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR $SOC_VERSION
            echo "单卡数据解析完成"
            echo "--------------------------------------------"
        else
            for file_dump in $TARGET_DIR/exception_info.*;
            do
                if [[ -f "$file_dump" ]]; then
                    python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                    python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR $SOC_VERSION
                    echo "单卡数据解析完成"
                    echo "--------------------------------------------"
                else
                    echo "error:路径 $TARGET_DIR 下没有dump数据"
                    echo "--------------------------------------------"
                fi
            done
        fi
    elif ls "$TARGET_DIR/1/exception_info."* >/dev/null 2>&1; then
        echo "开始解析多卡dump数据"
        for ((i = 0; i < file_num; i++))
        do
            if ls "$TARGET_DIR$i/exception_info."*.workspace.* >/dev/null 2>&1; then
                echo "开始解析 $i 卡数据"
                python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM $file_num $i $TARGET_DIR$i/ $SOC_VERSION
                echo "$i 卡数据解析完成"
                echo "--------------------------------------------"
            else
                for file_dump in $TARGET_DIR$i/exception_info.*;
                do
                    if [[ -f "$file_dump" ]]; then
                        echo "开始解析 $i 卡数据"
                        python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                        python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM $file_num $i $TARGET_DIR$i/ $SOC_VERSION
                        echo "$i 卡数据解析完成"
                        echo "--------------------------------------------"
                    else
                        echo "error:路径 $TARGET_DIR$i/ 下没有dump数据"
                        echo "--------------------------------------------"
                    fi
                done
            fi
        done
    elif ls "$TARGET_DIR/0/exception_info."* >/dev/null 2>&1; then
        echo "开始解析:单卡dump数据"
        if ls "$TARGET_DIR/0/exception_info."*.workspace.* >/dev/null 2>&1; then
            python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR/0/ $SOC_VERSION
            echo "单卡数据解析完成"
            echo "--------------------------------------------"
        else
            for file_dump in $TARGET_DIR/0/exception_info.*;
            do
                if [[ -f "$file_dump" ]]; then
                    python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                    python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR/0/ $SOC_VERSION
                    echo "单卡数据解析完成"
                    echo "--------------------------------------------"
                else
                    echo "error:路径 $TARGET_DIR/0/ 下没有dump数据"
                    echo "--------------------------------------------"
                fi
            done
        fi
    else
        echo "error:路径 $TARGET_DIR 下没有dump数据"
    fi
elif [ "$SOC_VERSION" = "$SOC_VERSION_950" ]; then
    echo "进入 A5 处理流程"
    if ls "$TARGET_DIR/mc2_exception_info"* >/dev/null 2>&1; then
        echo "开始解析:单卡dump数据"
        if ls "$TARGET_DIR/exception_info."*.workspace.* >/dev/null 2>&1; then
            python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR $SOC_VERSION
            echo "单卡数据解析完成"
            echo "--------------------------------------------"
        else
            for file_dump in $TARGET_DIR/exception_info.*;
            do
                if [[ -f "$file_dump" ]]; then
                    python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                    python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR $SOC_VERSION
                    echo "单卡数据解析完成"
                    echo "--------------------------------------------"
                else
                    echo "error:路径 $TARGET_DIR 下没有dump数据"
                    echo "--------------------------------------------"
                fi
            done
        fi
    elif ls "$TARGET_DIR/1/mc2_exception_info"* >/dev/null 2>&1; then
        echo "开始解析多卡dump数据"
        for ((i = 0; i < file_num; i++))
        do
            if ls "$TARGET_DIR$i/exception_info."*.workspace.* >/dev/null 2>&1; then
                echo "开始解析 $i 卡数据"
                python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM $file_num $i $TARGET_DIR$i/ $SOC_VERSION
                echo "$i 卡数据解析完成"
                echo "--------------------------------------------"
            else
                for file_dump in $TARGET_DIR$i/exception_info.*;
                do
                    if [[ -f "$file_dump" ]]; then
                        echo "开始解析 $i 卡数据"
                        python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                        python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM $file_num $i $TARGET_DIR$i/ $SOC_VERSION
                        echo "$i 卡数据解析完成"
                        echo "--------------------------------------------"
                    else
                        echo "error:路径 $TARGET_DIR$i/ 下没有dump数据"
                        echo "--------------------------------------------"
                    fi
                done
            fi
        done
    elif ls "$TARGET_DIR/0/mc2_exception_info"* >/dev/null 2>&1; then
        echo "开始解析:单卡dump数据"
        if ls "$TARGET_DIR/0/exception_info."*.workspace.* >/dev/null 2>&1; then
            python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR/0/ $SOC_VERSION
            echo "单卡数据解析完成"
            echo "--------------------------------------------"
        else
            for file_dump in $TARGET_DIR/0/exception_info.*;
            do
                if [[ -f "$file_dump" ]]; then
                    python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                    python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR/0/ $SOC_VERSION
                    echo "单卡数据解析完成"
                    echo "--------------------------------------------"
                else
                    echo "error:路径 $TARGET_DIR/0/ 下没有dump数据"
                    echo "--------------------------------------------"
                fi
            done
        fi
    else
        echo "error:路径 $TARGET_DIR 下没有dump数据"
    fi
fi