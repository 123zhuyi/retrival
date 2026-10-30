# 算子kernel编译失败

## 问题描述

在使用命令`--pkg`或`--opkernel`编译算子kernel过程中，出现"the kernel * not generate output json"，算子.o未编译成功。

## 原因分析

算子kernel代码编译错误。

## 解决措施

查看`build/binary/${soc}/bin/build_log`目录下的日志，修改代码错误后重新编译。
