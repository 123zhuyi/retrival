# 算子名称

Cast

## 1. 算子功能

Cast 进行数据类型转换

## 2. 目前数据类型支持

## 算子规格描述

| 算子类型 (OpType) | Cast |
|------------------|------|
| 算子输入 | name: x&lt;br&gt;shape: 任意&lt;br&gt;data type: 见下表&lt;br&gt;format: ND |
| 算子输出 | name: z&lt;br&gt;shape: 与 x 相同&lt;br&gt;data type: 见下表&lt;br&gt;format: ND |
| 核函数名 | cast_custom |

### 支持的输入输出类型组合（19 种）

| TILING_KEY | 输入类型 (x) | 输出类型 (z) |
|------------|--------------|--------------|
| 1          | half         | float        |
| 2          | half         | uint8_t      |
| 3          | half         | int8_t       |
| 4          | half         | int16_t      |
| 5          | half         | int32_t      |
| 6          | float        | half         |
| 7          | float        | int16_t      |
| 8          | float        | int32_t      |
| 9          | float        | int64_t      |
| 10         | uint8_t      | half         |
| 11         | int8_t       | half         |
| 12         | int16_t      | half         |
| 13         | int16_t      | float        |
| 14         | int32_t      | half         |
| 15         | int32_t      | float        |
| 16         | int32_t      | int16_t      |
| 17         | int32_t      | int64_t      |
| 18         | int64_t      | float        |
| 19         | int64_t      | int32_t      |
| 20         | float        | bfloat6      |
| 21         | bfloat6      | float        |
| 22         | bfloat6      | int32_t      |

支持任意 shape。

## 3. 目前功能支持

- 基础特性支持，处理对齐格式向量。
- 能处理尾块和非对齐数据

## 4. 工程结构

```text
├── cast_v2                              // Cast算子
│   ├── examples           // Cast算子测试文件
│   ├── op_host           // Cast算子host侧实现
│   ├── op_kernel          // Cast算子kernel侧实现
│   ├── tests
│   └── README             // 算子工程README
```

## 5. 开发者信息

- 开发者1：梁杨琳
- 邮箱：<3431470978@qq.com>

## 6. 更新日志

| 版本 | 日期       | 更新内容 | 开发者 |
|------|------------|----------|--------|
| v1.0 | 2025-12-07 | 初始版本 | 梁杨琳   |
