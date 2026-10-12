[toc]

---

修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
|-------|-------|-------|-------|
| 2025/12/10 | 1.0 | 初版 |邱前锦  |
|  |  |  |  |

# 1. 需求分析

## 1.1 功能分析

- grouped_matmul算子功能为实现分组矩阵乘计算。
- 计算公式为$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中g为分组个数。
- 根据切分方式不同，算子可以分为m轴切分和k轴切分，前者多应用于前向和部分反向计算，后者应用于反向计算。
- 根据需求场景本设计专注实现m轴切分的基础场景gmm算子，不包含量化等功能特性。

## 1.2 竞品方案分析

在编写triton算子之前，A5平台上aclnn已经率先完成了对该算子功能的支持。因此将aclnn中的实现作为参考设计模板和性能比对基准。其torch调用接口原型为：

```
npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=None, group_list_type=0, act_type=0, output_dtype=None, tuning_config=None) -> List[Tensor]
```

其中非量化场景的核心参数如下：

- `x(List[Tensor])`：必选参数。输入矩阵列表，表示矩阵乘法中的左矩阵。
- `weight(List[Tensor])`：必选参数。权重矩阵列表，表示矩阵乘法中的右矩阵。
- `group_list(List[int]/Tensor)`：可选参数。用于指定分组的索引，表示x的第0维矩阵乘法的索引情况。数据类型支持`int64`。
- `split_item(int)`：可选参数。用于指定切分模式。数据类型支持`int32`。
- `group_type(int)`：可选参数。代表需要分组的轴。数据类型支持`int32`。

# 2. 原型设计

```python
def group_gemm_split_m_triton(a, b, m_splits):
```

算子实现主要针对客户需求场景实现，只需求最基础的左右矩阵和group切分信息

# 3. 规格约束

|  类别                          |  参数                                                                                              |  约束描述                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
|  算子名称                      |  group_gemm_split_m_triton     |                                                          |
|  支持NPU型号                   |  910B/910_95                                                                          |                                                          |
|  CANN版本                      |  CANN8.5及以上                                                                      |                                                          |
|  输入支持的数据类型            |  FLOAT、FLOAT16、BFLOAT16                                |    |
|  输出支持的数据类型            |  FLOAT、FLOAT16、BFLOAT16                                |                                                          |
|  PTA版本                       |  pt2.6.0及以上版本                                                                                  |                                                          |
|  交付形式                      |  python直调                                                                     |                                                          |
|  精度标杆                      |  高精度双标杆  |                                                          |
|  性能目标                      |  平均0.9x竞品                                                                         |                                                          |

# 4. 特性实现方案

## 4.1 基本特性实现

### 4.1.1 tiling切分

triton算子要达到最佳性能，首要满足的条件是算子能充分利用所有的物理核，具体做法可以参考
[Wiki: Lighting Indexer的triton算子设计实现 (huawei.com)](https://wiki.huawei.com/domains/124182/wiki/290311/WIKI202511068930845)

```python
# Total number of blocks in sequence dimension (M)
NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)
# Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
NUM_BLOCKS = B * M * NUM_BLOCKS_N
    
# Current M-dimension block index
pid = tl.program_id(0)
NUM_CORE = tl.num_programs(0)
    
for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
    batch_idx = block_idx // (M * NUM_BLOCKS_N)
    bm_idx = block_idx // NUM_BLOCKS_N
    n_idx = block_idx % NUM_BLOCKS_N
```

其中的核心思想是以block切分各维度得到总任务块数量，每个物理核会根据自己的编号从总任务中选取自己要处理的任务块达成并行处理。因此需要考虑如何对数据进行分块，需要从计算公式出发，当前需求场景下n、k分别固定为768，1024。因此：

$$
y_i[m_i,768]=x[m_i,1024]×weight[1024,768],i=1...g
$$

可以切分的轴为m,n,k，其中m轴需要在每个g中单独分块，而k轴作为规约轴只在累加循环中不作为任务分块。记切块大小分别为`block_m`、`block_n`、`block_k`，总任务块数量为

$$
total\_blocks=∑^icdiv(m_i,block\_m) * cdiv(n,block\_n),i=1...g
$$

### 4.1.2 kernel方案

##### 4.1.2.1 片上内存分配策略

根据前面的任务分块，单次核任务处理的数据块为：

$$
y_i[block\_m,block\_n]=x[block\_m,1024]×weight[1024,block\_n],i=1...g
$$

片上只需要开辟[block\_m,block\_n]的累加器空间即可

##### 4.1.2.2 片上流水编排策略

每个任务块之间相互独立，只需要考虑单个任务块。块内执行的计算逻辑是在k轴上进行累加，这是一个标准的读数据-计算循环，需要使用doublebuffer进行加速

# 5 测试设计

## 5.1 测试标杆

mm算子采用高精度双标杆，真值采用高精度计算，GPU使用CPU进行模拟计算。

## 5.2 测试用例设计

根据需求，用例有两个典型场景，其每个group中的m大小满足分布N(50,20)和N(512，100)


