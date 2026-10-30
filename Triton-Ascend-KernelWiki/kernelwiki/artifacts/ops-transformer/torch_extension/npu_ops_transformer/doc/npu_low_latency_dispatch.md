# npu\_low\_latency_dispatch<a name="ZH-CN_TOPIC_0000002343094193"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √    |

## 功能说明<a name="zh-cn_topic_0000002203575833_section14441124184110"></a>

- API功能：需与[npu\_low\_latency_combine](npu_low_latency_combine.md)配套使用，完成MoE的并行部署下的token的dispatch和combine。
    - 支持动态量化场景，对token数据先进行量化（可选），进行EP（Expert Parallelism）域的alltoallv通信；
    - 支持特殊专家场景。
- 计算公式：
    - 动态量化场景：

      若`quant_mode`不为`2`，即非动态量化场景：

         $$
         \ quant\_out=
         \begin{cases}
         \ x, & \quad \text{if}\ quant\_mode = 0 \\
         \ CastToInt8(\ CastToFp32(x) \times \ scales ), & \quad \text{if } quant\_mode ≠ 0 \\
         \end{cases}
         $$

         $$\ alltoall\_x\_out= \ alltoallv(\ quant\_out)$$

         $$\ expand\_x= \ alltoall\_x\_out$$

      若`quant_mode`为`2`，即动态量化场景：

         $$\ x\_fp32= \ CastToFp32(x) \times \ scales$$

         $$\ dynamic\_scales\_value = 127.0/Max(Abs(x\_fp32))$$

         $$\ quant\_out=CastToInt8(\ x\_fp32 \times \ dynamic\_scales\_value )$$

         $$\ alltoall\_x\_out= \ alltoallv(\ quant\_out)$$

         $$\ alltoall\_dynamic\_scales\_out = alltoall(1.0/dynamic\_scales)$$

         $$\ expand\_x=\ alltoall\_x\_out$$

         $$\ dynamic\_scales=\ alltoall\_dynamic\_scales\_out$$

    - 特殊专家场景：

      零专家场景，即`zero_Expert_Num`不为0：

         $$Moe(ori\_x)=0$$

      拷贝专家场景，即`copy_Expert_Num`不为0：

         $$Moe(ori\_x)=ori\_x$$

      常量专家场景，即`const_expert_num`不为0：

         $$Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v$$

      参数ori\_x、const\_expert\_alpha\_1、const\_expert\_alpha\_2、const\_expert\_v见[npu\_low\_latency_combine](npu\_low\_latency_combine.md)文档。

## 函数原型<a name="zh-cn_topic_0000002203575833_section45077510411"></a>

```python
npu_low_latency_dispatch(x, topk_idx, num_experts, *, quant_mode = 0, comm_alg="", x_smooth_scale=None, x_active_mask=None, topk_weights=None, zero_expert_num=0, copy_expert_num=0, const_expert_num=0, elastic_info=None, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, expert_token_nums_type=1, num_max_dispatch_tokens_per_rank = 0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002203575833_section112637109429"></a>

- **x** (`Tensor`)：必选参数，表示计算使用的token数据，需根据`topk_idx`来发送给其他卡。要求为2维张量，shape为(BS, H)，表示有BS个token，数据类型支持`bfloat16`、`float16`，数据格式为$ND$，支持非连续的Tensor。
- **topk\_idx** (`Tensor`)：必选参数，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为(BS, K)，数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。对应[npu\_low\_latency\_combine](npu_low_latency_combine.md)的`topk_idx`输入，张量里value取值范围为\[0, num\_experts\)，且同一行中的K个value不能重复。
- **num\_experts** (`int`)：必选参数，MoE专家数量，取值范围[1, 1024]，并且满足以下条件：num\_experts\%(ep\_world\_size - shared\_expert\_rank\_num)\=0。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **quant\_mode** (`int`)：可选参数，表示量化模式。支持取值：0表示非量化（默认），2表示动态量化。当`quant_mode`为2，`dynamic_scales`不为None；当`quant_mode`为0，`dynamic_scales`为None。
- **comm\_alg** (`string`)：可选参数，表示通信亲和内存布局算法。当前版本支持""，"fullmesh_v1"，"fullmesh_v2"三种输入方式。
    - ""：默认值，使能fullmesh_v1模板；
    - "fullmesh_v1"：使能fullmesh_v1模板；
    - "fullmesh_v2"：使能fullmesh_v2模板；
- **x\_smooth\_scales** (`Tensor`)：可选参数，表示每个专家的权重，非量化场景不传，动态量化场景可传可不传。若传值要求为2维张量，如果有共享专家，shape为(shared\_expert\_num+num\_experts, H)，如果没有共享专家，shape为(num\_experts, H)，数据类型支持`float`，数据格式为$ND$，不支持非连续的Tensor。
- **x\_active\_mask** (`Tensor`)：可选参数，表示token是否参与通信，要求是一个1维或者2维张量。当输入为1维时，shape为(BS, ); 当输入为2维时，shape为(BS, K)。数据类型支持`bool`，数据格式要求为$ND$，支持非连续的Tensor。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true} 为非法输入；当输入为2D时，参数为true表示当前token对应的`topk_idx`参与通信，若当前token对应的K个`bool`值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。
- **topk\_weights** (`Tensor`)：暂不支持该参数，使用默认值即可。
- **zero\_expert\_num** (`int`)：可选参数，表示零专家的数量，取值范围[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，合法的零专家的ID值是[num\_experts, num\_experts+zero\_expert\_num)。

- **copy\_expert\_num** (`int`)：可选参数，表示拷贝专家的数量，取值范围[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，合法的拷贝专家的ID值是[num\_experts+zero\_expert\_num, num\_experts+zero\_expert\_num+copy\_expert\_num)。

- **const\_expert\_num** (`int`)：可选参数，表示常量专家的数量, 取值范围[0, MAX_INT32)，MAX_INT32 = 2^31 - 1，合法的常量专家的ID值是[moe\_expert\_num+zero\_expert\_num+copy\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num+const\_expert\_num)。
- **elastic\_info** (`Tensor`)：预留参数，当前版本不支持，传默认值None即可。

- **expert\_shard\_type** (`int`)：可选参数，表示共享专家卡排布类型。当前仅支持0，表示共享专家卡排在MoE专家卡前面。

- **shared\_expert\_num** (`int`)：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。取值范围[0, 4]，0表示无共享专家，默认值为1。

- **shared\_expert\_rank\_num** (`int`)：可选参数，表示共享专家卡数量。取值范围[0, ep\_world\_size)。取0表示无共享专家，不取0需满足shared\_expert\_rank\_num%shared\_expert\_num=0。

- **expert\_token\_nums\_type** (`int`)：可选参数，表示输出`expert_token_nums`的值类型，取值范围[0, 1]，0表示每个专家收到token数量的前缀和，1表示每个专家收到的token数量（默认）。

- **num\_max\_dispatch\_tokens\_per\_rank** (`int`)：可选参数，表示每张卡上的token数量。当每个rank的BS不同时，最大的BS大小，当每个rank上BS相同时，默认为0。

## 输出说明<a name="zh-cn_topic_0000002203575833_section22231435517"></a>

- **expand\_x** (`Tensor`)：表示本卡收到的token数据，要求为2维张量，shape为(A, H)，A表示在EP通信域可能收到的最大token数，数据类型支持`bfloat16`、`float16`、`int8`。量化时类型为`int8`，非量化时与`x`数据类型保持一致。数据格式为$ND$，支持非连续的Tensor。
- **dynamic\_scales** (`Tensor`)：表示计算得到的动态量化参数。当`quant_mode`不为0时才有该输出，要求为1维张量，shape为(A,)，数据类型支持`float`，数据格式支持$ND$，支持非连续的Tensor。
- **assist\_info\_for\_combine** (`Tensor`)：表示给同一专家发送的token个数，要求是一个1维张量，shape为(A \* 128, )。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。对应[npu\_low\_latency\_combine](npu_low_latency_combine.md)的`assist_info_for_combine`输入。

- **expert\_token\_nums** (`Tensor`)：本卡每个专家实际收到的token数量，要求为1维张量，shape为\(local\_expert\_num,\)，数据类型`int64`，数据格式支持$ND$，支持非连续的Tensor。
- **ep\_recv\_counts** (`Tensor`)：表示EP通信域各卡收到的token数（token数以前缀和的形式表示），要求为1维张量，数据类型`int32`，数据格式支持$ND$，支持非连续的Tensor。对应[npu\_low\_latency\_combine](npu_low_latency_combine.md)的`ep_send_counts`输入。要求shape为\(ep\_world\_size\*local\_expert\_num, \)。
- **expand\_scales** (`Tensor`)：表示`topk_weights`与`x`一起进行alltoallv之后的输出。暂不支持该输出，返回None。

## 约束说明<a name="zh-cn_topic_0000002203575833_section12345537164214"></a>

- 该接口支持推理场景下使用。
- 该接口支持静态图模式，`npu_low_latency_dispatch`和`npu_low_latency_combine`必须配套使用。
- 在不同产品型号、不同通信算法或不同版本中，`npu_low_latency_dispatch`的Tensor输出`assist_info_for_combine`、`ep_recv_counts`、`tp_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`npu_low_latency_combine`对应参数即可，模型其他业务逻辑不应对其存在依赖。
- 调用接口过程中使用的`num_experts`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`num_max_dispatch_tokens_per_rank`参数取值所有卡需保持一致, `expert_shard_type`、`num_max_dispatch_tokens_per_rank`网络中不同层中也需保持一致，且和[npu\_low\_latency\_combine](npu_low_latency_combine.md)对应参数也保持一致。
- 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
- num_experts + zero_expert_num + copy_expert_num + const_expert_num < MAX_INT32。
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，取值范围如下
        - 对于共享专家，要满足A=BS\*shared\_expert\_num/shared\_expert\_rank\_num。
        - 对于MoE专家，当`num_max_dispatch_tokens_per_rank`为0时，要满足A \>= BS \* ep\_world\_size \* min\(local\_expert\_num, K\)；当`num_max_dispatch_tokens_per_rank`不为0时，要满足A \>= num_max\_dispatch\_tokens\_per\_rank \* ep\_world\_size \* min\(local\_expert\_num, K\)。

    - H：表示hidden size隐藏层大小。取值为\[1024, 8192\]。

    - BS：表示batch sequence size，即本卡最终输出的token数量。取值范围为0<BS≤512。

    - K：表示选取topK个专家，取值范围为0<K≤16，同时满足0 < K ≤ num\_experts + zero_expert_num + copy_expert_num + const_expert_num。

    - local\_expert\_num：表示本卡专家数量。
        - 对于共享专家卡，local\_expert\_num为1。
        - 对于MoE专家卡，local\_expert\_num=num\_experts/\(ep\_world\_size-shared\_expert\_rank\_num)，应满足0 < local_expert_num * ep_world_size ≤ 2048。

- HCCL通信域缓存区大小:
    调用本接口前需检查HCCL\_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
    - 该场景仅支持通过环境变量HCCL\_BUFFSIZE配置，该环境变量按通信域粒度管理，每个通信域独占一组“2*HCCL\_BUFFSIZE”大小的内存。
    - ep通信域内，comm\_alg配置为"fullmesh_v1"或"": 设置大小要求 \>= 2 \* \(local\_expert\_num \* max\_bs \* ep\_world\_size \* Align512\(Align32\(2 \* H\) + 64\) + \(K + shared\_expert\_num\) \* max\_bs \* Align512\(2 \* H\)\)。
    - ep通信域内，comm\_alg配置为"fullmesh_v2": 设置大小要求 \>= 2 \* \(local\_expert\_num \* max\_bs \* ep\_world\_size \* 480Align512\(Align32\(2 \* H\) + 64\) + \(K + shared\_expert\_num\) \* max\_bs \* Align512\(2 \* H\)\)。
    - 其中 480Align512(x) = ((x+480-1)/480)\*512,Align512(x) = ((x+512-1)/512)\*512,Align32(x) = ((x+32-1)/32)\*32。
    - 通信域开设大小可通过调用MoeDistributeBuffer.get_low_latency_ccl_buffer_size接口计算。

- 本文公式中的“/”表示整除。

- 通信域使用约束：

    - 一个模型中的`npu_low_latency_dispatch`和`npu_low_latency_combine`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

    - 一个模型中的`npu_low_latency_dispatch`和`npu_low_latency_combine`算子仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

    - 一个通信域内的节点需在一个超节点内，不支持跨超节点。

- 版本配套约束：

     静态图模式下，从Ascend Extension for PyTorch 8.0.0版本开始，Ascend Extension for PyTorch框架会对静态图中最后一个节点输出结果做Meta推导与inferShape推导的结果强校验。当图中只有一个Dispatch算子，若CANN版本落后于Ascend Extension for PyTorch版本，会出现Shape不匹配报错，建议用户升级CANN版本，详细的版本配套关系参见《[Ascend Extension for PyTorch 版本说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-7.3.0/docs/zh/release_notes/release_notes.md)》中“相关产品版本配套说明”。

## 调用示例<a name="zh-cn_topic_0000002203575833_section14459801435"></a>

- 单算子模式调用

    ```python
    import os
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    from npu_ops_transformer.ops import MoeDistributeBuffer

    # 控制模式
    quant_mode = 2  # 2为动态量化
    is_dispatch_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0  # 共享专家数
    num_experts = 32  # moe专家数
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = num_experts // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 0


    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1


    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2


    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v


    def get_new_group(rank):
        for i in range(tp_world_size):
            # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
            tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
            tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_t = tp_group
                print(f"rank:{rank} tp_ranks:{tp_ranks}")
        return ep_group_t, tp_group_t


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info


    def get_dispatch_kwargs_warmup(
        x_warm_up, topk_idx_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()
        return {
            'x': x_warm_up,
            'topk_idx': topk_idx_warm_up,
            'x_active_mask': None,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'num_experts': num_experts,
            'quant_mode': 2,
            'num_max_dispatch_tokens_per_rank': 16,
        }


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        topk_idx = torch.tensor([[5, 7, 17, 4, 2, 6, 11, 16],
                                [10, 12, 13, 15, 19, 4, 18, 1],
                                [19, 33, 1, 17, 9, 5, 0, 32],
                                [19, 11, 17, 0, 10, 5, 7, 9],
                                [10, 16, 11, 17, 33, 8, 9, 3],
                                [12, 19, 5, 7, 1, 3, 18, 16],
                                [11, 9, 13, 16, 12, 33, 17, 14],
                                [16, 4, 9, 5, 0, 10, 11, 17]], dtype=torch.int32).npu()
        topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()

        scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        const_expert_alpha_1 = None
        const_expert_alpha_2 = None
        const_expert_v = None

        distribute_buffer = MoeDistributeBuffer(ep_group)

        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, _ = distribute_buffer.npu_low_latency_dispatch(
            x=x,
            topk_idx=topk_idx,
            shared_expert_num=0,
            shared_expert_rank_num=shared_expert_rank_num,
            num_experts=num_experts,
            quant_mode=quant_mode,
            num_max_dispatch_tokens_per_rank=globalBS,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num)

        if is_quant:
            expand_x = expand_x.to(input_dtype)

        x = distribute_buffer.npu_low_latency_combine(x=expand_x,
                                                topk_idx=topk_idx,
                                                assist_info_for_combine=assist_info_for_combine,
                                                ep_send_counts=ep_recv_counts,
                                                topk_weights=topk_weights,
                                                shared_expert_num=0,
                                                shared_expert_rank_num=shared_expert_rank_num,
                                                num_experts=num_experts,
                                                num_max_dispatch_tokens_per_rank=globalBS,
                                                ori_x=x,
                                                const_expert_alpha_1=const_expert_alpha_1,
                                                const_expert_alpha_2=const_expert_alpha_2,
                                                const_expert_v=const_expert_v,
                                                zero_expert_num=zero_expert_num,
                                                copy_expert_num=copy_expert_num,
                                                const_expert_num=const_expert_num)
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"num_max_dispatch_tokens_per_rank={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"num_experts={num_experts}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)
        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num 不能大于 ep_world_size")
            exit(0)
        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
            exit(0)
        if num_experts % moe_rank_num != 0:
            print("num_experts 必须是 moe_rank_num 的整数倍")
            exit(0)
        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```

- 图模式调用

    ```python
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time
    from npu_ops_transformer.ops import MoeDistributeBuffer

    # 控制模式
    quant_mode = 2  # 2为动态量化
    is_dispatch_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0  # 共享专家数
    num_experts = 32  # moe专家数
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = num_experts // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)

    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 0


    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self, group_ep):
            super().__init__()
            self.group = group_ep
            self.distribute_buffer = MoeDistributeBuffer(group_ep)

        def forward(self, x, topk_idx, group_ep, group_tp, ep_world_size, tp_world_size,
                    ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, num_experts,
                    scales, quant_mode, num_max_dispatch_tokens_per_rank, topk_weights, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = self.distribute_buffer.npu_low_latency_dispatch(
                x=x,
                topk_idx=topk_idx,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)

            output_combine_npu = self.distribute_buffer.npu_low_latency_combine(
                x=expand_x_npu,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]


    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1


    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2


    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v


    def get_new_group(rank):
        for i in range(tp_world_size):
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
            tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_t = tp_group
                print(f"rank:{rank} tp_ranks:{tp_ranks}")
        return ep_group_t, tp_group_t


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info


    def get_dispatch_kwargs_warmup(
        x_warm_up, topk_idx_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()

        return {
            'x': x_warm_up,
            'topk_idx': topk_idx_warm_up,
            'x_active_mask': None,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'num_experts': num_experts,
            'scales': None,
            'quant_mode': 2,
            'num_max_dispatch_tokens_per_rank': 1,
        }


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        topk_idx = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()

        topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        elastic_info = None
        const_expert_alpha_1 = None
        const_expert_alpha_2 = None
        const_expert_v = None

        model = MOE_DISTRIBUTE_GRAPH_Model(ep_group)
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(
            x, topk_idx, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
            rank // tp_world_size, rank % tp_world_size, 0, shared_expert_rank_num, num_experts, scales,
            quant_mode, globalBS, topk_weights, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
            zero_expert_num, copy_expert_num, const_expert_num
        )
        torch.npu.synchronize()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

        time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"num_max_dispatch_tokens_per_rank={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"num_experts={num_experts}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num 不能大于 ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
            exit(0)

        if num_experts % moe_rank_num != 0:
            print("num_experts 必须是 moe_rank_num 的整数倍")
            exit(0)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```
