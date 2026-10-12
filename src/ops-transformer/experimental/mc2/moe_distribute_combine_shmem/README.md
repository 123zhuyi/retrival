# 摘要

本篇文档提供如何基于shmem跑通MC2 dispatch和combine通信算子用例

## 一、环境准备

### 1.驱动和固件 

- 推荐版本：Ascend HDK 25.3.0

### 2.安装 CANN toolkit和ops包

 ```bash
# 确保安装包有可执行权限
chmod +x Ascend-cann-toolkit_8.5.0_linux-${arch}.run
# 安装命令
./Ascend-cann-toolkit_8.5.0_linux-${arch}.run --install --force --install-path=${install_path}
    
# 确保安装包有可执行权限
chmod +x Ascend-cann-${device_type}-ops_8.5.0_linux-${arch}.run
# 安装命令
./Ascend-cann-${device_type}-ops_8.5.0_linux-${arch}.run --install --force --install-path=${install_path}
   ```

### 3.编译shmem库

执行编译脚本build.sh，加上编译python包选项

 ```bash
 bash shmem/scritps/build.sh -package
 pip3 install package/*.whl
 ```
   
## 二、编译dispatch & combine工程

### 1.添加环境变量

```bash
# 配置CANN包环境变量，此为默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
   
# 配置shmem环境变量，
source shmem/install/set_env.sh
export CPLUS_INCLUDE_PATH="${SHMEM_HOME_PATH}/shmem/src/device:$CPLUS_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${SHMEM_HOME_PATH}/shmem/include/device:$CPLUS_INCLUDE_PATH"
```

### 2.编译自定义算子

此为在ops-transformer目录下，用户需要在cmake/func.cmake文件的MC2_OPS_LIST里加入“moe_distribute_dispatch_shmem”和“moe_distribute_combine_shmem”以支持自定义算子编译。

```bash
bash build.sh --pkg --soc=910_93 --ops="moe_distribute_dispatch_shmem,moe_distribute_combine_shmem" --experimental
```

### 3.执行装包命令

```bash
cd build_out
chmod +x *.run
./*.run
```

## 三、执行测试脚本

执行python用例

```bash
python3 scripts/test.py
```

以下为使用shmem替换hccl通信域改动点

```python
# 参数初始化
init_attr = shmem.InitAttr()
init_attr.my_rank = rank
init_attr.n_ranks = rank_size
init_attr.ip_port = 'tcp://127.0.0.1:50001'
init_attr.local_mem_size = 1024 * 1024 * 1024 // 内存池大小
init_attr.option_attr.data_op_engine_type = shmem.OpEngineType.MTE

#初始化并申请共享内存
shmem.shmem_init(init_attr)
shmem_space = shmem.shmem_create_tensor(shape, dtype, device_id) //申请内存大小
shmem_space.fill_(0)

#新增算子shmem入参
dispatch_kwargs = get_dispatch_kwargs(
	shmem_space = shmem_space,
    x = x,
    expert_ids = expert_ids,
	......
 )

```

## 三、torch_npu编包

### 1. 拉取目标版本的pytorch仓代码到本地，进入到op-plugin仓

```bash
git clone https://gitcode.com/Ascend/pytorch.git -b v2.6.0 --recursive
```

### 2. 修改{算子名}KernelOpApi.cpp，增加shmem_space入参，数据类型为int8。调用的aclnn函数名分别改为"aclnnMoeDistributeDispatchShmem"和"aclnnMoeDistributeCombineShmem"

### 3. 修改op_plugin_functions.yaml和_meta_registrations.py，增加shmem_space，数据类型为int8

## 四、整网脚本适配

### 1. 定义一个全局的init_shmem_once函数，里面做shmem内存分配初始化

```python
global shmem_space, cnt
if cnt != 0:
	return
cnt += 1
ret = shm.set_conf_store_tls(False, "")
# 参数初始化
init_attr = shmem.InitAttr()
init_attr.my_rank = rank
init_attr.n_ranks = rank_size
init_attr.ip_port = 'tcp://127.0.0.1:50001'
init_attr.local_mem_size = 1024 * 1024 * 1024 // 内存池大小
init_attr.option_attr.data_op_engine_type = shmem.OpEngineType.MTE

#初始化并申请共享内存
shmem.shmem_init(init_attr)
shmem_space = shmem.shmem_create_tensor(shape, dtype, device_id) //申请内存大小
shmem_space.fill_(0)
```

### 2. 将init_shmem_once函数放置DeepseekV3ForCausalLM类的__init__下

### 3. dispatch_args和combine_args里额外加"shmem_space"入参

### 4. 跑整网pytorch需使用2.6.0版本，python使用python3.11版本，启用单流eager模式
