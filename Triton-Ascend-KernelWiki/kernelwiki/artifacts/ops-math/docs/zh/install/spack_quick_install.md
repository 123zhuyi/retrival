# 使用Spack快速搭建开发环境

针对新加入的开发人员，可以根据以下步骤快速搭建开发环境

## 一、本地快速构建

### 1.本地快速搭建环境并构建

#### 方法一、运行一键式脚本快速构建

prepare_cann_env.sh会一键式搭建Spack环境并安装ops-math的全部依赖

```bash
# 进入代码根目录
cd ${local_repo_path}/ops-math
source spack/prepare_cann_env.sh
```

#### 方法二、手动配置环境（适合高级用户或自定义配置）

如果您希望更精细地控制环境配置，可以按照以下步骤手动配置：

##### 步骤1：下载并安装Spack（Spack已安装可跳过）

```bash
# 选择安装目录（默认为 $HOME）
export SPACK_INSTALL_DIR="$HOME"
# 下载Spack v1.1.0并激活Spack工具
cd $SPACK_INSTALL_DIR
git clone https://gitcode.com/GitHub_Trending/sp/spack.git -b v1.1.0 --depth=2
source $SPACK_INSTALL_DIR/spack/share/spack/setup-env.sh
# 验证Spack安装后的版本
spack --version
```

##### 步骤2：设置Spack软件包默认安装路径

Spack软件包默认安装路径在$SPACK_INSTALL_DIR/spack/opt，可以通过如下命令进行修改：

```bash
#普通用户
spack config --scope user add "config:install_tree:root:$HOME/.spack"
#root管理员
spack config --scope user add "config:install_tree:root:/opt/spack"
```

普通用户建议使用个人home下目录，避免被其他用户修改导致环境不稳定
管理员建议使用全局路径，便于使用[Spack串联能力](https://spack.readthedocs.io/en/latest/chain.html)统一安装软件提供给其他用户使用:

##### 步骤3：添加外部已安装工具到Spack环境中

```bash
spack compiler find  # 配置本地已安装的编译器，如gcc
```

##### 步骤4：配置Spack官方仓库gitcode镜像源

Spack默认官方仓库地址为[github地址](https://github.com/spack/spack-packages.git),可以通过如下方式修改默认官方仓库，使用GitCode镜像源起到加速作用：
修改~/.spack目录下的repos.yaml，如不存在则新建该文件，写入以下内容

```yaml
repos:
  # Spack内置仓库
  builtin:
    git: https://gitcode.com/spack/spack-packages.git
    branch: releases/v2025.11
```

验证仓库配置：`spack repo list`

##### 步骤5：下载并添加CANN社区Spack包仓库

```bash
# 克隆CANN Spack包仓库
cd $SPACK_INSTALL_DIR
git clone --depth=1 https://gitcode.com/cann/cann-spack-package.git

# 添加到Spack仓库列表
spack repo add $SPACK_INSTALL_DIR/cann-spack-package

# 验证添加成功
spack repo list | grep cann
```

##### 步骤6：创建并激活Spack环境

```bash
# 创建名为cann-dev-env的环境
spack env create cann-dev-env ${local_repo_path}/ops-math/spack/spack.yaml

# 激活环境
spack env activate cann-dev-env

# 验证环境激活
spack env status
```

##### 步骤7：设置终端自动加载（可选）

```bash
# 将Spack环境配置添加到.bashrc
echo "source $SPACK_INSTALL_DIR/spack/share/spack/setup-env.sh" >> ~/.bashrc

# 立即生效
source ~/.bashrc
```

##### 步骤8：配置开发模式并安装

```bash
# 设置ops-math为开发模式（指向本地代码）
spack develop -p ${local_repo_path}/ops-math cann-ops-math@master

# 添加包到环境（根据需要调整变体）
spack add cann-ops-math@master+pkg+jit

# 解析依赖关系
spack concretize -f

# 安装
spack install
```

### 2.查看产物位置

执行命令

```bash
spack location -i cann-ops-math
```

即可查看ops-math编译生成run包的位置，run包已自动安装到ASCEND_HOME_PATH下

## 二、修改代码后重新构建

### 1.卸载已构建的产物

```bash
spack uninstall -y cann-ops-math
```

### 2.移除已有变体并添加新变体

如果想指定不同的构建参数，可通过更换Spack包变体来实现
（不更改构建参数，可跳过此步骤）

```bash
spack change cann-ops-math@master +pkg +jit soc=ascend910b # 举例
```

### 3.重新进行依赖解析

```bash
spack concretize -f
```

### 4.重新构建

```bash
spack install
```

## 三、重新进入Spack环境

开启新终端需要重新进入Spack环境

```bash
spack env activate cann-dev-env
```

## 四、清理环境或卸载Spack

如果不想继续使用Spack，可使用`prepare_cann_env.sh`的clean参数卸载Spack并通过**重启终端**来清理环境变量

```bash
source prepare_cann_env.sh clean
```

## 五、ops-math的Spack构建变体

通过`spack info cann-ops-math`命令可以查看ops-math支持的参数，变体具体含义请参考[build.md](./build.md#参数说明).

## 六、Spack开发与调试命令指导

```shell
#查看当前环境列表：如果是从一键式脚本prepare_cann_env.sh生成，已准备好开发环境cann-dev-env
spack env list

#创建环境：Spack管理的环境之间互相隔离，以链接的形式与软件包关联
spack env create <env-name>

#启用环境：当前激活的环境字体为绿色，Spack的命令以当前激活环境为准，与所处位置无关
spack env activate <env-name>

#查看当前环境中都引入了哪些软件包：如果不在特定环境中执行则为查看所有环境
spack find                              #列出所有软件包
spack find -L                           #显示完整哈希
spack find <package-name>               #按包名过滤，不提供则为全部包
spack find --deps <package-name>        #显示该包的依赖树
spack find --explicit <package-name>    #显示手动安装的定级包 缩写：-e
spack find -p <package-name>            #显示完整安装路径
spack find -lv <package-name>           #显示完整哈希和变体信息

#向当前环境中添加软件包：
spack add <package-name>                #同样支持spec语法，指定版本，变体，编译器等

#从当前环境中移除软件包: Spack不会删除软件包，只是移除当前环境与软件包之间的依赖
spack remove <package-name>

#卸载软件包：需要所有的Spack环境都不依赖该软件包才可以由Spack卸载，会卸载本地安装好的包，慎用
spack uninstall <package-name>

#删除某个Spack环境：物理删除文件，慎用
spack env remove <env-name>

#取消激活当前环境：
spack env deactivate

#搜索Spack支持的软件包
spack list <package-name>
spack list 'py-*'        #列出所有Python包
spack list -d 'mpi'      #在描述中搜索mpi

#查看可用版本号
spack versions <package-name>

#清除Spack构建缓存：
spack clean
spack clean -all   #将旧源码与旧构建记录及缓存全部清除，慎用

#查看当前软件包信息：查看该软件包所有支持的版本，Spack默认倾向于最新版本
spack info <package-name>

#安装软件包，Spack自动选择最新版本
spack install                                                   #安装当前环境中所有未安装的软件包
spack install <package-name>                                    #指定具体的软件包名
spack install <package-name>@1.12.0%gcc@11.4.0                  #spec语法，指定版本与编译器版本
spack install <package-name>+mpi+fortran ^openmpi@4.1           #spec语法，指定变体
spack install <package-name1> <package-name2> <package-name3>   #同时安装多个包
spack -k install                                                #安装时跳过安全认证，慎用
spack install --verbose                                         #输出详细构建日志，缩写：-v
spack install --no-checksum                                     #安装时禁用网络校验和检查，慎用

#具体化当前环境：Spack的策略是默认倾向于重用已安装在本地的软件包，即使已有更高版本且满足依赖条件，除非本地包已不符合依赖条件
spack concretize

#从远端下载软件包源码到本地并注册为开发包：否则Spack引用的远端文件会作为缓存临时文件，会被spack clean清除
spack develop <package-name>

#从本地代码开发目录注册开发包
git clone <git-path>  #克隆源码并进入目录
cd /package/path
spack develop --no-clone <package-name>

#解析该软件包依赖信息：
spack spec <package-name>

# 如果git仓有更新，该命令可以根据repos.yaml中配置的远程仓库地址与分支进行同步更新
spack repo update <repo name>

#定位软件包安装位置：
spack location -b <package-name>

#查看编译器配置：
spack compilers                                                 #查看所有支持的编译器，同spack compiler list
spack compiler add /path/to/compiler/bin                        #手动添加编译器

#查看Spack复用的本机已安装的软件
spack external find

#查看Spack版本，验证安装：
spack --version

#编辑Spack配置: 会自动打开vim
spack config edit config
spack config edit packages
```

## 七、一键式脚本prepare_cann_env.sh生成路径（默认）

```shell
#cann开源仓目录：
$HOME/cann-spack-package

#Spack管理的软件包安装目录：
root用户：/opt/spack/linux-*
普通用户：$HOME/.spack/linux-*

#Spack配置目录：
$HOME/.spack

#Spack自身提供的builtin软件包目录：
$HOME/.spack/package_repos/opgus4u/repos/spack_repo/builtin/packages

#Spack本体目录：
$HOME/spack

#Spack环境目录：
$HOME/spack/var/spack/environments
```

## 八、FAQ

### 1.执行prepare_cann_env.sh出现报错怎么办

如果机器在以前安装过Spack可能会导致环境变量或配置文件残留，可在备份关键数据后执行```source prepare_cann_env.sh clean```并重启终端后重新执行

### 2.安装过程中出现ssl类报错怎么办

此类报错是由于机器缺少相关证书导致，请自行配置证书并重新运行

### 3.不想使用Spack了如何卸载

执行```source prepare_cann_env.sh clean```可自动卸载Spack，root用户还需要执行```rm -rf /opt/spack```卸载所有Spack软件包

## 九、更多Spack操作请参考官方文档

<https://spack.readthedocs.io/en/latest/>
