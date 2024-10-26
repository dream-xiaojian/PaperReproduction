#复现 

# 复现论文笔记汇总

- DeepDiffuse [https://github.com/dream-xiaojian/PaperReproduction/blob/master/deep-diffuse/README.md](https://github.com/dream-xiaojian/PaperReproduction/blob/master/deep-diffuse/README.md)

# 环境

复现中环境处理这件事很重要，由于对于每一个被复现的代码项目具有不同的时间跨度，导致说每一个代码项目的环境是不一样的（主要是所用python，库的版本不同）。不可能说每次要复现一个代码项目都要重头开始配置一遍环境
## 解决方案 - 本地

> 包管理工具、虚拟环境

- 包管理工具：可以快速的下载和管理不同版本的库。并且完成**项目虚拟环境的构建（管理所有的环境）**
- 虚拟环境：给代码项目创建一个**隔绝**（**Isolation**）其他项目的环境

### 包管理工具 - anaconda

> 选择anaconda进行包管理工具（同时自带的软件和库可以很好的进行深度学习相关的库安装）

（1）下载安装包，下载地址：[Index of /anaconda/archive/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D)

（2）傻瓜操作进行安装（这里我勾选了将conda加入到系统环境中）

#### 环境管理

> 使用的编辑器是viscode软件，注意在官网中并没有虚拟环境的概念，只是叫做environments

##### 虚拟环境

官方文档部分：[Environments — Anaconda documentation](https://docs.anaconda.com/working-with-conda/environments/)

（1）**安装Python扩展**

（2）**创建Conda环境**： 终端

```Bash
conda create --name <ENV_NAME> //最后一个单词是虚拟环境的名字
```

如果你想在创建环境的同时安装一些包，可以这样操作:

```Bash
conda create -n <ENV_NAME> python=<VERSION> <PACKAGE>=<VERSION>
```

（3）**激活Conda环境**

> 创建的所有虚拟环境都会被conda管理，但是被项目要使用某个虚拟环境，必须将其激活，（?)一次只能有一个环境被激活

```Bash
conda activate <ENV_NAME>
```

（4）共享环境（导出环境配置）

To share an environment and its software packages, you must export your environment’s configurations into a `.yml` file

```Bash
conda env export > environment.yml
```

（5）**选择解释器**

#### packages

（1）install packages

> 注意：**下载包时，包会自动下载到被激活的环境中（如果不做修改）**

```Bash
conda install package-name=2.3.4
```

修改下载到某个env

```Bash
conda install package-name=2.3.4 -n some-environment
```


## 解决方案 - 云端


