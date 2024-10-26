#复现/论文/代码

## 介绍😀

复现论文为（ccf-b，2018年发表）《DeepDiffuse: Predicting the ‘Who’ and ‘When’  in Cascades》[deepdiffuse-icdm18.pdf](https://faculty.cc.gatech.edu/~badityap/papers/deepdiffuse-icdm18.pdf)  。其论文代码：[raihan2108/deep-diffuse](https://github.com/raihan2108/deep-diffuse)

不同于传统的加强数据信息的密度，该论文主要仅仅利用两个信息属性进行传播预测
- 被传播的用户节点
- 传播的时间

其结构主要Cascade Analyzer Network和Cascade Predictor Network (CPN)组成，从代码层面看分别对应着encoder和decoder，其中deconder部分使用的是legacy_seq2seq的rnn_decoder部分。encoder部分则构建新的模块（序列切割，RNN，attention，位置编码）进行encoder操作。

## 复现工作

- 构建配置文件并运行结果 
- 代码注释
- 代码层面：数据处理理解
- 代码层面：模型构建理解（**部分完成**）
- 存在问题

## 配置与运行

> 如何将项目跑起来
#### 配置

> 因为DeepDiffuse（python3.6.2）是并没有给配置库文件，因此根据其python版本和发表时间，构建了配置库文件见requirements.txt

导入配置库

```Shell
pip install -r requirements.txt
```

#### 运行

运行main.py模块，其结果展示在glimpse-twitter.log日志中，同时params.ini文件可以对模型的参数进行配置

![image.png](https://cdn.jsdelivr.net/gh/dream-xiaojian/DrawingBed@master/20241026094808.png)


## 数据处理代码理解

### 数据介绍

> 其数据在文件夹的data/twitter下，包括4个文件

数据文件中，其中graph.txt并没有在代码中使用（注释掉了），test.txt和train.txt为twitter数据集自带的，而seen_nodes.txt文件为代码运行过程中产生的。**graph.txt并不是本模型使用的内容, 似乎是在对比实验中对比模型的输入**

#### 数据格式

- test.txt / train.txt：每一行为一个cascade，每一行元素通过空格隔开，其中第一个元素为query（未使用），后面的元素按照：（节点，节点时间戳）的组合重复
- seen_nodes.txt: 根据test.txt和 train.txt将所有节点汇总，每一行为节点编号（**节点并没有额外的信息, 只有节点的编号, 嵌入的时候直接使用的是截断正态分布的方式**）

### 数据流动

> 对应的文件是utils.py

#### overview

![image.png](https://cdn.jsdelivr.net/gh/dream-xiaojian/DrawingBed@master/20241025200514.png)

#### 细节

通过对代码的阅读，我认为这里主要做的是两部分：

（1）将所有**节点编号映射到索引** dist{节点编号:索引, ...}

> 为什么？**离散化**

对数据进行简单的代码分析：

```python
   print(f"最大数", max(seen_nodes)); //137039
   print(f"最小数", min(seen_nodes)); //37
   print(f"数量", len(seen_nodes)); //5942
```

 可以看到在5942个数中，其值域跨度为[37, 137039]。也就是数不是自然数。根据论文中的通过节点编号去获得其嵌入，那么二维数组的范围就要开到137039。可以发现其中很多节点编号都是不存在的（数据呈现离散化的情况），因此通过映射的方式将5942个数分别映射到0 ~ 5942。**既没有失去其独特性也保证空间可控。下面是核心代码

```python
node_index = {v: i for i, v in enumerate(seen_nodes)}
```

（2）将数据处理成级联矩阵

> 矩阵的一行表示一个级联，其元素存储这级联的信息

主要包括三个矩阵（剩下的2个label矩阵没整明白）：
 ![image.png](https://cdn.jsdelivr.net/gh/dream-xiaojian/DrawingBed@master/20241025194750.png)
#### 注意事项

（1）时间方面使用的是**时间间隔**，所以一个级联中节点数要比节点的时间戳少1，使得时间间隔数 === 节点数

## 模型代码理解

### overview

![image.png](https://cdn.jsdelivr.net/gh/dream-xiaojian/DrawingBed@master/20241025204711.png)

### 细节

#### tensorboard展开计算图

（1）创建 TensorBoard 日志记录器

```python
self.logdir = "./Logs"
self.summary_writer = tf.summary.FileWriter(self.logdir)
```

（2）记录计算图

```python
self.summary_writer.add_graph(tf.get_default_graph())
```

（3）运行

```bash
tensorboard --logdir=./Logs --host=127.0.0.1   
```

> 虽然通过tensorboard将其计算图打开，还是太复杂


## 存在问题

（1）目前模型部分代码细节（tensorflow框架下进行）并不是很理解

（2）公式和相应的代码的不太理解（主要是在目标函数方面）

（3）代码复现结果并没有出来，目前正改为云端跑