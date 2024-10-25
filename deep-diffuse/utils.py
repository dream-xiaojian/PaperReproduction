import pickle
import numpy as np
import configparser
import pandas as pd
import networkx as nx
from os.path import join, isfile


# 构建论文中使用的时间间隔
def process_timestamps(timestamps):
    arr = np.asarray(timestamps)
    # 做差 n个元素变成n - 1个元素
    diff = list(np.diff(arr))
    # 将时间间隔转换为分钟
    diff = [d / 60.0 for d in diff]

    # 这里处理重复的差值也不是啊？内层循环有啥作用😀
    for i in range(0, len(diff)-1):
        if diff[i] == diff[i+1]:
            for j in range(0, len(diff)):
                diff[j] += 0.25
    return diff

'''def write_seen_nodes(data_path, seq_len):
    seen_nodes = []
    with open(join(data_path, 'train.txt'), 'r') as read_file:
        for i, line in enumerate(read_file):
            query, cascade = line.strip().split(' ', 1)
            sequence = cascade.split(' ')[::2]
            seen_nodes.extend(sequence)
    with open(join(data_path, 'test.txt'), 'r') as read_file:
        for i, line in enumerate(read_file):
            query, cascade = line.strip().split(' ', 1)
            sequence = cascade.split(' ')[::2]
            seen_nodes.extend(sequence)
    seen_nodes = set(seen_nodes)
    with open(join(data_path, 'seen_nodes.txt'), 'w+') as write_file:
        for node in seen_nodes:
            write_file.write(node + '\n')
    print(len(seen_nodes))'''


def load_graph(data_path):
    # 加载节点（路径拼接）data/seen_nodes.txt
    node_file = join(data_path, 'seen_nodes.txt')
    with open(node_file, 'r') as f:
        seen_nodes = [int(x.strip()) for x in f]

    print(f"最大数", max(seen_nodes));
    print(f"最小数", min(seen_nodes));
    print(f"数量", len(seen_nodes));


    print(type(seen_nodes))
    #seen_nodes是一个列表，里面是所有的node节点
    # builds node index
    node_index = {v: i for i, v in enumerate(seen_nodes)}
    print(node_index)
    print(type(node_index))

    # 构建一个字典：{节点编号：节点编号在seen_nodes中的索引}

    # loads graph
    '''graph_file = join(data_path, 'graph.txt')
    pkl_file = join(data_path, 'graph.pkl')

    if isfile(pkl_file):
        G = pickle.load(open(pkl_file, 'rb'))
    else:
        G = nx.Graph()
        G.name = data_path
        n_nodes = len(node_index)
        G.add_nodes_from(range(n_nodes))
        with open(graph_file, 'r') as f:
            next(f)
            for line in f:
                u, v = map(int, line.strip().split())
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    G.add_edge(u, v)
        pickle.dump(G, open(pkl_file, 'wb+'))'''

    return node_index


def load_instances(data_path, file_type, node_index, seq_len, limit, ratio=1.0, testing=False):
    '''
        生成一个训练集（测试集）list，其中包括多条级联结构体
        [
            { 
                'sequence': prefix_c, 注意这里存储的是节点索引list
                'time': prefix_t, 时间间隔list
                'label_n': label_n,
                'label_t': label_t
            },
            { 
                'sequence': prefix_c, 注意这里存储的是节点索引list
                'time': prefix_t, 时间间隔list
                'label_n': label_n,
                'label_t': label_t
            },
            ...
        ]
    '''
    max_diff = 0
    pkl_path = join(data_path, file_type + '.pkl')
    if isfile(pkl_path):
        instances = pickle.load(open(pkl_path, 'rb'))
    else: 
        file_name = join(data_path, file_type + '.txt')
        instances = []
        with open(file_name, 'r') as read_file:
            for i, line in enumerate(read_file):
                query, cascade = line.strip().split(' ', 1)
                # 从下标为0开始每次隔2个元素取出一个
                cascade_nodes = list(map(int, cascade.split(' ')[::2]))
                # 从下标为1开始每次隔2个元素取出一个
                cascade_times = list(map(float, cascade.split(' ')[1::2]))
                if seq_len is not None:

                    '''
                        为什么这里时间戳的个数要比节点个数多一个？（不是一样的吗）

                        因为最后使用的是时间戳的差值作为输入，因此差值个数 == 时间戳个数 - 1
                        为了节点个数和差值个数相等，所以时间戳个数要比节点个数多一个(?)

                    '''
                    cascade_nodes = cascade_nodes[:seq_len+1]
                    cascade_times = cascade_times[:seq_len+2]
                    if len(cascade_times) == len(cascade_nodes):
                        cascade_nodes.pop()
                    cascade_times = process_timestamps(cascade_times) #处理时间戳

                    # 如果程序不满足这个条件len(cascade_nodes) == len(cascade_times，那么就会报错
                    assert len(cascade_nodes) == len(cascade_times), "Error: {} != {}".format(len(cascade_nodes), len(cascade_times))

                # 将节点编号转换为节点索引（节点索引应该是对应一个节点的嵌入）
                cascade_nodes = [node_index[x] for x in cascade_nodes]
                if not cascade_nodes or not cascade_times:
                    continue
                
                # max_diff应该是最大级联长度
                max_diff = max(max_diff, max(cascade_times))
                '''
                     生成一个list，其中包括一条级联结构体

                     [{ 'sequence': prefix_c, 注意这里存储的是节点索引list
                       'time': prefix_t, 时间间隔list
                       'label_n': label_n,
                       'label_t': label_t
                       }
                    ]
                '''

                ins = process_cascade(cascade_nodes, cascade_times, testing)
                #注意这里是extend, 表示是扩展，不是append
                instances.extend(ins)

                if limit is not None and i == limit:
                    break
        # pickle.dump(instances, open(pkl_path, 'wb+'))
    total_samples = len(instances)
    #随机选择样本索引
    indices = np.random.choice(total_samples, int(total_samples * ratio), replace=False)
    sampled_instances = [instances[i] for i in indices]
    return sampled_instances, max_diff



def process_cascade(cascade, timestamps, testing=False):
    size = len(cascade)
    examples = []
    for i, node in enumerate(cascade):
        if i == size - 1 and not testing:
            return examples
        if i < size - 1 and testing:
            continue
        prefix_c = cascade[: i + 1]
        prefix_t = timestamps[: i + 1]
        # predecessors = set(network[node]) & set(prefix_c)
        # others = set(prefix_c).difference(set(predecessors))

        '''if i == 0:
            times.extend([0.0])
        else:
            # print(i)
            times.extend([(timestamps[i-1] - timestamps[i])])'''

        if not testing:
            label_n = cascade[i + 1] # 最后一个节点的下一个节点索引 （不是回到自己了吗？）
            label_t = timestamps[i+1] #最后一个节点的下一个节点的时间戳
        else:
            label_n = None
            label_t = None

        example = {'sequence': prefix_c, 'time': prefix_t,
                   'label_n': label_n, 'label_t': label_t}

        if not testing:
            examples.append(example)
        else:
            return example


#加载参数函数
def load_params(param_file='params.ini'):
    options = {}
    config = configparser.ConfigParser()
    config.read(param_file)
    #数据文件夹
    options['data_dir'] = config['general']['data_dir']
    #数据集的名字
    options['dataset_name'] = config['general']['dataset_name']
    #每次训练的批次的样本数量
    options['batch_size'] = int(config['general']['batch_size'])
    #序列长度，每次训练时处理的时间步数
    options['seq_len'] = int(config['general']['seq_len'])
    #窗口长度 - 加强版，裁剪序列长度用的
    options['win_len'] = int(config['general']['win_len'])
    #单元类型（不知道干啥的？）
    options['cell_type'] = str(config['general']['cell_type'])
    #训练的轮数
    options['epochs'] = int(config['general']['epochs'])
    #学习率
    options['learning_rate'] = float(config['general']['learning_rate'])
    #应该是隐藏层的大小（维度还是数量？）
    options['state_size'] = int(config['general']['state_size'])
    #是否进行节点预测
    options['node_pred'] = config.getboolean('general','node_pred')
    #是否在每个epoch开始时打乱数据？
    options['shuffle'] = config.getboolean('general', 'shuffle')
    #嵌入层的维度大小 - 256
    options['embedding_size'] = int(config['general']['embedding_size'])
    #采样数量（选批的时候）
    options['n_samples'] = int(config['general']['n_samples'])
    #是否使用注意力机制
    options['use_attention'] = config.getboolean('general', 'use_attention')
    #时间损失函数（均方误差mse）
    options['time_loss'] = str(config['general']['time_loss'])
    #?
    options['num_glimpse'] = int(config['general']['num_glimpse'])
    #？隐藏层的大小
    options['hl_size'] = int(config['general']['h_size'])
    #？隐藏层的大小
    options['hg_size'] = int(config['general']['h_size'])
    #？所有hl合成的hg的大小
    options['g_size'] = int(config['general']['g_size'])
    #位置嵌入的维度
    options['loc_dim'] = int(config['general']['loc_dim'])
    #梯度裁剪的阈值，防止梯度爆炸
    options['clipping_val'] = float(config['general']['clipping_val'])
    #?: 最小学习率，用于学习率调度
    options['min_lr'] = float(config['general']['min_lr'])
    #测试频率，表示每多少个epoch进行一次测试（不进行训练，只是测试）
    options['test_freq'] = int(config['general']['test_freq'])
    #显示频率，表示每多少个epoch显示一次训练进度（训练的损失）
    options['disp_freq'] = int(config['general']['disp_freq'])

    return options


def prepare_minibatch(tuples, inference=False, options=None):
    '''
        准备一个batch的数据
        输入：
        tuples: 已经经过采样的samples的级联list
        [
                    1个级联
                    {  'sequence': prefix_c, 注意这里存储的是节点索引
                       'time': prefix_t,
                       'label_n': label_n,
                       'label_t': label_t
                   }
                   ,
                   2个级联
                   ...
        ]
        输入：
    '''
    seqs = [t['sequence'] for t in tuples]
    times = [t['time'] for t in tuples]
    lengths = list(map(len, seqs)) #每个级联的长度
    n_timesteps = max(lengths) #最大长度
    n_samples = len(tuples) #样本数量
    '''try:
        assert n_timesteps == options['seq_len']
    except AssertionError:
        print(n_timesteps, options['seq_len'])'''

    # prepare sequences data
    # 生成一个级联矩阵数组（都是0），行数为最大长度，列数为样本数量；m（级联长度） * n（每一级联）
    seqs_matrix = np.zeros((options['seq_len'], n_samples)).astype('int32')
    for i, seq in enumerate(seqs):
        # 将每个级联的节点索引填充到矩阵中，从前往后填充每一列，就是说给每一列填充一个级联的节点索引
        seqs_matrix[: lengths[i], i] = seq
    # 转置矩阵，变成m * n的矩阵
    seqs_matrix = np.transpose(seqs_matrix)

    # 生成一个时间矩阵数组（都是0），同理
    times_matrix = np.zeros((options['seq_len'], n_samples)).astype('float32')
    for i, time in enumerate(times):
        times_matrix[: lengths[i], i] = time
    times_matrix = np.transpose(times_matrix) 

    # prepare topo-masks data
    '''topo_masks = [t['topo_mask'] for t in tuples]
    topo_masks_tensor = np.zeros(
        (n_timesteps, n_samples, n_timesteps)).astype(np.float)
    for i, topo_mask in enumerate(topo_masks):
        topo_masks_tensor[: lengths[i], i, : lengths[i]] = topo_mask'''

    # prepare sequence masks，掩码矩阵，用于标识每个级联中有效的时间步。掩码矩阵中的元素为 1.0 表示该时间步有效，为 0.0 表示该时间步无效
    # 掩码矩阵可以很好的处理变长序列
    len_masks_matrix = np.zeros((n_timesteps, n_samples)).astype(np.float)
    for i, length in enumerate(lengths):
        len_masks_matrix[: length, i] = 1.
    len_masks_matrix = np.transpose(len_masks_matrix)

    # prepare labels data
    '''
        label标签是干嘛的？

    '''
    if not inference:
        labels_n = [t['label_n'] for t in tuples]
        labels_t = [t['label_t'] for t in tuples]
        labels_vector_n = np.array(labels_n).astype('int32')
        labels_vector_t = np.array(labels_t).astype('int32')
    else:
        labels_vector_t = None
        labels_vector_n = None

    return (seqs_matrix,
            times_matrix,
            len_masks_matrix,
            labels_vector_n, labels_vector_t)


#加载器
class Loader:
    def __init__(self, data, options=None):
        self.batch_size = options['batch_size']
        self.idx = 0
        self.data = data
        self.shuffle = True
        self.n = len(data)
        self.n_words = options['node_size'] #总结点数
        # 生成一个0到n-1的数组
        self.indices = np.arange(self.n, dtype="int32")
        self.options = options

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __call__(self):
        # 如果shuffle为True并且idx为0
        if self.shuffle and self.idx == 0:
            # 打乱数据
            np.random.shuffle(self.indices)

        # 生成一个batch的索引
        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        # 生成一个batch的样本
        batch_examples = [self.data[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        return prepare_minibatch(batch_examples,
                                 inference=False,
                                 options=self.options)


if __name__ == '__main__':
    # print("utils.py ----------")
    load_graph('data//twitter')