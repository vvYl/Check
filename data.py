import pickle
from os import path

import numpy as np


class Data(object):
    def __init__(self, dataname, limit, topk=10):
        """
        初始化数据对象
        :param dataname: 数据集名称
        :param limit: 使用用户的限制数量
        :param topk: 度数排名前 topk 的用户序号
        """
        self.data = None  # 邻接矩阵数据
        self.dataname = dataname  # 数据集名称
        self.dict_size = 0  # 节点数量
        self.limit = limit  # 限制使用的边数量
        self.topk = topk  # 返回度数排名前topk的用户序号
        self.adj_matrix_file = f'{dataname}_adj.pkl'  # 保存邻接矩阵的文件名
        self.degree_file = f'{dataname}_degree.pkl'  # 保存度数文件名

        # 加载或生成数据
        self.load_or_generate_data()

    def load_or_generate_data(self):
        """
        加载或生成邻接矩阵数据
        """
        if not path.exists(self.adj_matrix_file):
            print("Generating adjacency matrix...")
            self.generate_adj_matrix()
        else:
            print("Loading adjacency matrix from file...")
            self.data = pickle.load(open(self.adj_matrix_file, 'rb'))

        # 计算度数
        degrees = np.sum(self.data, axis=1)
        pickle.dump(degrees, open(self.degree_file, 'wb'))

        # 输出度数排名前 topk 的用户序号
        topk_users = np.argsort(degrees)[-self.topk:][::-1]
        # print(f"Top-{self.topk} users by degree: {topk_users}")

    def generate_adj_matrix(self):
        """
        根据数据文件生成邻接矩阵
        """
        # 读取图数据
        user_file_name = '%s-data/%s.txt' % (self.dataname, self.dataname)
        edges = []
        max_node = 0
        count = 0

        with open(user_file_name, 'r') as f:
            for line in f:
                if count >= self.limit:  # 如果读取的边数达到限制，停止读取
                    break
                if len(line.strip()) == 0 or line[0] == '#':
                    continue
                i, j = map(int, line.strip().split())
                edges.append((i, j))
                max_node = max(max_node, i, j)
                count += 1

        self.dict_size = max_node + 1
        adj_matrix = np.zeros((self.dict_size, self.dict_size), dtype=int)

        # 填充邻接矩阵
        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # 无向图

        # 保存邻接矩阵
        pickle.dump(adj_matrix, open(self.adj_matrix_file, 'wb'))
        self.data = adj_matrix
        print(f"Adjacency matrix saved to {self.adj_matrix_file}")