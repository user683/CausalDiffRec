import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import numpy as np
from torch.utils.data import Dataset


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


class UserItemDataset(Dataset):
    def __init__(self, interaction_matrix):
        """
        初始化数据集
        :param interaction_matrix: scipy.sparse.csr_matrix，用户-物品交互矩阵
        """
        self.interaction_matrix = interaction_matrix
        self.num_users, self.num_items = interaction_matrix.shape
        self.user_indices = []
        self.pos_item_indices = []
        self.neg_item_indices = []

        # 预处理数据，生成正负样本
        for user_id in range(self.num_users):
            # 使用稀疏矩阵格式获取正样本索引
            pos_items = interaction_matrix[user_id].indices
            remaining_items = list(set(range(self.num_items)) - set(pos_items))
            if not remaining_items:
                continue  # 如果没有剩余的物品可供选择，则跳过
            neg_items = np.random.choice(remaining_items, size=len(pos_items), replace=False)
            self.user_indices.extend([user_id] * len(pos_items))
            self.pos_item_indices.extend(pos_items)
            self.neg_item_indices.extend(neg_items)

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        user_id = self.user_indices[idx]
        pos_item_id = self.pos_item_indices[idx]
        neg_item_id = self.neg_item_indices[idx]

        return user_id, pos_item_id, neg_item_id
