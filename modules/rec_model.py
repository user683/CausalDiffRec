import torch
import torch.nn as nn
import numpy as np


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)


class LGCN_Encoder(nn.Module):
    def __init__(self, user_num, n_layers, norm_adj, user_embeddings=None, item_embeddings=None):
        super(LGCN_Encoder, self).__init__()
        self.user_num = user_num
        self.layers = n_layers
        self.norm_adj = norm_adj

        self.embedding_dict = nn.ParameterDict({
            # 'user_emb': nn.Parameter(torch.tensor(user_embeddings, dtype=torch.float32, requires_grad=True)),
            # 'item_emb': nn.Parameter(torch.tensor(item_embeddings, dtype=torch.float32, requires_grad=True)),
            'user_emb': nn.Parameter(user_embeddings.clone().detach().requires_grad_(True)),
            'item_emb': nn.Parameter(item_embeddings.clone().detach().requires_grad_(True))
        })
        self.sparse_norm_adj = self.convert_sparse_mat_to_tensor(self.norm_adj).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @staticmethod
    def convert_sparse_mat_to_tensor(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
        indices = torch.from_numpy(np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mat.data)
        shape = torch.Size(sparse_mat.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.user_num]
        item_all_embeddings = all_embeddings[self.user_num:]
        return torch.cat([user_all_embeddings, item_all_embeddings])
