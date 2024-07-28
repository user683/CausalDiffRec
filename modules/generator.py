import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse


# the environment generator
class Graph_Editer(nn.Module):
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n))
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, n, num_sample, k, edge_index, noise_level=0.8):
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(dtype=torch.float16, device=self.device)

        # add nose edge
        num_edges_to_modify = int(A.numel() * noise_level)
        indices_to_modify = torch.randint(0, A.numel(), (num_edges_to_modify,), device=self.device)
        values_to_modify = torch.randint(0, 2, (num_edges_to_modify,), dtype=torch.float16, device=self.device)
        A.view(-1)[indices_to_modify] = values_to_modify

        A_c = torch.ones(n, n, dtype=torch.float16, device=self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float16, device=self.device)
        col_idx = torch.arange(0, n, device=self.device).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        Bk_clone = Bk.clone().to(dtype=torch.float16, device=self.device)
        sum_bk = torch.sum(Bk_clone[S, col_idx], dim=1)
        logsumexp_bk = torch.logsumexp(Bk_clone, dim=0)
        log_p = sum_bk - logsumexp_bk

        return edge_index, log_p
