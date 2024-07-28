import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class Model(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, device, num_nodes):
        super(Model, self).__init__()
        self.device = device
        self.num_nodes = num_nodes

        # Graph Convolution layers
        self.base_gc = GCNConv(in_dim, hidden1_dim)
        self.mean_gc = GCNConv(hidden1_dim, hidden2_dim)
        self.log_std_gc = GCNConv(hidden1_dim, hidden2_dim)

        # Decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden2_dim + in_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, num_nodes),
            nn.ReLU()
        )

    def encoder(self, features, edge_index):
        h = self.base_gc(features, edge_index).to(self.device)
        self.mean = self.mean_gc(h, edge_index)
        self.log_std = self.log_std_gc(h, edge_index)

        gaussian_noise = torch.randn(self.mean.size()).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std)

        assert sampled_z.shape[0] == features.shape[0], "Shape mismatch between z and features"

        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, features, edge_index):
        z = self.encoder(features, edge_index)
        adj_rec = self.decoder(z)
        adj_filter = torch.nan_to_num(adj_rec, nan=0.0)
        return adj_filter

