import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_normal_, constant_


class DNN(nn.Module):
    """
    A deep neural network for the reverse process of latent diffusion.
    """

    def __init__(self, in_dims, out_dims, emb_size, env_size, time_type="cat", norm=False, act_func='tanh',
                 dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_emb_dim = emb_size
        self.env_size = env_size
        self.time_type = time_type
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            # in_dims_temp = [self.in_dims[0] + self.time_emb_dim + self.env_size] + self.in_dims[1:]
            in_dims_temp = [self.in_dims[0] + 2 + self.env_size] + self.in_dims[1:]
            # print("DNN里in_dims_temp的维度")
            # print(in_dims_temp)
            # print(self.in_dims[0])
            # print(self.in_dims[1:])
            # print(self.time_emb_dim)
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_modules = []
        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
            self.in_modules.append(nn.Linear(d_in, d_out))
            if act_func == 'tanh':
                self.in_modules.append(nn.Tanh())
            elif act_func == 'relu':
                self.in_modules.append(nn.ReLU())
            elif act_func == 'sigmoid':
                self.in_modules.append(nn.Sigmoid())
            elif act_func == 'leaky_relu':
                self.in_modules.append(nn.LeakyReLU())
            else:
                raise ValueError("Unsupported activation function")
        self.in_layers = nn.Sequential(*self.in_modules)

        self.out_modules = []
        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
            self.out_modules.append(nn.Linear(d_in, d_out))
            if act_func == 'tanh':
                self.out_modules.append(nn.Tanh())
            elif act_func == 'relu':
                self.out_modules.append(nn.ReLU())
            elif act_func == 'sigmoid':
                self.out_modules.append(nn.Sigmoid())
            elif act_func == 'leaky_relu':
                self.out_modules.append(nn.LeakyReLU())
            else:
                raise ValueError("Unsupported activation function")
        self.out_modules.pop()
        self.out_layers = nn.Sequential(*self.out_modules)

        self.dropout = nn.Dropout(dropout)

        self.apply(xavier_normal_initialization)

    def forward(self, x, timesteps, e):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x, dim=-1)
        x = self.dropout(x)

        if isinstance(e, tuple):
            e = torch.cat(e, dim=-1)
        assert isinstance(e, torch.Tensor), f"Expected Tensor as element 2, but got {type(e)}"
        assert e.size(0) == x.size(0), f"Mismatch in batch size between x ({x.size(0)}) and e ({e.size(0)})"
        h = torch.cat([x, emb, e], dim=-1)  # Concatenate x, time embedding, and environment variable
        h = self.in_layers(h)
        h = self.out_layers(h)
        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def xavier_normal_initialization(module):
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
