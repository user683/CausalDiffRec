import torch
import torch.nn as nn
import torch.nn.functional as F

# variational graph autoencoder
class EVAE(nn.Module):
    def __init__(self, latent_size=16, hidden_size=16):
        super(EVAE, self).__init__()

        # encoder
        self.fc1_mu = nn.Linear(latent_size, hidden_size)
        self.fc1_log_std = nn.Linear(latent_size, hidden_size)

        # decoder
        self.fc2 = nn.Linear(hidden_size, latent_size)

        #
        self.mu_prior = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.log_std_prior = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def encode(self, z):
        h1 = F.relu(z)
        mu = self.fc1_mu(h1)
        log_std = self.fc1_log_std(h1)
        return mu, log_std

    def decode(self, h):
        h3 = F.relu(h)
        recon = self.fc2(h3)
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, z):
        mu, log_std = self.encode(z)  #
        h = self.reparametrize(mu, log_std)
        recon = self.decode(h)
        return recon, mu, log_std

    def kl_divergence(self, mu, log_std):
        std = torch.exp(log_std)
        std_prior = torch.exp(self.log_std_prior)
        mu_prior = self.mu_prior

        kl = torch.sum(
            0.5 * (2 * (self.log_std_prior - log_std) +
                   (std ** 2 + (mu - mu_prior) ** 2) / std_prior ** 2 - 1)
        )
        return kl


def evae_loss(recon_x, x, kl_div):
    # print(recon_x.size())
    # print(x.size())
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    return recon_loss + kl_div


