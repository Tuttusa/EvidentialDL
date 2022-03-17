import torch.nn as nn
import torch.nn.functional as F
import torch


class DenseNormal(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseNormal, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.dense = nn.Linear(self.in_dim, 2 * self.out_dim)

    def forward(self, x):
        output = self.dense(x)
        mu, logsigma = output.chunk(2, dim=-1)
        sigma = F.softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)


class DenseNormalGamma(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseNormalGamma, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.dense = nn.Linear(self.in_dim, 4 * self.out_dim)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)

        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)

        return torch.cat([mu, v, alpha, beta], dim=-1)


class DenseDirichlet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseDirichlet, self).__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.dense = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        output = self.dense(x)
        evidence = torch.exp(output)
        alpha = evidence + 1
        prob = alpha / torch.unsqueeze(torch.sum(alpha, dim=1), -1)

        return alpha, prob


class DenseSigmoid(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseSigmoid, self).__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.dense = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        logits = self.dense(x)
        prob = F.sigmoid(logits)
        return [logits, prob]