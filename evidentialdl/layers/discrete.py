import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical

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

        S = torch.unsqueeze(torch.sum(alpha, dim=1), -1)
        K = alpha.shape[-1]

        prob = alpha / S
        epistemic_uncertainty = K / S
        aleatoric_uncertainty = Categorical(probs = p_tensor).entropy()

        return alpha, prob, epistemic_uncertainty, aleatoric_uncertainty


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
