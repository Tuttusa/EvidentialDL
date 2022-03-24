import torch
import numpy as np


def mse(y, y_pred, reduce=True):
    ax = list(range(1, len(y.shape)))

    mse = torch.mean((y - y_pred) ** 2, dim=ax)

    return torch.mean(mse) if reduce else mse


def rmse(y, y_):
    rmse = torch.sqrt(torch.mean((y - y_) ** 2))
    return rmse


def gaussian_nll(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))
    logprob = -torch.log(sigma) - 0.5 * torch.log(2 * np.pi) - 0.5 * ((y - mu) / sigma) ** 2
    loss = torch.mean(-logprob, dim=ax)
    return torch.mean(loss) if reduce else loss


def gaussian_nll_logvar(y, mu, logvar, reduce=True):
    ax = list(range(1, len(y.shape)))

    log_liklihood = 0.5 * (
            -torch.exp(-logvar) * (mu - y) ** 2 - torch.log(2 * torch.tensor(np.pi, dtype=logvar.dtype)) - logvar
    )
    loss = torch.mean(-log_liklihood, axis=ax)
    return torch.mean(loss) if reduce else loss


def nig_nll(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(np.pi / v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll


def kl_nig(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * torch.square(mu2 - mu1)) \
         + 0.5 * v2 / v1 \
         - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
         - 0.5 + a2 * torch.log(b1 / b2) \
         - (torch.lgamma(a1) - torch.lgamma(a2)) \
         + (a1 - a2) * torch.digamma(a1) \
         - (b1 - b2) * a1 / b1
    return KL


def nig_reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = torch.abs(y - gamma)

    if kl:
        kl = kl_nig(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + (alpha)
        reg = error * evi

    return torch.mean(reg) if reduce else reg


def evidential_regression_loss(coeff=1.0):
    def func(y_true, evidential_output):
        gamma, v, alpha, beta, aleatoric, epistemic = evidential_output
        loss_nll = nig_nll(y_true, gamma, v, alpha, beta)
        loss_reg = nig_reg(y_true, gamma, v, alpha, beta)
        return loss_nll + coeff * loss_reg

    return func
