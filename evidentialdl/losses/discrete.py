import torch


def KL(alpha, nb_classes):
    beta = torch.ones((1, nb_classes))
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def dirichlet_loss(nb_classes):
    def func(target, evidential_output, global_step, annealing_step):
        alpha, preds, epistemic_uncertainty, aleatoric_uncertainty = evidential_output

        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        m = alpha / S

        A = torch.sum((target - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

        annealing_coef = torch.minimum(torch.tensor(1.0), torch.tensor(global_step / annealing_step))

        alp = E * (1 - target) + 1
        C = annealing_coef * KL(alp, nb_classes)
        return torch.mean((A + B) + C)

        """
        check nb_classes
        check softmax for alp
        kl divergence
        """

    return func
