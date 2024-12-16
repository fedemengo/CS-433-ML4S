import torch
import numpy as np
from torch import nn

def gaussian_tv_penalty(predictions, alpha):
    diffs = predictions[:, 1:] - predictions[:, :-1]
    weights = torch.sqrt(2 * alpha * torch.e) * torch.exp(-alpha * diffs**2)
    return torch.mean(torch.abs(diffs) * weights)

def anti_constant_penalty(predictions, beta=1):
    # Encourages non-constant solutions by penalizing non unitary variances - input and target data needs to be normalized
    varmean= torch.mean(torch.var(predictions,dim=1))
    return torch.pow(varmean-1,2) # Most likely better

def combined_penalty(predictions, alpha, beta, lambda_tv, lambda_const):
    tv_loss = gaussian_tv_penalty(predictions, alpha)
    const_loss = anti_constant_penalty(predictions, beta)
    return (lambda_tv * tv_loss + lambda_const * const_loss)/(lambda_tv + lambda_const)

def blocky_loss(alpha=8.0, beta=1.0, lambda_tv=1.0, lambda_const=1.0, lambda_val=2.6157):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    lambda_tv = torch.tensor(lambda_tv)
    lambda_const = torch.tensor(lambda_const)
    lambda_val = torch.tensor(lambda_val)

    print(f"blocky loss params alpha={alpha} beta={beta} lambda_tv={lambda_tv} lambda_const={lambda_const} lambda_val={lambda_val}")

    def loss_fn(predictions, targets):
        criterion = nn.L1Loss()
        loss_data = criterion(predictions, targets)
        smoothness_loss = combined_penalty(predictions, alpha, beta, lambda_tv, lambda_const)
        return loss_data + lambda_val * smoothness_loss
    return loss_fn
