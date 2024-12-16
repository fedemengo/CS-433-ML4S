import torch
import numpy as np
from torch import nn

def gaussian_tv_penalty(predictions, alpha=8):
    # max is at 1/sqrt(2*alpha), needs to be searched for max gain
    # diffs is a first derivative approx
    diffs = predictions[:,1:,:] - predictions[:,:-1,:]

    # exponential weights to punish small variations
    weights = np.sqrt(2*alpha*np.e)*torch.exp(-alpha * diffs**2)

    return torch.mean(torch.abs(diffs) * weights) # normalized to have max value of 1 and min of zero, 

def anti_constant_penalty(predictions):
    # Encourages non-constant solutions by penalizing non unitary variances - input and target data needs to be normalized
    varmean= torch.mean(torch.var(predictions,dim=1))
    return torch.pow(varmean-1,2) # Most likely better
    
def combined_penalty(predictions, alpha=8, lambda_tv=1, lambda_const=1):
    tv_loss = gaussian_tv_penalty(predictions, alpha)
    const_loss = anti_constant_penalty(predictions)
    return (lambda_tv * tv_loss + lambda_const * const_loss)/(lambda_tv+lambda_const)


# For training plug and play -- maybe?
def blocky_loss_criterion(predictions, ground_truth, data_fidelity_criterion=nn.L1Loss(), smoothness_criterion=combined_penalty, lambda_val=2.6):
    return data_fidelity(predictions,ground_truth) + lambda_val * smoothness_criterion(predictions)

# for training as used until now
def blocky_loss(data_loss, smoothness_loss, lambda_val=2.6):
    return data_loss + lambda_val * smoothness_loss
