import numpy as np
import torch

def select_augmentation():
    return 42


def shift(X, y, shift_range=(-20, +20), groups=5, ratio=0.3):
    batch_size = X.shape[0]
    group_size = batch_size // groups
    
    shifted_X = X.clone()
    shifted_y = y.clone()
    
    for g in range(groups):
        start_idx = g * group_size
        end_idx = start_idx + group_size if g < groups - 1 else batch_size
        
        num_to_shift = int((end_idx - start_idx) * ratio)
        shift_indices = np.random.choice(range(start_idx, end_idx), size=num_to_shift, replace=False)
        
        for idx in shift_indices:
            shift = np.random.randint(shift_range[0], shift_range[1] + 1)
            shifted_X[idx] = torch.roll(X[idx], shifts=shift, dims=0)
            shifted_y[idx] = torch.roll(y[idx], shifts=shift, dims=0)
    
    return shifted_X, shifted_y