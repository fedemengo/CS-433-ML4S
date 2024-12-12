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


def temporal_scale(X, y, scale_range=(0.8, 1.2), ratio=0.3, keep_len=True):
    n_voxels, time_points, _ = X.shape
    
    if not keep_len:
        new_length = int(time_points * np.random.uniform(*scale_range))
        print("new_length", new_length)
        scaled_X = torch.nn.functional.interpolate(X.transpose(1,2), size=new_length).transpose(1,2)
        scaled_y = torch.nn.functional.interpolate(y.unsqueeze(1), size=new_length).squeeze(1)
    else:
        scaled_X, scaled_y = X.clone(), y.clone()
    
    for vox_idx in np.random.choice(range(n_voxels), size=int(n_voxels * ratio), replace=False):
        new_length = int(time_points * np.random.uniform(*scale_range)) 
        x_interp = torch.nn.functional.interpolate(X[vox_idx].T.unsqueeze(0), size=new_length).squeeze().T.unsqueeze(-1)
        y_interp = torch.nn.functional.interpolate(y[vox_idx].unsqueeze(0).unsqueeze(0), size=new_length).squeeze()
        
        if keep_len:
            if new_length < time_points:
                repeats = int(np.ceil(time_points / new_length))
                x_interp = x_interp.repeat(repeats, 1)[:time_points]
                y_interp = y_interp.repeat(repeats)[:time_points]
            else:
                x_interp = x_interp[:time_points]  # Only trim if longer
                y_interp = y_interp[:time_points]
            scaled_X[vox_idx] = x_interp
            scaled_y[vox_idx] = y_interp
        else:
            scaled_X[vox_idx] = x_interp
            scaled_y[vox_idx] = y_interp
            
    return scaled_X, scaled_y


