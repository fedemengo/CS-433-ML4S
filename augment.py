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


def augment_data(X_train, Y_train, shift_range=(-20, +20), amplitude_range=(0.5, 2), noise_std=(0.05,0.1), stretch_ratio=0.6, plot_times=False):
    """
    Augment the data by shifting and scaling the time series data.
    
    Parameters:
    - X_train, Y_train Tensors for the BOLD signal and predicted signals (Y).
    - shift_range: Tuple defining the range for temporal shifts (in number of time steps).
    - amplitude_range: Tuple defining the range for amplitude scaling (scaling factor).
    
    Returns:
    - Augmented tensors for X, Y
    """
    print("Data augmentation")
    # Apply time shifts and amplitude scaling to the training set
    augmented_X = []
    augmented_Y = []
    num_samples = X_train.shape[0]
    if plot_times:
        x,y = X_train[0,:],Y_train[0,:]
        plt.plot(x,label="original")
        plt.plot(y,label='block')
        plt.title("Original signal")

        plt.legend()
        plt.show()

    for i in range(num_samples):
        # 1. Apply random time shift
        shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        shifted_X = torch.roll(X_train[i], shifts=shift, dims=0)
        shifted_Y = torch.roll(Y_train[i], shifts=shift, dims=0)
        if plot_times:

            x,y = shifted_X,shifted_Y
            plt.plot(x,label="original")
            plt.plot(y,label='block')            
            plt.title("Signal after shift")

            plt.legend()
            plt.show()
        

        # 2. Apply random amplitude scaling
        scale_factor = np.random.uniform(amplitude_range[0], amplitude_range[1])
        scaled_X = shifted_X * scale_factor
        scaled_Y = shifted_Y * scale_factor
        if plot_times:

            x,y = scaled_X,scaled_Y
            plt.plot(x,label="original")
            plt.plot(y,label='block')
            plt.title("Signal after scaling")
            plt.legend()
            plt.show()
                
        # 4. Add Gaussian noise
        noise_level=np.random.uniform(noise_std[0], noise_std[1])
        noise_std_value = noise_level*torch.max(scaled_Y).item()
        noise_X = torch.normal(mean=0, std=noise_std_value, size=scaled_X.shape, device=scaled_X.device)
        noisy_X = scaled_X + noise_X

        if plot_times:
            x,y = noisy_X,scaled_Y
            plt.plot(x,label="original")
            plt.plot(y,label='block')
            plt.title(f"Signal with added noise, noise added: {20*np.log10((noise_level**2/torch.mean(scaled_X**2).item())):.4f}, noise lev{noise_level:.4f}")

            plt.legend()
            plt.show()
            plot_times-=1
        # Store augmented samples
        augmented_X.append(noisy_X)
        augmented_Y.append(scaled_Y)

    # Convert lists back to tensors
    augmented_X_tensor = torch.stack(augmented_X)
    augmented_Y_tensor = torch.stack(augmented_Y)
    
    return augmented_X_tensor, augmented_Y_tensor

