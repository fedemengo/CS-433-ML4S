import matplotlib.pyplot as plt
import numpy as np

def plot_brain_dist_comparison(original, processed):
    mask1 = (original != 0).astype(int)
    mask2 = (processed != 0).astype(int)
    
    fig = plt.figure(figsize=(12, 3))
    
    # First row - first dataset
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(np.sum(mask1, axis=(1, 2)))
    ax1.set_title('original X')
    ax1.grid(True)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(np.sum(mask1, axis=(0, 2)))
    ax2.set_title('original Y')
    ax2.grid(True)
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(np.sum(mask1, axis=(0, 1)))
    ax3.set_title('original Z')
    ax3.grid(True)
    
    # Second row - second dataset
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(np.sum(mask2, axis=(1, 2)))
    ax4.set_title('processed X')
    ax4.grid(True)
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(np.sum(mask2, axis=(0, 2)))
    ax5.set_title('processed Y')
    ax5.grid(True)
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(np.sum(mask2, axis=(0, 1)))
    ax6.set_title('processed Z')
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_voxel_timecourse(predicted_voxel, binary_design_matrix):
    plt.figure(figsize=(20, 5))

    n_voxels = predicted_voxel.shape[0]
    max_h = 1
    # for i in range(n_voxels):
        # max_h = max(max_h, np.max(predicted_voxel[i,:]))
        # plt.plot(predicted_voxel[i,:], alpha=0.6)
    
    
    df = binary_design_matrix
    for col in df.columns:
        plt.fill_between(range(len(df)), df[col] * max_h, label=col, alpha=0.1)

    plt.title('Individual Voxel Timecourses')
    plt.xticks(np.linspace(0, predicted_voxel.shape[1], num=10))
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=len(df.columns)//2)
    plt.show()


import random
import matplotlib.pyplot as plt

def plot_voxels(bold, block, pred, n_samples=10, seed=42, print_x=False):
    random.seed(seed)

    print(bold.shape, block.shape, pred.shape)

    n_samples = min(n_samples, len(bold))
    sample_indices = random.sample(range(len(bold)), n_samples)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_samples))
    
    plt.figure(figsize=(20, 5))
    
    for i, idx in enumerate(sample_indices):
        t = range(len(bold[idx]))
        if print_x:
            plt.plot(t, bold[idx], color=colors[i], linestyle='-', alpha=0.7, linewidth=0.5, label=f'Input {idx}')
        plt.plot(t, block[idx], color=colors[i], linestyle='-', alpha=0.7, label=f'True Sequence {idx}')
        plt.plot(t, pred[idx], color=colors[i], linestyle='--', alpha=0.7, label=f'Predicted {idx}')
    
    plt.xticks(np.arange(0, len(block[0]), step=max(1, len(block[0])//20)))
    plt.title(f'True vs Predicted Values (Sample of {n_samples} sequences)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    if n_samples < 10:
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()