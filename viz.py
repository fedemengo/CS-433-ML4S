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


def plot_voxel_timecourse(predicted_voxel, regressors):
    n_voxels = predicted_voxel.shape[0]
    plt.figure(figsize=(20, 5))
    
    time = np.arange(len(regressors))
    plt.fill_between(time, regressors, alpha=0.2, label='regressor')
    discared = 0
    for i in range(n_voxels):
        if np.any(np.abs(predicted_voxel[i,:]) > 20):
            discared += 1
            continue
        plt.plot(predicted_voxel[i,:])
    
    print(f"discarded {discared} / {n_voxels}")
    plt.title('Individual Voxel Timecourses')
    plt.xticks(np.linspace(0, predicted_voxel.shape[1], num=10))
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    