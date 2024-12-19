from preprocessing import prepare_subject_sample, SMOOTHING, TR, ACQUISITION, ALL_SUBJECTS, DATASET_DIR
from preprocessing import get_predicted_bold_response, get_predicted_neural_activity, plot_voxel_timecourse
from nilearn.plotting import plot_design_matrix
from utils import pjoin
import xarray as xr
import numpy as np
import os
from itertools import product

def normalize(X):
    epsilon = 1e-8
    return (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + epsilon)

def task_dataset(subjects, task):
    datasets = []

    dataset_path = pjoin(DATASET_DIR, f'dataset_{task}_{len(subjects)}_subjects_normed.nc')
    if os.path.isfile(dataset_path):
        print("dataset exists, skipping", dataset_path)
        return

    for acquisition in ["RL", "LR"]:
        for subj in subjects:
            (norm_bold, block, conv), glm, mask = prepare_subject_sample(
                subject=subj, 
                task=task, 
                acquisition=acquisition, 
                smoothing=smoothing, 
                voxel_quantile=98, 
                label_extraction=get_predicted_neural_activity,
                drop_non_paradigm=False,
            )
            norm_block, norm_conv = normalize(block), normalize(conv)
    
            dataset = xr.Dataset({
                'X': (['voxel', 'time'], norm_bold),  # original
                'Y': (['voxel', 'time'], norm_block),  # predicted
                'Y_conv': (['voxel', 'time'], norm_conv),  # fitted
            },
            coords={
                'subject': f"{subj}_{acquisition}",
                'task': task,
                'voxel': np.arange(norm_bold.shape[0]),
                'time': np.arange(norm_bold.shape[1]),
            },
            attrs={
                'description': f'Dataset for subject {subj}, task {task}, acquisition {acquisition}'
            })
    
            datasets.append(dataset)

    final_dataset = xr.concat(datasets, dim='subject', fill_value=np.nan)

    print("SAVING dataset to", dataset_path)
    final_dataset.to_netcdf(dataset_path)


if __name__ == "__main__":

    np.random.seed(42)

    tasks = ["GAMBLING"]
    n_subjs = 30
    
    print("select subset of subjets")
    ss_subjs = np.random.choice(ALL_SUBJECTS, size=n_subjs, replace=False)
    for task in tasks:
        print(ss_subjs, task)
        task_dataset(ss_subjs, task)
        print("done with task", task)
