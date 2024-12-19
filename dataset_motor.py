from utils import paradigm_dir, derivatives_dir, DATASET_DIR
from utils import get_subject_ids

from preprocessing import SMOOTHING
from preprocessing import (
    subject_run,
    prepare_subject_sample,
)

from os.path import join as pjoin
import numpy as np
import pandas as pd
import xarray as xr


def get_block_and_neural_activity(glm, active_mask_data, bdesign_matrix, original_data):
    n_volumes, n_regressors = glm.design_matrices_[0].shape
    n_conditions = n_regressors - 1  # drop constant regressor
    contrast_matrix = np.eye(n_conditions, n_regressors)

    # actual response (amplitude) in the voxel for each of the n_conditions
    betas_data = glm.compute_contrast(
        contrast_matrix, output_type="effect_size"
    ).get_fdata()
    active_voxel_beta = betas_data[active_mask_data > 0].reshape(-1, n_conditions)
    print("Bdesign shape: ", bdesign_matrix.T.to_numpy().shape)
    print("Active beta: ", active_voxel_beta.shape)
    predicted_neural_activity = active_voxel_beta @ bdesign_matrix.T.to_numpy()

    print(predicted_neural_activity.shape)
    # plot_voxel_timecourse(predicted_neural_activity, binary_design_matrix)
    predicted = glm.predicted[0]
    predicted_data = predicted.get_fdata()[active_mask_data > 0]
    print(predicted_data.shape)

    # plot_voxel_timecourse(predicted_data, binary_design_matrix)

    timepoints = original_data.shape[-1]
    original_data_masked = original_data[active_mask_data > 0].reshape(-1, timepoints)

    return original_data_masked, predicted_neural_activity, predicted_data


def main():
    print("generating motor dataset")

    types_acq = ["RL", "LR"]
    sub_list = get_subject_ids(paradigm_dir)
    print(sub_list)
    subjects = [
        subject_run(sub, type_acq) for sub in sub_list for type_acq in types_acq
    ]
    print(subjects[0])

    df = pd.read_csv(pjoin(derivatives_dir, "sorted_subjects_motor.csv"))
    df.head()
    keep_quantile = 0.7  # Keep only % of observations
    thr = np.quantile(df["activation"], keep_quantile)
    keep_df = df[df["activation"] > thr]
    keep_df.tail()
    subjects_kept = []
    for entry in keep_df.itertuples():
        subjects_kept.append(
            subject_run(
                sid=entry.subject, acquisition=entry.acquisition, task=entry.task
            )
        )

    # Initialize an empty list to store datasets
    datasets = []
    i = 0
    for entry in subjects_kept:
        print(f"Processing subject {i} out of {len(subjects_kept)}")
        i = i + 1
        subject_id = str(entry["id"])
        acquisition = entry["acquisition"]
        task = entry["task"]

        X_list, Y_list, Y_conv_list = prepare_subject_sample(
            subject_id,
            task,
            acquisition,
            SMOOTHING,
            voxel_quantile=98,
            label_extraction=get_block_and_neural_activity,
        )

        # xarray dataset for this subject
        dataset = xr.Dataset(
            {
                "X": (["voxel", "time"], X_list),  # original timeseries
                "Y": (["voxel", "time"], Y_list),  # predicted timeseries
                "Y_conv": (["voxel", "time"], Y_conv_list),  # fittedd timeseries
            },
            coords={
                "subject": f"{subject_id}_{acquisition}",
                "task": task,
                "voxel": np.arange(X_list.shape[0]),
                "time": np.arange(X_list.shape[1]),
            },
            attrs={
                "description": f"Dataset for subject {subject_id}, task {task}, acquisition {acquisition}"
            },
        )

        datasets.append(dataset)

    final_dataset = xr.concat(datasets, dim="subject", fill_value=np.nan)
    final_dataset.to_netcdf(
        pjoin(
            DATASET_DIR, f"dataset_{task}_{len(subjects_kept)}_subjects_both_noncens.nc"
        )
    )


if __name__ == "__main__":
    main()
