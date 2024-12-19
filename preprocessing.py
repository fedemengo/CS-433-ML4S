from utils import (
    process_gray_matter_mask,
    create_4d_volume,
    create_events_df,
    compute_task_fmap,
    compute_bins_threshold,
)
from utils import (
    subject_gm_mask_path,
    subject_task_active_mask_path,
    anat_dir,
)
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel


SMOOTHING = 10
TR = 0.72
ACQUISITION = "RL"
TRAIN_TASK = "motor"

ALL_SUBJECTS = [
    "100307",
    "100408",
    "101107",
    "101309",
    "101915",
    "103111",
    "103414",
    "103818",
    "105014",
    "105115",
    "106016",
    "108828",
    "110411",
    "111312",
    "111716",
    "113619",
    "113922",
    "114419",
    "115320",
    "116524",
    "117122",
    "118528",
    "118730",
    "118932",
    "120111",
    "122317",
    "122620",
    "123117",
    "123925",
    "124422",
    "125525",
    "126325",
    "127630",
    "127933",
    "128127",
    "128632",
    "129028",
    "130013",
    "130316",
    "131217",
    "131722",
    "133019",
    "133928",
    "135225",
    "135932",
    "136833",
    "138534",
    "139637",
    "140925",
    "144832",
    "146432",
    "147737",
    "148335",
    "148840",
    "149337",
    "149539",
    "149741",
    "151223",
    "151526",
    "151627",
    "153025",
    "154734",
    "156637",
    "159340",
    "160123",
    "161731",
    "162733",
    "163129",
    "176542",
    "178950",
    "188347",
    "189450",
    "190031",
    "192540",
    "196750",
    "198451",
    "199655",
    "201111",
    "208226",
    "211417",
    "211720",
    "212318",
    "214423",
    "221319",
    "239944",
    "245333",
    "280739",
    "298051",
    "366446",
    "397760",
    "414229",
    "499566",
    "654754",
    "672756",
    "751348",
    "756055",
    "792564",
    "856766",
    "857263",
    "899885",
]


def subject_run(sid, acquisition, task="MOTOR"):
    return {"id": sid, "acquisition": acquisition, "task": task}


def binary_design_matrix(events_df, TR, n_volumes):
    time_points = np.arange(0, n_volumes * TR, TR)
    conditions = sorted(events_df["trial_type"].unique())
    design_matrix = pd.DataFrame(0, index=time_points, columns=conditions)

    for _, event in events_df.iterrows():
        start_idx = np.searchsorted(time_points, event["onset"])
        end_idx = np.searchsorted(time_points, event["onset"] + event["duration"])
        design_matrix.iloc[start_idx:end_idx, conditions.index(event["trial_type"])] = 1

    return design_matrix


def get_predicted_bold_response(
    glm, active_mask_data, binary_design_matrix, original_bold
):
    predicted = glm.predicted[0]
    predicted_data = predicted.get_fdata()[active_mask_data > 0]
    print(predicted_data.shape)

    plot_voxel_timecourse(predicted_data, binary_design_matrix)

    return predicted_data


def normalize_bold(X):
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)


def get_predicted_neural_activity(
    glm: FirstLevelModel, active_mask_data, binary_design_matrix, original_bold
):
    n_volumes, n_regressors = glm.design_matrices_[0].shape
    n_conditions = n_regressors - 1  # drop constant regressor
    contrast_matrix = np.eye(n_conditions, n_regressors)

    # actual response (amplitude) in the voxel for each of the n_conditions
    betas_data = glm.compute_contrast(
        contrast_matrix, output_type="effect_size"
    ).get_fdata()
    active_voxel_beta = betas_data[active_mask_data > 0].reshape(-1, n_conditions)

    predicted_neural_activity = active_voxel_beta @ binary_design_matrix.T.to_numpy()

    print(predicted_neural_activity.shape)
    # plot_voxel_timecourse(predicted_neural_activity, binary_design_matrix)

    timepoints = original_bold.shape[-1]
    original_data_masked = original_bold[active_mask_data > 0].reshape(-1, timepoints)

    predicted = glm.predicted[0]
    predicted_data = predicted.get_fdata()[active_mask_data > 0]

    return (
        normalize_bold(original_data_masked),
        predicted_neural_activity,
        predicted_data,
    )


def get_predicted_neural_activity_per_condition(
    glm: FirstLevelModel, active_mask_data, binary_design_matrix, original_bold
):
    print(binary_design_matrix.shape)

    n_volumes, n_regressors = glm.design_matrices_[0].shape
    n_conditions = n_regressors - 1  # drop constant regressor
    contrast_matrix = np.eye(n_conditions, n_regressors)

    betas = glm.compute_contrast(contrast_matrix, output_type="effect_size")
    betas_data = betas.get_fdata()
    active_voxel_beta = betas_data[active_mask_data > 0].reshape(-1, n_conditions)

    # binary_design: (T × C) = (284 × 10)
    # betas: (V × C) = (1937 × 10)
    # out output: (C × T × V) = (10 × 284 × 1937) beta for each condition for each ts for each voxel_activiy

    predictions = np.zeros(
        (
            n_conditions,  # C = 10
            active_voxel_beta.shape[0],  # V = 1937
            n_volumes,  # T = 284
        )
    )

    binary_design_matrix = binary_design_matrix.to_numpy()
    for c in range(n_conditions):
        # binary_design[:, c] is (T,)
        # betas[:, c] is (V,)
        # outer product to get (T × V) for this condition
        # basically I want to be able to separate betas across condition (skip '@' = Sum_i beta_i cond_i)
        predictions[c] = np.outer(active_voxel_beta[:, c], binary_design_matrix[:, c])

    return predictions


def voxel_activation_glm(brain_mask, TR, fmri_vols, events):
    fmri_glm = FirstLevelModel(
        mask_img=brain_mask,
        t_r=TR,
        noise_model="ar1",
        standardize=False,
        hrf_model="spm",
        drift_model=None,
        minimize_memory=False,
    )

    return fmri_glm.fit(fmri_vols, events)


def create_active_voxel_mask(
    subject,
    task,
    acquisition,
    smoothing,
    voxel_quantile,
    base_gm_mask_img,
    fmap_img,
    threshold,
):
    task_active_thr_map = subject_task_active_mask_path(
        subject, task, acquisition, smoothing, voxel_quantile
    )
    if os.path.isfile(task_active_thr_map):
        return nib.load(task_active_thr_map)

    fmap_data = fmap_img.get_fdata()
    threshold_mask = fmap_data > threshold

    active_data = fmap_data
    active_data[~threshold_mask] = 0
    active_data[threshold_mask] = 1

    active_img = nib.Nifti1Image(
        active_data, affine=base_gm_mask_img.affine, header=base_gm_mask_img.header
    )
    active_img.to_filename(task_active_thr_map)

    return active_img


def plot_voxel_timecourse(predicted_voxel, binary_design_matrix=None):
    plt.figure(figsize=(20, 5))

    n_voxels = predicted_voxel.shape[0]
    max_h = 1
    for i in range(n_voxels):
        # max_h = max(max_h, np.max(predicted_voxel[i,:]))
        plt.plot(predicted_voxel[i, :], alpha=0.6)

    plt.title("Individual Voxel Timecourses")
    plt.xticks(np.linspace(0, predicted_voxel.shape[1], num=10))
    plt.xlabel("Time")
    plt.ylabel("Signal")

    if binary_design_matrix is not None:
        df = binary_design_matrix
        for col in df.columns:
            plt.fill_between(range(len(df)), df[col] * max_h, label=col, alpha=0.1)
        plt.legend(
            bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=len(df.columns) // 2
        )

    plt.show()


def prepare_subject_sample(
    subject,
    task,
    acquisition,
    smoothing,
    voxel_quantile=98,
    label_extraction=get_predicted_neural_activity,
    drop_non_paradigm=True,
):
    gm_mask = subject_gm_mask_path(subject)
    print("processing subject gray matter mask", gm_mask)
    process_gray_matter_mask(anat_dir, subject, border_size=2, save=True, plot=False)

    print(f"concatenating volumes for task {task}")
    fmri_vols = create_4d_volume(
        subject, task, acquisition, smoothing=smoothing, save=True
    )
    if isinstance(fmri_vols, str):
        fmri_vols = nib.load(fmri_vols)

    print("processing event conditions")
    events = create_events_df(
        subject, task, acquisition, drop_non_paradigm=drop_non_paradigm
    )

    bdesign_matrix = binary_design_matrix(events, TR, fmri_vols.shape[-1])

    print("first GLM to select active voxel")
    activity_glm = voxel_activation_glm(gm_mask, TR, fmri_vols, events)

    n_regressors = activity_glm.design_matrices_[0].shape[1]
    # plot_design_matrix(activity_glm.design_matrices_[0])
    # print(activity_glm.design_matrices_[0])
    n_conditions = n_regressors - 1  # drop constant regressor

    print(f"computing f-map and selecting {voxel_quantile}th voxel")
    contrast_matrix = np.eye(n_conditions, n_regressors)

    fmap = compute_task_fmap(
        subject, task, acquisition, smoothing, activity_glm, contrast_matrix
    )
    threshold = compute_bins_threshold(fmap, n_perc=voxel_quantile, show=False)

    # plot_fmap(fmap, threshold, display_mode="x", task=task, info={"subj": subject}, cut_cords=x_coords)
    # plot_fmap(fmap, threshold, display_mode="y", task=task, info={"subj": subject}, cut_cords=y_coords)
    # plot_fmap(fmap, threshold, display_mode="z", task=task, info={"subj": subject}, cut_cords=z_coords)

    gm_mask = nib.load(gm_mask)
    active_mask = create_active_voxel_mask(
        subject, task, acquisition, smoothing, voxel_quantile, gm_mask, fmap, threshold
    )
    active_mask_data = active_mask.get_fdata()

    return (
        label_extraction(
            activity_glm, active_mask_data, bdesign_matrix, fmri_vols.get_fdata()
        ),
        activity_glm,
        active_mask_data,
    )


def merge_dataset(datasets):
    max_len = max(ds.dims["time"] for ds in datasets)

    padded = []
    for ds in datasets:
        if ds.dims["time"] < max_len:
            pad_idx = np.arange(max_len - ds.dims["time"]) % ds.dims["time"]
            padded_data = {
                var: (
                    ds[var].dims,
                    np.concatenate(
                        [ds[var].values, ds[var].values[..., pad_idx]], axis=-1
                    ),
                )
                for var in ds.data_vars
                if "time" in ds[var].dims
            }

            padded.append(
                xr.Dataset(
                    padded_data,
                    coords={
                        "time": np.arange(max_len),
                        **{c: ds[c] for c in ds.coords if c != "time"},
                    },
                )
            )
        else:
            padded.append(ds)

    return xr.concat(padded, dim="task")  # Changed from 'subject' to 'task'
