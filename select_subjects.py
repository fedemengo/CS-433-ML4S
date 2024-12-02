from utils import show_task_activation, plot_fmap
from utils import process_gray_matter_mask, create_4d_volume, create_events_df, compute_task_fmap, compute_bins_threshold
from utils import subject_gm_mask_path, paradigm_dir, subject_task_active_mask_path, subject_task_regressor_path, subject_task_sample_prefix, subject_task_fmap
from utils import mkdir_no_exist, x_coords, y_coords, z_coords
from utils import resample_mask, get_atlas_activation, get_subject_ids
from viz import plot_voxel_timecourse

from matplotlib.backends.backend_pdf import PdfPages
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from glob import glob

import nibabel as nib
from nilearn import image
from nilearn.image import resample_to_img
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, run_glm
from nilearn.plotting import (
    plot_design_matrix,
    plot_stat_map,
    plot_img,
    plot_glass_brain,
)

HCP_dir = '/media/miplab-nas2/HCP-Data'
paradigm_dir = pjoin(HCP_dir,'HCP_100unrelated_TaskParadigms')
fMRI_dir = pjoin(HCP_dir,'HCP_100unrelated_preprocessed_ERG/data')

project_dir = '/media/RCPNAS/Data2/CS-433-ML4S/'
project_data_dir = pjoin(project_dir, 'data')
project_dataset_dir = pjoin(project_dir, "dataset")

derivatives_dir = pjoin(project_data_dir, 'derivatives')
motor_atlas_old = "/media/RCPNAS/Data2/CS-433-ML4S/HMAT/HMAT.nii"
task_atlas_path = {"MOTOR":resample_mask(motor_atlas_old)}
anat_dir = pjoin(derivatives_dir, 'anat')
func_dir = pjoin(derivatives_dir, 'func')
events_dir = pjoin(project_data_dir, 'events')

mkdir_no_exist(anat_dir)
mkdir_no_exist(func_dir)
mkdir_no_exist(events_dir)
mkdir_no_exist(project_dataset_dir)

subject = "127630"
task = "MOTOR"
acq = "RL"
smoothing = 10
TR = 0.72 

def subject_run(sid, acquisition):
    return {"id": sid, "acquisition": acquisition}

def voxel_activation_glm(mask_img, TR, fmri_vols, events):
    fmri_glm = FirstLevelModel(
        mask_img=mask_img,
        t_r=TR,
        noise_model='ar1', # or ols 
        standardize=False,
        hrf_model='spm',
        drift_model=None, # not necessary, nuisance covariates have already been removed
    )

    print(f"fitting GLM for task {task}")
    fitted_glm = fmri_glm.fit(fmri_vols, events)
    return fitted_glm
def raw_voxel_activity_glm(mask_img, TR, fmri_vols, events):
    fmri_glm = FirstLevelModel(
        mask_img=mask_img,
        t_r=TR,
        noise_model='ar1', # or ols 
        standardize=False,
        hrf_model=None,
        drift_model=None, # not necessary, nuisance covariates have already been removed
        minimize_memory=False,
    )

    print(f"fitting GLM for task {task}")
    fitted_glm = fmri_glm.fit(fmri_vols, events)
    return fitted_glm
def create_active_voxel_mask(subject, task, acquisition, smoothing, voxel_quantile, base_gm_mask_img, fmap_img, threshold):
    task_active_thr_map = subject_task_active_mask_path(subject, task, acquisition, smoothing, voxel_quantile)
    if os.path.isfile(task_active_thr_map):
        return nib.load(task_active_thr_map)

    gm_mask_data = base_gm_mask_img.get_fdata()
    fmap_data = fmap_img.get_fdata()

    threshold_mask = (fmap_data > threshold)

    active_data = fmap_data
    active_data[~threshold_mask] = 0
    active_data[threshold_mask] = 1

    active_img = nib.Nifti1Image(active_data, affine=base_gm_mask_img.affine, header=base_gm_mask_img.header)
    active_img.to_filename(task_active_thr_map)

    return active_img
def create_labeled_sample(save_prefix, bold_data, predicted_data, active_mask):
    voxel_coords = np.array(np.where((active_mask > 0))).T
    np.save(f'{save_prefix}voxel_mapping.npy', voxel_coords)

    X_list, Y_list = bold_data[active_mask > 0], predicted_data[active_mask > 0]
    for i, (X, Y) in enumerate(zip(X_list, Y_list)):
        np.save(f'{save_prefix}X_{i:04d}.npy', X)
        np.save(f'{save_prefix}Y_{i:04d}.npy', Y)

def prepare_subject_sample(subject, task, acquisition, smoothing, voxel_quantile=98, second_glm=False,pdf=None):
    gm_mask = subject_gm_mask_path(subject)
    print("processing subject gray matter mask", gm_mask)
    process_gray_matter_mask(anat_dir, subject, border_size=2, save=True, plot=False)

    print(f"concatenating volumes for task {task}")
    fmri_vols = create_4d_volume(subject, task, acquisition, smoothing, save=True)

    fmap_path = subject_task_fmap(subject, task, acquisition, smoothing)
    if not os.path.isfile(fmap_path):
        print("processing event conditions")
        independent_events = create_events_df(subject, task, acquisition)
        regressors = scipy.io.loadmat(subject_task_regressor_path(subject, task, acquisition))
        flat_regressors = regressors['Regressor'].flatten()
        print("first GLM to select active voxel")
        activity_glm = voxel_activation_glm(gm_mask, TR, fmri_vols, independent_events)
        print(f"computing f-map and selecting {voxel_quantile}th voxel")
        n_regressors = activity_glm.design_matrices_[0].shape[1]
        contrast_matrix = np.zeros((n_regressors-1, n_regressors))
        np.fill_diagonal(contrast_matrix, 1)
    else:
        print(f"Fmap already present at {fmap_path}: skipping glm")
        activity_glm=None
        contrast_matrix=None
    fmap = compute_task_fmap(subject, task, acquisition, smoothing, activity_glm, contrast_matrix,save=True)
    threshold = compute_bins_threshold(fmap, n_perc=voxel_quantile,pdf=pdf)

    atlas_activation = get_atlas_activation(fmap,task_atlas_path[task],threshold)
    print("Atlas activation [1-100]: ",atlas_activation)

    plot_fmap(fmap, threshold, display_mode="x", task=task, info={"subj": subject}, cut_cords=x_coords,pdf=pdf)
    plot_fmap(fmap, threshold, display_mode="y", task=task, info={"subj": subject}, cut_cords=y_coords,pdf=pdf)
    plot_fmap(fmap, threshold, display_mode="z", task=task, info={"subj": subject}, cut_cords=z_coords,pdf=pdf)

    fmap_data = fmap.get_fdata()
    fmap_data[fmap_data<threshold]=0
    fmap_data[fmap_data>0]=1
    fmap = nib.Nifti1Image(fmap_data,fmap.affine)
    active_mask_path = fmap_path.replace('fmap.nii','active_mask.nii')
    print("Saving fmap thresholded in ",active_mask_path)
    nib.save(fmap,active_mask_path)
    if second_glm:

        gm_mask = nib.load(gm_mask)
        active_mask = create_active_voxel_mask(subject, task, acquisition, smoothing, voxel_activation_glm, gm_mask, fmap, threshold)
        active_mask_data = active_mask.get_fdata()

        #plot_img(gm_mask)
        #plot_img(active_mask)

        print("second GLM to extract raw voxel activity")
        raw_voxel_glm = raw_voxel_activity_glm(active_mask, TR, fmri_vols, independent_events)
        predicted = raw_voxel_glm.predicted[0]
        predicted_data = predicted.get_fdata()

        plot_voxel_timecourse(predicted_data[active_mask_data > 0], flat_regressors)

        print("generating labeled sample from fmri run")
        sample_prefix = subject_task_sample_prefix(subject, task, acquisition, smoothing, voxel_quantile)
        create_labeled_sample(sample_prefix, fmri_vols.get_fdata(), predicted_data, active_mask_data)

    print("done")
    return atlas_activation
types_acq = ['RL','LR']
sub_list = get_subject_ids(paradigm_dir)
print(sub_list)
subjects = [subject_run(sub, type_acq) for sub in sub_list for type_acq in types_acq]
activations = []
with PdfPages(pjoin(derivatives_dir,'all_subjects_plots.pdf')) as pdf:
    i=0
    num_subjects = len(subjects)
    for entry in subjects:
        activations.append({'activation':prepare_subject_sample(entry.get("id"), task, entry.get("acquisition"), smoothing=10,pdf=pdf),
                       'subject':entry.get('id'),
                       'task': task,
                       'acquisition': entry.get('acquisition')})
        print(f"\n\n\nProcessed subject {i} out of {num_subjects}\n\n\n")
        i=i+1
sorted_results = sorted(activations, key=lambda x: x['activation'], reverse=True)

# Display sorted results
for entry in sorted_results:
    print(f"Activation: {entry['activation']:.3f}, Subject: {entry['subject']}, Task: {entry['task']}, Acquisition: {entry['acquisition']}")

output_file = pjoin(derivatives_dir,"sorted_subjects_motor.csv")
import csv
# Write the data to a CSV file
with open(output_file, mode='w', newline='') as file:
    # Create a writer object
    writer = csv.DictWriter(file, fieldnames=sorted_results[0].keys())
    
    # Write the header row
    writer.writeheader()
    
    # Write the data rows
    writer.writerows(sorted_results)


print(f"Data saved to {output_file}")
