from utils import show_task_activation, plot_fmap
from utils import process_gray_matter_mask, create_4d_volume, create_events_df, compute_task_fmap, compute_bins_threshold
from utils import subject_gm_mask_path, paradigm_dir, subject_task_active_mask_path, subject_task_regressor_path, subject_task_sample_prefix
from utils import mkdir_no_exist, x_coords, y_coords, z_coords
from utils import resample_mask, get_atlas_activation, get_subject_ids
from utils import voxel_activation_glm, regressors_to_binary_design
from utils import create_labeled_sample, create_active_voxel_mask



from viz import plot_voxel_timecourse


from matplotlib.backends.backend_pdf import PdfPages
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from glob import glob
import xarray as xr

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
smoothing = 10
TR = 0.72 
def subject_run(sid, acquisition,task='MOTOR'):
    return {"id": sid, "acquisition": acquisition, 'task':task}
def binary_design_matrix(events_df, TR, n_volumes):
    time_points = np.arange(0, n_volumes * TR, TR)
    conditions = sorted(events_df['trial_type'].unique())
    design_matrix = pd.DataFrame(0, index=time_points, columns=conditions)
    
    for _, event in events_df.iterrows():
        start_idx = np.searchsorted(time_points, event['onset'])
        end_idx = np.searchsorted(time_points, event['onset'] + event['duration'])
        design_matrix.iloc[start_idx:end_idx, conditions.index(event['trial_type'])] = 1
    
    return design_matrix
def get_predicted_bold_response(glm, active_mask_data, binary_design_matrix,original_data):    
    predicted = glm.predicted[0]
    predicted_data = predicted.get_fdata()[active_mask_data > 0]
    print(predicted_data.shape)

    #plot_voxel_timecourse(predicted_data, binary_design_matrix)

    _=0
    timepoints = original_data.shape[-1]
    original_data_masked = original_data[active_mask_data > 0].reshape(-1,timepoints)

    return original_data_masked,predicted_data,_

def get_predicted_neural_activity(glm, active_mask_data, binary_design_matrix,original_data):
    n_volumes, n_regressors = glm.design_matrices_[0].shape
    n_conditions = n_regressors - 1 # drop constant regressor
    contrast_matrix = np.eye(n_conditions, n_regressors)

    # actual response (amplitude) in the voxel for each of the n_conditions
    betas_data = glm.compute_contrast(contrast_matrix, output_type='effect_size').get_fdata()
    active_voxel_beta = betas_data[active_mask_data > 0].reshape(-1, n_conditions)

    predicted_neural_activity =  active_voxel_beta @ binary_design_matrix.T.to_numpy()

    print(predicted_neural_activity.shape)
    #plot_voxel_timecourse(predicted_neural_activity, binary_design_matrix)
    
    _=0
    timepoints = original_data.shape[-1]
    original_data_masked = original_data[active_mask_data > 0].reshape(-1,timepoints)

    return original_data_masked,predicted_neural_activity,_

def get_predicted_neural_activity_per_condition(glm, active_mask_data, binary_design_matrix,original_data):
    print(binary_design_matrix.shape)
    
    n_volumes, n_regressors = glm.design_matrices_[0].shape
    n_conditions = n_regressors - 1 # drop constant regressor
    contrast_matrix = np.eye(n_conditions, n_regressors)
    
    betas = activity_glm.compute_contrast(contrast_matrix, output_type='effect_size')
    betas_data = betas.get_fdata()
    active_voxel_beta = betas_data[active_mask_data > 0].reshape(-1, n_conditions)
    
    # binary_design: (T × C) = (284 × 10)
    # betas: (V × C) = (1937 × 10)
    # out output: (C × T × V) = (10 × 284 × 1937) beta for each condition for each ts for each voxel_activiy
    
    predictions = np.zeros((
        n_conditions,       # C = 10
        active_voxel_beta.shape[0],   # V = 1937
        n_volumes,       # T = 284
    ))

    binary_design_matrix = binary_design_matrix.to_numpy()
    for c in range(n_conditions):
        # binary_design[:, c] is (T,)
        # betas[:, c] is (V,)
        # outer product to get (T × V) for this condition
        # basically I want to be able to separate betas across condition (skip '@' = Sum_i beta_i cond_i)
        predictions[c] = np.outer(active_voxel_beta[:, c], binary_design_matrix[:, c])
    
    timepoints = original_data.shape[-1]
    original_data_masked = original_data[active_mask_data > 0].reshape(-1,timepoints)
    return original_data_masked,predictions,_

def get_block_and_neural_activity(glm, active_mask_data, binary_design_matrix,original_data):
    n_volumes, n_regressors = glm.design_matrices_[0].shape
    n_conditions = n_regressors - 1 # drop constant regressor
    contrast_matrix = np.eye(n_conditions, n_regressors)

    # actual response (amplitude) in the voxel for each of the n_conditions
    betas_data = glm.compute_contrast(contrast_matrix, output_type='effect_size').get_fdata()
    active_voxel_beta = betas_data[active_mask_data > 0].reshape(-1, n_conditions)
    print("Bdesign shape: ",binary_design_matrix.T.to_numpy().shape)
    print("Active beta: ",active_voxel_beta.shape)
    predicted_neural_activity =  active_voxel_beta @ binary_design_matrix.T.to_numpy()

    print(predicted_neural_activity.shape)
    #plot_voxel_timecourse(predicted_neural_activity, binary_design_matrix)
    predicted = glm.predicted[0]
    predicted_data = predicted.get_fdata()[active_mask_data > 0]
    print(predicted_data.shape)

    #plot_voxel_timecourse(predicted_data, binary_design_matrix)

    timepoints = original_data.shape[-1]
    original_data_masked = original_data[active_mask_data > 0].reshape(-1,timepoints)

    return original_data_masked,predicted_neural_activity,predicted_data

types_acq = ['RL','LR']
sub_list = get_subject_ids(paradigm_dir)
print(sub_list)
subjects = [subject_run(sub, type_acq) for sub in sub_list for type_acq in types_acq]
print(subjects[0])
import pandas as pd
df=pd.read_csv(pjoin(derivatives_dir,'sorted_subjects_motor.csv'))
df.head()
keep_quantile=0.7 # Keep only % of observations
thr=np.quantile(df['activation'],keep_quantile)
keep_df = df[df['activation']>thr]
keep_df.tail()
subjects_kept=[]
for entry in keep_df.itertuples():
    subjects_kept.append(subject_run(sid=entry.subject,acquisition=entry.acquisition,task=entry.task))
   
import numpy as np
import xarray as xr
import nibabel as nib

def voxel_activation_glm(mask_img, TR, fmri_vols, events):
    fmri_glm = FirstLevelModel(
        mask_img=mask_img,
        t_r=TR,
        noise_model='ar1', # or ols 
        standardize=False,
        hrf_model='spm',
        drift_model=None, # not necessary, nuisance covariates have already been removed
        minimize_memory=False,
    )

    print(f"fitting GLM for task {task}")
    fitted_glm = fmri_glm.fit(fmri_vols, events)
    return fitted_glm
def prepare_subject_sample(subject, task, acquisition, smoothing, voxel_quantile=98, label_extraction=get_predicted_neural_activity,show=True):
    gm_mask = subject_gm_mask_path(subject)
    print("processing subject gray matter mask", gm_mask)
    process_gray_matter_mask(anat_dir, subject, border_size=2, save=True, plot=False)

    print(f"concatenating volumes for task {task}")
    fmri_vols = create_4d_volume(subject, task, acquisition, smoothing, save=True)

    print("processing event conditions")
    
    events_f = create_events_df(subject, task, acquisition,drop_non_paradigm=True)

    print("first GLM to select active voxel")
    activity_glm = voxel_activation_glm(gm_mask, TR, fmri_vols, events_f)
    
    n_regressors = activity_glm.design_matrices_[0].shape[1]
    n_conditions = n_regressors - 1 # drop constant regressor

    print(f"computing f-map and selecting {voxel_quantile}th voxel")
    contrast_matrix = np.eye(n_conditions, n_regressors)
    
    fmap = compute_task_fmap(subject, task, acquisition, smoothing, activity_glm, contrast_matrix)
    threshold = compute_bins_threshold(fmap, n_perc=voxel_quantile, show=False)
    if show:
        plot_fmap(fmap, threshold, display_mode="x", task=task, info={"subj": subject}, cut_cords=x_coords)
        plot_fmap(fmap, threshold, display_mode="y", task=task, info={"subj": subject}, cut_cords=y_coords)
        plot_fmap(fmap, threshold, display_mode="z", task=task, info={"subj": subject}, cut_cords=z_coords)

    gm_mask = nib.load(gm_mask)
    active_mask = create_active_voxel_mask(subject, task, acquisition, smoothing, voxel_quantile, gm_mask, fmap, threshold)
    active_mask_data = active_mask.get_fdata()
    print("Selecting events without droping rest condition")
    events = create_events_df(subject, task, acquisition,drop_non_paradigm=False)
    regressors = scipy.io.loadmat(subject_task_regressor_path(subject, task, acquisition))
    regressors_ts = regressors['Regressor'].flatten()

    bdesign_matrix = binary_design_matrix(events, TR, fmri_vols.shape[-1])

    print("second GLM without censoring of rest condition")
    interest_glm = voxel_activation_glm(gm_mask, TR, fmri_vols, events)
    return label_extraction(interest_glm, active_mask_data, bdesign_matrix,fmri_vols.get_fdata())

# Initialize an empty list to store datasets
datasets = []
i=0
for entry in subjects_kept:
    print(f"Processing subject {i} out of {len(subjects_kept)}")
    i=i+1
    subject_id = str(entry['id'])
    acquisition = entry['acquisition']
    task = entry['task']
    
    #X_list, Y_list = prepare_subject_sample(subject_id, task, acquisition, smoothing, voxel_quantile=98, labeled_sample_extraction=get_predicted_neural_activity , show=False)
    X_list, Y_list, Y_conv_list = prepare_subject_sample(subject_id, task, acquisition, smoothing, voxel_quantile=98, label_extraction=get_block_and_neural_activity , show=False)


    # Create an xarray Dataset for this subject
    dataset = xr.Dataset(
        {
            'X': (['voxel', 'time'], X_list),  # Original timeseries
            'Y': (['voxel', 'time'], Y_list),  # Predicted timeseries
            'Y_conv': (['voxel', 'time'], Y_conv_list),  # Predicted timeseries

        },
        coords={
            'subject': f"{subject_id}_{acquisition}",
            'task': task,
            'voxel': np.arange(X_list.shape[0]),
            'time': np.arange(X_list.shape[1]),
        },
        attrs={
            'description': f'Dataset for subject {subject_id}, task {task}, acquisition {acquisition}'
        }
    )

    # Append the dataset to the list
    datasets.append(dataset)

# Combine datasets into one large xarray Dataset along the 'subject' dimension
final_dataset = xr.concat(datasets, dim='subject', fill_value=np.nan)
# Save to file
final_dataset.to_netcdf(pjoin(derivatives_dir,f'dataset_{task}_{len(subjects_kept)}_subjects_both_noncens.nc'))