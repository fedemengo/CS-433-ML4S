from preprocessing import prepare_subject_sample
from preprocessing import get_predicted_bold_response, get_predicted_neural_activity, plot_voxel_timecourse
from nilearn.plotting import plot_design_matrix
from utils import derivatives_dir, pjoin
import xarray as xr
import numpy as np
import os
from itertools import product

def normalize(X):
    epsilon = 1e-8
    return (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + epsilon)

def task_dataset(subjects, task):
    datasets = []
    
    dataset_path = pjoin(derivatives_dir, f'dataset_{task}_{len(subjects)}_subjects_normed.nc')
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
            )
            norm_block, norm_conv = normalize(block), normalize(conv)
    
            dataset = xr.Dataset({
                'X': (['voxel', 'time'], norm_bold),  # Original timeseries
                'Y': (['voxel', 'time'], norm_block),  # Predicted timeseries
                'Y_conv': (['voxel', 'time'], norm_conv),  # Predicted timeseries
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


acq = "RL"
smoothing = 10
TR = 0.72 

subjects = ["100307", "100408", "101107", "101309", "101915", "103111", "103414", "103818", "105014", "105115", "106016", "108828", "110411", "111312", "111716", "113619", "113922", "114419", "115320", "116524", "117122", "118528", "118730", "118932", "120111", "122317", "122620", "123117", "123925", "124422", "125525", "126325", "127630", "127933", "128127", "128632", "129028", "130013", "130316", "131217", "131722", "133019", "133928", "135225", "135932", "136833", "138534", "139637", "140925", "144832", "146432", "147737", "148335", "148840", "149337", "149539", "149741", "151223", "151526", "151627", "153025", "154734", "156637", "159340", "160123", "161731", "162733", "163129", "176542", "178950", "188347", "189450", "190031", "192540", "196750", "198451", "199655", "201111", "208226", "211417", "211720", "212318", "214423", "221319", "239944", "245333", "280739", "298051", "366446", "397760", "414229", "499566", "654754", "672756", "751348", "756055", "792564", "856766", "857263", "899885"]

# subjects = ["127630"]

tasks = ["WM"]


if __name__ == "__main__":
    for task in tasks:
        print(subjects, task)
        task_dataset(subjects, task)
        print("done with task", task)
