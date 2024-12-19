# Estimating Brain Activity Timecourses Using fMRI Data

## Table of Contents
- [Overview](#overview)
- [Key Objectives](#Objectives)
- [Dataset Description](#Dataset)
- [Methodology](#Methodology)
- [Usage](#usage)
- [Requirements](Requirements)
- [Repo Structure](#RepoStructure)


## Overview
This project aims to estimate brain activity timecourses during task paradigms using functional magnetic resonance imaging (fMRI) data from the Human Connectome Project (HCP). The hypothesis driving this work is that neuronal firing patterns correspond to voxel activations in fMRI scans, and this relationship can be modeled as a blind deconvolution problem. Machine learning techniques, including Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs), are employed to model the temporal dynamics and spatial characteristics of brain activity.

## Objectives
1. Estimate the beta values from the BOLD signal
2. Calculate the deconvolved neural activity timecourses during task paradigms.
3. Explore machine learning techniques (RNNs and CNNs) for modeling brain activity dynamics.

## Dataset
### Source
The dataset used in this project comes from the Human Connectome Project (HCP), a comprehensive resource for understanding the human brain's functional and structural connectivity.

### Participants
- **Number of participants**: 100
- **Tasks**: Cognitive and motor tasks, including working memory, language processing, motor function, emotion processing, gambling, and social cognition.

### Data Structure
- **Task fMRI**: Time-series measurements of brain activity during task paradigms.
- **Experimental Paradigms**: Each task has unique onset times, durations, and experimental conditions (e.g., subtasks defined by time-duration tuples).

## Methodology
### Preprocessing
1. **Smoothing**: Applied a Gaussian smoothing kernel (FWHM = 10 mm) to reduce noise and enhance signal-to-noise ratio.
2. **Masking**: Focused on gray matter regions by applying a predefined gray matter mask and removing noisy borders.

### Voxel Selection
1. Retained voxels in clusters of at least 10 neighboring voxels to eliminate spurious activations.
2. Thresholded the F-statistic maps (FDR = 0.05) and retained the top 2% of active voxels (98th percentile), resulting in approximately 1950 voxels per participant.

### Analysis Framework
1. **Blind Deconvolution**: Estimating brain activity without prior knowledge of the HRF.
2. **Machine Learning**:
   - **RNNs**: Capture temporal dependencies in the BOLD signal.
   - **CNNs**: Handle spatially structured data for improved analysis of voxel relationships.
   - 
## Usage
### Data Preprocessing
Run the preprocessing script to clean and smooth the data:
> python scripts/data_preprocessing.py

### Voxel Selection
Run the voxel selection script to keep only the most meaningful the data:

> python scripts/voxel_selection.py

### Models

...............


## Requirements
- **Programming Language**: Python 3.x
- **Libraries**:
  - NumPy
  - SciPy
  - scikit-learn
  - TensorFlow or PyTorch
  - nilearn
  - nibabel
- **System Requirements**: A machine with GPU support for training deep learning models.


## RepoStructure
The main files and directories in this project are as follows:
```plaintext
project-folder/
├── atlas/HMAT/
│   ├── HMAT_Manuscript.pdf       # 
│   ├── README.txt                # 
├── docs                          # papers and bibliography
├── loss/                         #
│   ├── blocky_loss.py            # 
│   ├── loss.py                   #
├── models/                       #
│   ├── trainer/                  #
|   │   ├── trainer.py            # 
│   ├── bi_lstm.py                # model 3 layers BiLSTM + fc 
│   ├── cnn_rnn.py                # model Conv + 2 layers LSTM + fc
|   ├── conv_lstm.py              # model Conv + 3 layers LSTM + fc
│   ├── lstm_1l.py                # model 1 layers LSTM + fc
|   ├── lstm_3l.py                # model 3 layers LSTM + fc
│   ├── lstm_att.py               # model LSTM + attention + fc
|   ├── lstm_conv.py              # model 
│   ├── pure_conv.py              # model 3 layers conv
|   ├── rnn_cnn_rnn.py            # model LSTM to create kernel + convolution + LSTM
│   ├── rnn_cnn_rnn_bi.py         # model BiLSTM to create kernel + convolution + BiLSTM
|── notebooks/
|   ├──        # 
│   ├──        #
|   ├──        # 
│   ├──  
├── augment.py                    # dataset augmentation
├── model_eval.py                 # 
├── model_grid_search.py          # 
├── model_selection.py            # 
├── obtain_dataset.py             # 
└── preprocessing.py              # preprocessing MOTOR task
├── preprocessing_multi_task.py   # preprocessing others tasks
├── run.py                        # 
├── select_subjects.pdf           # 
├── utils.py                      # 
├── viz.py                        #

