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

- **bi_lstm**: A Bidirectional LSTM model with 3 layers, followed by a fully connected (FC) layer.

- **cnn_rnn**: A Convolutional Neural Network (CNN) that feeds into a Recurrent Neural Network (RNN), followed by a fully connected (FC) layer.

- **conv_lstm**:  A CNN that feeds into an LSTM, followed by a fully connected (FC) layer.

- **lstm_1l**:  A simple LSTM model with 1 layer, followed by a fully connected (FC) layer.

- **lstm_3l**: An LSTM model with 3 layers, followed by a fully connected (FC) layer.

- **lstm_att**: An LSTM model with an attention mechanism, followed by a fully connected (FC) layer.

- **lstm_conv**: An LSTM model that feeds into a convolutional layer.

- **pure_conv**: A pure CNN model with 3 layers, followed by a fully connected (FC) layer.

- **rnn_cnn_rnn**: A hybrid model consisting of a 2-layer LSTM followed by a fully connected (FC) layer that predicts a convolutional kernel. The convolved signal then feeds into a 3-layer LSTM, followed by another FC layer.

- **rnn_cnn_rnn_bi**: Similar to `rnn_cnn_rnn`, but uses bidirectional LSTMs in its architecture.



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
├── models/                       #
│   ├── trainer/                  #
|   │   ├── trainer.py            # base trainer with shared logic
│   ├── bi_lstm.py                # model 3 layers BiLSTM + fc
|   ├── conv_lstm.py              # model Conv + 3 layers LSTM + fc
│   ├── lstm_1l.py                # model 1 layers LSTM + fc
|   ├── lstm_3l.py                # model 3 layers LSTM + fc
│   ├── lstm_att.py               # model LSTM + attention + fc
│   ├── pure_conv.py              # model 3 layers conv
|   ├── rnn_cnn_rnn.py            # model LSTM to create kernel + convolution + LSTM
│   ├── rnn_cnn_rnn_bi.py         # model BiLSTM to create kernel + convolution + BiLSTM
|── notebooks/
|   ├──        #
│   ├──        #
|   ├──        #
│   ├──
├── augment.py                    # dataset augmentation
└── dataset_motor.py              # dataset generation MOTOR task
├── dataset_multi_task.py         # dataset generation for others tasks
├── model_eval.py                 # model comparison
├── model_grid_search.py          # dense param grid search in optimal neighbourhood
├── model_selection.py            # optuna sparse param grid search
├── preprocessing.py              # extract timeseris from fMRI data
├── run.py                        # entrypoint for prediction and comparison
├── select_subjects.pdf           # subject selection
├── utils.py                      # various helpers
├── viz.py                        # viz utils for nb exploration

