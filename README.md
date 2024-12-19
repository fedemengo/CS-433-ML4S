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
Models can be found inside the ⁠ models ⁠ directory, with the subdirectory ⁠ trainer ⁠ containing the train class to train the models. The following models have been reported:
•⁠  ⁠*bi_lstm*: bidirectional LSTM, 3 layers + + Fully connected layer
•⁠  ⁠*cnn_rnn*: CNN that feeds into RNN + FC
•⁠  ⁠*conv_lstm*: CNN that feeds into LSTM + Fully connected layer
•⁠  ⁠*lstm_1l*: simple 1 layer LSTM + Fully connected layer
•⁠  ⁠*lstm_3l*: 3 layers LSTM + Fully connected layer
•⁠  ⁠*lstm_att*: LSTM with attention mechanism
•⁠  ⁠*lstm_conv*: LSTM that feeds into convolutional layer
•⁠  ⁠*pure_conv*: CNN 3 Layers + Fully connected layer
•⁠  ⁠*rnn_cnn_rnn*: LSTM 2layer + FC that predicts a convolutional kernel, the convolved signal feeds into a 3L LSTM + FC
•⁠  ⁠*rnn_cnn_rnn_bi*: same as above, using bidirectional LSTMs


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

