# Estimating Brain Activity Timecourses Using fMRI Data

## Project Overview
This project aims to estimate brain activity timecourses during task paradigms using functional magnetic resonance imaging (fMRI) data from the Human Connectome Project (HCP). The hypothesis driving this work is that neuronal firing patterns correspond to voxel activations in fMRI scans, and this relationship can be modeled as a blind deconvolution problem. Machine learning techniques, including Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs), are employed to model the temporal dynamics and spatial characteristics of brain activity.

## Key Objectives
1. Model the BOLD signal as a convolution of the Hemodynamic Response Function (HRF) and neural activity, incorporating noise:
   \[
   \text{BOLD}(t) = (h * a)(t) + \varepsilon(t)
   \]
2. Estimate deconvolved neural activity timecourses during task paradigms.
3. Explore machine learning techniques (RNNs and CNNs) for modeling brain activity dynamics.

## Dataset Description
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

## File Structure
The main files and directories in this project are as follows:

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

## Usage
### Data Preprocessing
Run the preprocessing script to clean and smooth the data:
> python scripts/data_preprocessing.py

### Voxel Selection
Run the voxel selection script to keep only the most meaningful the data:

> python scripts/voxel_selection.py

### Models

...............



