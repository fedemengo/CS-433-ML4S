import os
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import xarray as xr
import torch
from pathlib import Path
from sys import argv
import time
from torch import nn
from preprocessing import merge_dataset
from utils import DATASET_DIR
from augment import temporal_scale, augment_data
from model_selection import prepare_bold_input, prepare_target_input
from model_eval import eval_models

from models.bi_lstm import BiLSTM, BiLSTM_Trainer
from models.conv_lstm import ConvLSTM, ConvLSTM_Trainer
from models.lstm_att import LSTM_attention, LSTM_attention_Trainer
from models.lstm_1l import LSTM1l, LSTM1lTrainer
from models.lstm_3l import LSTM3l, LSTM3lTrainer
from models.pure_conv import PureConv, PureConv_Trainer
from models.rnn_cnn_rnn import RNNCNNDeconvolutionRNN, RNNCNNDeconvolutionRNNTrainer
from models.rnn_cnn_rnn_bi import (
    RNNCNNDeconvolutionRNNbi,
    RNNCNNDeconvolutionRNN_bi_Trainer,
)

def load_combined(subjects_per_task=1, random_state=42):
   path = f"{DATASET_DIR}/combined_s{subjects_per_task}_r{random_state}.nc"
   if os.path.exists(path):
       return xr.load_dataset(path)
   return None 

def store_combined(data, subjects_per_task, random_state): 
   path = f"{DATASET_DIR}/combined_s{subjects_per_task}_r{random_state}.nc"
   if not os.path.exists(path):
       data_reset = data.reset_index("combined_subjects")
       data_reset.to_netcdf(path)

def load_combined_data(subjects_per_task=20, random_state=42):
   data = load_combined(subjects_per_task, random_state)
   if data is not None:
       return data.set_index(combined_subjects=["task", "subject"])
       
   rng = np.random.RandomState(random_state)
   n_subjects = rng.choice(30, subjects_per_task, replace=False)

   merged = merge_dataset([
       xr.load_dataset(f"{DATASET_DIR}/dataset_MOTOR_30_subjects_normed.nc").isel(subject=n_subjects),
       xr.load_dataset(f"{DATASET_DIR}/dataset_LANGUAGE_30_subjects_normed.nc").isel(subject=n_subjects), 
       xr.load_dataset(f"{DATASET_DIR}/dataset_EMOTION_30_subjects_normed.nc").isel(subject=n_subjects),
       xr.load_dataset(f"{DATASET_DIR}/dataset_WM_30_subjects_normed.nc").isel(subject=n_subjects),
   ])

   data = merged.stack(combined_subjects=("task", "subject")).transpose(
       "combined_subjects", "voxel", "time"
   )
   store_combined(data, subjects_per_task, random_state)
   return data


def load_data(task="MOTOR_30", n_subjects=5):
    dataset = xr.open_dataset(f"{DATASET_DIR}/dataset_{task}_subjects_normed.nc")
    selected_subjects = np.random.choice(
        dataset.subject.values, size=n_subjects, replace=False
    )
    subset_dataset = dataset.sel(subject=selected_subjects)
    return subset_dataset


def preprocess_dataset(dataset):
    valid_mask = ~dataset.X.isnull().any(dim="time")
    print(f"Original shape: {dataset.X.shape}")

    dataset = dataset.isel(voxel=valid_mask.all(dim="subject"))

    print(f"Shape after dropping NaNs: {dataset.X.shape}")
    return dataset


def preprocess_combined_dataset(dataset):
    valid_mask = ~dataset.X.isnull().any(dim="time")
    print(f"Original shape: {dataset.X.shape}")

    dataset = dataset.isel(voxel=valid_mask.all(dim="combined_subjects"))

    print(f"Shape after dropping NaNs: {dataset.X.shape}")
    return dataset


def create_combined_train_test_split(dataset, test_task="LANGUAGE"):
    combined = dataset.task.values
    test_mask = np.array([task.startswith(test_task) for task in combined])

    full_index = dataset.combined_subjects.values
    return dataset.sel(combined_subjects=full_index[~test_mask]), dataset.sel(
        combined_subjects=full_index[test_mask]
    )


def create_train_test_split(dataset, test_size=0.2, random_state=None):
    subjects = dataset.subject.values
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(subjects, groups=subjects))

    train_subjects = subjects[train_idx]
    test_subjects = subjects[test_idx]

    return dataset.sel(subject=train_subjects), dataset.sel(subject=test_subjects)


def load_latest_model(weights_dir="./weights", prefix="multitask"):
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        return None
        
    model_files = list(weights_path.glob(f"{prefix}-*.pth"))
    if not model_files:
        return None
        
    latest_model = max(model_files, key=lambda x: x.stem)
    return torch.load(latest_model)

def trained_model(X_train, y_train, prefix):
    model = RNNCNNDeconvolutionRNN(
        input_size=1,
        hidden_size=64,
        kernel_size=40,
        output_size=1,
    )
    
    model_state = load_latest_model(prefix=prefix)
    if model_state:
        print("loading pretrained model")
        model.load_state_dict(model_state)
        return RNNCNNDeconvolutionRNNTrainer(model=model, config={})

    # load model with grid-optimized params
    train_config = {
        "batch_size": 16,
        "epochs": 40,
        "optimizer": "adam",
        "optimizer_params": {"lr": 0.00030706278416962776},
        "loss_fn": "blocky_loss",
        "loss_param": {
            "alpha": 8.0,
            "lambda_tv": 1.0,
            "lambda_const": 1,
            "lambda_val": 2.6,
        }
    }
    mt = RNNCNNDeconvolutionRNNTrainer(model=model, config=train_config)

    print("augmenting data")
    X_train_aug, y_train_aug = temporal_scale(X_train, y_train, ratio=0.6)
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    print("data augmentation done")

    # train on length 1x data
    mt.train(X_train_aug, y_train_aug)

    return mt
        

def predict(run_id, save_model=False):
    print("running end-to-end prediction", run_id, f"(save_model={save_model})")
    prefix = "multitask"
    
    dataset = preprocess_combined_dataset(load_combined_data())
    train, test = create_combined_train_test_split(dataset, test_task="LANGUAGE")
    X_train, X_test = (
        prepare_bold_input(train.X),
        prepare_bold_input(test.X),
    )
    y_train, y_test = prepare_target_input(train.Y), prepare_target_input(test.Y)
    print("Xs", X_train.shape, X_test.shape)
    print("ys", y_train.shape, y_test.shape)

    mt = trained_model(X_train, y_train, prefix=prefix)

    if save_model:
        model_path = f"./weights/{prefix}-{run_id}.pth"
        mt.save_model(model_path)

    y_pred = mt.predict(X_test)
    print(y_pred.shape)

    x_name = f"./preds/x-{run_id}.npy"
    y_true_name = f"./preds/y_true-{run_id}.npy"
    y_pred_name = f"./preds/y_pred-{run_id}.npy"

    np.save(x_name, X_test)
    np.save(y_true_name, y_test)
    np.save(y_pred_name, y_pred)
    print("PREDICTION RUN DONE", run_id)


def evaluate_models(run_id):
    print("running models evaluation", run_id)
    dataset = preprocess_dataset(load_data(task="MOTOR_100", n_subjects=100))

    train, test = create_train_test_split(dataset)
    X_train, X_test, y_train, y_test = train.X, test.X, train.Y, test.Y

    n_subjects_train, n_voxels_train, n_timepoints_train = X_train.shape
    n_subjects_test, n_voxels_test, n_timepoints_test = X_test.shape

    print("train shape", X_train.shape, "test shape", X_test.shape)
    print((n_subjects_train * n_voxels_train, n_timepoints_train, 1))
    print((n_subjects_test * n_voxels_test, n_timepoints_test, 1))

    language = preprocess_dataset(load_data(task="LANGUAGE_30", n_subjects=3))
    emotion = preprocess_dataset(load_data(task="EMOTION_30", n_subjects=3))
    wm = preprocess_dataset(load_data(task="WM_30", n_subjects=3))
    relational = preprocess_dataset(load_data(task="RELATIONAL_100", n_subjects=3))
    gambling = preprocess_dataset(load_data(task="GAMBLING_30", n_subjects=3))

    # select the best model
    models_and_trainers = [
        # (BiLSTM, BiLSTM_Trainer, {"base_criterion": nn.L1Loss()}),
        # (ConvLSTM, ConvLSTM_Trainer, {"base_criterion": nn.L1Loss()}),
        # (LSTM_attention, LSTM_attention_Trainer, {"base_criterion": nn.L1Loss()}),
        # (PureConv, PureConv_Trainer, {"base_criterion": nn.L1Loss()}),
        # (RNNCNNDeconvolutionRNNbi, RNNCNNDeconvolutionRNN_bi_Trainer, {}),
        # (
        #     RNNCNNDeconvolutionRNNbi,
        #     RNNCNNDeconvolutionRNN_bi_Trainer,
        #     {"base_criterion": nn.L1Loss()},
        # ),
        # (LSTM1l, LSTM1lTrainer, {"base_criterion": nn.L1Loss()}),
        # (LSTM3l, LSTM3lTrainer, {"base_criterion": nn.L1Loss()}),
        # (
        #     RNNCNNDeconvolutionRNN,
        #     RNNCNNDeconvolutionRNNTrainer,
        #     {"base_criterion": nn.L1Loss()},
        # ),
        # (
        #     RNNCNNDeconvolutionRNN,
        #     RNNCNNDeconvolutionRNNTrainer,
        #     {"base_criterion": nn.MSELoss()},
        # ),
        (RNNCNNDeconvolutionRNN, RNNCNNDeconvolutionRNNTrainer, {}),
    ]

    eval_models(
        run_id,
        models_and_trainers,
        X_train,
        y_train,
        X_test,
        y_test,
        (
            ("language", language),
            ("emotion", emotion),
            ("wm", wm),
            ("relational", relational),
            ("gambling", gambling),
        ),
    )

    print("EVAL RUN DONE", run_id)

def evaluate_model(run_id, save_model=False):
    prefix = "motor"
    print("evaluate model", prefix, run_id, f"(save_model={save_model})")
    
    dataset = preprocess_dataset(load_data(task="MOTOR_100", n_subjects=100))
    train, test = create_train_test_split(dataset)
    X_train, X_test, y_train, y_test = train.X, test.X, train.Y, test.Y
    
    X_train, X_test = (
        prepare_bold_input(train.X),
        prepare_bold_input(test.X),
    )
    y_train, y_test = prepare_target_input(train.Y), prepare_target_input(test.Y)
    print("Xs", X_train.shape, X_test.shape)
    print("ys", y_train.shape, y_test.shape)

    mt = trained_model(X_train, y_train, prefix=prefix)

    if save_model:
        model_path = f"./weights/{prefix}-{run_id}.pth"
        mt.save_model(model_path)

    y_pred = mt.predict(X_test)
    print(y_pred.shape)

    x_name = f"./preds/x-{run_id}.npy"
    y_true_name = f"./preds/y_true-{run_id}.npy"
    y_pred_name = f"./preds/y_pred-{run_id}.npy"

    np.save(x_name, X_test)
    np.save(y_true_name, y_test)
    np.save(y_pred_name, y_pred)
    print("MOTOR PREDICTION RUN DONE", run_id)

def main():
    run_id = str(int(time.time()))

    if len(argv) > 1 and argv[1] == "models_evals":
        evaluate_models(run_id)
        return

    if len(argv) > 1 and argv[1] == "eval":
        evaluate_model(run_id, save_model=os.getenv('SAVE_MODEL') is not None)
        return

    predict(run_id, save_model=os.getenv('SAVE_MODEL') is not None)


if __name__ == "__main__":
    main()
