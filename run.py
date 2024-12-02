from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import xarray as xr
import torch
from sys import argv, exit
from models.bi_lstm import BiLSTMTrainer, BiLSTMModel
from datetime import datetime

from augment import select_augmentation, shift

from models.bi_lstm import BiLSTMModel, BiLSTMTrainer
from models.cnn_rnn import CNNRNNModel, CNNRNNTrainer
from model_selection import select_model, prepare_bold_input, prepare_target_input

DATASET = 'data/derivatives/dataset_MOTOR_60_subjects_both.nc' # hrf convoluted + block, if we want to denoise data
N_SUBJECTS = 30

def load_data():
    dataset = xr.open_dataset(DATASET)
    selected_subjects = np.random.choice(dataset.subject.values, size=N_SUBJECTS, replace=False)
    subset_dataset = dataset.sel(subject=selected_subjects)
    return subset_dataset


def preprocess_dataset(dataset):
    valid_mask = ~dataset.X.isnull().any(dim='time')
    print(f"Original shape: {dataset.X.shape}")
    
    dataset = dataset.isel(voxel=valid_mask.all(dim='subject'))
    
    print(f"Shape after dropping NaNs: {dataset.X.shape}")
    return dataset

def create_train_test_split(dataset, test_size=0.2, random_state=None):
    subjects = dataset.subject.values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(subjects, groups=subjects))
    
    train_subjects = subjects[train_idx]
    test_subjects = subjects[test_idx]
    
    return dataset.sel(subject=train_subjects), dataset.sel(subject=test_subjects)

def oneshot():
    dataset = preprocess_dataset(load_data())
    train, test = create_train_test_split(dataset, test_size=0.05)
    X_train, X_test =  prepare_bold_input(train.X), prepare_bold_input(test.X),
    y_train, y_test = prepare_target_input(train.Y), prepare_target_input(test.Y)
    print("Xs", X_train.shape, X_test.shape)
    print("ys", y_train.shape, y_test.shape)

    # load model with reasonable params
    model = BiLSTMModel(
        input_size=1, 
        hidden_size=80, 
        output_size=1,
    )
    train_config = {
      "batch_size": 128,
      "dropout_prob": 0.4,
      "epochs": 80,
      "optimizer": "adam",
      "optimizer_params": {
        "lr": 0.00030706278416962776
      },
    }
    mt = BiLSTMTrainer(model=model, config=train_config)

    X_train_aug, y_train_aug = shift(X_train, y_train)

    # train on length 1x data
    mt.train(X_train_aug, y_train_aug)

    # test on length 2x data
    X_test = torch.cat([X_test, X_test], dim=1)
    y_test = torch.cat([y_test, y_test], dim=1)
    
    y_pred = mt.predict(X_test)
    print(y_pred.shape)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    x_name = f"./preds/x-{ts}.npy"
    y_true_name = f"./preds/y_true-{ts}.npy"
    y_pred_name = f"./preds/y_pred-{ts}.npy"

    np.save(x_name, X_test)
    np.save(y_true_name, y_test)
    np.save(y_pred_name, y_pred)
    print(f"run data saved with ts {ts}")

def select_model_augmentation():
    dataset = preprocess_dataset(load_data())

    train, test = create_train_test_split(dataset)
    X_train, X_test, y_train, y_test = train.X, test.X, train.Y, test.Y

    n_subjects_train, n_voxels_train, n_timepoints_train = X_train.shape
    n_subjects_test, n_voxels_test, n_timepoints_test = X_test.shape
    print("train shape", X_train.shape, "test shape", X_test.shape)
    print((n_subjects_train * n_voxels_train, n_timepoints_train, 1))
    print((n_subjects_test * n_voxels_test, n_timepoints_test, 1))

    # select the best model given raw data
    # models_and_trainers = [(BiLSTMModel, BiLSTMTrainer), (CNNRNNModel, CNNRNNTrainer)]
    models_and_trainers = [(BiLSTMModel, BiLSTMTrainer)]
    best_model, best_trainer_cls, model_params = select_model(
        models_and_trainers, X_train, y_train, n_trials=35, n_folds=5,
    )
    
    # select the best data augmentation for the best model - optimizing over all of them was taking forever
    # augmenters = [NoiseAugmenter, SyntheticAugmenterm, RollingAugmenter, DilationAugmenter, PickAugmenter]
    # best_augmenter, aug_params = select_augmentation(
    #     best_model, best_trainer_cls, X_train, y_train, augmenters
    # )
    
    final_model = best_trainer_cls.model_cls(**model_params)
    final_trainer = best_trainer_cls(final_model)
    
    # X_aug, y_aug = best_augmenter.augment(X_train, y_train)
    
    # final_trainer.train(X_aug, y_aug)
    final_trainer.train(X_train, X_train)
    
    test_predictions = final_trainer.predict(X_test)
    test_score = final_trainer.evaluate(X_test, y_test)

    # final_trainer.save('final_model.pt')
    
    return final_model, test_predictions, test_score

if __name__ == "__main__":
    if len(argv) > 1 and argv[1] == "oneshot":
        oneshot()
        exit(0)

    model, predictions, score = select_model_augmentation()