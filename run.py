from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import xarray as xr

# from augment import select_augmentation

from models.bi_lstm import BiLSTMModel, BiLSTMTrainer
from models.cnn_rnn import CNNRNNModel, CNNRNNTrainer
from model_selection import select_model

DATASET = 'data/derivatives/dataset_MOTOR_60_subjects_both.nc' # hrf convoluted + block, if we want to denoise data
N_SUBJECTS = 10

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

def main():
    dataset = preprocess_dataset(load_data())

    train, test = create_train_test_split(dataset)
    X_train, X_test, y_train, y_test = train.X, test.X, train.Y, test.Y

    n_subjects_train, n_voxels_train, n_timepoints_train = X_train.shape
    n_subjects_test, n_voxels_test, n_timepoints_test = X_test.shape
    print(X_train.shape, X_test.shape)
    print((n_subjects_train * n_voxels_train, n_timepoints_train, 1))
    print((n_subjects_test * n_voxels_test, n_timepoints_test, 1))

    # select the best model given raw data
    # models_and_trainers = [(BiLSTMModel, BiLSTMTrainer), (CNNRNNModel, CNNRNNTrainer)]
    models_and_trainers = [(BiLSTMModel, BiLSTMTrainer)]
    best_model, best_trainer_cls, model_params = select_model(
        models_and_trainers, X_train, y_train, n_trials=100, n_folds=5,
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
    model, predictions, score = main()