import numpy as np
import xarray as xr
from run import create_train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

from models.rnn_cnn_rnn import RNNCNNDeconvolutionRNN, RNNCNNDeconvolutionRNNTrainer
from model_selection import prepare_bold_input, prepare_target_input


DATASET = "data/derivatives/dataset_MOTOR_20_subjects_both.nc"
N_SUBJECTS = 10


def load_data():
    dataset = xr.open_dataset(DATASET)
    selected_subjects = np.random.choice(
        dataset.subject.values, size=N_SUBJECTS, replace=False
    )
    subset_dataset = dataset.sel(subject=selected_subjects)
    return subset_dataset


def preprocess_dataset(dataset):
    valid_mask = ~dataset.X.isnull().any(dim="time")
    print(f"Original shape: {dataset.X.shape}")

    dataset = dataset.isel(voxel=valid_mask.all(dim="subject"))

    print(f"Shape after dropping NaNs: {dataset.X.shape}")
    return dataset


class TorchModelWrapper(BaseEstimator):
    def __init__(
        self, model_class, trainer_class, hidden_size=64, kernel_size=20, batch_size=32
    ):
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.batch_size = batch_size

    def fit(self, X, y):
        model_params = {
            "input_size": 1,
            "hidden_size": self.hidden_size,
            "kernel_size": self.kernel_size,
            "output_size": 1,
        }

        self.model = self.model_class(**model_params)
        self.trainer = self.trainer_class(
            model=self.model, config={"batch_size": self.batch_size}
        )

        X_tensor = prepare_bold_input(X)
        y_tensor = prepare_target_input(y)
        self.trainer.train(X_tensor, y_tensor)
        return self

    def predict(self, X):
        X_tensor = prepare_bold_input(X)
        return self.trainer.predict(X_tensor)

    def score(self, X, y):
        X_tensor = prepare_bold_input(X)
        y_tensor = prepare_target_input(y)
        return -self.trainer.evaluate(
            X_tensor, y_tensor
        )  # Negative because sklearn maximizes


def grid_search(models_and_trainers, X_train, y_train, n_folds=5):
    param_grid = {
        "hidden_size": [32, 64, 48, 64, 80],
        "kernel_size": [10, 20, 40, 60, 80],
        "batch_size": [16, 32, 48, 64],
        # "loss_params__alpha": np.linspace(0, 16, 25),
        # "loss_params__lambda_tv": np.linspace(0, 3, 25),
        # "loss_params__lambda_const": np.linspace(0, 3, 25),
        # "loss_params__lambda_val": np.linspace(0, 3, 25),
    }

    model_class, trainer_class = models_and_trainers[0]
    estimator = TorchModelWrapper(model_class, trainer_class)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=n_folds,
        n_jobs=1,  # GPU models keep at 1
        verbose=2,
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_.model, None, grid_search.best_params_


def select_model_params():
    dataset = preprocess_dataset(load_data())

    train, test = create_train_test_split(dataset)
    X_train, X_test, y_train, y_test = train.X, test.X, train.Y, test.Y

    n_subjects_train, n_voxels_train, n_timepoints_train = X_train.shape
    n_subjects_test, n_voxels_test, n_timepoints_test = X_test.shape
    print(
        "train shape",
        (X_train.shape, y_train.shape),
        "test shape",
        (X_test.shape, y_test.shape),
    )

    print((n_subjects_train * n_voxels_train, n_timepoints_train, 1))
    print((n_subjects_test * n_voxels_test, n_timepoints_test, 1))

    # select the best model given raw data
    models_and_trainers = [(RNNCNNDeconvolutionRNN, RNNCNNDeconvolutionRNNTrainer)]
    best_model, best_trainer_cls, model_params = grid_search(
        models_and_trainers,
        X_train,
        y_train,
        n_folds=5,
    )


if __name__ == "__main__":
    select_model_params()
