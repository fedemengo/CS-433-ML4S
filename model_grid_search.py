from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import numpy as np
import xarray as xr
import torch
from run import preprocess_dataset, create_train_test_split
from augment import select_augmentation, shift, temporal_scale
from model_selection import pretty_print
from datetime import datetime
import json

from models.rnn_cnn_rnn import RNNCNNDeconvolutionRNN, RNNCNNDeconvolutionRNNTrainer
from models.bi_lstm import BiLSTMModel, BiLSTMTrainer
from models.cnn_rnn import CNNRNNModel, CNNRNNTrainer
from model_selection import select_model, prepare_bold_input, prepare_target_input


DATASET = 'data/derivatives/dataset_MOTOR_20_subjects_both.nc' 
N_SUBJECTS = 10

def load_data():
    dataset = xr.open_dataset(DATASET)
    selected_subjects = np.random.choice(dataset.subject.values, size=N_SUBJECTS, replace=False)
    subset_dataset = dataset.sel(subject=selected_subjects)
    return subset_dataset


def grid_search(models_and_trainers, X_train, y_train, n_folds=5):
    def cleanup_gpu():
        return
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Define grid search parameters
    param_grid = {
        'hidden_size': [32, 64, 48, 64, 80],
        'kernel_size': [10, 20, 40, 60, 80],
        'batch_size': [16, 32, 48, 64]
    }
    
    best_score = float('inf')
    best_results = None
    subjects = X_train.subject.values
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(X_train, groups=subjects))

    # Prepare data tensors for each fold
    X_tensors = {}
    y_tensors = {}
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_fold_train = X_train.isel(subject=train_idx)
        X_fold_val = X_train.isel(subject=val_idx)
        y_fold_train = y_train.isel(subject=train_idx)
        y_fold_val = y_train.isel(subject=val_idx)
        
        X_tensors[fold_idx] = {
            'train': prepare_bold_input(X_fold_train),
            'val': prepare_bold_input(X_fold_val)
        }
        y_tensors[fold_idx] = {
            'train': prepare_target_input(y_fold_train),
            'val': prepare_target_input(y_fold_val)
        }

    run_id = str(int(datetime.now().timestamp()))
    results_log = []
    
    for ModelClass, TrainerClass in models_and_trainers:
        from itertools import product
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in product(*param_grid.values())]

        for params in param_combinations:
            # cleanup_gpu()  # Clean before each parameter combination
            
            full_params = {
                'input_size': 1,
                'output_size': 1,
                'loss_fn': 'blocky_loss',
                'training_logic': 'fixed',
                'dropout_prob': 0.1,
                **params,
            }
            
            pretty_print(full_params)
            
            model_params = {
                'input_size': full_params['input_size'],
                'hidden_size': full_params['hidden_size'],
                'kernel_size': full_params['kernel_size'],
                'output_size': full_params['output_size'],
            }
            
            fold_scores = []
            for fold_idx in range(n_folds):
                try:
                    cleanup_gpu()  # Clean before each fold
                    
                    model = ModelClass(**model_params)
                    trainer = TrainerClass(model=model, config=full_params)
                    
                    X_fold_train = X_tensors[fold_idx]['train']
                    X_fold_val = X_tensors[fold_idx]['val']
                    y_fold_train = y_tensors[fold_idx]['train']
                    y_fold_val = y_tensors[fold_idx]['val']
                    
                    # Move to CPU after getting scores
                    trainer.train(X_fold_train, y_fold_train)
                    fold_score = trainer.evaluate(X_fold_val, y_fold_val)
                    print(f"Fold {fold_idx}/{n_folds} score: {fold_score}")
                    
                    fold_scores.append(fold_score)
                
                finally:
                    # Cleanup regardless of success/failure
                    if 'model' in locals(): del model
                    if 'trainer' in locals(): del trainer
                    cleanup_gpu()
            
            if fold_scores:  # Only if we have valid scores
                mean_score = np.mean(fold_scores)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                print(f"Mean score: {mean_score}")
                
                result = {
                    'parameters': params,
                    'score': float(mean_score),  # Convert to float for JSON serialization
                    'fold_scores': [float(score) for score in fold_scores]
                }
                results_log.append(result)

                with open(f'./params_search/results_{run_id}_{timestamp}.json', 'w') as f:
                    json.dump(result, f, indent=2)

                if mean_score < best_score:
                    best_score = mean_score
                    best_results = (ModelClass, TrainerClass, full_params)
                    print(f"New best score: {best_score}")
                    print("Best parameters:")
                    pretty_print(full_params)

    with open(f'./params_search/results_{run_id}_all.json', 'w') as f:
        json.dump(results_log, f, indent=2)
    
    cleanup_gpu()  # Final cleanup
    
    if best_results is None:
        return None, None, None
        
    ModelClass, TrainerClass, best_params = best_results
    
    model_params = {
        'input_size': best_params['input_size'],
        'hidden_size': best_params['hidden_size'],
        'output_size': best_params['output_size'],
        'kernel_size': best_params['kernel_size']
    }
    
    model = ModelClass(**model_params)
    trainer = TrainerClass(model=model, config=best_params)
    
    return model, trainer, best_params

def select_model_params():
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
    models_and_trainers = [(RNNCNNDeconvolutionRNN, RNNCNNDeconvolutionRNNTrainer)]
    best_model, best_trainer_cls, model_params = grid_search(
        models_and_trainers, X_train, y_train, n_folds=5,
    )
    
    # select the best data augmentation for the best model - optimizing over all of them was taking forever
    # augmenters = [NoiseAugmenter, SyntheticAugmenterm, RollingAugmenter, DilationAugmenter, PickAugmenter]
    # best_augmenter, aug_params = select_augmentation(
    #     best_model, best_trainer_cls, X_train, y_train, augmenters
    # )
    
    # final_model = best_trainer_cls.model_cls(**model_params)
    final_trainer = best_trainer_cls(final_model)
    
    # X_aug, y_aug = best_augmenter.augment(X_train, y_train)
    
    # final_trainer.train(X_aug, y_aug)
    # final_trainer.train(X_train, X_train)
    
    # test_predictions = final_trainer.predict(X_test)
    # test_score = final_trainer.evaluate(X_test, y_test)

    # final_trainer.save('final_model.pt')
    
    # return final_model, test_predictions, test_score
    print(model_params)

select_model_params()