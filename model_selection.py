from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from functools import partial
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import xarray as xr
import numpy as np
import json
import logging
from datetime import datetime

optuna.logging.get_logger("optuna").addHandler(logging.FileHandler("optuna.log"))

def log_trial(study, trial):
    print(f"\nTrial {trial.number}:")
    print(f"Params: {trial.params}")
    print(f"Value: {trial.value}")

def pretty_print(data):
    formatted_json = json.dumps(data, 
        indent=2,
        sort_keys=True,
        separators=(',', ': '),
        ensure_ascii=False
    )
    print(formatted_json)

def prepare_bold_input(X_data):
    assert isinstance(X_data, xr.DataArray)
    n_subjects, n_voxels, n_timepoints = X_data.shape
    
    X_flat = X_data.values.reshape(-1, n_timepoints)
    X_scaled = np.vstack([RobustScaler().fit_transform(ts.reshape(-1, 1)).ravel() 
                         for ts in X_flat])
    
    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(-1)
    assert X_tensor.shape == (n_subjects * n_voxels, n_timepoints, 1)
    return X_tensor

def prepare_target_input(y_data):
    assert isinstance(y_data, xr.DataArray)
    n_subjects, n_voxels, n_timepoints = y_data.shape
    
    y_flat = y_data.values.reshape(-1, n_timepoints)
    y_tensor = torch.FloatTensor(y_flat)
    assert y_tensor.shape == (n_subjects * n_voxels, n_timepoints)
    return y_tensor

def select_model(models_and_trainers, X_train, y_train, n_trials=100, n_folds=5):
    best_score = float('inf')
    best_results = None

    subjects = X_train.subject.values
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(X_train, groups=subjects))

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
    
    for ModelClass, TrainerClass in models_and_trainers:
        def objective(trial):
            params = TrainerClass.get_optuna_params(trial)
            model_params = {
                'input_size': params['input_size'],
                'hidden_size': params['hidden_size'],
                'output_size': params['output_size'],
                'dropout_prob': params['dropout_prob']
            }
            
            pretty_print(params)

            fold_scores = []
            for fold_idx in range(n_folds):
                model = ModelClass(**model_params)
                trainer = TrainerClass(model=model, config=params)
                
                X_fold_train = X_tensors[fold_idx]['train']
                X_fold_val = X_tensors[fold_idx]['val']
                y_fold_train = y_tensors[fold_idx]['train']
                y_fold_val = y_tensors[fold_idx]['val']
                
                trainer.train(X_fold_train, y_fold_train)
                fold_score = trainer.evaluate(X_fold_val, y_fold_val)
                print(f"fold scole: {fold_score}")
                
                # Enable early stopping within folds
                trial.report(fold_score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                fold_scores.append(fold_score)
                torch.cuda.empty_cache()
            
            return np.mean(fold_scores)

        sampler = TPESampler(n_startup_trials=10)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )

        try:
            from joblib import parallel_backend
            with parallel_backend('threading', n_jobs=-1):
                study.optimize(
                    objective, 
                    n_trials=n_trials, 
                    callbacks=[log_trial],
                    catch=(Exception,)
                )
        except ImportError:
            study.optimize(
                objective, 
                n_trials=n_trials, 
                callbacks=[log_trial],
                catch=(Exception,)
            )
        study.trials_dataframe().to_csv(f'optuna_trials_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        if study.best_value < best_score:
            best_score = study.best_value
            best_results = (ModelClass, TrainerClass, study.best_params)
    
    if best_results is None:
        return None, None, None
        
    ModelClass, TrainerClass, best_params = best_results
    
    model_params = {
        'input_size': best_params['input_size'],
        'hidden_size': best_params['hidden_size'],
        'output_size': best_params['output_size'],
        'dropout_prob': best_params['dropout_prob']
    }
    
    model = ModelClass(**model_params)
    trainer = TrainerClass(model=model, config=best_params)
    
    return model, trainer, best_params