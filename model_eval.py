import torch
import numpy as np
from model_selection import prepare_bold_input, prepare_target_input
from augment import temporal_scale, augment_data
import matplotlib.pyplot as plt
import os
from utils import save_data


def plot_random_samples(
    run_model_prefix, task, flat_res, concat_res, N=3, save_dir="plots"
):
    os.makedirs(save_dir, exist_ok=True)

    for dataset, prefix in [(flat_res, "flat"), (concat_res, "concat")]:
        x, y, pred = dataset
        indices = np.random.choice(len(x), N)

        for i, idx in enumerate(indices):
            plt.figure(figsize=(10, 5))
            plt.plot(x[idx].cpu(), label="Input", alpha=0.5, linestyle="--")
            plt.plot(y[idx].cpu(), label="Target", alpha=1, linestyle="--")
            plt.plot(pred[idx].cpu().squeeze(), label="Prediction", alpha=1)
            plt.title(f"{task.capitalize()} - sample {idx}")
            plt.legend()
            plt.savefig(f"{save_dir}/{run_model_prefix}_{prefix}_sample_{i}.png")
            plt.tight_layout()
            plt.close()


def eval_models(
    run_id, models_and_trainers, X_train, y_train, X_test, y_test, other_tasks
):
    X_train_tensor = prepare_bold_input(X_train)
    y_train_tensor = prepare_target_input(y_train)

    X_test_tensor = prepare_bold_input(X_test)
    y_test_tensor = prepare_target_input(y_test)

    print("augmenting data")
    X_train_aug_tensor, y_train_aug_tensor = temporal_scale(
        X_train_tensor, y_train_tensor, ratio=0.6
    )
    X_train_aug_tensor, y_train_aug_tensor = augment_data(
        X_train_aug_tensor, y_train_aug_tensor
    )
    print("data augmentation done")

    other_tasks_dataset = {}
    for task_name, task_data in other_tasks:
        other_tasks_dataset[task_name] = (
            prepare_bold_input(task_data.X),
            prepare_target_input(task_data.Y),
        )

    results = {}
    for ModelClass, TrainerClass, trainer_params in models_and_trainers:
        model = ModelClass()
        trainer = TrainerClass(model=model, **trainer_params)
        model_name = str(trainer)
        print("\nRUNNING", model_name, "\n")
        run_model_prefix = f"{run_id}_{model_name}"

        trainer.train(X_train_aug_tensor, y_train_aug_tensor)
        model_result, flat_res, conca_res = trainer.full_eval(
            X_test_tensor, y_test_tensor
        )

        plot_random_samples(run_model_prefix, "MOTOR", flat_res, conca_res)

        run_model_tasks_prefix = f"{run_id}_{model_name}_tasks"
        tasks_results = {"motor": model_result}
        for name, data in other_tasks_dataset.items():
            run_model_prefix_task = f"{run_id}_{model_name}_{name}"
            task_result, flat_res_task, conca_res_task = trainer.full_eval(
                data[0], data[1]
            )
            plot_random_samples(
                run_model_prefix_task, name, flat_res_task, conca_res_task
            )
            tasks_results[name] = task_result

        save_data(run_model_tasks_prefix, tasks_results)
        save_data(run_model_prefix, model_result)
        results[model_name] = model_result

        try:
            torch.cuda.empty_cache()
        except:
            pass

    save_data(f"{run_id}_all", results)

    return results
