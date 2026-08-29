import optuna
import numpy as np
import torch
from train import build_batch, train
from stress import total_stress_loss
from utils.train_set_config import *
from utils.dir_manage import list_datasets
import json
import random
import torch.nn as nn

def quick_cv_score(
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    n_hidden: int,
    n_epochs: int,
    activation = "relu",
    dropout = 0.0,
    folds: list[str] | None = None,
    repeats: int = 3,
    trial: optuna.Trial | None = None,
) -> float:
    """Leave-one-dataset-out CV, scored on normalize+MAE. Lower is better."""
    folds = folds if folds is not None else SINGLE_SETS
    fold_scores = []

    for fold_idx, test_set in enumerate(folds):
        train_set = [d for d in list_datasets() if d not in test_set]
        test_batch = build_batch(test_set, variants=False, adjusted=False)

        rep_scores = []
        for _ in range(repeats):
            model = train(train_set, lr, weight_decay, hidden_dim, n_hidden, n_epochs, activation, dropout)
            model.eval()
            with torch.no_grad():
                score = total_stress_loss(
                    model, test_batch, rescale_type="normalize", loss_type="triplet"
                ).item()
            rep_scores.append(score)

        fold_mean = float(np.mean(rep_scores))
        fold_scores.append(fold_mean)

        # Let Optuna prune bad trials early, partway through folds
        if trial is not None:
            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_scores))

def objective(trial: optuna.Trial, search_folds: list[str], repeats: int) -> float:

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    n_hidden = trial.suggest_int("n_hidden", 1, 4)
    n_epochs = trial.suggest_int("n_epochs", 100, 1500, step=100)
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh", "leaky"])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    return quick_cv_score(
        lr, weight_decay, hidden_dim, n_hidden, n_epochs, activation, dropout,
        folds=search_folds, repeats=repeats, trial=trial,
    )

def optimize(
    n_trials: int = 50,
    n_folds: float = 5,
    search_repeats: int = 3,
    seed: int = 42,
) -> dict:
    """
    Two-phase hyperparameter search:
      1. Cheap search over a subset of folds / fewer repeats to find good hyperparameters.
      2. Full validation of the winning config on all folds with the full repeat count.

    Training always uses rescale_type="none", loss_type="MAE".
    Tuning/selection always uses rescale_type="normalize", loss_type="triplet".
    """
    rng = random.Random(seed)
    search_folds = rng.sample(SINGLE_SETS, n_folds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(
        lambda t: objective(t, search_folds, search_repeats),
        n_trials=n_trials,
    )

    best_params = study.best_params
    print(f"Best params from search: {best_params}")
    print(f"Search score (normalize/MAE, {n_folds} folds, {search_repeats} reps): "
          f"{study.best_value:.4f}")

    return {
        "params": best_params,
        "search_score": study.best_value,
        "study": study,
    }

def save_result(result: dict, path: str = "tuning_result.json") -> None:
    to_save = {
        "params": result["params"],
        "search_score": result["search_score"],
    }
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Saved to {path}")

if __name__ == "__main__":

    result = optimize(
        n_trials=150,
        n_folds=4, # 4 of 21 for validation
        search_repeats=2,
    )
    save_result(result)