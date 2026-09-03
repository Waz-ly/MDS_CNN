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
    out_dim: int,
    n_epochs: int,
    activation = "relu",
    folds: list[str] | None = None,
) -> float:
    """Leave-one-dataset-out CV, scored on normalize+MAE. Lower is better."""
    folds = folds if folds is not None else SINGLE_SETS
    fold_scores = []

    for idx, test_set in enumerate(folds):
        train_set = [d for d in list_datasets() if d not in test_set]
        test_batch = build_batch(test_set, variants=False, adjusted=False)

        model = train(train_set, lr, weight_decay, hidden_dim, n_hidden, out_dim, n_epochs, activation)
        model.eval()
        with torch.no_grad():
            score, _ = total_stress_loss(
                model, test_batch, rescale_type="normalize", loss_type="MAE"
            )
        fold_scores.append(score.item())

    return float(np.mean(fold_scores))

def objective(trial: optuna.Trial, search_folds: list[str], out_dim) -> float:

    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    n_hidden = trial.suggest_int("n_hidden", 1, 4)
    n_epochs = trial.suggest_int("n_epochs", 1000, 5000, step=200)

    return quick_cv_score(
        lr, weight_decay, hidden_dim, n_hidden, out_dim, n_epochs, "relu",
        folds=search_folds,
    )

def optimize(
    n_trials: int = 50,
    n_folds: float = 5,
    out_dim: int = 3,
    seed: int = 42,
) -> dict:
    """
    Two-phase hyperparameter search:
      1. Cheap search over a subset of folds / fewer repeats to find good hyperparameters.
      2. Full validation of the winning config on all folds with the full repeat count.

    Training always uses rescale_type="none", loss_type="MAE".
    Tuning/selection always uses rescale_type="normalize", loss_type="MAE".
    """
    rng = random.Random(seed)
    search_folds = rng.sample(SINGLE_SETS, n_folds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )

    study.optimize(
        lambda t: objective(t, search_folds, out_dim),
        n_trials=n_trials,
    )

    best_params = study.best_params
    print(f"Best params from search: {best_params}")
    print(f"Search score (normalize/MAE, {n_folds} folds): "
          f"{study.best_value:.4f}")

    return {
        "params": best_params,
        "search_score": study.best_value,
        "study": study,
    }

def save_result(result: dict, path: str = "data/tuning_result.json") -> None:
    to_save = {
        "params": result["params"],
        "search_score": result["search_score"],
    }
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Saved to {path}")

if __name__ == "__main__":

    result = optimize(
        n_trials=50,
        n_folds=9, # 9 of 18 for validation
        out_dim=256
    )
    save_result(result, "data/tuning_256D.json")
