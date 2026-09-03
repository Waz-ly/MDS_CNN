from train import build_batch, train
from stress import total_stress_loss
from model import EmbeddingNet, OpenL3Wrapper
import torch
from utils.train_set_config import *
import numpy as np
from utils.dir_manage import list_datasets
import json
from utils.dir_config import *

def test():

    with open(f"data/default_settings.json") as f:
        settings = json.load(f)["params"]

    # need to fix settings to dims

    model = EmbeddingNet()
    model.load_state_dict(torch.load("model/mapper.pth"))
    model.eval()

    test_batch = build_batch(TEST_SET, variants=False, adjusted=False)

    print(f"normal loss: {total_stress_loss(model, test_batch, rescale_type='normalize')[0].item()}")
    print(f"triplet score: {total_stress_loss(model, test_batch, rescale_type='normalize', loss_type='triplet')[0].item()}")
    print(f"NDCG score: {total_stress_loss(model, test_batch, rescale_type='normalize', loss_type='NDCG')[0].item()}")

def trial(
        content_type="music",
        dims=256
    ):

    results = []
    trials = 1

    with open(f"data/tuning_{dims}D.json") as f:
        settings = json.load(f)["params"]

    for test_set in SINGLE_SETS:

        train_set = [dataset for dataset in list_datasets() if dataset not in test_set]
        open("./data/rescale.log", "w").close()

        normal_loss: torch.Tensor = 0
        triplet_loss: torch.Tensor = 0
        NDCG_loss: torch.Tensor = 0

        n_normal = 0
        n_triplet = 0
        n_NDCG = 0

        for i in range(trials):

            model = train(train_set, out_dim=dims, content_type=content_type, **settings)

            model.eval()

            test_batch = build_batch(test_set, variants=False, adjusted=False, content_type=content_type)
            
            l, n = total_stress_loss(model, test_batch, rescale_type="normalize", loss_type="MAE")
            normal_loss, n_normal = normal_loss + l * n, n_normal + n
            l, n = total_stress_loss(model, test_batch, rescale_type="normalize", loss_type="triplet")
            triplet_loss, n_triplet = triplet_loss + l * n, n_triplet + n
            l, n = total_stress_loss(model, test_batch, rescale_type="normalize", loss_type="NDCG")
            NDCG_loss, n_NDCG = NDCG_loss + l * n, n_NDCG + n

        results.append([
            normal_loss.item() / n_normal, n_normal,
            triplet_loss.item() / n_triplet, n_triplet,
            NDCG_loss.item() / n_NDCG, n_NDCG,
        ])

    results = np.array(results)

    averages = [
        np.sum(results[:, 0] * results[:, 1]) / np.sum(results[:, 1]),
        np.sum(results[:, 1]),
        np.sum(results[:, 2] * results[:, 3]) / np.sum(results[:, 3]),
        np.sum(results[:, 3]),
        np.sum(results[:, 4] * results[:, 5]) / np.sum(results[:, 5]),
        np.sum(results[:, 5]),
    ]

    results = np.vstack([results, averages])

    with open("./data/trial_results.csv", "a") as f:
        np.savetxt(f, results, fmt="%.4f")

if __name__ == "__main__":

    trial(content_type="music", dims=256)
        
