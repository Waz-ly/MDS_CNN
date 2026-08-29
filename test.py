from train import build_batch, train
from stress import total_stress_loss
from model import EmbeddingNet, OpenL3Wrapper
import torch
from utils.train_set_config import *
import numpy as np
from utils.dir_manage import list_datasets
import json

def test():

    with open("model/model_config.json") as f:
        config = json.load(f)

    model = EmbeddingNet.from_config_dict(config)
    model.load_state_dict(torch.load("model/mapper.pth"))
    model.eval()

    test_batch = build_batch(TEST_SET, variants=False, adjusted=False)

    print(f"loss: {total_stress_loss(model, test_batch, rescale_type='none')[0].item()}")
    print(f"match loss: {total_stress_loss(model, test_batch, rescale_type='match')[0].item()}")
    print(f"normal loss: {total_stress_loss(model, test_batch, rescale_type='normalize')[0].item()}")
    print(f"triplet score: {total_stress_loss(model, test_batch, rescale_type='normalize', loss_type='triplet')[0].item()}")
    print(f"NDCG score: {total_stress_loss(model, test_batch, rescale_type='normalize', loss_type='NDCG')[0].item()}")

def trial():

    results = []
    trials = 3

    for test_set in SINGLE_SETS:

        train_set = [dataset for dataset in list_datasets() if dataset not in test_set]
        open("rescale.log", "w").close()

        loss: torch.Tensor = 0
        match_loss: torch.Tensor = 0
        normal_loss: torch.Tensor = 0
        triplet_loss: torch.Tensor = 0
        NDCG_loss: torch.Tensor = 0

        n_loss = 0
        n_match = 0
        n_normal = 0
        n_triplet = 0
        n_NDCG = 0

        for i in range(trials):

            model = train(train_set)

            model.eval()

            test_batch = build_batch(test_set, variants=False, adjusted=True)
            
            l, n = total_stress_loss(model, test_batch, rescale_type="none", loss_type="MAE")
            loss, n_loss = loss + l * n, n_loss + n
            l, n = total_stress_loss(model, test_batch, rescale_type="match", loss_type="MAE")
            match_loss, n_match = match_loss + l * n, n_match + n
            l, n = total_stress_loss(model, test_batch, rescale_type="normalize", loss_type="MAE")
            normal_loss, n_normal = normal_loss + l * n, n_normal + n
            l, n = total_stress_loss(model, test_batch, rescale_type="normalize", loss_type="triplet")
            triplet_loss, n_triplet = triplet_loss + l * n, n_triplet + n
            l, n = total_stress_loss(model, test_batch, rescale_type="normalize", loss_type="NDCG")
            NDCG_loss, n_NDCG = NDCG_loss + l * n, n_NDCG + n

        rescale_data = np.loadtxt("rescale.log")
        scales, offsets = rescale_data[:, 0], rescale_data[:, 1]

        results.append([
            loss.item() / n_loss, n_loss,
            match_loss.item() / n_match, n_match,
            normal_loss.item() / n_normal, n_normal,
            triplet_loss.item() / n_triplet, n_triplet,
            NDCG_loss.item() / n_NDCG, n_NDCG,
            scales.mean(),
            offsets.mean()
        ])

    results = np.array(results)

    averages = [
        np.sum(results[:, 0] * results[:, 1]) / np.sum(results[:, 1]),
        np.sum(results[:, 1]),
        np.sum(results[:, 2] * results[:, 3]) / np.sum(results[:, 3]),
        np.sum(results[:, 3]),
        np.sum(results[:, 4] * results[:, 5]) / np.sum(results[:, 5]),
        np.sum(results[:, 5]),
        np.sum(results[:, 6] * results[:, 7]) / np.sum(results[:, 7]),
        np.sum(results[:, 7]),
        np.sum(results[:, 8] * results[:, 9]) / np.sum(results[:, 9]),
        np.sum(results[:, 9]),
        np.mean(results[:, 10]),
        np.mean(results[:, 11])
    ]

    results = np.vstack([results, averages])

    with open("trial_results.csv", "a") as f:
        np.savetxt(f, results, fmt="%.4f")

if __name__ == "__main__":

    trial()
        
