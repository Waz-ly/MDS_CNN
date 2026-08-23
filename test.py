from train import EmbeddingNet, build_batch, stress_loss
import torch
import numpy as np
import utils.dir_manage
from utils.train_set_config import *


if __name__ == "__main__":

    model_size = 0
    with open('model/model_info.txt', 'r') as file:
        lines = file.readlines()
        model_size = int(lines[0].split()[2])

    model = EmbeddingNet(layer_dims=[model_size, 64, 64, 64, 8])
    model.load_state_dict(torch.load("model/mapper.pth"))
    model.eval()

    test_sets = [dataset for dataset in utils.dir_manage.list_datasets() if dataset in TEST_SET]
    vecs, target, pair_mask = build_batch(test_sets, variants=False)

    coords = model(vecs)
    loss = stress_loss(coords, target, pair_mask)

    print(loss.item())
