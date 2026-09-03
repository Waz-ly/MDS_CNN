import os
import numpy as np
from utils.dir_config import *
import utils.dir_manage

adjustment = {
    "Barthet2010": 1.1120,
    "Grey1977": 0.8290,
    "Grey1978": 0.8605,
    "Iverson1993_Onset": 0.2514,
    "Iverson1993_Remainder": 1.0424,
    "Iverson1993_Whole": 0.7811,
    "Lakatos2000_Comb": 0.9791,
    "Lakatos2000_Harm": 0.6829,
    "Lakatos2000_Perc": 0.9543,
    "McAdams1995": 5.3650,
    "Patil2012_A3": 0.8724,
    "Patil2012_DX4": 1.0166,
    "Patil2012_GD4": 0.9591,
    "Saitis2020_e2set1_general": 0.7626,
    "Siedenburg2016_e2set1": 0.7626,
    "Siedenburg2016_e2set2": 0.7780,
    "Siedenburg2016_e2set3": 0.9564,
    "Siedenburg2016_e3": 0.9564,
    "Vahidi2020": 0.4473,
    "Zacharakis2014_english": 0.8766,
    "Zacharakis2014_greek": 0.8766
}

from utils.dir_manage import list_datasets
from utils.dir_config import *
from utils.train_set_config import *
import json
from train import train, build_batch
from stress import get_scale_mismatch
import torch

def get_adjustment(
        content_type="music",
        n_trials=1,
    ):

    with open(f"data/default_settings.json") as f:
        settings = json.load(f)["params"]

    scaling = {}

    for test_set in SINGLE_SETS:

        train_set = [dataset for dataset in list_datasets() if dataset not in test_set]
        test_batch = build_batch(test_set, variants=False, adjusted=False, content_type=content_type)

        for dataset, [embeddings, target_matrix] in zip(test_set, test_batch):

            scale = 0

            for i in range(n_trials):

                model = train(train_set, content_type=content_type, **settings)
                model.eval()
                
                predicted_coords = model(embeddings)
                predicted_matrix = torch.cdist(predicted_coords, predicted_coords, p=2)

                scale += get_scale_mismatch(predicted_matrix, target_matrix)

            scale /= n_trials
            scaling[dataset] = scale.item()

    with open(f"data/scaling.json", "w") as f:
        json.dump(scaling, f, indent=2)

def adjust():

    for dataset in utils.dir_manage.list_datasets():

        dissimilarity_matrix = np.loadtxt(os.path.join(DATA_DIR, f"{dataset}_dissimilarity_matrix.txt"))

        adjusted_dissimilarity_matrix = adjustment[dataset] * dissimilarity_matrix

        np.savetxt(
            os.path.join(ADJUST_DATA_DIR, f"{dataset}_dissimilarity_matrix.txt"),
            adjusted_dissimilarity_matrix,
            fmt="%.15f"
        )

if __name__ == "__main__":

    get_adjustment()
