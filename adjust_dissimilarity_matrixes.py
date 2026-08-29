import os
import numpy as np
from utils.dir_config import *
import utils.dir_manage

adjustment = {
    "Barthet2010": [0.8037, -0.0228],
    "Grey1977": [0.8079, -0.2427],
    "Grey1978": [0.7165, -0.2686],
    "Iverson1993_Onset": [0.239, -0.4121],
    "Iverson1993_Remainder": [0.9179, 0.0122],
    "Iverson1993_Whole": [0.6034, -0.1195],
    "Lakatos2000_Comb": [1.0888, -0.059],
    "Lakatos2000_Harm": [0.8324, -0.2259],
    "Lakatos2000_Perc": [1.9099, -0.0829],
    "McAdams1995": [8.1218, 0.293],
    "Patil2012_A3": [0.8398, -0.1993],
    "Patil2012_DX4": [0.7496, -0.1554],
    "Patil2012_GD4": [0.9512, -0.1114],
    "Saitis2020_e2set1_general": [0.995, -0.1441],
    "Siedenburg2016_e2set1": [0.995, -0.1441],
    "Siedenburg2016_e2set2": [0.8974, -0.1516],
    "Siedenburg2016_e2set3": [1.3704, -0.1111],
    "Siedenburg2016_e3": [1.3704, -0.1111],
    "Vahidi2020": [0.3809, -0.4748],
    "Zacharakis2014_english": [1.231, -0.1407],
    "Zacharakis2014_greek": [1.231, -0.1407]
}

for dataset in utils.dir_manage.list_datasets():

    dissimilarity_matrix = np.loadtxt(os.path.join(DATA_DIR, f"{dataset}_dissimilarity_matrix.txt"))

    entry_mask = dissimilarity_matrix > 1e-8
    scale, offset = adjustment[dataset]

    adjusted_dissimilarity_matrix = (
        scale * (dissimilarity_matrix - dissimilarity_matrix[entry_mask].mean())
        + dissimilarity_matrix[entry_mask].mean() + offset
    ) * entry_mask

    np.savetxt(
        os.path.join(ADJUST_DATA_DIR, f"{dataset}_dissimilarity_matrix.txt"),
        adjusted_dissimilarity_matrix,
        fmt="%.15f"
    )

