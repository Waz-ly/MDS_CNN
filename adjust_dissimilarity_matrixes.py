import os
import numpy as np
from utils.dir_config import *
import utils.dir_manage

adjustment = {
    "Barthet2010": 1.0985,
    "Grey1977": 0.8570,
    "Grey1978": 0.8780,
    "Iverson1993_Onset": 0.3289,
    "Iverson1993_Remainder": 1.0618,
    "Iverson1993_Whole": 0.7800,
    "Lakatos2000_Comb": 0.9694,
    "Lakatos2000_Harm": 0.7839,
    "Lakatos2000_Perc": 0.9874,
    "McAdams1995": 7.0405,
    "Patil2012_A3": 0.8975,
    "Patil2012_DX4": 0.9676,
    "Patil2012_GD4": 0.9875,
    "Saitis2020_e2set1_brightness": 0.9639,
    "Siedenburg2016_e2set1": 0.9159,
    "Siedenburg2016_e2set2": 0.8793,
    "Siedenburg2016_e2set3": 0.9966,
    "Siedenburg2016_e3": 0.9530,
    "Vahidi2020": 0.5299,
    "Zacharakis2014_english": 0.8436,
    "Zacharakis2014_greek": 0.6977
}

for dataset in utils.dir_manage.list_datasets():

    dissimilarity_matrix = np.loadtxt(os.path.join(DATA_DIR, f"{dataset}_dissimilarity_matrix.txt"))

    adjusted_dissimilarity_matrix = dissimilarity_matrix * adjustment[dataset]

    np.savetxt(
        os.path.join(ADJUST_DATA_DIR, f"{dataset}_dissimilarity_matrix.txt"),
        adjusted_dissimilarity_matrix,
        fmt="%.15f"
    )

