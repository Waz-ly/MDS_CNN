TRAIN_SET = [
    "Barthet2010",
    "Grey1977",
    "Grey1978",
    "Iverson1993_Onset",
    "Iverson1993_Remainder",
    "Iverson1993_Whole",
    "Lakatos2000_Comb",
    "Lakatos2000_Harm",
    "Lakatos2000_Perc",
    "McAdams1995",
    "Patil2012_A3",
    "Patil2012_DX4",
    "Patil2012_GD4",
    "Saitis2020_e2set1_general",
    "Siedenburg2016_e2set1",
    # "Siedenburg2016_e2set2",
    "Siedenburg2016_e2set3",
    "Siedenburg2016_e3",
    "Vahidi2020",
    "Zacharakis2014_english",
    "Zacharakis2014_greek"
]

TEST_SET = ["Siedenburg2016_e2set2"]


SINGLE_SETS = [
    ["Barthet2010"],
    ["Grey1977"],
    ["Grey1978"],
    ["Iverson1993_Onset"],
    ["Iverson1993_Remainder"],
    ["Iverson1993_Whole"],
    ["Lakatos2000_Comb"],
    ["Lakatos2000_Harm"],
    ["Lakatos2000_Perc"],
    ["McAdams1995"],
    ["Patil2012_A3"],
    ["Patil2012_DX4"],
    ["Patil2012_GD4"],
    ["Saitis2020_e2set1_general", "Siedenburg2016_e2set1"],
    ["Siedenburg2016_e2set2"],
    ["Siedenburg2016_e2set3", "Siedenburg2016_e3"],
    ["Vahidi2020"],
    ["Zacharakis2014_english", "Zacharakis2014_greek"]
]

# Following pairs of datasets use the same audio set:
# Zacharakis2014_english, Zacharakis2014_greek
# Siedenburg2016_e3, Siedenburg2016_e2set3
# Saitis2020_e2set1_brightness, Siedenburg2016_e2set1