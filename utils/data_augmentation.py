import itertools

def get_variant_meshgrid():

    no_variant = {
        "delay": 0,
        "volume": 1,
        "pitch": 0,
    }

    variation_dict = {
        "delay": [0, 0.2],
        "volume": [1],
        "pitch": [0, 1, -1],
    }

    keys = variation_dict.keys()
    variations = [dict(zip(keys, values)) for values in itertools.product(*variation_dict.values())]

    return variations, no_variant