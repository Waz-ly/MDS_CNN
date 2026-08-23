import os
from utils.dir_config import *

def list_datasets():

    items = os.listdir(DATA_DIR)

    dataset_files = [
        item.replace("_dissimilarity_matrix.txt", "")
        for item in items
        if item.endswith("_dissimilarity_matrix.txt")
    ]

    return dataset_files

def list_dataset_audio(dataset):

    audio_dir = os.path.join(AUDIO_DIR, dataset)

    if not os.path.isdir(audio_dir):
        raise Exception(f"dataset {dataset} not found")

    items = os.listdir(audio_dir)

    audio_files = [
        item
        for item in items
        if item.lower().endswith(".aiff")
    ]

    return sorted(audio_files)

def list_dataset_embedding(dataset):

    embed_dir = os.path.join(EMBEDDING_DIR, dataset)

    if not os.path.isdir(embed_dir):
        raise Exception(f"dataset {dataset} not found")

    items = os.listdir(embed_dir)

    embedding_files = [
        item
        for item in items
        if item.lower().endswith(".npy")
    ]

    return sorted(embedding_files)