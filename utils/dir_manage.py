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

def list_dataset_embedding(
        dataset,
        content_type="music"
    ):

    embed_dir_path = os.path.join(EMBEDDING_DIR, content_type, dataset)

    if not os.path.isdir(embed_dir_path):
        raise Exception(f"dataset {dataset} not found")

    items = os.listdir(embed_dir_path)

    embedding_files = [
        item
        for item in items
        if item.lower().endswith(".npy")
    ]

    return sorted(embedding_files)