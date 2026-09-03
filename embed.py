import os
import librosa
import numpy as np
import openl3
import utils.dir_manage
import utils.data_augmentation
from utils.dir_config import *
import hashlib
import shutil
import torch
from stress import stress_loss

def hash_dataset_audio(dataset):

    h = hashlib.md5()

    for audio_file in sorted(utils.dir_manage.list_dataset_audio(dataset)):

        with open(os.path.join(AUDIO_DIR, dataset, audio_file), "rb") as f:
            h.update(f.read())

    return h.hexdigest()

def embed(
        content_type: str="music"
    ):

    variations, no_variant = utils.data_augmentation.get_variant_meshgrid()
    pitch_variations = set(variant["pitch"] for variant in variations)

    seen_datasets = {}

    for dataset in utils.dir_manage.list_datasets():

        save_dir = os.path.join(EMBEDDING_DIR, content_type, dataset)
        dataset_hash = hash_dataset_audio(dataset)

        if dataset_hash in seen_datasets:

            shutil.copytree(os.path.join(EMBEDDING_DIR, content_type, seen_datasets[dataset_hash]), save_dir, dirs_exist_ok=True)
            print(f"{dataset} identical to {seen_datasets[dataset_hash]}: copied embeddings")
            continue

        seen_datasets[dataset_hash] = dataset
        os.makedirs(save_dir, exist_ok=True)

        for audio_file in utils.dir_manage.list_dataset_audio(dataset):

            audio_path = os.path.join(AUDIO_DIR, dataset, audio_file)
            audio_name = audio_file.split(".")[0]

            audio, sr = librosa.load(audio_path, duration=4.4)
            audio = np.pad(audio, (0, int(4.4 * sr - len(audio))))

            pitch_shifted_audio = {
                pitch: librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch)
                for pitch in pitch_variations
            }

            batch_audio, batch_names = [], []
            for variation in variations:

                save_file = (
                    audio_name + ".npy"
                    if variation == no_variant
                    else audio_name + f"_d={variation['delay']}_v={variation['volume']}_p={variation['pitch']}.npy"
                )

                modified_audio = pitch_shifted_audio[variation["pitch"]]
                modified_audio = np.pad(modified_audio, (int(variation["delay"] * sr), 0))
                modified_audio = np.pad(modified_audio, (0, int(4.6 * sr - len(modified_audio))))
                modified_audio *= variation["volume"]

                batch_audio.append(modified_audio)
                batch_names.append(save_file)

            embeddings, _ = openl3.get_audio_embedding(
                batch_audio, sr, content_type=content_type, embedding_size=512, batch_size=len(batch_audio)
            )

            for embedding, save_file in zip(embeddings, batch_names):
                np.save(os.path.join(save_dir, save_file), embedding.flatten())

def pca_compress(x, n_components=2):

    x_centered = x - x.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(x_centered, q=n_components)
    return x_centered @ V[:, :n_components]

def embedding_analyze(
        content_type: str="music"
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

    normal_loss: torch.Tensor = 0
    triplet_loss: torch.Tensor = 0
    NDCG_loss: torch.Tensor = 0

    n_normal = 0
    n_triplet = 0
    n_NDCG = 0

    for dataset in utils.dir_manage.list_datasets():

        dataset_embeddings = torch.stack([
            torch.from_numpy(np.load(os.path.join(EMBEDDING_DIR, content_type, dataset, file))).float()
            for file in utils.dir_manage.list_dataset_embedding(dataset, content_type=content_type)
            if len(file.split("=")) < 4
        ])

        # dataset_embeddings = pca_compress(dataset_embeddings, n_components=2)

        target_matrix = torch.from_numpy(
            np.loadtxt(
                os.path.join(
                    DATA_DIR,
                    f"{dataset}_dissimilarity_matrix.txt"
                )
            )
        ).float()

        l, n = stress_loss(dataset_embeddings, target_matrix, rescale_type="normalize", loss_type="MAE")
        normal_loss, n_normal = normal_loss + l * n, n_normal + n
        l, n = stress_loss(dataset_embeddings, target_matrix, rescale_type="normalize", loss_type="triplet")
        triplet_loss, n_triplet = triplet_loss + l * n, n_triplet + n
        l, n = stress_loss(dataset_embeddings, target_matrix, rescale_type="normalize", loss_type="NDCG")
        NDCG_loss, n_NDCG = NDCG_loss + l * n, n_NDCG + n

    print(f"MAE (normal): {normal_loss.item() / n_normal}")
    print(f"triplet: {triplet_loss.item() / n_triplet}")
    print(f"NDCG: {NDCG_loss.item() / n_NDCG}")

if __name__ == "__main__":

    embedding_analyze(content_type="music")
    embedding_analyze(content_type="env")