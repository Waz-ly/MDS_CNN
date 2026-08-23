import os
import librosa
import numpy as np
import openl3
import utils.dir_manage
import utils.data_augmentation
from utils.dir_config import *
import hashlib
import shutil

def hash_dataset_audio(dataset):

    h = hashlib.md5()

    for audio_file in sorted(utils.dir_manage.list_dataset_audio(dataset)):

        with open(os.path.join(AUDIO_DIR, dataset, audio_file), "rb") as f:
            h.update(f.read())

    return h.hexdigest()

if __name__ == "__main__":

    variations, no_variant = utils.data_augmentation.get_variant_meshgrid()
    pitch_variations = set(variant["pitch"] for variant in variations)

    seen_datasets = {}

    for dataset in utils.dir_manage.list_datasets():

        save_dir = os.path.join(EMBEDDING_DIR, dataset)
        dataset_hash = hash_dataset_audio(dataset)

        if dataset_hash in seen_datasets:

            shutil.copytree(os.path.join(EMBEDDING_DIR, seen_datasets[dataset_hash]), save_dir, dirs_exist_ok=True)
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
                batch_audio, sr, content_type="music", embedding_size=512, batch_size=len(batch_audio)
            )

            for embedding, save_file in zip(embeddings, batch_names):
                np.save(os.path.join(save_dir, save_file), embedding.flatten())