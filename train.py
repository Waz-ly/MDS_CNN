import torch
import numpy as np
import os
import utils.dir_manage
import utils.data_augmentation
from utils.dir_config import *
from utils.train_set_config import *
import json
from model import EmbeddingNet
from stress import total_stress_loss

def build_batch(
        datasets: list[str],
        variants: bool=True,
        adjusted: bool=False,
        content_type: str="music",
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

    variations, no_variant = utils.data_augmentation.get_variant_meshgrid()
    n_variants = len(variations)

    batches = []

    for dataset in datasets:

        dataset_embeddings = torch.stack([
            torch.from_numpy(np.load(os.path.join(EMBEDDING_DIR, content_type, dataset, file))).float()
            for file in utils.dir_manage.list_dataset_embedding(dataset)
            if len(file.split("=")) < 4 or variants
        ])

        target_matrix = (
            torch.repeat_interleave(
            torch.repeat_interleave(
                torch.from_numpy(
                    np.loadtxt(
                        os.path.join(
                            DATA_DIR if not adjusted else ADJUST_DATA_DIR,
                            f"{dataset}_dissimilarity_matrix.txt"
                        )
                    )
                ).float(),
                n_variants if variants else 1, dim=0
            ),  n_variants if variants else 1, dim=1
            )
        )

        batches.append([dataset_embeddings, target_matrix])

    return batches

def train(
        datasets: list[str]=TRAIN_SET,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        in_dim: int = 21504,
        hidden_dim: int = 32,
        n_hidden: int = 3,
        out_dim: int = 8,
        n_epochs: int = 1000,
        content_type = "music",
    ) -> EmbeddingNet:

    training_batch = build_batch(datasets, variants=True, adjusted=True, content_type=content_type)
    in_dim = training_batch[0][0].shape[1]

    layer_dims = [in_dim] + [hidden_dim] * n_hidden + [out_dim]
    model = EmbeddingNet(layer_dims=layer_dims)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):

        optimizer.zero_grad()
        loss, _ = total_stress_loss(model, training_batch, rescale_type="none", loss_type="MAE")
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.6f}")

    print("done training...")

    return model

if __name__ == "__main__":

    model = train(TRAIN_SET)

    with open("model/model_config.json", "w") as f:
        json.dump(model.to_config_dict(), f)

    torch.save(model.state_dict(), "model/mapper.pth")
