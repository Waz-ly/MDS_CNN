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
        adjusted: bool=False
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

    variations, no_variant = utils.data_augmentation.get_variant_meshgrid()

    batches = []

    for dataset in datasets:

        dataset_embeddings = torch.stack([
            torch.from_numpy(np.load(os.path.join(EMBEDDING_DIR, dataset, file))).float()
            for file in utils.dir_manage.list_dataset_embedding(dataset)
            if variants or len(file.split("=")) < 4
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
                len(variations) if variants else 1, dim=0
            ),  len(variations) if variants else 1, dim=1
            )
        )

        batches.append([dataset_embeddings, target_matrix])

    return batches

def train(
        datasets: list[str]=TRAIN_SET,
        lr: float = 2.34e-5,
        weight_decay: float = 1.7e-4,
        hidden_dim: int = 32,
        n_hidden: int = 2,
        epochs: int = 1000,
        activation = "relu",
        dropout = 0.041
    ) -> EmbeddingNet:

    training_batch = build_batch(datasets, variants=False, adjusted=True)
    in_dim = training_batch[0][0].shape[1]

    layer_dims = [in_dim] + [hidden_dim] * n_hidden + [8]
    model = EmbeddingNet(layer_dims=layer_dims, activation=activation, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
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
