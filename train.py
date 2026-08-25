import torch
import torch.nn as nn
import numpy as np
import os
import utils.dir_manage
import utils.data_augmentation
from utils.dir_config import *
from utils.train_set_config import *

class EmbeddingNet(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()

        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        layers.pop()
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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

def get_scale_mismatch(
        predicted_coords: torch.Tensor,
        target_matrix: torch.Tensor,
    ):

    entry_mask = target_matrix > 1e-8
    predicted_matrix = torch.cdist(predicted_coords, predicted_coords, p=2)

    with torch.no_grad():
        average_predicted_distance = torch.mean(predicted_matrix[entry_mask])
        average_target_distance = torch.mean(target_matrix[entry_mask])
        scale = average_predicted_distance / average_target_distance

    print(f"target matrix rescaled by: {scale}")

    return scale

def stress_loss(
        predicted_coords: torch.Tensor,
        target_matrix: torch.Tensor,
        rescale: bool = False,
    ) -> tuple[torch.Tensor, int]:

    entry_mask = target_matrix > 1e-8
    predicted_matrix = torch.cdist(predicted_coords, predicted_coords, p=2)

    scale = 1 if not rescale else get_scale_mismatch(predicted_coords, target_matrix)
    adjusted_target_matrix = target_matrix * scale

    return torch.mean(torch.abs(predicted_matrix[entry_mask] - adjusted_target_matrix[entry_mask])), entry_mask.sum().item()

def total_stress_loss(
        model: torch.nn.Module,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
        rescale: bool = False,
    ) -> torch.Tensor:

    total_loss: torch.Tensor = 0.0
    total_pairs = 0

    for [embeddings, target_matrix] in batch:

        predicted_coords = model(embeddings)
        loss, n_pairs = stress_loss(predicted_coords, target_matrix, rescale)

        total_loss += loss * n_pairs
        total_pairs += n_pairs

    mean_loss = total_loss / total_pairs

    return mean_loss

if __name__ == "__main__":

    training_batch = build_batch(TRAIN_SET, variants=False, adjusted=True)
    in_dim = training_batch[0][0].shape[1]

    model = EmbeddingNet(layer_dims=[in_dim, 64, 64, 64, 8])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = total_stress_loss(model, training_batch, rescale=False)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.6f}")

    print("done training...")

    with open("model/model_info.txt", "w") as file:
        file.write(f"model size: {in_dim}")

    torch.save(model.state_dict(), "model/mapper.pth")