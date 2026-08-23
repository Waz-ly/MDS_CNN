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

        x = self.net(x)
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)

def build_batch(datasets, variants=True):

    variations, no_variant = utils.data_augmentation.get_variant_meshgrid()

    total_embeddings = sum(
        len(utils.dir_manage.list_dataset_embedding(dataset))
        for dataset in datasets
    )
    total_embeddings //= 1 if variants else len(variations)

    embeddings_list = []
    true_dissimilarity_matrix = torch.zeros((total_embeddings, total_embeddings))

    offset = 0
    for dataset in datasets:

        dataset_embedding_files = utils.dir_manage.list_dataset_embedding(dataset)
        dataset_embeddings = torch.stack([
            torch.from_numpy(np.load(os.path.join(EMBEDDING_DIR, dataset, file))).float()
            for file in dataset_embedding_files
            if variants or len(file.split("=")) < 4
        ])

        dissimilarity_matrix = torch.from_numpy(
            np.loadtxt(os.path.join(DATA_DIR, f"{dataset}_dissimilarity_matrix.txt"))
        ).float()

        true_dissimilarity_matrix[
            offset : offset + len(dataset_embeddings),
            offset : offset + len(dataset_embeddings)
        ] = torch.repeat_interleave(
            torch.repeat_interleave(
                dissimilarity_matrix,
                len(variations) if variants else 1, dim=0
            ),  len(variations) if variants else 1, dim=1
        )

        offset += len(dataset_embeddings)
        embeddings_list.append(dataset_embeddings)

    embeddings_list = torch.cat(embeddings_list, dim=0)
    dissimilarity_exists_mask = true_dissimilarity_matrix > 1e-8

    return embeddings_list, true_dissimilarity_matrix, dissimilarity_exists_mask

def stress_loss(coords, target_dissim, pair_mask):
    pred_dist = torch.cdist(coords, coords, p=2)
    return torch.mean(torch.abs(pred_dist[pair_mask] - target_dissim[pair_mask]))

if __name__ == "__main__":

    training_sets = [dataset for dataset in utils.dir_manage.list_datasets() if not dataset in TEST_SET]
    vecs, target, pair_mask = build_batch(training_sets, variants=False)

    model = EmbeddingNet(layer_dims=[vecs.shape[1], 64, 64, 64, 8])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    for epoch in range(5000):
        optimizer.zero_grad()
        coords = model(vecs)
        coords = coords / coords.norm(dim=1, keepdim=True).clamp_min(1e-8)
        loss = stress_loss(coords, target, pair_mask)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.6f}")

    print("done training...")

    with open("model/model_info.txt", "w") as file:
        file.write(f"model size: {vecs.shape[1]}")

    torch.save(model.state_dict(), "model/mapper.pth")