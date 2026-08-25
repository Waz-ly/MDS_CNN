from train import EmbeddingNet, build_batch, total_stress_loss
import torch
from utils.train_set_config import *


if __name__ == "__main__":

    model_size = 0
    with open('model/model_info.txt', 'r') as file:
        lines = file.readlines()
        model_size = int(lines[0].split()[2])

    model = EmbeddingNet(layer_dims=[model_size, 64, 64, 64, 8])
    model.load_state_dict(torch.load("model/mapper.pth"))
    model.eval()

    test_batch = build_batch(TEST_SET, variants=False, adjusted=False)

    loss = total_stress_loss(model, test_batch, rescale=False)
    print(loss.item())
    loss = total_stress_loss(model, test_batch, rescale=True)
    print(loss.item())
