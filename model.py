import torch
import torch.nn as nn
import numpy as np
import librosa
import openl3

class EmbeddingNet(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()

        self.layer_dims = layer_dims

        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        layers.pop()
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

TARGET_SR = 22050
TARGET_SAMPLES = int(4.6 * TARGET_SR)

class OpenL3Wrapper(nn.Module):
    def __init__(self, trained_net, sr=48000, embedding_size=512):

        super().__init__()
        self.net = trained_net
        self.sr = sr
        self.embedding_size = embedding_size

    def _fix_length(self, audio_np):

        n = len(audio_np)
        if n < TARGET_SAMPLES:
            pad = TARGET_SAMPLES - n
            audio_np = np.pad(audio_np, (0, pad), mode="constant")
        elif n > TARGET_SAMPLES:
            start = (n - TARGET_SAMPLES) // 2  # center-crop
            audio_np = audio_np[start:start + TARGET_SAMPLES]
        return audio_np

    def forward(self, audio):

        audio_np = audio.squeeze(0).detach().cpu().numpy()

        if self.sr != TARGET_SR:
            audio_np = librosa.resample(audio_np, orig_sr=self.sr, target_sr=TARGET_SR)

        audio_np = self._fix_length(audio_np)

        emb, ts = openl3.get_audio_embedding(
            audio_np, TARGET_SR,
            content_type="music",
            embedding_size=512,
        )
        # emb shape: (n_frames, embedding_size)
        emb_tensor = torch.from_numpy(emb).float().flatten().unsqueeze(0)
        return self.net(emb_tensor)
