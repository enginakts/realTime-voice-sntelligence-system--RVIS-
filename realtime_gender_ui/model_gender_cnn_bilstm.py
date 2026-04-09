from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    Notebook'taki mimarinin aynısı.
    Input: (B, 1, F=80, T)
    Output: (B, 2)
    """

    def __init__(self, n_mels: int = 80, n_classes: int = 2):
        super().__init__()
        self._n_mels = n_mels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # F/2, T/2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # F/4, T/4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # F/8, T/4
        )

        self.lstm = nn.LSTM(
            input_size=64 * (n_mels // 8),
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cnn(x)  # (B, C, F', T')
        b, c, fp, tp = z.shape
        z = z.permute(0, 3, 1, 2).contiguous().view(b, tp, c * fp)  # (B, T', C*F')
        z, _ = self.lstm(z)  # (B, T', 256)
        z = z.mean(dim=1)  # temporal pooling
        return self.head(z)


def load_gender_model(checkpoint_path: str, device: str = "cpu") -> CNNBiLSTM:
    """
    `checkpoint_path` aşağıdakilerden biri olabilir:
    - **state_dict** (analiz.py / eğitim çıktısı): torch.save(model.state_dict(), path)
    - **full model**: torch.save(model, path)  (nadiren)
    - **checkpoint dict**: {'state_dict': ..., ...}
    """
    obj = torch.load(checkpoint_path, map_location=device)

    # Full model kaydı
    if isinstance(obj, nn.Module):
        obj.to(device)
        obj.eval()
        return obj  # type: ignore[return-value]

    # Checkpoint dict
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    model = CNNBiLSTM(n_mels=80, n_classes=2)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

