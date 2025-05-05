import torch
import torch.nn as nn


class ColorizationNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Энкодер: сжимаем информацию
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # [1, 64, 64] -> [32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                # -> [32, 32, 32]

            nn.Conv2d(32, 64, 3, padding=1), # -> [64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                # -> [64, 16, 16]
        )

        # Декодер: восстанавливаем цвета
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1),  # -> [64, 16, 16] -> [64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1),  # -> [32, 32, 32] -> [32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),  # -> [3, 64, 64]
            nn.Sigmoid()  # чтобы значения были от 0 до 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
