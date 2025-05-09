import torch
import torch.nn as nn



class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()

        # Encoder: уменьшает размер, но увеличивает глубину (кол-во каналов)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # (1, 32, 32) -> (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (128, 16, 16)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # (256, 8, 8)
        )

        # Decoder: восстанавливает размер до исходного (32x32), но с 2 каналами
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # (128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),             # (2, 32, 32)
            nn.Tanh()  # чтобы выход был от -1 до 1, как и ab каналы
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
