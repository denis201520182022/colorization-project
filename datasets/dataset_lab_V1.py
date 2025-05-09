

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from skimage import color

class CIFAR10_Lab(Dataset):
  def __init__(self, train=True):
    self.data = CIFAR10(root='./data', train=train, download=True)
    self.images = self.data.data

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_rgb = self.images[idx]/255.0
    img_lab = color.rgb2lab(img_rgb)

    L = img_lab[:,:, 0] / 100.0
    ab = img_lab[:,:, 1:] / 128.0

    L_tenzor = torch.tensor(L, dtype=torch.float32).unsqueeze(0)
    ab_tenzor = torch.tensor(ab, dtype=torch.float32).permute(2,0,1)

    return L_tenzor, ab_tenzor