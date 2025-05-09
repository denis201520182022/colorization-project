import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from skimage import color
from torchvision.transforms.functional import rgb_to_grayscale

class ColorizationDataset(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    img_color, _ = self.dataset[idx]
    img_gray = rgb_to_grayscale(img_color)

    
    
    return img_gray, img_color

