from skimage.color import lab2rgb
import torch
import numpy as np



def lab_to_rgb_tensor(L_tensor, ab_tensor):
    L_np = L_tensor.squeeze(0).squeeze(0).cpu().numpy() * 100
    ab_np = ab_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 128

    lab_img = np.zeros((32, 32, 3), dtype=np.float32)
    lab_img[:, :, 0] = L_np
    lab_img[:, :, 1:] = ab_np

    return lab2rgb(lab_img)