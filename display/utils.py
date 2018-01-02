import numpy as np
import torch

def img_to_numpy(img_torch):

    img = img_torch[0, ]
    img = img.cpu().numpy()
    img = np.transpose(img, [1, 2, 0])

    return img


def img_to_torch(img_numpy):

    img = np.transpose(img_numpy, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)

    return img