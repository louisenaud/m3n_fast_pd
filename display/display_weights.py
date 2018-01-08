import matplotlib

matplotlib.use('tkAgg')

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import numpy as np
import utils
from src.weights import Weights
from src.scorer import Scorer
from src.interpolator import Interpolator

img0_path = '../data/test/tsukuba1.png'

# Read images
img0 = plt.imread(img0_path)
img0 = img0.astype(dtype=np.float32)

# Convert to pytorch format
img0 = Variable(utils.img_to_torch(img0))

#
weights = Weights(num_input_channel=3)

w = weights.forward(img0)
w = w.data.numpy()

plt.imshow(w[0, :, :, 0])
plt.show()

plt.imshow(w[0, :, :, 1])
plt.show()
