
import matplotlib
matplotlib.use('tkAgg')

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import numpy as np
import utils

from src.mrf import MRF
from src.scorer import Scorer
from src.unaries import Unary
from src.pairwise import Pairwise
from src.weights import Weights
from src.distance_matrix import Distance
from src.interpolator import Interpolator

img0_path = '../data/test/tsukuba1.png'
img1_path = '../data/test/tsukuba2.png'

# Read images
img0 = plt.imread(img0_path)
img0 = img0.astype(dtype=np.float32)

img1 = plt.imread(img1_path)
img1 = img1.astype(dtype=np.float32)


#plt.imshow(img0)
#plt.show()

# Convert to pytorch format
img0 = Variable(utils.img_to_torch(img0))
img1 = Variable(utils.img_to_torch(img1))

#
scorer = Scorer(num_input_channel=3)
unary = Unary(scorer)

weights = Weights(num_input_channel=3)
distance = Distance()
pairwise = Pairwise(weights, distance)

mrf = MRF(unary, pairwise)

x_min = mrf.forward(img0, img1, 0, 14)


plt.imshow(x_min[0, :, :])
plt.show()

