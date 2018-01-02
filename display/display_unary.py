
import matplotlib
matplotlib.use('tkAgg')

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import numpy as np
import utils
from src.unaries import Unary
from src.scorer import Scorer
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
img0 = utils.img_to_torch(img0)
img1 = utils.img_to_torch(img1)

#
scorer = Scorer(num_input_channel=3)
unary = Unary(scorer)

score = unary.forward(img0, img1, 10.)
score = score.data[0, :, :].numpy()

flow = Variable(torch.zeros(1, img0.size()[2], img0.size()[3], 2))
flow[:, ] = 0.
flow[:, :, :, 0] = 10.

interp = Interpolator()
imgw = interp.forward(img0, flow)

imgw = utils.img_to_numpy(imgw.data)

plt.imshow(score)
plt.show()

