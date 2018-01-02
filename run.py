"""
Project:    m3n_fast_pd
File:       run.py
Created by: louise
"""

import argparse
import random
import string
import time

import torch
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import transforms

from src.scorer import Scorer
from src.energy import Energy
from src.margin import Margin
from src.m3n import M3N
from src.mrf import MRF
from src.unaries import Unary
from src.pairwise import Pairwise
from src.weights import Weights
from src.distance_matrix import Distance
from src.KittiDataset import KITTI, indexer_KITTI



def id_generator(size=6, chars=string.ascii_letters + string.digits):
    """
    Function to generate randon names for images to save.
    :param size: int, #of characters in the
    :param chars:
    :return:
    """
    return ''.join(random.choice(chars) for _ in range(size))


parser = argparse.ArgumentParser(description='Run Primal Dual Net.')
parser.add_argument('--use_cuda', type=bool, default=False,
                        help='Flag to use CUDA, if available')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='Number of epochs in the Primal Dual Net')
parser.add_argument('--save_flag', type=bool, default=True,
                    help='Flag to save or not the result images')
parser.add_argument('--log', type=bool, help="Flag to log loss in tensorboard", default=False)
parser.add_argument('--out_folder', help="output folder for images",
                    default="firetiti__20it_50_epochs_sigma005_006_smooth_loss_lr_10-3_batch300_dataset_/")
parser.add_argument('--clip', type=float, default=0.1,
                    help='Value of clip for gradient clipping')
args = parser.parse_args()

# Supplemental imports
if args.log:
    from tensorboard import SummaryWriter
    # Keep track of loss in tensorboard
    writer = SummaryWriter()
# Set parameters:
max_epochs = args.max_epochs

# Transform dataset

transformations = transforms.Compose([transforms.CenterCrop((16, 16)), transforms.ToTensor()])
transformations_d = transforms.Compose([transforms.CenterCrop((16, 16))])
dd = KITTI("/Users/louisenaud1/Documents/data_disparity/KITTI/", indexer=indexer_KITTI, transform=transformations,
           depth_transform=transformations_d)
if args.use_cuda:
    dtype = torch.cuda.DoubleTensor
else:
    dtype = torch.DoubleTensor

train_loader = DataLoader(dd, batch_size=8, num_workers=4)
scorer = Scorer()
weights = Weights()
distance = Distance()
unaries = Unary(scorer)
pairwise = Pairwise(weights, distance)
energy = Energy(unaries, pairwise)
margin = Margin()
mrf = MRF(unaries, pairwise)
net = M3N(energy, margin)

# loss criterion for data
criterion = torch.nn.MSELoss(size_average=True)
# loss criterion for data smoothness
criterion_g = torch.nn.MSELoss(size_average=True)
# Adam Optimizer with initial learning rate of 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# Scheduler to decrease the leaning rate at each epoch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
params = list(net.parameters())
# Store losses and energies for plotting
loss_history = []
it = 0


# Start training
t0 = time.time()
net.train()

x_min = torch.zeros([1])
x_max = 5000. * torch.ones([1])
for epoch in range(max_epochs):
    loss_epoch = 0.
    for batch_id, (batch0, batch1, batchd) in enumerate(train_loader):
            # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        batch0, batch1, batchd = Variable(batch0).type(dtype), Variable(batch1).type(dtype), \
                                 Variable(batchd).type(dtype)
        x_opt = mrf.forward(batch0, batch1, 0., 200.)
        x_opt = Variable(x_opt, requires_grad=True).type(dtype)
        # reset optimizer
        optimizer.zero_grad()
        # compute estimate for disparity
        batchd.squeeze_(1)
        output = net.forward(batch0, batch1, batchd, x_opt)
        # compute loss
        loss = criterion(x_opt, batchd)
        # backpropagation
        loss.backward()
        # optimizer step
        optimizer.step()
        loss_epoch += loss

        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_id, len(train_loader.dataset),
        #         100. * batch_id / len(train_loader), loss.data[0]))

    print("-------- Loss Epoch = ", loss_epoch)
    scheduler.step()


t1 = time.time()
print("Elapsed time in minutes :", (t1 - t0) / 60.)
