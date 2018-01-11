"""
Project:    m3n_fast_pd
File:       run.py
Created by: louise
"""

import argparse
import random
import string
import time

import matplotlib
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.KittiDataset import KITTI, indexer_KITTI
from src.datasets.middlebury_dataset import Middlebury, indexer_middlebury
from src.distance_matrix import Distance
from src.energy import Energy
from src.m3n import M3N
from src.margin import Margin
from src.mrf import MRF
from src.pairwise import Pairwise
from src.scorer import Scorer
from src.unaries import Unary
from src.weights import Weights

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def id_generator(size=6, chars=string.ascii_letters + string.digits):
    """
    Function to generate randon names for images to save.
    :param size: int, #of characters in the
    :param chars:
    :return:
    """
    return ''.join(random.choice(chars) for _ in range(size))


parser = argparse.ArgumentParser(description='Run Primal Dual Net.')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='Flag to use CUDA, if available')
parser.add_argument('--max_epochs', type=int, default=30,
                    help='Number of epochs in the Primal Dual Net')
parser.add_argument('--save_flag', type=bool, default=True,
                    help='Flag to save or not the result images')
parser.add_argument('--log', type=bool, help="Flag to log loss in tensorboard",
                    default=True)
parser.add_argument('--out_folder', help="output folder for images",
                    default="results/")
parser.add_argument('--clip', type=float, default=0.1,
                    help='Value of clip for gradient clipping')
args = parser.parse_args()

# Supplemental imports
if args.log:
    from tensorboard import SummaryWriter

    # Keep track of loss in tensorboard
    writer = SummaryWriter("M3N_zizi_mid_1000_1000")
# Set parameters:
max_epochs = args.max_epochs

# Transform dataset
patch_size_1 = 1000
patch_size_2 = 1000
transformations = transforms.Compose([transforms.CenterCrop((patch_size_1, patch_size_2)), transforms.ToTensor()])
transformations_d = transforms.Compose([transforms.CenterCrop((patch_size_1, patch_size_2))])
#dd = KITTI("/media/louise/data/datasets/KITTI/stereo/training", indexer=indexer_KITTI, transform=transformations,
#           depth_transform=transformations_d)
#dd = KITTI("/Users/louisenaud1/m3n_fast_pd/data/", indexer=indexer_KITTI, transform=transformations,
#           depth_transform=transformations_d)
dd = KITTI("/media/louise/data/datasets/KITTI/stereo/training", indexer=indexer_KITTI, transform=transformations,
           depth_transform=transformations_d)
dd = Middlebury("/media/louise/data/datasets/middlebury", indexer=indexer_middlebury, transform=transformations,
           depth_transform=transformations_d)
if args.use_cuda:
    dtype = torch.cuda.DoubleTensor
else:
    dtype = torch.DoubleTensor

train_loader = DataLoader(dd, batch_size=1, num_workers=4, shuffle=False)
scorer = Scorer()
weights = Weights()
distance = Distance()
unaries = Unary(scorer)
pairwise = Pairwise(weights, distance)
energy = Energy(unaries, pairwise)
margin = Margin()
mrf = MRF(unaries, pairwise).type(dtype)
net = M3N(energy, margin).type(dtype)

# loss criterion for data
criterion = torch.nn.MSELoss(size_average=True)
# Adam Optimizer with initial learning rate of 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
# Scheduler to decrease the leaning rate at each epoch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.08)
params = list(net.parameters())
# Store losses and energies for plotting
loss_history = []
it = 0

# Start training
t0 = time.time()
net.train()

x_min = torch.zeros([1])
x_max = 255. * torch.ones([1])
for epoch in range(max_epochs):
    loss_epoch = 0.
    for batch_id, (batch0, batch1, batchd, batchm) in enumerate(train_loader):
        if batch_id % 100 == 0:
            # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
            if args.use_cuda:  # Speed-up if cuda is in use
                batch0 = batch0.pin_memory()
                batch1 = batch1.pin_memory()
                batchd = batchd.pin_memory()
                batchm = batchm.pin_memory()

            # For computing x_opt
            batch0, batch1, batchd = Variable(batch0, volatile=True).type(dtype), Variable(batch1, volatile=True).type(dtype), \
                                     Variable(batchd, volatile=True).type(dtype)
            batchm = Variable(batchm).type(dtype)

            if args.use_cuda:  # Speed-up if cuda is in use
                batch0 = batch0.cuda(async=True)
                batch1 = batch1.cuda(async=True)
                batchd = batchd.cuda(async=True)
                batchm = batchm.cuda(async=True)

            # Compute MRF MAP inference
            x_opt = mrf.forward(batch0, batch1, 1., 255., batchd, margin)
            x_opt = Variable(x_opt, requires_grad=False).type(dtype)  # x_opt doesn't require gradient wrt net parameters
            # reset optimizer
            optimizer.zero_grad()

            # compute estimate for disparity
            batch0.volatile = False
            batch1.volatile = False
            batchd.volatile = False

            # Disparity and mask don't have a channel dimension
            batchd.squeeze_(1)
            batchm.squeeze_(1)
            output = net.forward(batch0, batch1, batchd, x_opt, batchm)
            print("output =", output)
            # compute loss
            loss = criterion(output, Variable(torch.zeros(output.size())).type(dtype))
            # backpropagation
            loss.backward()
            # optimizer step
            optimizer.step()
            loss_epoch += loss.data[0]
            # if args.log and it % 10 == 0:
            #     writer.add_scalar("loss_batch", torch.sum(loss).data[0], it)
            it += 1

        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_id, len(train_loader.dataset),
        #         100. * batch_id / len(train_loader), loss.data[0]))

    print("-------- Loss Epoch = ", loss_epoch)
    x_est = mrf.forward(Variable(dd[0][0].unsqueeze_(0)).type(dtype), Variable(dd[0][1].unsqueeze_(0)).type(dtype), 1., 255.)
    fn_out = "res_mid_1000_1000_mask_2_epoch_" + str(epoch) + ".png"
    plt.figure()
    plt.imshow(x_est.squeeze_(0).cpu().numpy())
    plt.savefig(fn_out)
    if args.log:
        writer.add_scalar("loss_epoch", loss_epoch, it)
    scheduler.step()

t1 = time.time()
print("Elapsed time in minutes :", (t1 - t0) / 60.)

img1 = Variable(dd[0][0].unsqueeze_(0)).type(dtype)
img2 = Variable(dd[0][1].unsqueeze_(0)).type(dtype)
x_est = mrf.forward(img1, img2, 1., 255.)
plt.figure()
plt.imshow(x_est.squeeze_(0).cpu().numpy())
plt.savefig('res_mid_1000_1000_100e_2_mask.png')
plt.show()
