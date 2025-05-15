import os
import numpy as np
import scipy.io as sio


# Check the number of the samples in the train, test, and dev sets
train_gt = sio.loadmat('./data/processed_data/prisma.mat')["prisma"]["train_gt"][0, 0]
test_gt = sio.loadmat('./data/processed_data/prisma.mat')["prisma"]["test_gt"][0, 0]
dev_gt = sio.loadmat('./data/processed_data/prisma.mat')["prisma"]["dev_gt"][0, 0]
print("Train Set: ", np.unique(train_gt, return_counts=True))
print("Dev Set: ",np.unique(dev_gt, return_counts=True))
print("Test Set: ",np.unique(test_gt, return_counts=True))