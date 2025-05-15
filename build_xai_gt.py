import numpy as np
import scipy.io as sio
from part_1_modeling.utils.utils import plot_gt, ids_to_crops, palette


dev_gt = sio.loadmat('./data/processed_data/prisma.mat')["prisma"]["dev_gt"][0, 0]

attrs_gt =  np.zeros_like(dev_gt) - 1
min_examples = 512 # Number of examples to select for each class for xai"

for i in range(6):
    X = np.stack(np.where(dev_gt == i), axis=1)
    pos = np.random.choice(X.shape[0], size=min_examples, replace=False)  
    examples = X[pos]
    attrs_gt[examples[:, 0], examples[:, 1]] = i
    
plot_gt(attrs_gt+1, ids_to_crops, palette, title="xai_gt")
np.save("./data/xai_gt.npy", attrs_gt)
