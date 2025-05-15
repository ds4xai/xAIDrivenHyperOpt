import os
import numpy as np
import scipy.io as sio
from part_1_modeling.data_processing.custom_dataset import split_gt
from part_1_modeling.utils.utils import ids_to_crops, palette, plot_gt

def load_data(hsi_path, train_gt_path, test_gt_path):
    """Load data from .npy files."""
    hsi = np.load(hsi_path)
    train_gt = np.load(train_gt_path)
    test_gt = np.load(test_gt_path)
    return hsi, train_gt, test_gt

def split_and_visualize_data(train_gt, test_gt):
    """Split training data and visualize the results."""
    train_gt, dev_gt = split_gt(train_gt, train_size=.6, split_strategy="disjoint",
                                train_side="left", absence_of_crop_label=0)

    plot_gt(test_gt, ids_to_crops, palette, title="Ground Truth Test")
    plot_gt(dev_gt, ids_to_crops, palette, title="Ground Truth Dev")
    plot_gt(train_gt, ids_to_crops, palette, title="Ground Truth Train")

    return train_gt, dev_gt

def preprocess_data(hsi, train_gt, dev_gt, test_gt):
    """Preprocess data by removing insignificant bands and normalizing."""
    # Remap labels
    dev_gt -= 1
    test_gt -= 1
    train_gt -= 1

    # Remove insignificant bands
    ################### Delete insignificant bands #########################
    ## see Arthur paper: https://api.semanticscholar.org/CorpusID:264356070
    bands_to_remove = list(range(11)) + list(range(57, 64)) + list(range(101, 15)) + list(range(145, 163)) + list(range(210, 234))
    hsi = np.delete(hsi, bands_to_remove, axis=0)

    # Normalize data
    mins = np.min(hsi, axis=(1, 2), keepdims=True)
    maxs = np.max(hsi, axis=(1, 2), keepdims=True)
    hsi = (hsi - mins) / (maxs - mins)
    hsi = np.clip(hsi, 0, 1)

    return hsi, train_gt, dev_gt, test_gt

def calculate_statistics(hsi, train_gt):
    """Calculate mean and standard deviation for each spectral band."""
    train_mask = train_gt != -1
    means = [np.mean(hsi[i, train_mask]) for i in range(hsi.shape[0])]
    stds = [np.std(hsi[i, train_mask]) for i in range(hsi.shape[0])]
    return np.array(means), np.array(stds)

def save_data_as_mat(hsi, train_gt, dev_gt, test_gt, means, stds, output_dir):
    """Save processed data to a .mat file."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    data_dict = {
        'prisma': 
            {
                'dataset_name': 'Prisma',
                'description': 'Prisma dataset on 27 march 2020 with corrected labels',
                'data_shape': hsi.shape,
                'num_bands': hsi.shape[0],
                'num_classes': len(np.unique(train_gt)),
                'class_info': str(ids_to_crops),
                'hsi': hsi,
                'train_gt': train_gt,
                'dev_gt': dev_gt,
                'test_gt': test_gt,
                'means': means,
                'stds': stds
            },
        }
    file_path = os.path.join(output_dir, "prisma.mat")
    sio.savemat(file_path, data_dict)

def main():
    hsi, train_gt, test_gt = load_data("./data/hsi.npy", "./data/train_crops.npy", "./data/test_crops.npy")
    train_gt, dev_gt = split_and_visualize_data(train_gt, test_gt)
    hsi, train_gt, dev_gt, test_gt = preprocess_data(hsi, train_gt, dev_gt, test_gt)
    means, stds = calculate_statistics(hsi, train_gt)
    save_data_as_mat(hsi, train_gt, dev_gt, test_gt, means, stds, "./data/processed_data")
    print(np.unique(train_gt, return_counts=True))
    print(np.unique(test_gt, return_counts=True))
    print(np.unique(dev_gt, return_counts=True))

if __name__ == "__main__":
    main()
