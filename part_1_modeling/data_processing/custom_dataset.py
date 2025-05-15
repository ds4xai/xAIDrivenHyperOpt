########################################################################################
##########                           Importations                             ##########
########################################################################################
### Python community modules
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

### Project-specific modules
from part_1_modeling.utils.utils import set_seed

########################################################################################
##########                              Classes                               ##########
########################################################################################

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for handling hyperspectral image (HSI) data.

    Args:
        hsi (np.ndarray): Hyperspectral image data.
        stats (dict): Dictionary containing mean and standard deviation for standardization.
        gt (np.ndarray): Ground truth labels.
        split (str): Dataset split type ('train', 'test', etc.).
        patch_size (int): Size of the patches to extract (must be odd).
        standardize (bool): Whether to standardize the patches.
        stride (int): Stride for patch extraction.
        absence_of_crop_label (int): Label for absence of crop.
        padding_mode (str): Padding mode ('constant', etc.).
        padding_value (int): Value to use for constant padding.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, hsi: np.ndarray, stats: dict, gt: np.ndarray, split: str = "train",
                 patch_size: int = 3, standardize: bool = False, stride: int = 1,
                 absence_of_crop_label: int = -1, padding_mode: str = 'constant',
                 padding_value: int = 0, seed: int = None, **kwargs):
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")

        if seed is not None:
            set_seed(seed)

        self.hsi_tensor = torch.from_numpy(hsi).float()
        self.gt_tensor = torch.from_numpy(gt).long()
        self.split_type = split.lower()
        self.no_crop_label = absence_of_crop_label
        self.patch_size = patch_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.standardize = standardize

        # Prepare mean and standard deviation tensors
        self.mean_tensor = torch.from_numpy(stats["means"]).squeeze().unsqueeze(-1).unsqueeze(-1)
        self.std_tensor = torch.from_numpy(stats["stds"]).squeeze().unsqueeze(-1).unsqueeze(-1)

        # Add padding around the HSI
        self.pad_size = patch_size // 2
        self.padded_hsi = self.pad_hsi(self.hsi_tensor, self.pad_size, padding_mode, padding_value)

        # Compute the positions of patches in the original HSI
        self.patch_positions = self.compute_patch_positions(self.gt_tensor, stride)

    def pad_hsi(self, hsi_tensor, pad_size, mode, value):
        """Pads the input HSI with the specified mode and value."""
        return torch.nn.functional.pad(
            hsi_tensor,
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode=mode,
            value=value
        )

    def compute_patch_positions(self, gt_tensor, stride):
        """Computes positions of valid patches in the HSI based on the ground truth and stride."""
        valid_rows, valid_cols = torch.where(gt_tensor != self.no_crop_label)
        valid_positions = list(zip(valid_rows.tolist(), valid_cols.tolist()))

        if stride > 1:
            rows, cols = gt_tensor.shape
            valid_positions = [
                (row, col) for row in range(0, rows, stride) 
                for col in range(0, cols, stride) 
                if (row, col) in valid_positions
            ]

        if self.split_type == "train":
            np.random.default_rng().shuffle(valid_positions)

        return valid_positions

    def __len__(self):
        """Returns the total number of patches."""
        return len(self.patch_positions)

    def __getitem__(self, idx):
        """Returns a patch and its label."""
        center_row, center_col = self.patch_positions[idx]

        padded_center_row = center_row + self.pad_size
        padded_center_col = center_col + self.pad_size

        patch = self.padded_hsi[
            :,
            padded_center_row - self.pad_size:padded_center_row + self.pad_size + 1,
            padded_center_col - self.pad_size:padded_center_col + self.pad_size + 1
        ]

        if self.standardize:
            patch = (patch - self.mean_tensor) / self.std_tensor

        label = self.gt_tensor[center_row, center_col]

        return patch, label.long()

########################################################################################
##########                            Functions                               ##########
########################################################################################

def split_gt(gt: np.ndarray, train_size: float = 0.7, split_strategy: str = 'random_stratify',
             train_side: str = "right", absence_of_crop_label: int = -1):
    """
    Splits the ground truth into training and testing sets based on the specified strategy.

    Args:
        gt (np.ndarray): Ground truth array.
        train_size (float): Proportion of training data.
        split_strategy (str): Strategy for splitting ('random_stratify', 'disjoint', etc.).
        train_side (str): Side for training data ('right', 'left', 'top', 'bottom').
        absence_of_crop_label (int): Label for absence of crop.

    Returns:
        tuple: Training ground truth, testing ground truth.
    """
    split_strategy = split_strategy.lower()
    train_side = train_side.lower()

    if absence_of_crop_label is None:
        absence_of_crop_label = -1

    xy_indices = np.argwhere(gt != absence_of_crop_label)
    y = gt[xy_indices[:, 0], xy_indices[:, 1]]
    unique_classes = np.unique(y)

    train_gt = np.copy(gt)
    test_gt = np.copy(gt)
    train_mask = np.zeros(gt.shape, dtype=bool)
    test_mask = np.zeros(gt.shape, dtype=bool)

    if split_strategy == 'random_stratify':
        # Ensure all classes have sufficient samples for stratification
        class_counts = np.bincount(y)
        if np.any(class_counts < 2):
            raise ValueError("All classes must have at least 2 samples for stratification.")
        
        train_indices, test_indices = train_test_split(
            xy_indices, train_size=train_size, stratify=y, random_state=42
        )
        train_mask[train_indices[:, 0], train_indices[:, 1]] = True
    elif split_strategy == 'random':
        train_indices, test_indices = train_test_split(
            xy_indices, train_size=train_size, random_state=42
        )
        train_mask[train_indices[:, 0], train_indices[:, 1]] = True
    
    elif split_strategy == 'disjoint':

        test_size = 1-train_size 
        
        H, W = gt.shape

        test_indices = []
        for class_label in unique_classes:
            # Mask for the current class
            class_mask = (gt == class_label)
            total_pixels = class_mask.sum()

            if train_side == 'right':
                for col in range(1, W):
                    left_half_count = np.count_nonzero(class_mask[:, :col])
                    if (left_half_count / total_pixels) <= test_size:
                        continue
                    else:
                        break
                class_mask[:, col:] = False
                test_indices.extend(np.column_stack(np.where(class_mask)))
            elif train_side == 'left':
                for col in range(W - 1, 0, -1):
                    right_half_count = np.count_nonzero(class_mask[:, col:])
                    if (right_half_count / total_pixels) <= test_size:
                        continue
                    else:
                        break
                class_mask[:, :col] = False
                test_indices.extend(np.column_stack(np.where(class_mask)))
            elif train_side == 'top':
                for row in range(H - 1, 0, -1):
                    bottom_half_count = np.count_nonzero(class_mask[row:, :])
                    if (bottom_half_count / total_pixels) <= test_size:
                        continue
                    else:
                        break
                class_mask[:row, :] = False
                test_indices.extend(np.column_stack(np.where(class_mask)))
            elif train_side == 'bottom':
                for row in range(1, H):
                    top_half_count = np.count_nonzero(class_mask[:row, :])
                    if (top_half_count / total_pixels) <= test_size:
                        continue
                    else:
                        break
                class_mask[row:, :] = False
                test_indices.extend(np.column_stack(np.where(class_mask)))
            else:
                raise ValueError("Unsupported train side: choose 'right', 'left', 'top' or 'bottom'")

        test_indices = np.array(test_indices)
        test_mask[test_indices[:, 0], test_indices[:, 1]] = True
        train_mask = ~test_mask
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    train_gt[~train_mask] = absence_of_crop_label
    test_gt[train_mask] = absence_of_crop_label

    return train_gt, test_gt