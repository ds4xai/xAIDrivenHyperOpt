import os
import json
from part_1_modeling.data_processing.custom_dataset import CustomDataset
import torch
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from trainer import get_dataloader
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from part_1_modeling.models.models import get_model
from part_1_modeling.utils.utils import get_device, set_seed
from part_2_xai.xai_utils import integrated_gradient

################################## utils ##################################
def load_data(data_path, xai_gt_path):
    
    data = sio.loadmat(data_path)
    prisma_data = data["prisma"]
    hsi = prisma_data["hsi"][0, 0]
    xai_gt = np.load(xai_gt_path)
    stats = {
        "means": prisma_data["means"][0, 0],
        "stds": prisma_data["stds"][0, 0]
    }
    return hsi, xai_gt, stats


def get_dataloader(config=None, seed=42):
    """
    Create dataloaders for training, validation, and testing.

    Args:
        world_size (int): Number of processes in DDP.
        rank (int): Rank of the current process.
        config (dict): Configuration dictionary.
        use_ddp (bool): Whether to use DDP.
        seed (int): Random seed.

    Returns:
        tuple: train_dataloader, dev_dataloader, test_dataloader, config
    """
    hsi, xai_gt, stats = load_data(config["data_path"], config["xai_gt_path"])
 
    # Build datasets
    xai_dataset = CustomDataset(hsi, stats, xai_gt, patch_size=config["patch_size"], split="test", standardize=True, absence_of_crop_label=-1, seed=seed)

    # Batch sizes
    batch_size = config["batch_size"]
 
    xai_dataloader = DataLoader(xai_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)

    return xai_dataloader



def compute_and_update_spectral_attrs(attr_fn, logs, model, inputs, targets, predictions, baseline=None, device=None):
    # Select only correctly predicted samples
    correct_predictions_mask = predictions == targets
    
    if correct_predictions_mask.sum() == 0:  # No correctly predicted samples
        return logs
    
    correct_inputs, correct_targets = inputs[correct_predictions_mask], targets[correct_predictions_mask]
    unique_classes = np.unique(correct_targets)
    
    for cls in unique_classes:
        class_mask = correct_targets == cls
        class_inputs = correct_inputs[class_mask]
        if attr_fn.__name__.lower() == "integrated_gradient":
            attrs = attr_fn(class_inputs, int(cls), model, baseline=baseline, device=device)
        else:
            raise ValueError(f"Unknown attribution function: {attr_fn.__name__}")
        
        # Retain only positive contributions
        attrs = np.maximum(attrs, 0)
        # Average over spatial dimensions for spectral contributions
        spectral_attrs = attrs.mean(axis=(2, 3))
        logs["attrs"][cls].extend(spectral_attrs.tolist())
    
    return logs


def attributions(attr_fn, hyperparams: dict = {}, save_dir: str = None, gpu_id: int = 0, seed: int = 42):
    device = get_device(gpu_id=gpu_id)
    dataset_name = hyperparams["dataset_name"]
    model_name = hyperparams["model_name"]
    patch_size = hyperparams["patch_size"]
    
    print(f"{'#' * 100}\nDataset: {dataset_name} | xAI Method: {attr_fn.__name__} | Model: {model_name} | Patch Size: {patch_size}\n{'#' * 100}")
    
    # Get dataloaders
    xai_loader = get_dataloader(config=hyperparams, seed=seed)
    
    hyperparams["n_batches"] = len(xai_loader)

    # Initialize the model
    model, _, _, _, hyperparams = get_model(hyperparams["model_name"], ignored_classes=[-1], **hyperparams)
    model = model.to(device)
    
    # Load model checkpoint
    ckpt_dir = os.path.join("runs/modeling", dataset_name, model_name, f'patch_size_{patch_size}', 'checkpoint')
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpt_files:
        print(f"No checkpoint found in {ckpt_dir}")
        return
    best_ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
    
    # Load weights
    state_dict = torch.load(best_ckpt_path, map_location=device, weights_only=False)['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove DDP prefix
    model.load_state_dict(state_dict)
    
    logs = {
        "attrs": {cls: [] for cls in range(hyperparams["n_classes"])},
        "f1_scores": None,
    }
    
    all_predictions, all_targets = [], []
    progress_bar = tqdm(xai_loader, desc="Computing attributions", colour="magenta", leave=True, total=len(xai_loader))
    
    with torch.no_grad():
        for batch in xai_loader:
            inputs, targets = batch[0].to(device), batch[1].cpu().numpy()
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            logs = compute_and_update_spectral_attrs(attr_fn, logs, model, inputs, targets, predictions, baseline=None, device=device)
            
            progress_bar.update(1)
    progress_bar.close()
    
    # Compute F1 scores
    logs["f1_scores"] = f1_score(all_targets, all_predictions, average=None).tolist()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "data.json")
    try:
        with open(save_path, "w") as f:
            json.dump(logs, f)
    except Exception as e:
        print(f"Error saving attributions: {e}")
    
    print(f"Attributions saved at {save_path}")


if __name__ == "__main__":
    gpu_id = 0  
    seed = 42
    patch_sizes = list(range(5, 22, 2))
    
    
    # Model hyperparameters
    hyperparams = {
        "prisma": {
            "cnn2d": {
                'data_path': './data/processed_data/prisma.mat',
                'xai_gt_path': './data/xai_gt.npy',
                'dataset_name': None,
                'patch_size': None, 
                'model_name': None,
                'n_bands': 174,
                'n_classes': 6,
                'n_comps': 15,
                'batch_size': 4,
                'ratio': 0.1,
                'optim_metric': "f1_macro",
            },
            "vit": {
                'data_path': './data/processed_data/prisma.mat',
                'xai_gt_path': './data/xai_gt.npy',
                'dataset_name': None,
                'patch_size': None,
                'model_name': None,
                'n_bands': 174,
                'n_classes': 6,
                'n_comps': 15,
                'batch_size': 4,
                'ratio': 0.1,
                'optim_metric': "f1_macro",    
            }
        },
    }
    
    xai_methods = {
        "ig": integrated_gradient,
    }
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Compute attributions for each dataset, model, XAI method, and patch size
    for xai_method, attr_fn in xai_methods.items():
        for dataset_name, dataset_hyps in hyperparams.items():
            for model_name, model_hyps in dataset_hyps.items():
                for patch_size in patch_sizes:
                    model_hyps.update({
                        "dataset_name": dataset_name,
                        "patch_size": patch_size,
                        "model_name": model_name,
                        "attrs_fn": xai_method,
                    })
                    
                    save_dir = os.path.join("runs", "xai", dataset_name, model_name, xai_method, f"patch_size_{patch_size}")
                    if os.path.exists(save_dir):
                        continue
                    
                    attributions(attr_fn, model_hyps, save_dir, gpu_id=gpu_id, seed=seed)

