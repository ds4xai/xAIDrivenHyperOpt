import os
import torch
import scipy.io as sio
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from part_1_modeling.models.models import get_model, trainer, tester
from part_1_modeling.data_processing.custom_dataset import CustomDataset
from part_1_modeling.utils.utils import compute_class_weights, get_device, set_seed


def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


################################## utils ##################################
def load_data(data_path):
    """
    Load data from a .mat file.

    Args:
        data_path (str): Path to the .mat file.

    Returns:
        tuple: hsi, train_gt, dev_gt, test_gt, stats
    """
    data = sio.loadmat(data_path)
    prisma_data = data["prisma"]
    hsi = prisma_data["hsi"][0, 0]
    train_gt = prisma_data["train_gt"][0, 0]
    dev_gt = prisma_data["dev_gt"][0, 0]
    test_gt = prisma_data["test_gt"][0, 0]
    stats = {
        "means": prisma_data["means"][0, 0],
        "stds": prisma_data["stds"][0, 0]
    }
    return hsi, train_gt, dev_gt, test_gt, stats


def get_dataloader(world_size, rank, config=None, use_ddp=False, use_class_weights=False, seed=42):
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
    hsi, train_gt, dev_gt, test_gt, stats = load_data(config["data_path"])
    if use_class_weights:
        config["class_weights"] = compute_class_weights(train_gt, normalize=True, ignored_classes=[-1]).to(config["device"])

    # Build datasets
    train_dataset = CustomDataset(hsi, stats, train_gt, patch_size=config["patch_size"], split="train", standardize=True, absence_of_crop_label=-1, seed=seed)
    dev_dataset = CustomDataset(hsi, stats, dev_gt, patch_size=config["patch_size"], split="dev", standardize=True, absence_of_crop_label=-1, seed=seed)
    test_dataset = CustomDataset(hsi, stats, test_gt, patch_size=config["patch_size"], split="test", standardize=True, absence_of_crop_label=-1, seed=seed)

    # Batch sizes
    train_batch_size = config["batch_size"]
    dev_batch_size = min(config["batch_size"] * 3, 256)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, pin_memory=True, sampler=train_sampler, num_workers=2, prefetch_factor=2)
        dev_dataloader = DataLoader(dev_dataset, batch_size=dev_batch_size, pin_memory=True, sampler=dev_sampler, num_workers=2, prefetch_factor=2)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2)
        dev_dataloader = DataLoader(dev_dataset, batch_size=dev_batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)

    test_dataloader = DataLoader(test_dataset, batch_size=dev_batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)

    return train_dataloader, dev_dataloader, test_dataloader, config


def training(rank, config, world_size, save_path=None, use_ddp=True, gpu_id=0, seed=42):
    """
    Main training function.

    Args:
        rank (int): Rank of the current process.
        config (dict): Configuration dictionary.
        world_size (int): Number of processes in DDP.
        save_path (str): Path to save the model.
        use_ddp (bool): Whether to use DDP.
        gpu_id (int): GPU ID to use.
        seed (int): Random seed.
    """
    set_seed(seed)

    if use_ddp:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    print("Training started...")
    config["device"] = rank if use_ddp else get_device(gpu_id)
    device = config["device"]

    # Get dataloaders
    train_dataloader, dev_dataloader, test_dataloader, config = get_dataloader(world_size, rank, config=config, use_ddp=use_ddp, seed=seed)
    config["n_batches"] = len(train_dataloader)

    # Initialize the model
    model, criterion, optimizer, scheduler, config = get_model(config["model_name"], ignored_classes=[-1], **config)
    model = model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    print(f"Patch size: {config['patch_size']}")

    if not save_path:
        save_path = os.path.join("runs", "modeling", config['dataset_name'], config['model_name'], f"patch_size_{str(config['patch_size'])}")

    # Training loop
    best_chkpt_file = trainer(
        model,
        optimizer,
        criterion,
        train_dataloader,
        config,
        save_path,
        scheduler,
        dev_dataloader,
        eval_step=config['eval_step'],
        pin_memory=True,
        optim_metric=config["optim_metric"],
        show_cls_report=False,
        display_iter=len(train_dataloader) // 2,
    )
    if use_ddp:
        cleanup()

    print("Training finished.")    

if __name__ == "__main__":
    # Define hyperparameters
    seed = 42
    gpu_id = 0
    use_ddp = False
    patch_sizes = list(range(5, 22, 2))

    hyperparams = {
        "prisma": {
            "cnn2d": {
                'dataset_name': None,
                'data_path': './data/processed_data/prisma.mat',
                'patch_size': None,
                'model_name': None,
                'n_epochs': 30,
                'eval_step': 1,
                'n_bands': 174,
                'n_classes': 6,
                'n_comps': 15,
                'batch_size': 256,
                'optim_metric': "f1_macro",
            },
            "vit": {
                'dataset_name': None,
                'data_path': './data/processed_data/prisma.mat',
                'patch_size': None,
                'model_name': None,
                'n_epochs': 30,
                'eval_step': 1,
                'n_bands': 174,
                'n_classes': 6,
                'n_comps': 15,
                'batch_size': 8,
                'optim_metric': "f1_macro",
            }
        },
    }

    # Launch training
    
    # Set the random seed for reproducibility
    set_seed(seed)
    for dataset_name, models in hyperparams.items():
        for model_name, model_config in models.items():
            for patch_size in patch_sizes:
                model_config["dataset_name"] = dataset_name
                model_config["patch_size"] = patch_size
                model_config["model_name"] = model_name

                save_path = os.path.join("runs", "modeling", model_config['dataset_name'], model_config['model_name'], f"patch_size_{str(model_config['patch_size'])}")
                model_config["save_path"] = save_path

                if os.path.exists(save_path):
                    continue

                print(f"{'#' * 100}\nDataset: {dataset_name} | Model: {model_name} | Patch Size: {patch_size}\n{'#' * 100}")
                if use_ddp:
                    rank = int(os.getenv("RANK", 0))
                    world_size = int(os.getenv("WORLD_SIZE", 4))
                    mp.spawn(training, args=(model_config, world_size, save_path, use_ddp, gpu_id, seed), nprocs=world_size)
                else:
                    training(gpu_id, model_config, 1, save_path, use_ddp, gpu_id, seed=seed)
