import os
import torch
import json
import numpy as np
import scipy.io as sio
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, DataLoader
from part_1_modeling.utils.utils import get_device, set_seed
from part_1_modeling.models.models import get_model, tester, trainer
from torch.nn.parallel import DistributedDataParallel as DDP
from part_1_modeling.data_processing.custom_dataset import CustomDataset



def cleanup():
    dist.destroy_process_group()

################################## utils ##################################
def load_data(data_path):

    # load data: hsi and gt
    data = sio.loadmat(data_path)
    hsi = data["prisma"]["hsi"][0,0]
    test_gt = data["prisma"]["test_gt"][0,0]
    stats = {
        "means": data["prisma"]["means"][0,0],
        "stds": data["prisma"]["stds"][0,0]
        }

    return hsi, test_gt, stats


def get_dataloader(hyps: dict=None, seed: int=42):
    
    hsi, test_gt, stats = load_data(hyps["data_path"])

    # build datasets
    test_dataset = CustomDataset(hsi, stats, test_gt, split="test", patch_size=hyps["patch_size"], standardize=True, absence_of_crop_label=-1, seed=seed) 
    test_dataloader = DataLoader(test_dataset, batch_size=hyps["batch_size"]*3, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)
    
    return test_dataloader




def testing(hyps: dict, save_path: str=None, gpu_id: int=0, seed: int=42):
    set_seed(seed)

    hyps["device"] = get_device(gpu_id)
    device = hyps["device"]
    hyps["weights"] = torch.Tensor(hyps["weights"])

    # get dataloaders
    test_dataloader = get_dataloader(hyps=hyps, seed=seed)

    # Initialize the model
    model, _, _, _, hyps = get_model(hyps["model_name"], **hyps)

    model = model.to(device)

    print(f"Patch size: {hyps['patch_size']}")

     #Load model and checkpoint
    ckpt_dir = os.path.join(save_path, 'checkpoint')
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpt_files:
        print(f"Aucun checkpoint trouvé dans {ckpt_dir}")
    ckpt_file = ckpt_files[0]
    best_ckpt_path = os.path.join(ckpt_dir, ckpt_file)


    # Load best model weights from checkpoint
    state_dict = torch.load(best_ckpt_path, map_location=device, weights_only=False)['model_state_dict'] # take juste weights
    # Supprimer le préfixe 'module.' pour les entrainement ddp
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # Load model weights
    print(hyps['model_name'], " : " ,model.load_state_dict(state_dict))

    ### testing
    tester(
        model,
        test_dataloader,
        save_path,
        device=device,
        show_cls_report=True,
        pin_memory=True,
    )




if __name__ == "__main__":

    gpu_id = 0
    seed = 42
    patch_sizes = list(range(5, 22, 2))
    
    
    # model hyps
    hyps = {
        "prisma": {
            "cnn2d": {
                'data_path': './data/processed_data/prisma.mat',
                'dataset_name': None,
                'patch_size': None,
                'model_name': None,
             

                },
            "vit": {
                'data_path': './data/processed_data/prisma.mat',
                'dataset_name': None,
                'patch_size': None,
                'model_name': None,
                }
            },
        }

    # start testing
    # Loop through each dataset and model
    for dataset_name in hyps.keys():
        hyps_data = hyps[dataset_name]
        for model_name in hyps_data.keys(): 
            hyps_model = hyps_data[model_name]
            for patch_size in patch_sizes:
                save_path = os.path.join("runs", "modeling", dataset_name, model_name, f"patch_size_{str(patch_size)}")
                filename = os.path.join(save_path, "training_log", "logs.json")
                
                # Load hyperparameters from the JSON file
                # Check if the file exists and is not empty
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    with open(filename, 'r') as f:
                        hyps_model = json.load(f)[model_name]["hyperparameters"]
                else:
                    print("########################### Could not open file. Training has not terminated; the training log file does not exist. ########################")
                    continue


                print(f"{'#'*100}\nDataset: {dataset_name} | model: {model_name} | patch_size: {patch_size} \n{'#'*100}")
                testing(hyps_model, save_path, gpu_id, seed=seed)
