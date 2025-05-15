########################################################################################
##########                           Importations                             ##########
########################################################################################
### python community modules
import os
import json
import time
import torch
import joblib
import shutil
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.nn import init
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report

from part_1_modeling.utils.utils import compute_metrics




########################################################################################
##########                            Modèles                               ##########
########################################################################################

## model 1 : 2D classic conv net
class CNN2D(nn.Module):
    """
    CNN2D: A 2D Convolutional Neural Network for Hyperspectral Image Classification.

    This model processes hyperspectral image patches using 2D convolutions. It extracts 
    spatial features through multiple convolutional layers and dense layers. 
    The network includes dropout for regularization to prevent overfitting.

    Args:
        n_bands (int): Number of spectral bands in the hyperspectral image (input channels).
        n_classes (int): Number of output classes.
        patch_size (int, optional): Size of the input patch (assumed to be square). Default is 7.
    """

    def __init__(self, n_bands: int = 234, patch_size: int = 7, n_comps: int = 15, n_classes: int = 6):
        super(CNN2D, self).__init__()
        
        # Hyperparameters
        self.input_channels = n_bands
        self.reduced_channels = n_comps if n_comps else n_bands
        self.apply_reduction = n_comps > 0
        self.num_classes = n_classes
        self.patch_size = patch_size
        
        # Dimensionality reduction block (if not done upstream)
        if self.apply_reduction:
            self.reduction_layer = nn.Sequential(
                nn.Conv2d(self.input_channels, self.reduced_channels, kernel_size=1),
                nn.Tanh()
            )

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.reduced_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(), 

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        # Apply dimensionality reduction if required
        if self.apply_reduction:
            x = self.reduction_layer(x)
            
        # Extract features
        x = self.feature_extractor(x)
        
        # Apply global average pooling
        x = self.gap(x).flatten(1)
        
        # Classify features
        x = self.classifier(x) 
        return x


## model 2 : Transformer
class AttentionBlock(nn.Module):
    """
    AttentionBlock: Implements a single transformer block with multi-head attention and feed-forward layers.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        hidden_dim (int): Dimensionality of the feed-forward layer.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        # Multi-Head Attention Layer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Apply multi-head attention with residual connection
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # Apply feed-forward layer with residual connection
        x = x + self.feed_forward(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    VisionTransformer: Implements a Vision Transformer (ViT) for image classification.

    Args:
        n_bands (int): Number of spectral bands in the input image.
        n_comps (int): Number of components after dimensionality reduction.
        embed_dim (int): Dimensionality of the embeddings.
        hidden_dim (int): Dimensionality of the feed-forward layer.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        patch_size (int): Size of the image patches.
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(
        self,
        n_bands=234,
        n_comps=15,
        embed_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        patch_size=7,
        num_classes=6,
        dropout=0.2,
    ):
        super().__init__()

        # Hyperparameters
        self.input_channels = n_bands
        self.apply_reduction = n_comps > 0
        self.reduced_channels = n_comps if n_comps else n_bands
        self.patch_size = patch_size
        self.vit_patch_size = 3
        self.num_patches = ((patch_size + (self.vit_patch_size - (patch_size % self.vit_patch_size)) % self.vit_patch_size) // self.vit_patch_size) ** 2
        self.input_dim = self.reduced_channels * self.vit_patch_size ** 2
        
        # Dimensionality reduction block (if not done upstream)
        if self.apply_reduction:
            self.reduction_layer = nn.Sequential(
                nn.Conv2d(self.input_channels, self.reduced_channels, kernel_size=1),
                nn.Tanh()
            )

        # Input embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim),
            nn.GELU()
        )
        
        # Transformer layers
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim))

    def forward(self, x):
        # Apply dimensionality reduction if required
        if self.apply_reduction:
            x = self.reduction_layer(x)  # Shape: [B, reduced_channels, H, W]
            
        # Convert image to patches
        x = img_to_patch(x, self.vit_patch_size)  # Shape: [B, num_patches, input_dim]
        B, T, _ = x.shape
        
        # Embed patches
        x = self.embedding_layer(x)  # Shape: [B, num_patches, embed_dim]

        # Add CLS token and positional embeddings
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)  # Shape: [B, 1 + num_patches, embed_dim]
        x = x + self.positional_embedding[:, :T + 1]

        # Apply transformer layers
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Shape: [1 + num_patches, B, embed_dim]
        x = self.transformer(x)

        # Classification using CLS token
        cls_output = x[0]  # Shape: [B, embed_dim]
        out = self.classification_head(cls_output)
        return out

# f3: Patch extraction
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Converts an image tensor into patches.

    Args:
        x (Tensor): Input image tensor of shape [B, C, H, W].
        patch_size (int): Size of each patch.
        flatten_channels (bool): Whether to flatten the patch channels.

    Returns:
        Tensor: Patches of shape [B, num_patches, patch_dim].
    """
    B, C, H, W = x.shape

    # Padding to ensure that image dimensions are divisible by patch size
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size

    # Pad image dimensions 
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0) 
    H_padded, W_padded = x.shape[2], x.shape[3]

    # Extract patches
    x = x.reshape(B, C, H_padded // patch_size, patch_size, W_padded // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)              # [B, H'*W', C, p_H, p_W]
    
    # Flatten patches
    if flatten_channels:
        x = x.flatten(2, 4)          # [B, H'*W', C*p_H*p_W]
    
    return x



########################################################################################
##########                            fonctions                               ##########
########################################################################################
#### f1: Weights initializer
def init_weights(model):
    """
    Initialize the weights of a neural network model using Kaiming normal initialization for convolutional layers,
    Xavier normal initialization for linear layers, and constant initialization for batch normalization layers.

    Parameters:
    model (nn.Module): The neural network module to initialize.

    Returns:
    None. The weights of the module are modified in-place.
    """
    if isinstance(model, nn.Conv3d):
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            init.constant_(model.bias, 0)
    elif isinstance(model, nn.Conv2d):
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            init.constant_(model.bias, 0)
    elif isinstance(model, nn.Conv1d):
        init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm3d):
        init.constant_(model.weight, 1)
        init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        init.constant_(model.weight, 1)
        init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm1d):
        init.constant_(model.weight, 1)
        init.constant_(model.bias, 0)
    elif isinstance(model, nn.Linear):
        init.xavier_normal_(model.weight)
        if model.bias is not None:
            init.constant_(model.bias, 0)

#### f2: get model
def get_model(name, n_classes=6, n_bands=174, ignored_classes=[-1], **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    #  hyperparameters 
    n_bands = kwargs.setdefault("n_bands", n_bands)
    n_classes = kwargs.setdefault("n_classes", n_classes)
    kwargs.setdefault("ignored_classes", ignored_classes)
    device = kwargs.setdefault("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    weights = torch.ones(n_classes)
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)  # Default: equal weight for all classes
    kwargs.setdefault("apply_weight_initialization", True)
    n_batches = kwargs.setdefault("n_batches", None)

    if name.lower() == "cnn2d":
        lr = kwargs.setdefault("learning_rate", 3e-4)
        patch_size = kwargs.setdefault("patch_size", 11)  
        n_comps = kwargs.setdefault("n_comps", 15) 
        n_epochs = kwargs.setdefault("n_epochs", 50)  
        weight_decay = kwargs.setdefault("weight_decay", 1e-6)
        model = CNN2D(n_bands, patch_size, n_comps, n_classes).to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights)
        scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr * 100, epochs=n_epochs, steps_per_epoch=n_batches)
        kwargs.setdefault("batch_size", 32) 

    elif name.lower() == "vit":
        lr = kwargs.setdefault("learning_rate", 3e-4) 
        n_epochs = kwargs.setdefault("n_epochs", 100)  
        weight_decay = kwargs.setdefault("weight_decay", 1e-6)  
        patch_size = kwargs.setdefault("patch_size", 7)  
        n_comps = kwargs.setdefault("n_comps", 15)  
        d_model = kwargs.setdefault("d_model", 128)  
        n_encoder_layers = kwargs.setdefault("n_encoder_layers", 6)  
        n_heads = kwargs.setdefault("n_heads", 4)  
        d_ff = kwargs.setdefault("d_ff", 256) 
        emb_dropout = kwargs.setdefault("emb_dropout", 0.1)  
        model = VisionTransformer(
            patch_size=patch_size,
            n_bands=n_bands,
            n_comps=n_comps,
            num_classes=n_classes,
            embed_dim=d_model,
            num_layers=n_encoder_layers,
            num_heads=n_heads,
            hidden_dim=d_ff,
            dropout=emb_dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights)
        scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr * 100, epochs=n_epochs, steps_per_epoch=n_batches)
        kwargs.setdefault("batch_size", 32) 

    else:
        print("This architecture is not implemented")
    
    # Initialize weights
    if kwargs.get("apply_weight_initialization"):
        model.apply(init_weights)
        print("Layers Initialized \n")
            
    # Number of model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kwargs.setdefault("total_params", total_params)
    print(f"Number of trainable parameters: {total_params:,}".replace(",", " "))
    
    return model, criterion, optimizer, scheduler, kwargs


#### f2: trainer fonctions
def trainer(
    net,
    optimizer,
    criterion,
    train_dataloader,
    hyps, 
    save_path,
    scheduler=None,
    dev_dataloader=None,
    eval_step=1,
    pin_memory=False,
    optim_metric="f1_macro",
    show_cls_report=False,
    display_iter=10,
    
):
    
    if criterion is None or optimizer is None: 
        raise Exception("Missing {}. You must specify a {}.".format("criterion" if criterion is None else "optimiser", "loss function" if criterion is None else "optimizer"))

    n_epochs = hyps["n_epochs"]
    device = hyps["device"]
    optim_metric = optim_metric.lower()
    
    # save directory
    os.makedirs(save_path, exist_ok=True)
    
    net.to(device)
    
    temp_optim_metric = -1
    train_losses = []
    l_train_optim_metric = []
    val_losses = []
    l_val_optim_metric = []
    net_name = hyps["model_name"].lower() if "model_name" in hyps else str(net.__class__.__name__).lower()
    best_model_path = None

    # Dstart training
    start_time = time.time()
    
    # Training
    for e in range(1, n_epochs + 1):
        
        # Evaluation
        if dev_dataloader is not None and (e%eval_step == 0 or e==1 or e==n_epochs):
            val_optim_metric, val_loss = validator(net, dev_dataloader, criterion, optim_metric=optim_metric, \
                device=device, show_cls_report=show_cls_report, pin_memory=pin_memory)
            l_val_optim_metric.append(val_optim_metric)
            val_losses.append(val_loss)
            print(f"Validation loss: {val_loss} ---- {optim_metric}: {val_optim_metric}")
            
            # save best model by metric of optimisation
            if val_optim_metric > temp_optim_metric: 
                best_model_path = save_model(
                    net,
                    save_path,
                    epoch=e,
                    optim_metric={
                        optim_metric: val_optim_metric
                        },
                    )
                temp_optim_metric = val_optim_metric
            print("\n")  

        # train mode
        net.train()
        running_loss = 0.0
        all_targets, all_outputs = [], []

        pbar = tqdm(train_dataloader, desc=f"Training - Epoch [{e}/{n_epochs}]", colour="green", leave=True, total=len(train_dataloader))
        for i, batch in enumerate(train_dataloader, start=1):
            if pin_memory:
                inputs, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()


            running_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_outputs.extend(outputs)
            all_targets.extend(targets.cpu().numpy())

            pbar.update(1) 

            # display log
            if i % display_iter == 0:
                pbar.set_postfix(Progress=f"{100.0 * i / len(train_dataloader):.0f}%", Loss=f"{running_loss / i:.6f}")
        
        avg_loss = running_loss / len(train_dataloader)
        metric = compute_metrics(all_targets, all_outputs)[optim_metric] 
        train_losses.append(avg_loss)
        l_train_optim_metric.append(metric)

        pbar.close()  
  
    # empty culculator
    torch.cuda.empty_cache()
    end_time = time.time()
    training_time = np.round(end_time - start_time, 2)

    # logs path
    training_log_path = os.path.join(save_path, "training_log", "logs.json")

    
    if os.path.exists(training_log_path):
        try:
            with open(training_log_path, 'r') as f:
                logs = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from {training_log_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found: {training_log_path}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while reading {training_log_path}: {e}")
    else:
        logs = {}
        
    hyps["device"] = str(hyps["device"])  # for serialisation
    if "weights" in hyps:
        hyps["weights"] = hyps["weights"].tolist()  # for serialisation

    # save logs
    if net_name not in logs: 
        logs[net_name] = {"losses": {"train": [], "val": []}, 
                          optim_metric: {"train": [], "val": []}, 
                          "hyperparameters": hyps,
                          "training_time": training_time
                         }
    
    logs[net_name]["losses"]["train"] = train_losses
    logs[net_name][optim_metric]["train"] = l_train_optim_metric  
    if dev_dataloader is not None:
        logs[net_name]["losses"]["val"] = val_losses
        logs[net_name][optim_metric]["val"] = l_val_optim_metric   

    # Save hyps 
    os.makedirs(os.path.dirname(training_log_path), exist_ok=True)
    with open(training_log_path, 'w') as f:
        json.dump(logs, f, indent=4)
    
    return best_model_path

def validator(
    net, 
    dev_dataloader, 
    criterion, 
    optim_metric="f1_macro",
    device="cpu", 
    show_cls_report=False, 
    pin_memory=False
    ):
    
    net.to(device)
    net.eval()
    all_targets, all_outputs = [], []
    running_loss = 0.0
    
    pbar = tqdm(dev_dataloader, desc=f"validation: ", colour="yellow", leave=True, total=len(dev_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader, start=1):
            if pin_memory == True:
                inputs, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)  
            running_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_outputs.extend(outputs)
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({"Loss": running_loss / i})
            pbar.update(1)
       

    # compute metrics
    metric = compute_metrics(all_targets, all_outputs)[optim_metric]
    avg_loss = running_loss / len(dev_dataloader) 

    # display report
    if show_cls_report:
        print(classification_report(all_targets, all_outputs, zero_division=1))

    return metric, avg_loss


def tester(net, 
           test_dataloader, 
           save_path, 
           device="cpu", 
           show_cls_report=False, 
           pin_memory=False):
    
    net.to(device)
    net.eval()
    
    all_targets, all_outputs = [], []
    test_log_dir = os.path.join(save_path, "test_log")

    pbar = tqdm(test_dataloader, desc=f"testing: ", colour="blue", leave=True, total=len(test_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader, start=1):
            if pin_memory == True:
                inputs, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)

            outputs = net(inputs)
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_outputs.extend(outputs)
            all_targets.extend(targets.cpu().numpy())
            
            pbar.update(1)
       

    # compute metrics
    results = compute_metrics(all_targets, all_outputs)
    
    if show_cls_report:
        print(classification_report(all_targets, all_outputs, zero_division=1))
        
    # logs file path
    net_name = str(net.__class__.__name__).lower()
    filename = os.path.join(test_log_dir, "logs.json")
    
    # load or initialize logs file
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            logs = json.load(f)
    else:
        logs = {}
        
    # for serialization
    results["confusion_matrix"] = results["confusion_matrix"].tolist()
    results["f1_scores"] = results["f1_scores"].tolist()
          
    # add results in logs
    if net_name not in logs:
        logs[net_name] = {"metrics": {}}
    logs[net_name]["metrics"] = results
    
    # Save log
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    with open(filename, 'w') as f:
        json.dump(logs, f, indent=4)
        
    return results



def save_model(net, save_path, **kwargs):    
    if 'epoch' not in kwargs or 'optim_metric' not in kwargs:
        raise ValueError("Missing required kwargs: 'epoch' or 'optim_metric' must be provided for saving a torch model.")
    
    epoch = kwargs.get('epoch')
    net_name = kwargs["model_name"].lower() if "model_name" in kwargs else str(net.__class__.__name__).lower()
    metric, value = list(kwargs["optim_metric"].items())[0]
    
    chkpt_dir = os.path.join(save_path, "checkpoint")

    # Create model directory if it doesn't exist
    os.makedirs(chkpt_dir, exist_ok=True)
    
    # Clear the contents of the model directory
    try:
        for filename in os.listdir(chkpt_dir):
            file_path = os.path.join(chkpt_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        print(f"Error clearing the directory {chkpt_dir}: {e}")
    

    # Save PyTorch model weights
    if isinstance(net, torch.nn.Module):
        chkpt_name = f"{net_name}_epoch_{epoch}_{metric}_{value:.3f}.pth"
        chkpt_path = os.path.join(chkpt_dir, chkpt_name)
        print(f"Saving neural network weights in {chkpt_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': kwargs.get('optimizer_state_dict'),
            metric: value,
        }, chkpt_path)
        return chkpt_path

    
    # Save non-PyTorch model (e.g., scikit-learn model)
    else:
        print(f"Saving model params in {chkpt_name}.pkl")
        chkpt_name = f"{net_name}_epoch_{epoch}_{metric}_{value : .3f}.pkl"
        chkpt_path = os.path.join(chkpt_dir, chkpt_name)
        joblib.dump(net, chkpt_path)
        return chkpt_path



