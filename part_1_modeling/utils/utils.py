########################################################################################
##########                           Importations                             ##########
########################################################################################
import os
import torch
import random
import rasterio
import spectral
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight




########################################################################################
##########                            variables                               ##########
########################################################################################
# crop map to indices
ids_to_crops =  {
    0: "background",
    1: "durum wheat",
    2: "oranges",
    3: "permanent grassland",
    4: "rice",
    5: "sunflower",
    6: "olives",
}


# crop map to color
palette = {
    0: (255, 255, 255), 
    1: (230, 0, 0),      
    2: (0, 150, 0),       
    3: (0, 0, 200),      
    4: (255, 200, 0),     
    5: (0, 200, 200),    
    6: (200, 0, 200),      
}





########################################################################################
##########                            Fonctions                               ##########
########################################################################################

#### f1: for reproductible results
def set_seed(seed: int):
    """
    Sets the seed for reproducibility across various random number generators 
    and libraries commonly used in machine learning.

    Args:
        seed (int): The seed value to initialize the random number generators.

    This function affects:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generator (both CPU and GPU)
    - CuDNN's deterministic behavior in PyTorch
    """
    # Set the seed for Python's random module
    random.seed(seed)
    
    # Set the seed for NumPy's random number generator
    np.random.seed(seed)
    
    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # Set the seed for PyTorch on GPU (if applicable)
    torch.cuda.manual_seed(seed)
    
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#### f2: 
def rgb_to_hex(rgb):
    """Convert RGB color to hexadecimal format."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

#### f3: 
def convert_2d_to_color(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for id, color in palette.items():
        mask_id = arr_2d == id
        arr_3d[mask_id] = color

    return arr_3d

#### f4: choice of material on which the calculation will be made
def get_device(gpu_id):
    """
    Get the appropriate device for computation.

    This function checks if a CUDA GPU device is available. If it is, the function
    returns a CUDA device. If not, the function returns a CPU device.

    Returns:
    torch.device: The device for computation. It can be either 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        print(f"Computation on CUDA GPU device - N°{gpu_id}\n")
        device = torch.device(f'cuda:{gpu_id}')
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\ \n")
        device = torch.device('cpu')
    return device

#### f5: file loader
def open_file(dataset_path):
    """
    Open and load a dataset from a file.

    This function supports opening Matlab (.mat) files, TIFF (.tif, .tiff) files,
    and ENVI (.hdr) files. The function returns the dataset as a numpy array.

    Parameters:
    dataset_path (str): The path to the dataset file.

    Returns:
    numpy.ndarray: The dataset loaded from the file.

    Raises:
    ValueError: If the file format is not supported.

    """
    _, ext = os.path.splitext(dataset_path)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset_path)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        with rasterio.open(dataset_path) as src:
            return src.read()  
    elif ext == '.hdr':
        img = spectral.open_image(dataset_path)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

#### f6: plot ground truth
def plot_gt(gt, ids_to_crops, palette, title=None, save_path=None):
    """
    Plot a ground truth (GT) map with corresponding crop labels and a color legend.

    Args:
        gt (np.ndarray): 2D ground truth array where each pixel's value corresponds to a class label.
        ids_to_crops (dict): Mapping of class IDs (keys) to crop names (values).
        palette (dict): Mapping of class IDs (keys) to RGB colors (values) for visualization.
                        Example: {1: (255, 0, 0), 2: (0, 255, 0), ...}.
        title (str, optional): Title for the plot. Defaults to "Ground truth".
        save_path (str, optional): If provided, the plot is saved to this path as an image. 
                                   Otherwise, the plot is displayed.

    Returns:
        None
    """
    unique_classes = np.unique(gt)
    if len(unique_classes) > len(list(ids_to_crops.keys())):
        raise("Warning: There are more unique classes than crop labels in the palette.")
    # Set default title if none is provided
    title = title if title else "Ground truth"

    # Convert the ground truth labels to a color image using the palette
    img = convert_2d_to_color(gt, palette)

    # Create legend elements (colored patches corresponding to classes)
    legend_elements = [
        Patch(facecolor=np.array(color)/255, label=f'{ids_to_crops[i]}') 
        for i, color in palette.items() if i in unique_classes
    ]

    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(title.capitalize(), fontsize=20, fontweight='bold')
    plt.axis('off')  # Turn off axis lines and labels

    # Add a legend to the right of the plot
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.1, 1))

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

#### f7: compute metrics
def compute_metrics(true_labels, predicted_labels, num_classes=None):
    """
    Compute and return various classification metrics, including accuracy, 
    confusion matrix, F1 scores, and kappa coefficient.

    Args:
        true_labels (list or np.ndarray): Ground truth labels.
        predicted_labels (list or np.ndarray): Predicted labels.
        num_classes (int, optional): Total number of classes. Defaults to max(true_labels) + 1.

    Returns:
        dict: A dictionary containing the following metrics:
            - "confusion_matrix": Confusion matrix of shape (num_classes, num_classes).
            - "overall_accuracy": Fraction of correctly classified samples.
            - "average_accuracy": Mean accuracy across all classes.
            - "accuracy_percentage": Global accuracy as a percentage.
            - "f1_scores": F1 score for each class.
            - "f1_macro": Macro-average F1 score across all classes.
            - "kappa": Cohen's kappa coefficient.
    """
    # Initialize results dictionary
    metrics = {}

    # Convert inputs to numpy arrays for consistency
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Determine the number of classes if not provided
    num_classes = np.max(true_labels) + 1 if num_classes is None else num_classes

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))
    metrics["confusion_matrix"] = conf_matrix

    # Overall Accuracy (OA): Fraction of correctly classified samples
    total_correct = np.trace(conf_matrix)  # Sum of diagonal elements (true positives)
    total_samples = np.sum(conf_matrix)  # Total number of samples
    metrics["overall_accuracy"] = total_correct / total_samples

    # Average Accuracy (AA): Mean accuracy per class
    class_totals = np.sum(conf_matrix, axis=1)  # Total samples per class
    class_accuracies = np.diag(conf_matrix) / np.maximum(class_totals, 1)  # Avoid division by zero
    metrics["average_accuracy"] = np.mean(class_accuracies)

    # Global Accuracy as a percentage
    metrics["accuracy_percentage"] = (total_correct / total_samples) * 100

    # F1 Scores: Compute per-class F1 scores
    f1_scores = []
    for i in range(num_classes):
        precision = conf_matrix[i, i] / np.maximum(np.sum(conf_matrix[:, i]), 1)  # Avoid division by zero
        recall = conf_matrix[i, i] / np.maximum(np.sum(conf_matrix[i, :]), 1)  # Avoid division by zero
        f1 = 2 * (precision * recall) / np.maximum((precision + recall), 1e-10)  # Avoid division by zero
        f1_scores.append(f1)
    metrics["f1_scores"] = np.array(f1_scores)
    metrics["f1_macro"] = np.mean(f1_scores)

    # Kappa Coefficient: Measure of inter-rater agreement
    observed_agreement = np.trace(conf_matrix) / total_samples
    expected_agreement = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_samples ** 2)
    metrics["kappa"] = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return metrics


#### f8: compute weights
def compute_class_weights(gt, normalize=False, ignored_classes=[]):
    """
    Compute inverse frequency weights for each class based on the ground truth labels.
    This is useful for balancing class contributions during training.

    Args:
        gt (np.ndarray): Ground truth labels (2D or 1D array).
        normalize (bool, optional): If True, the computed weights are normalized to sum to 1. 
                                    Defaults to False.
        n_classes (int, optional): Total number of classes. Not used explicitly here, 
                                   but can help when further extensions are needed. Defaults to None.
        ignored_classes (list, optional): List of class labels to ignore when computing weights.
                                          These labels will be excluded from weight calculations.
                                          Defaults to an empty list.

    Returns:
        torch.Tensor: A tensor of class weights of shape `(n_classes,)`.
    """
    # Exclude ignored classes from the ground truth labels
    labels = gt[~np.isin(gt, ignored_classes)]

    # Compute class weights using scikit-learn's compute_class_weight
    weights = compute_class_weight(
        class_weight='balanced',  # Balances weights inversely proportional to class frequencies
        classes=np.unique(labels),  # Unique class labels
        y=labels  # Flattened ground truth labels
    )
    
    # Normalize weights if requested
    if normalize:
        weights /= np.sum(weights)
    
    # Convert weights to a PyTorch tensor for compatibility with PyTorch models
    return torch.from_numpy(weights).float()


### f9: serialize object to JSON
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj
