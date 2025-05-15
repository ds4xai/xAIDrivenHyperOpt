import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

###########################" Utils ###########################"
def load_metrics(metrics_file):
    """Load metrics from a JSON file."""
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} does not exist.")
        return None
    with open(metrics_file, "r") as f:
        return json.load(f)

def validate_data_lengths(patch_sizes, *data_lists):
    """Ensure all data lists match the length of patch_sizes."""
    for data_list in data_lists:
        if len(data_list) != len(patch_sizes):
            raise ValueError("Mismatch in number of patch sizes and data points.")
        
        
        
def plot_and_save_results(model_name, patch_sizes, xai_method, eval_metric="js_div", dataset_name="prisma"):
    """
    Plot and save results showing the correlation between F1 scores and evaluation metrics across patch sizes.
    """
    l_eval_metrics, l_f1_dev, l_f1_test = [], [], []

    for ps in patch_sizes:
        metrics_file = f"./runs/xai/{dataset_name}/{model_name}/{xai_method}/patch_size_{ps}/xai_eval_metrics.json"
        metrics_data = load_metrics(metrics_file)
        if metrics_data is None:
            continue

        # Extract and process metrics
        if eval_metric == "entropy":
            l_eval_metrics.append(metrics_data["entropy"])
        else:
            l_eval_metrics.append(1 - metrics_data[eval_metric])  # Invert eval metric for plotting
        l_f1_dev.append(metrics_data["f1_validation"])
        l_f1_test.append(metrics_data["f1_test"])

    # Validate data lengths
    validate_data_lengths(patch_sizes, l_eval_metrics, l_f1_dev, l_f1_test)

    # Convert lists to numpy arrays
    l_eval_metrics, l_f1_dev, l_f1_test = map(np.array, (l_eval_metrics, l_f1_dev, l_f1_test))


    max_f1_test = patch_sizes[np.argmax(l_f1_test)]

    print(max_f1_test)

    # Create a single plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot F1 scores
    ax1.set_xlabel('Patch Size', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Macro F1', fontsize=22, fontweight='bold', color='tab:blue')
    ax1.plot(patch_sizes, l_f1_dev, label='Macro F1 (Val Set)', color='tab:blue', marker='x', markersize=10, linestyle='-')
    ax1.plot(patch_sizes, l_f1_test, label='Macro F1 (Test Set)', color='tab:cyan', marker='v', markersize=10, linestyle='dashed')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=20)
    ax1.grid(visible=True, linestyle='--', alpha=0.6)
    ax1.set_xticks(patch_sizes)
    ax1.set_xticklabels([str(ps) for ps in patch_sizes], fontsize=20, fontweight='bold')

    for ps, f1_dev, f1_test in zip(patch_sizes, l_f1_dev, l_f1_test):
        ax1.text(ps, f1_dev, f"{f1_dev:.2f}", color='tab:blue', fontsize=18, fontweight='bold', ha='center', va='bottom')
        ax1.text(ps, f1_test, f"{f1_test:.2f}", color='tab:cyan', fontsize=18, fontweight='bold', ha='center', va='bottom')

    # Plot evaluation metric
    ax2 = ax1.twinx()
    metric_label = {
        "js_div": "$Consistency^{JSdiv}$"
    }.get(eval_metric, "Evaluation Metric")
    ax2.set_ylabel(metric_label, fontsize=20, fontweight='bold', color='tab:red')
    ax2.plot(patch_sizes, l_eval_metrics, label=metric_label, color='tab:red', marker='s', linestyle='-', markersize=10)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=20)

    for ps, eval_metric_value in zip(patch_sizes, l_eval_metrics):
        ax2.text(ps, eval_metric_value, f"{eval_metric_value:.2f}", color='tab:red', fontsize=18, fontweight='bold', ha='center', va='bottom')

    # Highlight optimal patch sizes
    ax1.axvline(x=max_f1_test, color='green', linestyle='--', label=f'Max F1 (test)')
    
    
    h1, l1 = ax1.get_legend_handles_labels() 
    h2, l2 = ax2.get_legend_handles_labels()
   
    ax1.legend(h1+h2, l1+l2, loc= 'best',  fontsize=16)


    plt.title(f"{model_name.upper()}",fontsize=22, fontweight='bold')
    fig.tight_layout()

    # Save the plot
    save_path = os.path.join("images", f"{dataset_name}_{xai_method}_{model_name}_correlation_f1_vs_{eval_metric}.svg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path,dpi=300)
    plt.close(fig)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    
    dataset_name = "prisma" 
    patch_sizes = list(range(5, 22, 2))
      
    for dataset_name in ["prisma"]:
        for model_name in ["cnn2d", "vit"]:
            for xai_method in ["ig"]:
                for eval_metric in ["js_div"]:
                    # Call the function to plot and save results
                    plot_and_save_results(model_name, patch_sizes, xai_method, eval_metric, dataset_name)
