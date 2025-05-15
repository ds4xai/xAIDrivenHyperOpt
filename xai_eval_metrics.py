import os
import json
import numpy as np
from part_2_xai.xai_utils import compute_js_divergence

xai_methods = ["ig"]
datasets = ["prisma"]
models = ["cnn2d", "vit"]
num_bootstrap_samples = 250
num_classes = 6
patch_sizes = list(range(5, 22, 2))

def generer_doublets_disjoints(liste):
    doublets = []
    # On parcourt la liste deux par deux
    for i in range(0, len(liste) - 1, 2):
        doublets.append((liste[i], liste[i+1]))
    return doublets



for xai_method in xai_methods:
    for dataset_name in datasets:
        for model_name in models:
            print(f"Processing {dataset_name} with {model_name} using {xai_method}...")

            results = {}
            for patch_size in patch_sizes:
            
                # Load F1 test score
                test_log_path = f"./runs/modeling/{dataset_name}/{model_name}/patch_size_{patch_size}/test_log/logs.json"
                with open(test_log_path, "r") as f:
                    model_name_ = "visiontransformer" if model_name == "vit" else model_name
                    f1_test_score = json.load(f)[model_name_]["metrics"]["f1_macro"]

                # Load F1 validation score
                training_log_path = f"./runs/modeling/{dataset_name}/{model_name}/patch_size_{patch_size}/training_log/logs.json"
                with open(training_log_path, "r") as f:
                    f1_scores = json.load(f)[model_name]["f1_macro"]["val"]
                    f1_validation_score = max(f1_scores)

                # Load attributions data
                xai_data_path = f"./runs/xai/{dataset_name}/{model_name}/{xai_method}/patch_size_{patch_size}/data.json"
                with open(xai_data_path, "r") as f:
                    xai_data = json.load(f)
             
                    
                # Compute js_div for each class
                js_div_per_class = []
                for cls, attrs_values in xai_data["attrs"].items():
                    attrs = np.array(attrs_values)
                    # Skip if no attributions found
                    if attrs.shape[0]==0:
                        continue
                    
                    # Compute js_div for each bootstrap sample
                    bootstrap_indices = np.random.choice(attrs.shape[0], size=(num_bootstrap_samples, attrs.shape[0]), replace=True)
                    js_div= []
                    for bootstrap in bootstrap_indices:
                        attrs_bootstrap = attrs[bootstrap]
                        
                        # js div
                        js_div.append(np.mean(np.array([
                            compute_js_divergence(attrs[i], is_normalized=False)
                            for i in range(attrs_bootstrap.shape[0])
                        ])))
                        
            
                    js_div_per_class.append(np.mean(js_div))
                
                # Aggregate js_div 
                mean_js_div = np.mean(js_div_per_class)
                
                
                # Store results
                results = {
                    "js_div": mean_js_div,
                    "f1_validation": f1_validation_score,
                    "f1_test": f1_test_score,
                }
            
              
                # Save results to JSON file
                save_path = f"./runs/xai/{dataset_name}/{model_name}/{xai_method}/patch_size_{patch_size}/xai_eval_metrics.json"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Results saved to {save_path}")
        
            

