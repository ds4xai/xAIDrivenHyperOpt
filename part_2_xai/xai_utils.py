import torch
import numpy as np
import torch.nn.functional as F
from captum.attr import IntegratedGradients, NoiseTunnel
from holisticai.explainability.metrics.global_feature_importance._importance_spread import spread_divergence


def integrated_gradient(input, target, model, baseline=None, device='cpu'):
    """
    Computes the integrated gradients for a given input, target, and model.
    """
    model.eval()
    model.to(device)
    input = input.to(device) if input.requires_grad else input.requires_grad_(True).to(device)
    if baseline is None:
        baseline = torch.zeros_like(input, device=device)  

    ig = IntegratedGradients(model)
    
    # Use NoiseTunnel to add noise and improve the stability of attributions.
    # NoiseTunnel applies noise to the input and averages the attributions over multiple noisy samples.
    # This reduces the variance of the attributions and makes them more robust.
    ig_with_nt = NoiseTunnel(ig)
    
    attributions, delta = ig_with_nt.attribute(
        input, 
        nt_type='smoothgrad_sq',  # Noise type: SmoothGrad with squared gradients.
        target=target, 
        nt_samples=10,           # Number of noisy samples to generate.
        stdevs=0.2,              # Standard deviation of the noise added.
        return_convergence_delta=True
    )
    
    return attributions.detach().cpu().numpy()


def compute_js_divergence(attributions=None, is_normalized=True):
    """
    Compute the Jensen-Shannon divergence between two distributions.
    """
    
    if attributions is None:
        raise ValueError("Attributions cannot be None.")

    # Normalize the attributions to sum to 1
    if is_normalized:
        attributions = attributions / np.sum(attributions, axis=1, keepdims=True)

    js_divergence = spread_divergence(attributions)

    return js_divergence