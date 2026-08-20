"""
房价预测调参实验

Assembled from your step-by-step solutions.
"""

# Step 2 - standardize_numeric (not yet solved)
# TODO: implement

# Step 3 - one_hot_encode (not yet solved)
# TODO: implement

# Step 4 - build_features (not yet solved)
# TODO: implement

# Step 5 - get_net
import torch
import torch.nn as nn

def get_net(n_features: int) -> nn.Module:
    layer = nn.Linear(n_features, 1)
    nn.init.normal_(layer.weight, mean=0, std=0.01)
    nn.init.normal_(layer.bias, mean=0, std=0.01)
    return layer

# Step 6 - example_kernel (not yet solved)
# TODO: implement

# Step 7 - example_kernel (not yet solved)
# TODO: implement

# Step 8 - example_kernel (not yet solved)
# TODO: implement

# Step 9 - best_learning_rate (not yet solved)
# TODO: implement

# Step 10 - example_kernel (not yet solved)
# TODO: implement

# Step 11 - example_kernel (not yet solved)
# TODO: implement
