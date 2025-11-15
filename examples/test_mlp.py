#!/usr/bin/env python3
"""Test the compiled Lumen MLP"""

import torch
import sys
sys.path.insert(0, '.')

from mlp_standalone import mlp_forward

# Create test data
batch = 32
features = 784
hidden = 256
classes = 10

# Random input
x = torch.randn(batch, features)

# Initialize parameters with Xavier/Kaiming
W1 = torch.nn.init.xavier_uniform_(torch.empty(features, hidden))
b1 = torch.zeros(hidden)
W2 = torch.nn.init.xavier_uniform_(torch.empty(hidden, classes))
b2 = torch.zeros(classes)

# Run forward pass
output = mlp_forward(x, W1, b1, W2, b2)

print(f"✅ Forward pass successful!")
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Expected shape: ({batch}, {classes})")
print(f"   Shapes match: {output.shape == (batch, classes)}")

# Test gradients
output.sum().backward()
print(f"✅ Backward pass successful!")
print(f"   W1 gradient shape: {W1.grad.shape if W1.grad is not None else 'None'}")
