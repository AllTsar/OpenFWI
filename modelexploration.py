import torch
import matplotlib.pyplot as plt
import numpy as np


#invnet_model = torch.load("trained_models/invnet_checkpoint.pth", map_location='cpu', weights_only=False)
fft_model = torch.load("trained_models/fft_checkpoint.pth", map_location='cpu', weights_only=False)
state_dict = fft_model['model']

# The 7x1 conv weight tensor: shape (32, 5, 7, 1)
# (out_channels=32, in_channels=5, kH=7, kW=1)
w = state_dict['convblock1.layers.0.weight']
b = state_dict['convblock1.layers.0.bias']

print(w.shape)  # torch.Size([32, 5, 7, 1])

w = w.squeeze(-1).detach().numpy() # (32, 5, 7)
w_min = w.min()
w_max = w.max()

# Plot each of the 32 filters, averaged across input channels
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
axes = axes.flatten()

for i in range(32):
    ax = axes[i]
    # Plot each input channel as a separate line
    # for c in range(5):
    #     ax.plot(w[i, c, :], alpha=0.5, linewidth=1)
    ax.plot(w[i, 1, :], alpha=0.5, linewidth=2, color="red")
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_title(f'Filter {i}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(w_min, w_max)

plt.suptitle('convblock1 learned filters (7x1)', fontsize=14)
plt.tight_layout()
plt.show()