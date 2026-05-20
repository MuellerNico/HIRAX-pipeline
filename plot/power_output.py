import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
from draco.core.io import get_telescope
from drift.core import manager
import glob
import datetime
import zipfile
import h5py

# plot the power output from the KFStokesMask flagging task

folder = "/home/uranus/HIRAX/out/2025-11-16_19-23/"
filename = "lowpass_power_lsd_9287.h5"

with h5py.File(folder + filename, "r") as f:
    print(f"Dataset shape: {f['vis_weight'].shape}")
    vis_weight = f["vis_weight"][:, 0, :]
    vis = f["vis"][:, 0, :]
    print(f"Loaded vis_weight with shape: {vis_weight.shape}")

# plot vis_weight
fig, ax = plt.subplots(figsize=(6, 10))

im = ax.imshow(vis_weight.T, aspect='auto', origin='lower', cmap='gray')
ax.set_title("Power vis_weight")
ax.set_xlabel("Freq")
ax.set_ylabel("Time")
cbar = fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(folder + "power_vis_weight.png", dpi=160)
plt.close(fig)

# plot vis data
data = vis.T
extent = (800, 400, 0, 360)

filtered_data = 10*np.log10(np.abs(data)[np.isfinite(data) & (data != 0.)])
if filtered_data.size == 0:
    print("Warning: No valid data points for quantile calculation")
#vmin, vmax = np.quantile(10*np.log10(np.abs(data)[np.isfinite(data) & (data != 0.)]), (0.01, 0.99))
vmin, vmax = np.quantile(filtered_data, (0.01, 0.99))

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 6))
im = axs[0].imshow(10*np.log10(np.abs(data)), aspect='auto', origin='lower', vmax=vmax, vmin=vmin, interpolation='none')
axs[0].set_title(r'Power $|\mathcal{V}|$')
plt.colorbar(im, ax=axs[0], label='[dB]')

im = axs[1].imshow(np.angle(data), aspect='auto', origin='lower', vmax=vmax, vmin=vmin, interpolation='none')
axs[1].set_title(r'Power Phase[$\mathcal{V}$]')
plt.colorbar(im, ax=axs[1], label='[deg.]')

for ax in axs:
        ax.set_ylabel('R.A. [deg]')
        ax.set_xlabel('Freq. [MHz]')
        ax.set_ylim(360, 0)
plt.savefig(folder + "power_vis.png")