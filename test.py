import os
import h5py
import numpy as np

folder = "data/20250617T135003Z_Klerefontein_corr/"

for filename in os.listdir(folder):
    if filename.endswith(".h5"):
        filepath = os.path.join(folder, filename)
        with h5py.File(filepath, "r") as f:
            weights = f["flags"]["vis_weight"][()]
            print(f"File: {filename}")
            print(weights.shape)
            n_zero = np.sum(weights == 0)
            n_nan = np.sum(np.isnan(weights))
            n_inf = np.sum(np.isinf(weights))
            print(f"Number of zero weights in {filename}: {n_zero} ({n_zero / weights.size:.2%}%)")
            print(f"Number of NaN weights in {filename}: {n_nan} ({n_nan / weights.size:.2%}%)")
            print(f"Number of Inf weights in {filename}: {n_inf} ({n_inf / weights.size:.2%}%)")
            # filter inf
            weights = weights[np.isfinite(weights)]
            # mean and median of weights
            print(f"Weight range: [{np.min(weights):.3e}, {np.max(weights):.3e}]")
            mean_weight = np.mean(weights)
            median_weight = np.median(weights)
            print(f"Mean weight: {mean_weight}")
            print(f"Median weight: {median_weight}")
            print("-----")

