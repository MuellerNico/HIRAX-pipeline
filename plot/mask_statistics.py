import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

# This file provides some numbers and plots about the RFI masks generated 
# by the KFStokesIMask task and the frequency masking. 

data_folder = "/home/uranus/HIRAX/out/2025-12-01_14-16/"
filename_pattern_freq = "rfi_mask_freq_*.h5"
filename_pattern_stokesi = "rfi_mask_stokesi_*.h5"

lsd_list = [9287, 9288, 9289, 9290, 9137, 9138, 9139, 9140]

def main():
    fnames_freq, fnames_stokesi = get_mask_files()
    mask_sum = np.zeros((1024, 17920))
    frac_flagged_list = []
    frac_flagged_stokesi_list = []
    for freq_file in fnames_freq:
        mask_freq = load_mask_data(freq_file)
        print(mask_freq.shape)
        frac_flagged = np.sum(mask_freq) / mask_freq.size
        frac_flagged_list.append(frac_flagged)
        print(f"Frequency Mask - Fraction flagged: {frac_flagged:.4f}")
        # cut mask_freq to match expected shape if needed
        mask_freq = mask_freq[:, :17920]        
        mask_sum += mask_freq
    for stokesi_file in fnames_stokesi:
        mask_stokesi = load_mask_data(stokesi_file)
        frac_flagged = np.sum(mask_stokesi) / mask_stokesi.size
        frac_flagged_stokesi_list.append(frac_flagged)
        print(f"Stokes I Mask - Fraction flagged: {frac_flagged:.4f}")
    print(f"Average fraction flagged (frequency masks): {np.mean(frac_flagged_list):.4f}")
    print(f"Stddev fraction flagged (frequency masks): {np.std(frac_flagged_list):.4f}")
    print(f"Difference in average fraction flagged between Stokes I and frequency masks: {np.mean(frac_flagged_stokesi_list) - np.mean(frac_flagged_list):.4f}")
    num_files = len(fnames_freq)
    frac_flagged_per_channel = np.sum(mask_sum, axis=1) / (mask_sum.shape[1] * num_files)

    # bar plot
    plt.figure(figsize=(6, 4))
    num_bins = 100
    bin_edges = np.linspace(0, 1024, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    binned_values = []
    for i in range(num_bins):
        start_ch = int(bin_edges[i])
        end_ch = int(bin_edges[i + 1])
        binned_values.append(np.mean(frac_flagged_per_channel[start_ch:end_ch]))

    freq_low = 400  # MHz
    freq_high = 800  # MHz
    bin_centers_mhz = freq_high - (bin_centers / 1024) * (freq_high - freq_low)

    plt.bar(bin_centers_mhz, binned_values, width=(freq_high - freq_low) / num_bins, 
        color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xlim(freq_high, freq_low) 
    plt.xlabel('Freq. [MHz]  (binned)')
    plt.ylabel('Fraction of flagged samples')
    # plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/freq_flagged_hist.png", dpi=160)

    # plot a heatmap of the difference between the two masks (just for the first file)
    # diff_mask = mask_freq.astype(int)
    # plt.figure(figsize=(10, 6))
    # im = plt.imshow(diff_mask.T, aspect='auto', origin='lower', cmap='bwr', vmin=-1, vmax=1)
    # plt.colorbar(im, label='Mask Difference (Stokes I - Freq)')
    # plt.xlabel('Frequency')
    # plt.ylabel('Time')
    # plt.title('Difference between Stokes I Mask and Frequency Mask')
    # plt.savefig("mask_difference.png", dpi=160)
    # plt.close()

def get_mask_files():
    fnames_freq = glob.glob(data_folder + filename_pattern_freq)
    fnames_stokesi = glob.glob(data_folder + filename_pattern_stokesi)
    if len(lsd_list) > 0: # filter lsd if specified
        fnames_freq = [f for f in fnames_freq if int(f.split('_')[-1].split('.')[0]) in lsd_list]
        fnames_stokesi = [f for f in fnames_stokesi if int(f.split('_')[-1].split('.')[0]) in lsd_list]
    fnames_freq.sort()
    fnames_stokesi.sort()
    print(f"Found {len(fnames_freq)} frequency mask files")
    print(f"Found {len(fnames_stokesi)} Stokes I mask files")
    return fnames_freq, fnames_stokesi

def load_mask_data(filename):
    with h5py.File(filename, "r") as f:
        mask = f["mask"][:]
    return mask

if __name__ == "__main__":
    main()