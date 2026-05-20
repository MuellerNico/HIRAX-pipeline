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

data_folder = "/home/uranus/HIRAX/out/"
filename_pattern = "sstream_masked_*.zarr.zip" 
filename_pattern_ps = "delay_ps_gibbs_*.h5"
config_path = "/home/uranus/kf-pipe-lite/telescope_configs/bt/241211_kf_config1"
plot_root = 'plots'
suffix = "" # optional suffix for output filenames

lsd_list = [9287] # only use certain days. if list empty: use all days. assumes lsd as last part of filename

def is_valid_zip(fname):
    try:
        with zipfile.ZipFile(fname, 'r') as z:
            z.testzip()
        return True
    except Exception:
        print(f"Corrupted zip file: {fname}")
        return False

def get_files():
    fnames = glob.glob(data_folder + filename_pattern)
    if len(lsd_list) > 0: # filter lsd if specified
        fnames = [f for f in fnames if int(f.split('_')[-1].split('.')[0]) in lsd_list]
    fnames.sort()
    fnames = [f for f in fnames if is_valid_zip(f)] # remove corrupted files 
    print(f"Found {len(fnames)} valid files")
    for f in fnames:
        print(f)
    return fnames

def get_power_spectrum_files():
    fnames = glob.glob(data_folder + filename_pattern_ps)
    if len(lsd_list) > 0: # filter lsd if specified
        fnames = [f for f in fnames if int(f.split('_')[-1].split('.')[0]) in lsd_list]
    fnames.sort()
    print(f"Found {len(fnames)} power spectrum files")
    for f in fnames:
        print(f)
    return fnames

# Mask will be true when sun is up
def get_sun_up_mask(ra, lsd, observer):
    sun_up_mask = np.ones(ra.size, dtype=bool)
    lsd_start_unix = observer.lsd_to_unix(lsd)
    lsd_end_unix = observer.lsd_to_unix(lsd) + 24*60*60

    sun_rise_unix = observer.solar_rising(lsd_start_unix - 60, lsd_end_unix + 60)[0]
    sun_set_unix = observer.solar_setting(lsd_start_unix - 60, lsd_end_unix + 60)[0]

    sun_rise_ra = (sun_rise_unix - lsd_start_unix) * 360/(24*60*60)
    sun_set_ra = (sun_set_unix - lsd_start_unix) * 360/(24*60*60)

    print(f"LSD unix start: {lsd_start_unix}, end: {lsd_end_unix}")
    print(f"Sun rise unix: {sun_rise_unix}, Sun set unix: {sun_set_unix}")
    print(f"Sun rise RA: {sun_rise_ra:.2f} deg, Sun set RA: {sun_set_ra:.2f} deg")

    if sun_rise_ra < sun_set_ra:
        sun_up_mask[(ra <= sun_rise_ra) | (ra >= sun_set_ra)] = False
    else:
        sun_up_mask[(ra < sun_rise_ra) & (ra > sun_set_ra)] = False
    return sun_up_mask, sun_rise_ra, sun_set_ra

# save a simple figure of the sun up mask for reference
def plot_sun_up_mask(sun_up_mask, sun_rise_ra, sun_set_ra, ra):
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    cmap = plt.cm.gray.reversed()
    ax.imshow(sun_up_mask[:, None], extent=(800, 400, ra[0], ra[-1]), aspect='auto', cmap=cmap, origin='lower')
    plt.colorbar(ax.images[0], ax=ax, label='Sun Up Mask')
    # horizontal lines for sun rise/set
    ax.axhline(sun_rise_ra, color='orange', linestyle='--', label='Sun Rise')
    ax.axhline(sun_set_ra, color='red', linestyle='--', label='Sun Set')
    ax.set_ylabel('R.A. [deg]')
    ax.set_xlabel('Freq. [MHz]')
    ax.set_ylim(360, 0)
    ax.legend()
    return fig

def create_dirs(fnames, telescope):
    root = zarr.Group(fnames[0], '/')
    for bl_ind in range(root.index_map['prod'].size):
        input_a = root.index_map['prod'][bl_ind]['input_a']
        input_b = root.index_map['prod'][bl_ind]['input_b']
        dish_a = 'E' if telescope.feedpositions[input_a][0] == 9 else 'W'
        dish_b = 'E' if telescope.feedpositions[input_b][0] == 9 else 'W'
        pol_a, pol_b = telescope.polarisation[input_a], telescope.polarisation[input_b]
        for time_segment in ['sun_up', 'sun_down', 'full_day']:
            os.makedirs(f'{plot_root}/{time_segment}/{dish_a}{pol_a}_{dish_b}{pol_b}/', exist_ok=True)
    os.makedirs(f'{plot_root}/PS/', exist_ok=True)

def plot_waterfall(ra, data):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(40,25)) # larger figsize to accurately show masked pixels
    extent = (800, 400, ra[0], ra[-1])
    # median subtraction
    # meds = np.nanmedian(data, axis=0) # Note: warning about all NaN slice is expected for whole frequency channels masked
    # data -= meds[None, :] # subtract median per freq

    # Filter data for quantile calculation, handling empty arrays
    filtered_data = 10*np.log10(np.abs(data)[np.isfinite(data) & (data != 0.)])
    # Check if filtered data is empty
    if filtered_data.size == 0:
        print("Warning: No valid data points for quantile calculation. Skipping plot.")
        return fig
    
    #vmin, vmax = np.quantile(10*np.log10(np.abs(data)[np.isfinite(data) & (data != 0.)]), (0.01, 0.99))
    vmin, vmax = np.quantile(filtered_data, (0.01, 0.99))
    aspect = 2

    nan_mask = np.isnan(data)
    nan_color = 'magenta'
    cmap_nan = plt.cm.colors.ListedColormap(['none', nan_color]) # colormap for mask overlay

    im = axs[0].imshow(10*np.log10(np.abs(data)), extent=extent, aspect=aspect, origin='lower', vmin=vmin, vmax=vmax, interpolation='none')
    # overlay NaN values
    axs[0].imshow(nan_mask, extent=extent, aspect=aspect, origin='lower', cmap=cmap_nan, interpolation='none', alpha=1.0)
    axs[0].set_title(r'$|\mathcal{V}|$')
    plt.colorbar(im, ax=axs[0], label='[dB]')

    im = axs[1].imshow(np.degrees(np.angle(data)), extent=extent, aspect=aspect, origin='lower', interpolation='none')
    axs[1].imshow(nan_mask, extent=extent, aspect=aspect, origin='lower', cmap=cmap_nan, interpolation='none', alpha=1.0)
    axs[1].set_title(r'Phase[$\mathcal{V}$]')
    plt.colorbar(im, ax=axs[1], label='[deg.]')

    for ax in axs:
        ax.set_ylabel('R.A. [deg]')
        ax.set_xlabel('Freq. [MHz]')
        ax.set_ylim(360, 0)
    return fig

def plot_full_ps(spectrum, delay):
    # plot power spectrum
    fig, ax = plt.subplots(figsize=(10, 6))
    for bl_ind in range(spectrum.shape[0]):  
        ax.semilogy(delay, spectrum[bl_ind, :].T, alpha=0.7, label=f'BL {bl_ind}')
    # plot mean
    mean_ps = spectrum.mean(axis=0)
    ax.semilogy(delay, mean_ps, 'k-', linewidth=2, label='Mean')
    ax.set_xlabel(r'Delay [$\mu s$]')
    ax.set_ylabel('Power')
    ax.set_xlim(left=-0.1, right=0.1) # control zoom
    ax.set_ylim(bottom=0, top=10)
    ax.grid(True, alpha=0.3) 
    ax.legend()
    return fig

def plot_ps(spectrum, delay):
    # plot power spectrum for one baseline
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(delay, spectrum, 'b-', alpha=0.7)
    ax.set_xlabel('Delay')
    ax.set_ylabel('Power')
    ax.set_ylim(bottom=0, top=10)
    ax.grid(True, alpha=0.3) 
    return fig

def main():
    telescope = get_telescope(manager.ProductManager.from_config(config_path))
    fnames = get_files()
    fnames_ps = get_power_spectrum_files()

    create_dirs(fnames, telescope)

    for i, fname in enumerate(fnames):
        print(f"Processing {fname.split('/')[-1]}")
        root = zarr.Group(fname, '/')
        ra = root.index_map.ra[:]
        lsd = root.attrs['lsd']
        lsd_date = datetime.datetime.fromtimestamp(telescope.lsd_to_unix(lsd), datetime.UTC).isoformat().split('T')[0]
        out_file_tag = f"lsd_{lsd:.0f}{suffix}"     
    
        # plot full power spectrum
        power_spectrum = None
        delay = None
        if len(fnames_ps) == len(fnames):
            print("Processing power spectrum")
            fname_ps = fnames_ps[i] # hdf5 file
            with h5py.File(fname_ps, "r") as f:
                power_spectrum = f["spectrum"][:]  
                delay = f["index_map"]["delay"][:]
                fig = plot_full_ps(power_spectrum, delay) 
                fig.suptitle(f"Power Spectrum {lsd_date}")
                fig.savefig(f'{plot_root}/PS/power_spectrum_{out_file_tag}.png', dpi=160)
                plt.close(fig)  

        # sun up mask
        sun_up_mask, sun_rise_ra, sun_set_ra = get_sun_up_mask(ra, lsd, telescope)
        fig_mask = plot_sun_up_mask(sun_up_mask, sun_rise_ra, sun_set_ra, ra)
        fig_mask.suptitle(lsd_date)
        fig_mask.savefig(f'{plot_root}/sun_up_mask_{out_file_tag}.png', dpi=160)
        plt.close(fig_mask)

        for bl_ind in range(root.index_map['prod'].size):
            wt = root.vis_weight[:, bl_ind, :].T
            vis = root.vis[:, bl_ind, :].T
            set_nan = np.where(wt == 0, np.nan, 1.)
            vis *= set_nan

            input_a = root.index_map['prod'][bl_ind]['input_a']
            input_b = root.index_map['prod'][bl_ind]['input_b']
            dish_a = 'E' if telescope.feedpositions[input_a][0] == 9 else 'W'
            dish_b = 'E' if telescope.feedpositions[input_b][0] == 9 else 'W'
            pol_a, pol_b = telescope.polarisation[input_a], telescope.polarisation[input_b]
            title = f"{lsd_date}\n{dish_a}{pol_a}$\\times${dish_b}{pol_b}"

            # plot waterfalls
            plot_configs = [('sun_up', vis*np.where(sun_up_mask, 1., np.nan)[:, None]),
                            ('sun_down', vis*np.where(sun_up_mask, np.nan, 1.)[:, None]),
                            ('full_day', vis)]

            for time_segment, data in plot_configs:
                fig = plot_waterfall(ra, data)
                fig.suptitle(title)
                fig.savefig(f'{plot_root}/{time_segment}/{dish_a}{pol_a}_{dish_b}{pol_b}/waterfall_{out_file_tag}.png', dpi=160)
                plt.close(fig)

            if power_spectrum is not None:
                # plot power spectrum for this baseline
                ps_fig = plot_ps(power_spectrum[bl_ind, :], delay)
                ps_fig.suptitle(f"Power Spectrum {title}")
                ps_fig.savefig(f'{plot_root}/{time_segment}/{dish_a}{pol_a}_{dish_b}{pol_b}/ps_{out_file_tag}.png', dpi=160)
                plt.close(ps_fig)

            # print progress
            print(f"{dish_a}{pol_a}x{dish_b}{pol_b} ", end='', flush=True)

        print("") # new line
        root.store.close() 

if __name__ == "__main__":
    main()