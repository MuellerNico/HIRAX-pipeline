import numpy as np
import h5py as h5

from astropy.time import Time
from astropy import units
from astropy import coordinates as coords

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import dates
import matplotlib as mpl
from matplotlib.lines import Line2D
from cycler import cycler


import imageio

import scipy as sp
from scipy import signal

import glob


def load_hd5_list(file_list, test_freqs=None, only_autos=False, flip_freq_ax = False):
    ## KF correlator doesnt save digital gains here. separate file
    
    print('There are {} files in the list'.format(len(file_list)))

    # load the first one, get coordinates and gains
    data_file = h5.File(file_list[0],'r')
    freqs = data_file['index_map/freq']['centre']
    products = data_file['index_map/prod'][()]
    gain_coeff = data_file['gain_coeff'][()]
    gain_exp = data_file['gain_exp'][()]
    
    data_file.close()

    ## get subset of frequencies
    if test_freqs is not None:
        freq_idxs = []
        for test_freq in test_freqs:
            freq_idxs += [np.argmin((freqs - test_freq)**2)]
    else:
        freq_idxs = [i for i in range(len(freqs))]
    freqs = freqs[freq_idxs]

    
    vis_list = []
    unix_times_list = []

    ## get auto indexes
    if only_autos:
        prod_idxs = [np.argwhere((products['input_a'] == i) & (products['input_b'] == i))[0][0] for i in range(4)]
    else:
        prod_idxs = [i for i in range(10)]
    products = products[prod_idxs]

    ##  make an index grid for vis
    freq_idx_grid, prod_idx_grid = np.ix_(freq_idxs, prod_idxs)
    
    vis_list = []
    unix_times_list = []
    for file in file_list:
        data_file = h5.File(file,'r')
        vis = data_file['vis'][()][:, freq_idx_grid, prod_idx_grid]
        unix_times = data_file['index_map/time']['ctime']
        data_file.close()
        vis_list += [vis]
        unix_times_list += [unix_times]
    
    vis = np.concatenate(vis_list, axis=0)
    unix_times = np.concatenate(unix_times_list, axis=0)
    times = Time(unix_times, format='unix')

    if flip_freq_ax:
        freqs = np.flip(freqs)
        vis = np.flip(vis, axis=1)
        gain_coeff = np.flip(gain_coeff)
        gain_exp = np.flip(gain_exp)

    print('Merged vis. shape is {}'.format(vis.shape))
    print('Obs. data span is {} - {} UTC'.format(times[0].iso, times[-1].iso))
    
    return vis, times, freqs, products, gain_coeff, gain_exp


def load_digital_gains(path, flip_freq_ax=False):
    ## coeff has shape (1,n_freq, n_inputs)
    ## exp has shape (1, n_inputs)
    
    gain_file = h5.File(path,'r')
    gain_coeff = gain_file['gain_coeff'][()]
    gain_exp = gain_file['gain_exp'][()]
    gain_file.close()

    gain = gain_coeff*2**(gain_exp[:,None,:])

    if flip_freq_ax:
        gain = np.flip(gain, axis=1)

    print('Digital gains have shape {}'.format(gain.shape))

    return gain

def undo_digital_gains(vis, products, gains):
    vis2 = np.zeros_like(vis)
    for prod_idx in range(vis.shape[2]):
        in1, in2 = products[prod_idx]
        vis2[:,:,prod_idx] = vis[:,:,prod_idx]/gains[:,:,in1]/gains[:,:,in2]
    return vis2

def get_auto_idxs(products):
    return np.argwhere(products['input_a'] == products['input_b']).flatten().tolist()


#####



###

def get_telescope_loc():
    # From: Google Earth approximate location
    
    lat = -(30 + 58/60 +20/(60*60)) 
    lon = 21 + 59/60 + 42/(60*60)
    height = 1303
    
    loc = coords.EarthLocation.from_geodetic(
        lon=lon*units.deg,
        lat=lat*units.deg,
        height=height*units.m,
        ellipsoid='WGS84')
    
    return loc

def get_obs_data(targets, location, times):
    
    targets_ra_dec = []
    targets_alt_az = []
    pointings_alt_az = []
    transit_idxs = []

    times_sast = times + 2*units.hour
    
    for target in targets:
    
        if target=='Sun':
            target_ra_dec = None
            target_alt_az = coords.get_body(target, times).transform_to(coords.AltAz(obstime=times, location=location))
        else:
            target_ra_dec = coords.SkyCoord.from_name(target)
            target_alt_az = target_ra_dec.transform_to(coords.AltAz(obstime=times, location=location))
        
        pointing_alt_az = coords.AltAz(alt=target_alt_az.alt.deg.max()*units.deg*np.ones(len(times)), 
                                         az=target_alt_az.az[target_alt_az.alt.deg.argmax()]*np.ones(len(times)), 
                                         location=location, obstime=times)
        
        targets_ra_dec += [target_ra_dec]
        targets_alt_az += [target_alt_az]
        pointings_alt_az += [pointing_alt_az]
    
        transit_idx = np.argmin(pointings_alt_az[0].separation(target_alt_az).value)
    
        transit_idxs += [transit_idx]
    
        print('{}:\n {}\n {}\n'.format(target, pointing_alt_az[np.argmax(target_alt_az.alt.deg)], times_sast[transit_idx].datetime))
    
    return targets_ra_dec, targets_alt_az, pointings_alt_az, transit_idxs

def butterworth_filter(data, filter_type, char_freqs, order=4, sample_rate=1, axis=None):
    ## adapted from Devin's code
    
    if axis is None:
        axis_size = len(data)
        axis = -1
    else:
        axis_size = data.shape[axis]

    # FT data on specified axis
    vis_fft = np.fft.fft(data, axis=axis)
    
    # scipy signal processing implementation details...
    b, a = signal.butter(N=order, Wn=char_freqs, btype=filter_type, analog=True)
    freqs = np.fft.fftfreq(axis_size, d=sample_rate)
    _, h = signal.freqs(b, a, freqs)

    # filter data on axis specified
    filtered_vis_fft = vis_fft*(np.abs(h).reshape(-1,1,1)**2)

    return np.fft.ifft(filtered_vis_fft, axis=axis)

def get_geometric_delays(baseline, freq, target_altaz):
    az = target_altaz.az.radian
    alt = target_altaz.alt.radian  

    los_unit_vector = np.column_stack([
        np.cos(alt) * np.sin(az),  # East component
        np.cos(alt) * np.cos(az),  # North component
        np.sin(alt)               # Up component
    ])

    delays = np.dot(los_unit_vector, baseline) / sp.constants.c

    return delays


#### plots



cmap = mpl.colormaps["tab10"]

def plot_waterfall(data, times, freqs, 
                   cmap='magma', 
                   vlims=None, 
                   norm='linear', 
                   show_colorbar=False, 
                   show_labels=False, 
                   show_date=False,
                   hsv=False,
                   fig=None, 
                   ax=None, 
                   scale=1,
                   figsize=(10,7)):

    extent = (freqs[0], freqs[-1], dates.date2num(times.datetime[-1]), dates.date2num(times.datetime[0]))
    aspect = np.abs((extent[1] - extent[0]) / (extent[3] - extent[2]))*scale

    if vlims is None:
        vlims = [np.min(data), np.max(data)]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if not hsv: 
        cb = ax.imshow(data, extent=extent, aspect=aspect,  cmap=cmap, interpolation='none', vmin=vlims[0], vmax=vlims[1], norm=norm)
    
        if show_colorbar:
            plt.colorbar(cb)
    else:
        # hsv plot
        phase_norm = (np.angle(data) + np.pi)/2/np.pi
        intensity_norm = np.abs(data)/np.percentile(np.abs(data), 98)
        
        hsv_image = np.zeros((data.shape[0], data.shape[1], 3))
        hsv_image[..., 0] = phase_norm  # Hue corresponds to phase
        hsv_image[..., 1] = 1.0  # Saturation is fixed at 1
        hsv_image[..., 2] = intensity_norm # Value corresponds to intensity
        
        rgb_image = hsv_to_rgb(hsv_image)
        ax.imshow(rgb_image, extent=extent, aspect=aspect, interpolation='none')

        
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    
    if show_labels:
        ax.set_ylabel('SAST')
        ax.set_xlabel('Frequency [MHz]')

    if show_date:
        ax.text(-0.015, 1.02, times.iso[0].split(' ')[0], ha='right', va='bottom', transform=ax.transAxes)

    return fig, ax


def plot_timeseries(data, times, ylims=None, ylabel=None, scatter=False, figsize=(16,9), legend=True):

    labels = ['400-450 MHz', '450-500 MHz', '500-550 MHz', '550-600 MHz','600-650 MHz', '650-700 MHz', '700-750 MHz', '750-800 MHz']
    cmap = mpl.colormaps['Dark2_r']
    n = len(labels)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    legend_elements = [Line2D([0], [0], marker='o', color='none', label=label, markerfacecolor=color, markersize=8) for label, color in zip(labels, colors)]

    fig, ax = plt.subplots(figsize=figsize, nrows=1, sharex=True)
    
    x = times.to_datetime()

    for freq_idx in range(data.shape[1]):
        color = cmap(freq_idx/data.shape[1]) 
        if scatter: ax.scatter(x, data[:,freq_idx], alpha=.7, s=2, color=color)
        ax.plot(x, data[:,freq_idx], alpha=.2, color=color)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    
    if ylims: ax.set_ylim(ylims)
    if ylabel: ax.set_ylabel(ylabel)
    
    fig.tight_layout()
    
    if legend: ax.legend(handles=legend_elements, loc='upper left')

    return fig, ax
    


def plot_passband(data, freqs, products, prod_idxs, percentile=50, shade_intervals=None, ylims=None, figsize=(10,5), fig=None, ax=None):

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    
    for i, prod_idx in enumerate(prod_idxs):
        to_plot = np.percentile(data[:,:,prod_idx], percentile, axis=0)

        color = cmap(prod_idx % cmap.N)
        ax.plot(freqs, to_plot, alpha=1, label=str(products[prod_idx]), color=color)
    
        if shade_intervals:
            to_plot_max = np.percentile(data[:,:,prod_idx], shade_intervals[1], axis=0)
            to_plot_min = np.percentile(data[:,:,prod_idx], shade_intervals[0], axis=0)
            ax.fill_between(freqs, to_plot_min, to_plot_max, alpha=.4, color=color)
            
        ax.set_yscale('log')
        if ylims:
            ax.set_ylim(ylims)

    ax.legend(bbox_to_anchor=(1.1,.98))
    ax.grid()

    return fig, ax

def plot_passband_3up(vis, freqs, products, gains, vis2, only_autos=True, ylims=None, percentile=50, shade_intervals=None):

    fig, ax = plt.subplots(figsize=(16,9), nrows=3, sharex=True)

    n_inputs = 4
    auto_idxs = get_auto_idxs(products)
    if only_autos: 
        prod_idxs = auto_idxs
    else:
        prod_idxs = list(range(len(products)))
    
    for i in range(4):
        color = cmap(auto_idxs[i] % cmap.N)
        ax[0].plot(freqs, np.real(gains[0,:,i]), label='input {}'.format(i), color=color)
    ax[0].legend(bbox_to_anchor=(1.1,.98))
    ax[0].grid()
    ax[0].set_title('Digital Gains')
    
    plot_passband(vis, freqs, products, prod_idxs, percentile=percentile, shade_intervals=shade_intervals, ylims=ylims, ax=ax[1])
    ax[1].set_title('|Vis|')
    plot_passband(vis2, freqs, products, prod_idxs, percentile=percentile, shade_intervals=shade_intervals, ylims=ylims, ax=ax[2])
    ax[2].set_title('|Vis/G_i/G_j|')

    fig.tight_layout()

    return fig, ax

def plot_waterfall_triangle(data, times, freqs, products, cmap='gist_ncar', vlims=None, norm='log', sat_percent=False, figsize=(10,10), hsv=False):

    fig, ax = plt.subplots(figsize=figsize, nrows=4, ncols=4, sharey=True, sharex=True)
     
    for i in range(4):
        for j in range(i, 4):
            idx = np.argwhere((products['input_a'] == i) & (products['input_b'] == j))[0][0]
            to_plot = data[:,:,idx]
    
            if sat_percent:
                vlims = [np.percentile(to_plot, sat_percent), np.percentile(to_plot, 100-sat_percent)]
    
            plot_waterfall(to_plot, times, freqs, cmap=cmap, vlims=vlims, norm=norm, show_colorbar=False, show_labels=False, show_date=False, ax=ax[j,i], hsv=hsv)
    
            ax[j,i].title.set_text('{}x{}'.format(i,j))
            if i==0:
                ax[j,i].set_ylabel('SAST')
                ax[j,i].yaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    
    ax[0,0].text(-0.015, 1.02, times.iso[0].split(' ')[0], ha='right', va='bottom', transform=ax[0,0].transAxes)

    for i in range(4):
        for j in range(i):
            ax[j,i].set_axis_off()
    
    fig.tight_layout()

    return fig, ax

def plot_waterfall_columns(data, times_sast, freqs, prod_idxs, cmap='gist_ncar', vlims=None, norm='linear', sat_percent=None, scale=2, hsv=False):

    fig, ax = plt.subplots(figsize=(16,9), ncols=len(prod_idxs), nrows=1, sharey=True)

    for i, prod_idx in enumerate(prod_idxs):
        
        to_plot = data[:,:,prod_idx]

        if sat_percent:
            vlims = [np.percentile(np.real(to_plot), sat_percent), np.percentile(np.real(to_plot),100-sat_percent)]
    
        if i==0:
            plot_waterfall(to_plot, times_sast, freqs, cmap=cmap, ax=ax[i], show_labels=True, norm='linear', vlims=vlims, show_colorbar=False, show_date=True, scale=scale, hsv=hsv)
        else:
            plot_waterfall(to_plot, times_sast, freqs, cmap=cmap, ax=ax[i], show_labels=False, norm='linear', vlims=vlims, show_colorbar=False, show_date=False, scale=scale, hsv=hsv)
            ax[i].set_xlabel('Frequency [MHz]')
    
    return fig, ax
