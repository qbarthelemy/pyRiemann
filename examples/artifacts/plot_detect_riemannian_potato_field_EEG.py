"""
===============================================================================
Artifact Detection with Riemannian Potato Field
===============================================================================

Example of Riemannian Potato Field [1] applied on EEG time-series to detect
artifacts in online processing. It is compared to the Riemannian Potato [2].
"""
# Authors: Quentin Barthélemy & David Ojeda
#
# License: BSD (3-clause)

import os
import numpy as np
from mne.datasets import sample                     # tested with mne 0.21
from mne.io import read_raw_fif
from mne import Epochs, make_fixed_length_events
from pyriemann.estimation import Covariances
from pyriemann.clustering import Potato, PotatoField
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


###############################################################################


def set_channels(raw):
    """Select by eye and rename the usual 21 channels of the 10-20 montage"""
    ch_idx = [1, 2, 3, 8, 10, 12, 14, 16, 26, 28, 30, 32, 34, 44, 46, 48, 50,
        52, 57, 58, 59]
    ch_names_old = ['EEG 0' + str(i).zfill(2) for i in ch_idx]
    raw.pick_channels(ch_names_old, ordered=True)
    ch_names_new = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7',
        'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    ch_renaming = {name_old: name_new
        for (name_old, name_new) in zip(ch_names_old, ch_names_new)}
    raw.rename_channels(ch_renaming)
    return raw


def filter_bandpass(signal, low_freq, high_freq, channels=None, method="iir"):
    """Filter signal on specific channels nad in a specific frequency band"""
    sig = signal.copy()
    if channels is not None:
        sig.pick_channels(channels)
    return sig.filter(l_freq=low_freq, h_freq=high_freq, method=method)


def plot_sig(axis, time, signal, offset, ch_names):
    axis.cla()
    axis.plot(time, signal + offset, lw=0.75)
    axis.set(xlabel='Time (s)', ylabel='EEG channels')
    axis.set_xlim([time[0], time[-1]])
    axis.set_yticks(offset)
    axis.set_yticklabels(ch_names)


def plot_rect(axis, RP_label, RPF_label):
    ylims = axis.get_ylim()
    height = ylims[1] - ylims[0]
    if not RP_label:
        axis.add_patch(
            Rectangle((time[0, 0] - 0.98 * test_time_start,
                ylims[0] + 0.02 * height),
                width=duration, height=0.96 * height, edgecolor='r',
                facecolor='none', linestyle='dashed'))
        axis.text(time[0, 0] - 0.85 * test_time_start,
                  ylims[1] - 0.05 * height, 'RP', color='r', size=16)
    if not RPF_label:
        axis.add_patch(
            Rectangle((time[0, 0] - test_time_start, ylims[0] + 0.01 * height),
                width=1.01 * duration, height=0.98 * height,
                edgecolor='violet', facecolor='none', linestyle='dashed'))
        axis.text(time[0, 0] - test_time_start + 1.03 * duration,
            ylims[1] - 0.05 * height, 'RPF', color='violet', size=16)
    if RP_label and RPF_label:
        axis.add_patch(
            Rectangle((time[0, 0] - test_time_start, ylims[0] + 0.02 * height),
                width=duration, height=0.96 * height, edgecolor='b',
                facecolor='none'))


###############################################################################
# Load EEG data
# -------------

raw_fname = os.path.join(sample.data_path(), 'MEG', 'sample',
                         'sample_audvis_raw.fif')
raw = read_raw_fif(raw_fname, preload=True, verbose=False)
sfreq = int(raw.info['sfreq'])


###############################################################################
# Offline processing of EEG data
# ------------------------------

# Apply common average reference on EEG channels
raw.pick_types(meg=False, eeg=True).set_eeg_reference(ref_channels='average',
                                                      projection=False) #TODO

# Select the usual 21 channels of the 10-20 montage
raw = set_channels(raw)
ch_names = raw.ch_names
ch_count = len(ch_names)
raw.plot_sensors(ch_type='eeg', show_names=True)

# Define time-series epoching with a sliding window
duration = 2.5    # duration of epochs
interval = 0.25   # interval between epochs
events = make_fixed_length_events(raw, id=1, duration=interval)


###############################################################################
# Offline Calibration of Potatoes
# -------------------------------

z_th = 2.0       # z-score threshold
split_set = 40   # nb of matrices to train the potato
train_set = slice(split_set)

# Riemannian potato (RP): select all channels and filter between 1 and 35 Hz.
rp_sig = filter_bandpass(raw, 1., 35.)
rp_epochs = Epochs(rp_sig, events, tmin=0., tmax=duration, baseline=None,
                   verbose=False)
rp_covs = Covariances(estimator='lwf').transform(5e5 * rp_epochs.get_data())
RP = Potato(metric='riemann', threshold=z_th).fit(rp_covs[train_set])

# Riemannian potato field (RPF): it combines several potatoes of low dimension,
# eahc one being designed to capture specific artifact typically affecting
# specific spatial areas (ie, subsets of channels) and/or specific frequency
# bands.
p_th = 0.01       # probability threshold
rpf_config = {
    'eye_blinks': {'ch_names': ['Fp1', 'Fpz', 'Fp2'], # for eye-blinks
                   'low_freq': 1.,
                   'high_freq': 20.},
    'left': {'ch_names': ['F7', 'T7', 'P7'], # for muscular artifacts
             'low_freq': 55.,                # in left area
             'high_freq': 95.},
    'right': {'ch_names': ['F8', 'T8', 'P8'], # for muscular artifacts
              'low_freq': 55.,                # in right area
              'high_freq': 95.},
    'occipital': {'ch_names': ['O1', 'Oz', 'O2'], # for muscular artifacts
                  'low_freq': 55.,                # in occipital area
                  'high_freq': 95.},
    'global_lf': {'ch_names': None, # for low-frequency artifacts
                  'low_freq': 0.5,  # in all channels
                  'high_freq': 3.},
    'global_hf': {'ch_names': None, # for global high-frequency artifacts
                  'low_freq': 25.,  # in all channels
                  'high_freq': 95.}
   }

rpf_covs = []
for _, p in rpf_config.items():
    rpf_sig = filter_bandpass(raw, p.get('low_freq'), p.get('high_freq'),
                              channels=p.get('ch_names'))
    rpf_epochs = Epochs(rpf_sig, events, tmin=0., tmax=duration, baseline=None,
                        verbose=False)
    cov_ = Covariances(estimator='lwf').transform(5e5 * rpf_epochs.get_data())
    rpf_covs.append(cov_)
RPF = PotatoField(metric='riemann', z_threshold=z_th, p_threshold=p_th,
                  n_potatoes=len(rpf_config)).fit(rpf_covs[train_set])


###############################################################################
# Online Artifact Detection with Potatoes
# ---------------------------------------

test_cov_max = 500      # nb of matrices to visualize in this example
test_time_start = -2    # start time to display time-series
test_time_end = 5       # end time to display time-series

test_set = range(split_set, test_cov_max)
eeg_data = 3e5 * raw.get_data()
eeg_offset = - 15 * np.linspace(1, ch_count, ch_count, endpoint=False)

# Plot online detection (an interactive window is required)
fig = plt.figure(figsize=(12, 10))
fig.suptitle('Online artifact detection, RP vs RPF', fontsize=16)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# Detect artifacts/outliers on test set, with a loop to imitate an online
# acquisition, processing and artifact detection of EEG time-series.
# Remak that all these potatoes are static: they are not updated when data is
# not artifacted.
for t in test_set:

    # Online artifact detection
    RP_label = RP.predict(rp_covs[t][np.newaxis, ...])
    RPF_label = RPF.predict([c[t][np.newaxis, ...] for c in rpf_covs])

    # Update data
    time_start = t * interval + test_time_start
    time_end = t * interval + test_time_end
    time = np.linspace(time_start, time_end,
        int((time_end - time_start) * sfreq), endpoint=False)[np.newaxis, :]
    # Update plot
    plot_sig(ax, time.T,
        eeg_data[:, int(time_start * sfreq):int(time_end * sfreq)].T,
        eeg_offset.T, ch_names)
    plot_rect(ax, RP_label, RPF_label)

    plt.pause(0.5)
    plt.draw()


###############################################################################
# References
# ----------
# [1] Q. Barthélemy, L. Mayaud, D. Ojeda, M. Congedo, "The Riemannian potato
# field: a tool for online signal quality index of EEG", IEEE TNSRE, 2019.
# [2] A. Barachant, A. Andreev, M. Congedo, "The Riemannian Potato: an
# automatic and adaptive artifact detection method for online experiments using
# Riemannian geometry", Proc. TOBI Workshop IV, 2013.
