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
# from pyriemann.utils.covariance import normalize_trace
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


###############################################################################


def normalize_trace(matrices): # TODO SUPP
    traces = np.trace(matrices, axis1=-2, axis2=-1)
    while traces.ndim != matrices.ndim:
        traces = traces[..., np.newaxis]
    return matrices / traces


def set_channels(raw):
    """Select and rename the usual 21 channels of the 10-20 montage"""
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
    """Filter signal on specific channels and in a specific frequency band"""
    sig = signal.copy()
    if channels is not None:
        sig.pick_channels(channels)
    return sig.filter(l_freq=low_freq, h_freq=high_freq, method=method)


def plot_labels(ax, rp_label, rpf_label):
    labels = []
    ylims = ax.get_ylim()
    height = ylims[1] - ylims[0]
    if not rp_label:
        r1 = ax.axhspan(ylims[0] + 0.06 * height, ylims[1] - 0.05 * height,
            edgecolor='r', facecolor='none',
            xmin=-test_time_start / test_duration - 0.005,
            xmax=(duration - test_time_start) / test_duration - 0.005)
        labels.append(r1)
        ax.text(0.25, 0.95, 'RP', color='r', size=16, transform=ax.transAxes)
    if not rpf_label:
        r2 = ax.axhspan(ylims[0] + 0.05 * height, ylims[1] - 0.06 * height,
            edgecolor='m', facecolor='none',
            xmin=-test_time_start / test_duration + 0.005,
            xmax=(duration - test_time_start) / test_duration + 0.005)
        labels.append(r2)
        ax.text(0.65 , 0.95, 'RPF', color='m', size=16, transform=ax.transAxes)
    if rp_label and rpf_label:
        r3 = ax.axhspan(ylims[0] + 0.05 * height, ylims[1] - 0.05 * height,
            edgecolor='k', facecolor='none',
            xmin=-test_time_start / test_duration,
            xmax=(duration - test_time_start) / test_duration)
        labels.append(r3)
    return labels


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
    projection=False)

# Select the usual 21 channels of the 10-20 montage
raw = set_channels(raw)
ch_names = raw.ch_names
ch_count = len(ch_names)
raw.plot_sensors(ch_type='eeg', show_names=True, title='EEG channels')

# Define time-series epoching with a sliding window
duration = 2.5    # duration of epochs
interval = 0.2    # interval between epochs
events = make_fixed_length_events(raw, id=1, duration=interval)


###############################################################################
# Offline Calibration of Potatoes
# -------------------------------

z_th = 2.0       # z-score threshold
t = 40           # nb of matrices to train the potato
train_set = range(t)

# Riemannian potato (RP): select all channels and filter between 1 and 35 Hz.
rp_sig = filter_bandpass(raw, 1., 35.)
rp_epochs = Epochs(rp_sig, events, tmin=0., tmax=duration, baseline=None,
    verbose=False)
rp_covs = Covariances(estimator='lwf').transform(5e5 * rp_epochs.get_data())
# Trace-normalize covariance matrices
rp_covs = normalize_trace(rp_covs)
# RP training
rp = Potato(metric='riemann', threshold=z_th).fit(rp_covs[train_set])

# Riemannian potato field (RPF): it combines several potatoes of low dimension,
# each one being designed to capture specific artifact typically affecting
# specific spatial areas (ie, subsets of channels) and/or specific frequency
# bands.
p_th = 0.01       # probability threshold
rpf_config = {
    'eye_blinks': {'ch_names': ['Fp1', 'Fpz', 'Fp2'], # for eye-blinks
                   'low_freq': 1.,
                   'high_freq': 20.},
    'left':       {'ch_names': ['F7', 'T7', 'P7'], # for muscular artifacts
                   'low_freq': 55.,                # in left area
                   'high_freq': 95.},
    'right':      {'ch_names': ['F8', 'T8', 'P8'], # for muscular artifacts
                   'low_freq': 55.,                # in right area
                   'high_freq': 95.},
    'occipital':  {'ch_names': ['O1', 'Oz', 'O2'], # for muscular artifacts
                   'low_freq': 55.,                # in occipital area
                   'high_freq': 95.},
    'global_lf':  {'ch_names': None, # for low-frequency artifacts
                   'low_freq': 0.5,  # in all channels
                   'high_freq': 3.},
    'global_hf':  {'ch_names': None, # for global high-frequency artifacts
                   'low_freq': 25.,  # in all channels
                   'high_freq': 95.}
   }

rpf_covs = []
for _, p in rpf_config.items():
    rpf_sig = filter_bandpass(raw, p.get('low_freq'), p.get('high_freq'),
        channels=p.get('ch_names'))
    rpf_epochs = Epochs(rpf_sig, events, tmin=0., tmax=duration, baseline=None,
        verbose=False)
    covs_ = Covariances(estimator='lwf').transform(5e5 * rpf_epochs.get_data())
    rpf_covs.append(normalize_trace(covs_))
# RPF training
rpf = PotatoField(metric='riemann', z_threshold=z_th, p_threshold=p_th,
    n_potatoes=len(rpf_config)).fit([c[train_set] for c in rpf_covs])


###############################################################################
# Online Artifact Detection with Potatoes
# ---------------------------------------

# Detect artifacts/outliers on test set, with an animation to imitate an online
# acquisition, processing and artifact detection of EEG time-series.
# Remak that all these potatoes are static: they are not updated when EEG is
# not artifacted.

test_covs_max = 300     # nb of matrices to visualize in this example
test_time_start = -2    # start time to display signal
test_time_end = 5       # end time to display signal

test_duration = test_time_end - test_time_start
time_start = t * interval + test_time_start
time_end = t * interval + test_time_end
time = np.linspace(time_start, time_end, int((time_end - time_start) * sfreq),
    endpoint=False)
eeg_data = 3e5 * raw.get_data()
sig = eeg_data[:, int(time_start * sfreq):int(time_end * sfreq)]
eeg_offset = - 15 * np.linspace(1, ch_count, ch_count, endpoint=False)

# Plot online detection (an interactive display is required)
fig, ax = plt.subplots(figsize=(12, 10), nrows=1, ncols=1)
fig.suptitle('Online artifact detection, RP vs RPF', fontsize=16)
ax.set(xlabel='Time (s)', ylabel='EEG channels')
ax.set_xlim([time[0], time[-1]])
ax.set_yticks(eeg_offset)
ax.set_yticklabels(ch_names)
pl = ax.plot(time.T, sig.T + eeg_offset.T, lw=0.75)
labels = []

# Plot online detection (an interactive display is required)
def online_update(self):
    global t, time, sig, labels

    # Online artifact detection
    rp_label = rp.predict(rp_covs[t][np.newaxis, ...])
    rpf_label = rpf.predict([c[t][np.newaxis, ...] for c in rpf_covs])

    # Update data
    time_start = t * interval + test_time_end
    time_end = (t + 1) * interval + test_time_end
    time_ = np.linspace(time_start, time_end, int(interval * sfreq),
        endpoint=False)
    time = np.r_[time[int(interval * sfreq):], time_]
    sig = np.hstack((sig[:, int(interval * sfreq):],
        eeg_data[:, int(time_start * sfreq):int(time_end * sfreq)]))
    t += 1

    # Update plot
    for c in range(ch_count):
        pl[c].set_data(time, sig[c] + eeg_offset[c])
        pl[c].axes.set_xlim(time[0], time[-1])
    for lbl in labels:
        lbl.remove()
    for txt in ax.texts:
        txt.set_visible(False)
    labels = plot_labels(ax, rp_label, rpf_label)
    return pl

potato = FuncAnimation(fig, online_update, frames=test_covs_max,
    interval=interval, blit=False, repeat=False)
plt.show()


###############################################################################
# References
# ----------
# [1] Q. Barthélemy, L. Mayaud, D. Ojeda, M. Congedo, "The Riemannian potato
# field: a tool for online signal quality index of EEG", IEEE TNSRE, 2019.
#
# [2] A. Barachant, A. Andreev, M. Congedo, "The Riemannian Potato: an
# automatic and adaptive artifact detection method for online experiments using
# Riemannian geometry", Proc. TOBI Workshop IV, 2013.
