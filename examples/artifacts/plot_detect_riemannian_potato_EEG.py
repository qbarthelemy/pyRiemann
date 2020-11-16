"""
===============================================================================
Artifact Detection with Riemannian Potato
===============================================================================

Example of Riemannian Potato [1] applied on EEG time-series to detect artifacts
in online processing. It is computed only for two channels to display intuitive
visualizations.
"""
# Authors: Quentin Barthélemy & David Ojeda
#
# License: BSD (3-clause)

from functools import partial

import os
import numpy as np
from mne.datasets import sample                    # tested with mne 0.21
from mne.io import read_raw_fif
from mne import Epochs, make_fixed_length_events
from pyriemann.estimation import Covariances
from pyriemann.clustering import Potato
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle


###############################################################################


@partial(np.vectorize, excluded=['potato'])
def get_zscores(cov_00, cov_01, cov_11, potato):
    cov = np.array([[cov_00, cov_01], [cov_01, cov_11]])
    return potato.transform(cov[np.newaxis, ...])


def plot_potato_2D(axis, caxis, X, Y, P_zscore, P_center, P_colors, covs, *,
                   title, xlabel, ylabel, cblabel, alphas=None):
    axis.cla()
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    qcs = axis.contourf(X, Y, P_zscore, levels=20, vmin=P_zscore.min(),
                        vmax=P_zscore.max(), cmap='RdYlBu_r', alpha=0.5)
    axis.contour(X, Y, P_zscore, levels=[z_th], colors='r')
    if alphas is not None:
        colors = [to_rgb(c) for c in P_colors]
        P_colors = [(c[0], c[1], c[2], a) for c, a in zip(colors, alphas)]
    axis.scatter(covs[:, 0, 0], covs[:, 1, 1], c=P_colors)
    axis.scatter(P_center[0, 0], P_center[1, 1], c='k', s=100)
    if caxis:
        caxis.cla()
        cbar = fig.colorbar(qcs, cax=caxis)
    else:
        cbar = fig.colorbar(qcs, ax=axis)
    cbar.ax.set_ylabel(cblabel)


def plot_sig(axis, time, signal, label):
    axis.cla()
    axis.plot(time, signal, lw=0.75)
    axis.set_ylabel(label)
    axis.set_xlim([time[0], time[-1]])
    axis.set_ylim([-15, 15])
    axis.add_patch(
            Rectangle((time[0] - test_time_start, -14), width=duration,
                      height=28, edgecolor='r', facecolor='none'))


###############################################################################
# Load EEG data
# -------------

raw_fname = os.path.join(sample.data_path(), 'MEG', 'sample',
                         'sample_audvis_filt-0-40_raw.fif')
raw = read_raw_fif(raw_fname, preload=True, verbose=False)
sfreq = int(raw.info['sfreq'])


###############################################################################
# Offline processing of EEG data
# ------------------------------

# Apply common average reference on EEG channels
raw.pick_types(meg=False, eeg=True).apply_proj()
# Select two EEG channels for the example, preferably without artifact at the
# beginning to have a reliable calibration
ch_names = ['EEG 010', 'EEG 015']

# Apply band-pass filter between 1 and 35 Hz
raw.filter(1., 35., method='iir', picks=ch_names)

# Epoch time-series with a sliding window
duration = 2.5    # duration of epochs
interval = 0.25   # interval between epochs
events = make_fixed_length_events(raw, id=1, duration=interval)
epochs = Epochs(raw, events, tmin=0., tmax=duration, baseline=None,
                verbose=False)
epochs_data = 5e5 * epochs.get_data(picks=ch_names)

# Estimate spatial covariance matrices
covs = Covariances(estimator='lwf').transform(epochs_data)


###############################################################################
# Offline Calibration of Potato
# -----------------------------

z_th = 2.5       # z-score threshold
split_set = 40   # nb of matrices to train the potato

# Calibrate potato by unsupervised training on first matrices: compute a
# reference matrix, mean and standard deviation of distances to this reference.
train_set = range(split_set)
rp = Potato(metric='riemann', threshold=z_th).fit(covs[train_set])
rp_center = rp._mdm.covmeans_[0]
ep = Potato(metric='euclid', threshold=z_th).fit(covs[train_set])
ep_center = ep._mdm.covmeans_[0]

rp_labels = rp.predict(covs[train_set])
rp_colors = ['b' if l==1 else 'r' for l in rp_labels.tolist()]
ep_labels = ep.predict(covs[train_set])
ep_colors = ['b' if l==1 else 'r' for l in ep_labels.tolist()]

# 2D projection of the z-score map of the Riemannian potato, for 2x2 covariance
# matrices (in blue if clean, in red if artifacted) and their reference matrix
# (in black). The colormap defines the z-score and a chosen isocontour defines
# the potato. It reproduces Fig 1 of reference [2].

# Zscores in the horizontal 2D plane going through the reference
X, Y = np.meshgrid(np.linspace(1, 31, 100), np.linspace(1, 31, 100))
rp_zscore2D = get_zscores(X, np.full_like(X, rp_center[0, 1]), Y, potato=rp)
rp_zscore2D_m = np.ma.masked_where(~np.isfinite(rp_zscore2D), rp_zscore2D)
ep_zscore2D = get_zscores(X, np.full_like(X, ep_center[0, 1]), Y, potato=ep)

# Plot offline calibration
xlabel = 'Cov({},{})'.format(ch_names[0], ch_names[0])
ylabel = 'Cov({},{})'.format(ch_names[1], ch_names[1])

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
fig.suptitle('Offline calibration of potatoes', fontsize=16)
plot_potato_2D(axs[0], None, X, Y, rp_zscore2D_m, rp_center, rp_colors,
               covs[train_set], xlabel=xlabel, ylabel=ylabel,
               title='2D projection of Riemannian potato', 
               cblabel='Z-score of Riemannian distance to reference')
plot_potato_2D(axs[1], None, X, Y, ep_zscore2D, ep_center, ep_colors,
               covs[train_set], xlabel=xlabel, ylabel=ylabel,
               title='2D projection of Euclidean potato', 
               cblabel='Z-score of Euclidean distance to reference')
plt.show()


###############################################################################
# Online Artifact Detection with Potato
# -------------------------------------

test_cov_max = 250      # nb of matrices to visualize in this example
test_cov_visu = 30      # nb of matrices to display simultaneously
test_time_start = -2    # start time to display time-series
test_time_end = 5       # end time to display time-series

test_set = range(split_set, test_cov_max)
alphas = np.linspace(0, 1, test_cov_visu)
eeg_data = 3e5 * raw.get_data(picks=ch_names)
rp_colors = []
ep_colors = []

# Plot online detection (an interactive window is required)
fig = plt.figure(figsize=(12, 10))
fig.suptitle('Online artifact detection by potatoes', fontsize=16)
ax_sig0 = fig.add_axes([0.125,0.722857,0.775,0.157143])
ax_sig1 = fig.add_axes([0.125,0.518571,0.775,0.157143])
ax_rp = fig.add_axes([0.125,0.11,0.281818,0.361429])
ax_ep = fig.add_axes([0.547727,0.11,0.281818,0.361429])
cax_rp = fig.add_axes([0.424432,0.11,0.0143818,0.361429])
cax_ep = fig.add_axes([0.847159,0.11,0.0143818,0.361429])

# Detect artifacts/outliers on test set, with a loop to imitate an online
# acquisition, processing and artifact detection of EEG time-series.
# The potato is static: it is not updated when data is not artifacted.
for t in test_set:

    # Online artifact detection
    rp_label = rp.predict(covs[t][np.newaxis, ...])
    ep_label = ep.predict(covs[t][np.newaxis, ...])

    # Update data
    rp_colors.append('b' if rp_label==1 else 'r')
    ep_colors.append('b' if ep_label==1 else 'r')
    if len(rp_colors) > test_cov_visu:
        rp_colors.pop(0)
        ep_colors.pop(0)
    b = max(test_set[0], t - test_cov_visu + 1)
    time_start = t * interval + test_time_start
    time_end = t * interval + test_time_end
    time = np.linspace(time_start, time_end,
                       int((time_end - time_start) * sfreq),
                       endpoint=False)[np.newaxis, :]
    # Update plot
    plot_sig(ax_sig0, time.T,
             eeg_data[0, int(time_start * sfreq):int(time_end * sfreq)].T,
             label=ch_names[0])
    ax_sig0.set_xlabel('Time (s)')
    plot_sig(ax_sig1, time.T,
             eeg_data[1, int(time_start * sfreq):int(time_end * sfreq)].T,
             label=ch_names[1])
    ax_sig1.set_xticks([])
    plot_potato_2D(ax_rp, cax_rp, X, Y, rp_zscore2D_m, rp_center, rp_colors,
                   covs[b:t+1], xlabel=xlabel, ylabel=ylabel,
                   title='2D projection of Riemannian potato',
                   cblabel='Z-score of Riemannian distance to reference',
                   alphas=alphas[-len(rp_colors):])
    plot_potato_2D(ax_ep, cax_ep, X, Y, ep_zscore2D, ep_center, ep_colors,
                   covs[b:t+1], xlabel=xlabel, ylabel=ylabel,
                   title='2D projection of Euclidean potato',
                   cblabel='Z-score of Euclidean distance to reference',
                   alphas=alphas[-len(ep_colors):])
    plt.pause(0.5)
    plt.draw()


###############################################################################
# References
# ----------
# [1] A. Barachant, A. Andreev, M. Congedo, "The Riemannian Potato: an
# automatic and adaptive artifact detection method for online experiments using
# Riemannian geometry", Proc. TOBI Workshop IV, 2013.
# [2] Q. Barthélemy, L. Mayaud, D. Ojeda, M. Congedo, "The Riemannian potato
# field: a tool for online signal quality index of EEG", IEEE TNSRE, 2019.
