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
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


###############################################################################


@partial(np.vectorize, excluded=['potato'])
def get_zscores(cov_00, cov_01, cov_11, potato):
    cov = np.array([[cov_00, cov_01], [cov_01, cov_11]])
    return potato.transform(cov[np.newaxis, ...])


def plot_potato_2D(axis, X, Y, p_zscore, p_center, p_colors, covs, *,
                   title, xlabel, ylabel, cblabel):
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    qcs = axis.contourf(X, Y, p_zscore, levels=20, vmin=p_zscore.min(),
        vmax=p_zscore.max(), cmap='RdYlBu_r', alpha=0.5)
    axis.contour(X, Y, p_zscore, levels=[z_th], colors='k')
    axis.scatter(covs[:, 0, 0], covs[:, 1, 1], c=p_colors)
    axis.scatter(p_center[0, 0], p_center[1, 1], c='k', s=100)
    cbar = fig.colorbar(qcs, ax=axis)
    cbar.ax.set_ylabel(cblabel)


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
interval = 0.2    # interval between epochs
epochs = Epochs(raw, make_fixed_length_events(raw, id=1, duration=interval),
    tmin=0., tmax=duration, baseline=None, verbose=False)
epochs_data = 5e5 * epochs.get_data(picks=ch_names)

# Estimate spatial covariance matrices
covs = Covariances(estimator='lwf').transform(epochs_data)


###############################################################################
# Calibration of Potato
# ---------------------

z_th = 2.5       # z-score threshold
t = 30           # nb of matrices to train the potato

# Calibrate potato by unsupervised training on first matrices: compute a
# reference matrix, mean and standard deviation of distances to this reference.
train_set = range(t)
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

# Plot calibration
xlabel = 'Cov({},{})'.format(ch_names[0], ch_names[0])
ylabel = 'Cov({},{})'.format(ch_names[1], ch_names[1])

fig, axs = plt.subplots(figsize=(12, 5), nrows=1, ncols=2)
fig.suptitle('Offline calibration of potatoes', fontsize=16)
plot_potato_2D(axs[0], X, Y, rp_zscore2D_m, rp_center, rp_colors,
    covs[train_set], xlabel=xlabel, ylabel=ylabel,
    title='2D projection of Riemannian potato',
    cblabel='Z-score of Riemannian distance to reference')
plot_potato_2D(axs[1], X, Y, ep_zscore2D, ep_center, ep_colors,
    covs[train_set], xlabel=xlabel, ylabel=ylabel,
    title='2D projection of Euclidean potato',
    cblabel='Z-score of Euclidean distance to reference')
plt.show()


###############################################################################
# Online Artifact Detection with Potato
# -------------------------------------

# Detect artifacts/outliers on test set, with an animation to imitate an online
# acquisition, processing and artifact detection of EEG time-series.
# The potato is static: it is not updated when EEG is not artifacted, damaging
# its efficiency over time.

test_covs_max = 300     # nb of matrices to visualize in this example
test_covs_visu = 30     # nb of matrices to display simultaneously
test_time_start = -2    # start time to display signal
test_time_end = 5       # end time to display signal

time_start = t * interval + test_time_start
time_end = t * interval + test_time_end
time = np.linspace(time_start, time_end, int((time_end - time_start) * sfreq),
    endpoint=False)
eeg_data = 3e5 * raw.get_data(picks=ch_names)
sig = eeg_data[:, int(time_start * sfreq):int(time_end * sfreq)]
covs_visu = np.empty([0, 2, 2])
rp_colors, ep_colors = [], []
alphas = np.linspace(0, 1, test_covs_visu)

# Prepare animation for online detection
fig = plt.figure(figsize=(12, 10), constrained_layout=False)
fig.suptitle('Online artifact detection by potatoes', fontsize=16)
gs = fig.add_gridspec(nrows=4, ncols=40, top=0.90, hspace=0.3, wspace=1.0)
ax_sig0 = fig.add_subplot(gs[0, :], xlabel='Time (s)', ylabel=ch_names[0],
    xlim=(time[0], time[-1]), ylim=(-15, 15))
pl_sig0, = ax_sig0.plot(time, sig[0], lw=0.75)
ax_sig0.axhspan(-14, 14, edgecolor='r', facecolor='none',
    xmin=-test_time_start / (test_time_end - test_time_start),
    xmax=(duration - test_time_start) / (test_time_end - test_time_start))
ax_sig1 = fig.add_subplot(gs[1, :], ylabel=ch_names[1],
    xlim=(time[0], time[-1]), ylim=(-15, 15))
pl_sig1, = ax_sig1.plot(time, sig[1], lw=0.75)
ax_sig1.set_xticks([])
ax_sig1.axhspan(-14, 14, edgecolor='r', facecolor='none',
    xmin=-test_time_start / (test_time_end - test_time_start),
    xmax=(duration - test_time_start) / (test_time_end - test_time_start))
ax_rp = fig.add_subplot(gs[2:4, 0:15], xlabel=xlabel, ylabel=ylabel,
    title='2D projection of Riemannian potato')
qcs_rp = ax_rp.contourf(X, Y, rp_zscore2D_m, levels=20, cmap='RdYlBu_r',
    vmin=rp_zscore2D_m.min(), vmax=rp_zscore2D_m.max(), alpha=0.5)
ax_rp.contour(X, Y, rp_zscore2D_m, levels=[z_th], colors='k')
sc_rp = ax_rp.scatter([], [], c=rp_colors)
ax_rp.scatter(rp_center[0, 0], rp_center[1, 1], c='k', s=100)
cax_rp = fig.add_subplot(gs[2:4, 15])
fig.colorbar(qcs_rp, cax=cax_rp)
cax_rp.set_ylabel('Z-score of Riemannian distance to reference')
ax_ep = fig.add_subplot(gs[2:4, 21:36], xlabel=xlabel, ylabel=ylabel,
    title='2D projection of Euclidean potato')
qcs_ep = ax_ep.contourf(X, Y, ep_zscore2D, levels=20, cmap='RdYlBu_r',
    vmin=ep_zscore2D.min(), vmax=ep_zscore2D.max(), alpha=0.5)
ax_ep.contour(X, Y, ep_zscore2D, levels=[z_th], colors='k')
sc_ep = ax_ep.scatter([], [], c=ep_colors)
ax_ep.scatter(ep_center[0, 0], ep_center[1, 1], c='k', s=100)
cax_ep = fig.add_subplot(gs[2:4, 36])
fig.colorbar(qcs_ep, cax=cax_ep)
cax_ep.set_ylabel('Z-score of Euclidean distance to reference')

# Plot online detection (an interactive window is required)
def online_update(self):
    global t, time, sig, covs_visu, rp_colors, ep_colors

    # Online artifact detection
    rp_label = rp.predict(covs[t][np.newaxis, ...])
    ep_label = ep.predict(covs[t][np.newaxis, ...])

    # Update data
    time_start = t * interval + test_time_end
    time_end = (t + 1) * interval + test_time_end
    time_ = np.linspace(time_start, time_end, int(interval * sfreq),
        endpoint=False)
    time = np.r_[time[int(interval * sfreq):], time_]
    sig = np.hstack((sig[:, int(interval * sfreq):],
        eeg_data[:, int(time_start * sfreq):int(time_end * sfreq)]))
    covs_visu = np.vstack((covs_visu, covs[t][np.newaxis, ...]))
    rp_colors.append('b' if rp_label==1 else 'r')
    ep_colors.append('b' if ep_label==1 else 'r')
    if len(covs_visu) > test_covs_visu:
        covs_visu = covs_visu[1:]
        rp_colors.pop(0)
        ep_colors.pop(0)
    colors = [to_rgb(c) for c in rp_colors]
    rp_colors_ = [(c[0], c[1], c[2], a)
        for c, a in zip(colors, alphas[-len(rp_colors):])]
    colors = [to_rgb(c) for c in ep_colors]
    ep_colors_ = [(c[0], c[1], c[2], a)
        for c, a in zip(colors, alphas[-len(ep_colors):])]
    t += 1

    # Update plot
    pl_sig0.set_data(time, sig[0])
    pl_sig0.axes.set_xlim(time[0], time[-1])
    pl_sig1.set_data(time, sig[1])
    pl_sig1.axes.set_xlim(time[0], time[-1])
    sc_rp.set_offsets(np.c_[covs_visu[:, 0, 0], covs_visu[:, 1, 1]])
    sc_rp.set_color(rp_colors_)
    sc_ep.set_offsets(np.c_[covs_visu[:, 0, 0], covs_visu[:, 1, 1]])
    sc_ep.set_color(ep_colors_)

    return pl_sig0, pl_sig1, sc_rp, sc_ep

potato = FuncAnimation(fig, online_update, frames=test_covs_max,
    interval=interval, blit=False, repeat=False)
plt.show()


###############################################################################
# References
# ----------
# [1] A. Barachant, A. Andreev, M. Congedo, "The Riemannian Potato: an
# automatic and adaptive artifact detection method for online experiments using
# Riemannian geometry", Proc. TOBI Workshop IV, 2013.
#
# [2] Q. Barthélemy, L. Mayaud, D. Ojeda, M. Congedo, "The Riemannian potato
# field: a tool for online signal quality index of EEG", IEEE TNSRE, 2019.
