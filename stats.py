# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause
import numpy as np


def sliding_window_correlation(data, sampling_frequency=256.0, time_step=1.0):
    """
    Parameters
    ----------
    data : np.ndarray
        Should be a 2-dimensional array of shape channel x samples.
    sampling_frequency : float
        The sampling frequency of the data (in Hz). Defaults to 256.
    time_step : float | int
        Window length for analysis (in seconds). Defaults to 1.0.
    Returns
    -------
    channel_correlations : np.ndarray
        Numpy array containing the channel by channel correlations.
    """
    # get data dimensions
    n_channels, n_samples = data.shape

    # based on the sampling rate and window length (in seconds):
    # determine the number of data point that should be included
    # in the analysis
    samples_for_corr = int(time_step * sampling_frequency)

    # get the index of the samples that marks the start of each window
    # for correlation analysis
    sample_idx = np.arange(0, n_samples, samples_for_corr)

    # number of windows to use for analysis
    n_corr_steps = len(sample_idx)

    # reshape data to individual windows
    dat_windowed = data.reshape((n_channels, n_corr_steps, samples_for_corr))

    # placeholder for results
    channel_correlations = np.zeros((n_corr_steps, n_channels, n_channels))

    # compute correlations for windowed data
    for step in range(n_corr_steps):
        # get window data
        eeg_portion = np.squeeze(dat_windowed[:, step, :])
        # compute correlation coefficients
        corrs = np.corrcoef(eeg_portion)
        channel_correlations[step, :, :] = corrs

    return channel_correlations


# -- WIP --
# def noise_correlation:
    # noise_covs = mne.compute_covariance(
    #     epochs, tmax=0., method=('empirical', 'shrunk'),
    #     return_estimators=True, rank=None)
    #
    # noise_diag = np.diag(noise_covs[0].data)
    # np.sqrt(noise_diag)
    # noise_corr = np.linalg.inv(np.sqrt(np.diag(noise_diag))) @ noise_covs[
    #     0].data @ np.linalg.inv(np.sqrt(np.diag(noise_diag)))
