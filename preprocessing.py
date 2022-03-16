# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause
import errno
import os

from pathlib import Path

import warnings

import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation as mad

from mne.io.base import BaseRaw
from mne.io import read_raw_bdf
from mne.filter import filter_data, notch_filter
from mne.time_frequency import psd_array_welch
from mne import find_events, events_from_annotations, Annotations

from mne_bids import BIDSPath, write_raw_bids

from config import montage, sourcedata_fname
from stats import sliding_window_correlation


def robust_z_score(values):
    values = np.array(values)
    robust_z = 0.67449 * (values - np.nanmedian(values)) / mad(values)
    return robust_z


def sourcedata_to_bids(sourcedata_path,
                       subject, task, datatype, extension,
                       bids_path=None, events_channel=None, min_duration=0.0,
                       event_id=None, include_demographics=True, node=False):
    """
    Parameters
    ----------
    sourcedata_path : path-like
        The path to the root directory of the dataset storage location.
        The sourcedata/ directory should be structured according to the
        `BIDS`_ standard for electroencephalography
        (see :footcite:`pernet2019`).
    subject : int | str
        The subject ID. Corresponds to “sub”.
    task : str
        The experimental task. Corresponds to “task”.
    datatype : str
        Type of data to look for
    extension : str
        The extension of the filename (e.g., ".bdf").
    bids_path : path-like | None
        The root directory of the BIDS dataset. If None, `sourcedata_path` is
        used as `bids_path`
    events_channel : None | str
        The name of the channel to use for identifying events in the data
        (e.g., usually 'Status' for .bdf). Alternatively, one can pass
       `events_channel="Annotations"` if events are to be extracted from the
        file's `Annotations` (see Notes).
    min_duration : float
        The minimum duration of a change in the events channel required to
        consider it as an event (in seconds). Only used if `events_channel` is
        provided.
    event_id : dict | None
        Can be:

        - **dict**: map descriptions (keys) to integer event codes (values).
          Only the descriptions present will be mapped, others will be ignored.
        - **None**: Map descriptions to unique integer values based on their
          ``sorted`` order.
    include_demographics : bool
        Whether `demographics/` directory is provided for each subject in the
        sourcedata
    node : bool
        Whether to return the data structure for further processing.

    Returns
    -------
    bids_path : Path
        The path of the created data file.

    Notes
    -----
    **Data structure**
    The required structure of the `sourcedata/` directory is::

        |sourcedata/
        |--- sub-01/
        |------ eeg/
        |--------- sub-01_dpx_eeg.bdf
        |--- sub-02/
        |------ eeg/
        |--------- sub-02_dpx_eeg.bdf
        |--- sub-03/
        |------ eeg/
        ...

    Other data modalities can be included as follows::

        |sourcedata/
        |--- sub-01/
        |------ demographics/
        |--------- sub-01_dpx_demographics.tsv
        |------ eeg/
        |--------- sub-01_dpx_eeg.bdf
        ...

    **Annotations**
    Annotations are added to an instance of :class:`mne.io.Raw` as the attribute
    :attr:`raw.annotations <mne.io.Raw.annotations>`
    (see https://mne.tools/stable/generated/mne.Annotations.html).
    """
    if bids_path is None:
        bids_path = sourcedata_path

    # check if directory exists
    if not os.path.isdir(sourcedata_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), sourcedata_path)

    # get path for file in question
    file_name = sourcedata_fname.format(sourcedata_path=sourcedata_path,
                                        subject=subject,
                                        task=task,
                                        datatype=datatype,
                                        extension=extension)

    # 1) import the data ------------------------------------------------------
    raw = read_raw_bdf(file_name, preload=False)

    # 2) get data specs, add correct channel types and EGG-montage ------------
    # sampling rate
    sfreq = raw.info['sfreq']
    # channels names
    channels = raw.info['ch_names']

    # identify channel types based on matching names in montage
    types = []
    for channel in channels:
        if channel in montage.ch_names:
            types.append('eeg')
        elif channel.startswith('EOG') | channel.startswith('EXG'):
            types.append('eog')
        else:
            types.append('stim')

    # add channel types and eeg-montage
    raw.set_channel_types(
        {channel: typ for channel, typ in zip(channels, types)})
    raw.set_montage(montage)

    # 3) add subject information to `raw.info` --------------------------------
    if include_demographics:
        # compute approx. date of birth
        # get measurement date from dataset info
        date_of_record = raw.info['meas_date']
        # convert to date format
        date = date_of_record.strftime('%Y-%m-%d')

        # here, we compute only and approximate of the subject's birthday
        # this is to keep the date anonymous (at least to some degree)
        demographics = sourcedata_fname.format(sourcedata_path=sourcedata_path,
                                               subject=subject,
                                               task=task,
                                               datatype='demographics',
                                               extension='.tsv')
        demo = pd.read_csv(demographics, sep='\t', header=0)
        age = demo[demo.subject_id == 'sub-' + str(subject).rjust(3, '0')].age
        sex = demo[demo.subject_id == 'sub-' + str(subject).rjust(3, '0')].sex

        year_of_birth = int(date.split('-')[0]) - int(age)
        approx_birthday = (year_of_birth,
                           int(date[5:].split('-')[0]),
                           int(date[5:].split('-')[1]))

        # add modified subject info to dataset
        raw.info['subject_info'] = dict(id=subject,
                                        sex=int(sex),
                                        birthday=approx_birthday)

        # frequency of power line
        raw.info['line_freq'] = 50.0

    # 4) add events as annotations --------------------------------------------
    if events_channel is None and event_id is None:
        pass
    elif events_channel is None and event_id is not None:
        warnings.warn('Ignoring `event_id` as no `events_channel` was '
                      'provided.')
    else:
        if events_channel is not None and events_channel in raw.ch_names:
            # extract events from events channel
            events = find_events(raw,
                                 stim_channel=events_channel,
                                 output='onset',
                                 min_duration=min_duration)
        elif events_channel == 'Annotations':
            # check if a valid event_ids were provided
            if not isinstance(event_id, dict) or event_id is None:
                raise ValueError(
                    "Invalid `event_id` structure provided. `event_id` must be"
                    "a `dict` or None")
            # extract events from the file's annotations
            events = events_from_annotations(raw,
                                             event_id=event_id,
                                             regexp=None)
        else:
            raise ValueError("`events_channel` must be one of the channels in "
                             "the dataset (i.e., one of `raw.ch_names`), "
                             "'Annotations', or None. Stopping execution.")

        # events to data frame
        events = pd.DataFrame(events,
                              columns=['onset', 'duration', 'description'])
        # onset to seconds
        events['onset_in_s'] = events['onset'] / sfreq
        # sort by onset
        events = events.sort_values(by=['onset_in_s'])

        if event_id is not None:
            # only keep relevant events
            events = events.loc[events['description'].isin(event_id.values())]

        # crate annotations object
        annotations = Annotations(events['onset_in_s'],
                                  events['duration'],
                                  events['description'])
        # apply to raw data
        raw.set_annotations(annotations)

    # 5) save raw data to a BIDS-compliant folder structure -------------------
    output_path = BIDSPath(subject=f'{subject:03}',
                           task=task,
                           datatype=datatype,
                           root=bids_path)

    # include events if provided
    write_raw_bids(raw,
                   output_path,
                   overwrite=True)

    if node is False:
        return output_path
    else:
        return raw, output_path


# main function which implements different methods
def find_bad_channels(raw, picks='eeg', sfreq=None, channels=None,
                      detrend=False, method='correlation',
                      mad_threshold=1e-15, std_threshold=1e-15,
                      r_threshold=0.4, percent_threshold=0.01, time_step=1.0,
                      high_frequency_threshold=50.0,
                      return_z_scores=False,
                      n_jobs=1):
    """
    Parameters
    ----------
    raw : mne.io.Raw | np.ndarray
        An instance of mne.io.Raw containing the raw data where bad (i.e.,
        noisy channels are presumed. Alternatively, data can be supplied as a
        2-D array (channels x samples). In the latter case, `sampling_freq` and
        a list channel names must be provided.
    picks : list | 'eeg'
        A list of channel names to be included in the analysis,
        Can be a str 'eeg' to use all EEG channels. Defaults to 'eeg'.
    sfreq : float | None
    channels : list | str
    detrend : bool
    method : 'str
    mad_threshold : float
    std_threshold : float
    r_threshold : float
    percent_threshold : float
    time_step : int | float
    high_frequency_threshold : float
        In Herz. Defaults to 50.
    return_z_scores : bool
    n_jobs : int
    """
    # arguments to be passed to pick_types
    kwargs = {pick: True for pick in [picks]}

    # check that tha input data can be handled by the function
    if isinstance(raw, BaseRaw):
        # only keep data from desired channels
        inst = raw.copy().pick_types(**kwargs)
        data = inst.get_data()
        channels = inst.ch_names
        sfreq = inst.info['sfreq']

    elif isinstance(raw, np.ndarray):
        if channels is None:
            raise ValueError('If "raw" is not an instance of mne.io.Raw, '
                             'a list of channel names must be provided')
        if sfreq is None:
            raise ValueError('If "raw" is not an instance of mne.io.Raw, the '
                             'sampling frequency for the data must be provided')
        data = raw
    else:
        raise ValueError('inst must be an instance of BaseRaw or a numpy array')

    # remove slow drifts if specified
    if detrend:
        dat = filter_data(data, sfreq=sfreq, l_freq=1.0, h_freq=None)
    else:
        dat = data

    # save shape of data
    n_channels, n_samples = dat.shape
    if n_channels != len(channels):
        raise ValueError("Number and channels and data dimensions don't match")

    # make sure method arguments are in a list
    if not isinstance(method, list):
        method = [method]

    # placeholder for results
    bad_channels = dict()

    # 1) find channels with zero or near zero activity
    if 'flat' in method:
        # compute estimates of channel activity
        mad_flats = mad(dat, scale=1, axis=1) < mad_threshold
        std_flats = np.std(dat, axis=1) < std_threshold

        # flat channels identified
        flats = np.argwhere(np.logical_or(mad_flats, std_flats))
        flats = np.asarray([channels[int(flat)] for flat in flats])

        # warn user if too many channels were identified as flat
        if flats.shape[0] > (n_channels / 2):
            warnings.warn('Too many channels have been identified as "flat"! '
                          'Make sure the input values in "inst" are provided '
                          'on a volt scale. '
                          'Otherwise try choosing another (meaningful) '
                          'threshold for identification.')

        bad_channels.update(flat=flats)

    # 3) find bad channels by deviation (high variability in amplitude)
    if 'deviation' in method:

        # mean absolute deviation (MAD) scores for each channel
        mad_scores = [mad(dat[i, :]) for i in range(n_channels)]

        # compute robust z-scores for each channel
        rz_scores = robust_z_score(mad_scores)

        # channels identified by deviation criterion
        bad_deviation = [channels[i]
                         for i in np.where(np.abs(rz_scores) >= 5.0)[0]]

        bad_channels.update(deviation=np.asarray(bad_deviation))

        if return_z_scores:
            bad_channels.update(deviation_z_scores=rz_scores)

    # 3) find channels with low correlation to other channels
    if 'correlation' in method:

        # check that sampling frequency argument was provided
        if sfreq is None:
            raise ValueError('If "inst" is not an instance of BaseRaw a '
                             'sampling frequency must be provided. Usually '
                             'the sampling frequency of the EEG recording in'
                             'question.')

        # compute channel to channel correlations
        ch_corrs = sliding_window_correlation(dat, time_step=time_step,
                                              sampling_frequency=sfreq)
        # placeholder for results
        max_r = np.ones((ch_corrs.shape[0], ch_corrs.shape[1]))

        # loop through individual windows, extract the absolute correlations,
        # and estimate the maximum correlation (defined as the 98th
        # percentile of the channel-by-channel correlations)
        for step in range(max_r.shape[0]):
            # set diagonal to zero
            corr_no_diag = np.subtract(ch_corrs[step, :, :],
                                       np.diag(np.diag(ch_corrs[step, :, :])))
            # get absolute correlations
            abs_corr = np.abs(corr_no_diag)
            # get 98th percentile
            max_r[step, :] = np.percentile(abs_corr, 98, axis=0,
                                           method='median_unbiased')

        # check which channels correlate badly with the other channels (i.e.,
        # are below correlation threshold) in a certain fraction of windows
        # (bad_time_threshold)
        thresholded_correlations = max_r < r_threshold
        frac_bad_corr_windows = np.mean(thresholded_correlations, axis=0)

        # find the corresponding channel names and return
        bad_idxs = np.argwhere(frac_bad_corr_windows > percent_threshold)
        uncorrelated_channels = [channels[int(bad)] for bad in bad_idxs]

        bad_channels.update(correlation=np.asarray(uncorrelated_channels))

    if 'high_frequency_noise' in method:
        if sfreq < 100.0:
            warnings.warn('The sampling rate is to low to noise with a '
                          'frequency > 50.0 Hz. High-frequency noise detection'
                          'skipped.')
            pass

        # compute frequecy power
        asds, freqs = psd_array_welch(dat, sfreq=sfreq, n_jobs=n_jobs)
        asds = np.sqrt(asds) * 1e6

        # compute noise ratios
        noise_ratios = []
        for i in range(asds.shape[0]):
            high_f = asds[i, freqs >= high_frequency_threshold].sum()
            low_f = asds[i, freqs < high_frequency_threshold].sum()
            noise_ratio = high_f / low_f
            noise_ratios.append(noise_ratio)

        # compute robust z-scores
        rz_scores_hf = robust_z_score(noise_ratios)

        # channels identified by high frequency criterion
        bad_freq = [channels[i]
                    for i in np.where(np.abs(rz_scores_hf) >= 5.0)[0]]

        bad_channels.update(high_frequency_noise=np.asarray(bad_freq))

    return bad_channels


def robust_reference(raw, line_noise=None, n_jobs=1):
    """
    raw : mne.io.Raw | np.ndarray
        An instance of mne.io.Raw containing the raw data where bad (i.e.,
        noisy channels are presumed. Alternatively, data can be supplied as a
        2-D array (channels x samples). In the latter case, `sampling_freq` and
        a list channel names must be provided
    line_noise : float | list

    :return:
    """

    # make a copy of the data, only keeping EEG channels
    raw_copy = raw.copy().pick_types(eeg=True)

    if line_noise is not None:
        raw_no_line_noise = raw_copy.notch_filter(
            freqs=line_noise,
            picks=['eeg'],
            n_jobs='cuda')




