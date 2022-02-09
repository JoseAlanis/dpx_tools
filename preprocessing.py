# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause

import errno
import os

from pathlib import Path

import warnings

from mne.io import read_raw_bdf
from mne import find_events, events_from_annotations, Annotations

from mne_bids import BIDSPath, write_raw_bids

import pandas as pd

from config import montage, sourcedata_fname


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
