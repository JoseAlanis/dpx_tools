
from mne_bids import BIDSPath, read_raw_bids

from preprocessing import sourcedata_to_bids, find_bad_channels

import numpy as np

from pyprep.prep_pipeline import PrepPipeline

from config import event_ids, montage

from pyprep.find_noisy_channels import NoisyChannels

sourcedata_path = '../data/sourcedata'
bids_path = '../data/dpx_bids'

raw, _ = sourcedata_to_bids(sourcedata_path=sourcedata_path,
                            subject=1, task='dpx', datatype='eeg',
                            extension='.bdf',
                            bids_path=bids_path,
                            events_channel='Status', min_duration=0.001,
                            event_id=event_ids, include_demographics=True,
                            node=True)

del raw

# read in the data

# create bids path
raw_fname = BIDSPath(root=bids_path,
                     subject='001',
                     task='dpx',
                     datatype='eeg',
                     extension='.bdf')
# get the data
raw = read_raw_bids(raw_fname)
raw.load_data()


line_noise = np.arange(50, raw.info['sfreq'] / 2, 50)


# copy the data for preprocessing
raw_copy = raw.copy()

# get raw dat specs
sfreq = raw_copy.info['sfreq']
all_channels = raw_copy.ch_names

raw_signal = raw.get_data()


# only keep eeg channels
raw_eeg = raw.copy().pick_types(eeg=True)
raw_eeg.filter(l_freq=1.0, h_freq=50.0)

bad_channels = find_bad_channels(raw_eeg,
                                 method='correlation')['correlation']

from stats import sliding_window_correlation

mne.filter.filter_data(
    raw.copy().get_data(picks='eeg'),
    sfreq=raw.info['sfreq'],
    l_freq=1.0,
    h_freq=None)



raw_eeg = raw.copy().pick_types(eeg=True)

noisy_detector = NoisyChannels(
    raw.copy().pick_types(eeg=True),
    do_detrend=True,
    random_state=None,
    matlab_strict=False,
)

noisy_detector.find_all_bads(ransac=False)

bad_corr = find_bad_channels(raw,
                             picks='eeg',
                             detrend=True,
                             r_threshold=0.4,
                             percent_threshold=0.01,
                             time_step=1.0,
                             method='correlation')['correlation']

bad_dev = find_bad_channels(raw,
                            picks='eeg',
                            detrend=True,
                            method='deviation')['deviation']

bad_noise = find_bad_channels(raw,
                              picks='eeg',
                              detrend=True,
                              method='high_frequency_noise')['high_frequency_noise']






raw.load_data()

sample_rate = raw.info['sfreq']

prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": np.arange(50, sample_rate / 2, 50),
}

# Make a copy of the data
raw_copy = raw.copy()

prep = PrepPipeline(raw_copy, prep_params, montage)
prep.fit()



# def extract_data_blocks(raw, events_ids=None, limits=None):
#     """Extract periods of data according to certain inclusion criteria.
#     The continuous structure of the data is kept by concatenating the extracted
#     data chunks.
#     Data inclusion criteria are defined in 'limits' (e.g., certain period of time
#     following an event, certain sequence of events, etc.). See notes for details.
#
#     Parameters
#     ----------
#     raw : mne.io.Raw object
#        Instance of Raw containing continuous data (e.g., EEG-data)
#     events_ids : None | dict
#     limits: None | float | int | dict
#        Method that should be used for extracting periods of data
#
#     Returns
#     -------
#     raw : instance of Raw
#        The result of the concatenation (first Raw instance passed in).
#     """
#     from mne import find_events, events_from_annotations, concatenate_raws
#
#     # Step 0: Check if limits were provided
#     if limits is None:
#         raise ValueError('Whoops, no limits were provided, execution stopped.')
#
#     # 3) Create events info
#     # extract events
#     events = find_events(raw,
#                          stim_channel='Status',
#                          output='onset',
#                          min_duration=0.001)
#
#     # Step 1:
#     # relevant events
#     ids = {'70': 1,
#            '71': 2,
#            '72': 3,
#            '73': 4,
#            '74': 5,
#            '75': 6,
#            '76': 7
#            }
#
#     # extract events
#     events = events_from_annotations(raw, event_id=ids)
