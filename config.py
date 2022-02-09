"""
========================
Study configuration file
========================

Configuration parameters and global variable values for the study.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
from os import path as op

from mne.channels import make_standard_montage

# eeg channel names and locations ---------------------------------------------
montage = make_standard_montage('biosemi64')

# relevant events in the paradigm ---------------------------------------------
event_ids = {'correct_target_button': 13,
             'correct_non_target_button': 12,
             'incorrect_target_button': 113,
             'incorrect_non_target_button': 112,
             'cue_0': 70,
             'cue_1': 71,
             'cue_2': 72,
             'cue_3': 73,
             'cue_4': 74,
             'cue_5': 75,
             'probe_0': 76,
             'probe_1': 77,
             'probe_2': 78,
             'probe_3': 79,
             'probe_4': 80,
             'probe_5': 81,
             'start_record': 127,
             'pause_record': 245}

# templates for filenames -----------------------------------------------------
# where to retrieve the sourcedata files from
sourcedata_fname = op.join('{sourcedata_path}', 'sub-{subject:03d}',
                           '{datatype}',
                           'sub-{subject:03d}_{task}_{datatype}{extension}')
