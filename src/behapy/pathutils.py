import logging
from pathlib import Path
import re
from collections import namedtuple


def get_session_root(base, sub, ses):
    return Path(base) / 'sub-{sub}/ses-{ses}'.format(sub=sub, ses=ses)


def get_events_path(root, sub, ses, task, run):
    base = get_session_root(Path(root) / 'rawdata', sub, ses)
    template = 'sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.csv'
    return base / template.format(sub=sub, ses=ses, task=task, run=run)


def get_raw_fibre_path(root, sub, ses, task, run, label, channel, ext):
    base = get_session_root(Path(root) / 'rawdata', sub, ses)
    template = ('fp/sub-{sub}_ses-{ses}_task-{task}_run-{run}_label-{label}_'
                'channel-{channel}.{ext}')
    return base / template.format(sub=sub, ses=ses, task=task, run=run,
                                  label=label, channel=channel, ext=ext)


def get_rejected_intervals_path(root, sub, ses, task, run, label):
    root = get_session_root(root / 'derivatives/preprocess', sub, ses)
    template = ('sub-{sub}_ses-{ses}_task-{task}_run-{run}_label-{label}_'
                'rejected-intervals.csv')
    return root / template.format(sub=sub, ses=ses, task=task, run=run,
                                  label=label)


def get_preprocessed_fibre_path(root, sub, ses, task, run, label, ext):
    root = get_session_root(root / 'derivatives/preprocess', sub, ses)
    template = ('fp/sub-{sub}_ses-{ses}_task-{task}_run-{run}_label-{label}'
                '.{ext}')
    return root / template.format(sub=sub, ses=ses, task=task, run=run,
                                  label=label, ext=ext)


def list_recordings(base, subject='*', session='*', task='*', run='*', label='*', ext='npy'):
    Recording = namedtuple("Recording", ["subject", "session", "task", "run", "label", "channel", "file_path"])

    # Set the pattern for the data files
    base = Path(base)
    pattern = (f'sub-{subject}/ses-{session}/fp/'
               f'sub-{subject}_ses-{session}_'
               f'task-{task}_run-{run}_label-{label}_channel-*.{ext}')
    # Search for files that match the pattern
    data_files = list(base.glob(str(pattern)))
    # Regex pattern to extract variables from the file names
    regex_pattern = (r"sub-([^_]+)_ses-([^_]+)_task-([^_]+)_run-([^_]+)_"
                     r"label-([^_]+)_channel-([^_]+)\.") + ext
    # Extract variables from the file names and store them in a list of namedtuples
    extracted_data = []
    for file_path in data_files:
        match = re.search(regex_pattern, str(file_path.name))
        if match:
            sub, ses, task, run, lab, channel = match.groups()
            data_file = Recording(sub, ses, task, run, lab, channel, file_path)
            extracted_data.append(data_file)

    return extracted_data


def list_raw(root, subject='*', session='*', task='*', run='*', label='*'):
    return list_recordings(Path(root) / 'rawdata', subject,
                           session, task, run, label, 'npy')


def list_preprocessed(root, subject='*', session='*', task='*', run='*',
                      label='*'):
    Recording = namedtuple("Recording", ["subject", "session", "task", "run", "label", "file_path"])

    # Set the pattern for the data files
    base = Path(root) / 'derivatives/preprocess'
    pattern = (f'sub-{subject}/ses-{session}/fp/'
               f'sub-{subject}_ses-{session}_'
               f'task-{task}_run-{run}_label-{label}.parquet')
    # Search for files that match the pattern
    data_files = list(base.glob(str(pattern)))
    # Regex pattern to extract variables from the file names
    regex_pattern = (r"sub-([^_]+)_ses-([^_]+)_task-([^_]+)_run-([^_]+)_"
                     r"label-([^_]+)\.parquet")
    # Extract variables from the file names and store them in a list of namedtuples
    extracted_data = []
    for file_path in data_files:
        match = re.search(regex_pattern, str(file_path.name))
        if match:
            sub, ses, task, run, lab = match.groups()
            data_file = Recording(sub, ses, task, run, lab, file_path)
            extracted_data.append(data_file)

    return extracted_data


def get_session_meta_path(root):
    return Path(root) / 'session-meta.csv'


def preprocess_config_path(root):
    return Path(root) / 'etc/preprocess.json'
