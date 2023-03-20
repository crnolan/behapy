import logging
from pathlib import Path


def session_root(base, sub, ses):
    return Path(base) / 'sub-{sub}/ses-{ses}'.format(sub=sub, ses=ses)


def events_path(base, sub, ses, task, run):
    root = session_root(base, sub, ses)
    template = 'sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.csv'
    return root / template.format(sub=sub, ses=ses, task=task, run=run)


def fibre_path(base, sub, ses, task, run, label, channel, ext):
    root = session_root(base, sub, ses)
    template = ('fp/sub-{sub}_ses-{ses}_task-{task}_run-{run}_label-{label}_'
                'channel-{channel}.{ext}')
    return root / template.format(sub=sub, ses=ses, task=task, run=run,
                                  label=label, channel=channel, ext=ext)

