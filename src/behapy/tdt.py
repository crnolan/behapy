from typing import List, Union
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sig
from tdt import read_block
import json
from collections import defaultdict
from .pathutils import get_raw_fibre_path, get_events_path


def load_session_tank_map(filename: str) -> pd.DataFrame:
    """Loads a file mapping sessions to the tank files.

    Args:
        filename (str): path to the session mapping file
    """
    dtypes = {
        'block': str,
        'subject': str,
        'session': str,
        'task': str,
        'run': str,
        'type': str,
        'tdt_id': str,
        'channel': str,
        'label': str
    }
    info = pd.read_csv(filename, dtype=dtypes)
    resolved_path = Path(filename).resolve().parent
    info.block = info.block.apply(lambda x: resolved_path / x)
    return info


def load_event_names(filename: str) -> Union[List[str], None]:
    with open(filename) as file:
        events_dict = json.load(file)
    try:
        event_names = events_dict['event_names']
    except KeyError:
        logging.warning('No event names found in events file, using '
                        'default names')
        return None
    return event_names


def get_epoch_df(epoch, event_names=None):
    if event_names is None:
        event_names = [str(i) for i in range(8)]
    bits = [list('{:08b}'.format(x.astype(int))) for x in epoch.data]
    bits = np.array(bits).astype(int)
    # Add a row of zeros to ensure all events have offsets
    bits = np.concatenate([bits, np.zeros_like(bits[:1, :])])
    onsets = np.concatenate([epoch.onset, epoch.onset[-1:]])
    oo = np.concatenate([bits[:1, :], np.diff(bits, axis=0)])
    event_onsets = [onsets[x == 1] for x in oo.T]
    event_offsets = [onsets[x == -1] for x in oo.T]
    event_id = [[event_names[i]] * len(x)
                for i, x in zip(np.flip(np.arange(8)), event_onsets)]
    # Now make a DataFrame, adding the event ID as a column
    df = pd.DataFrame({
        'onset': np.concatenate(event_onsets),
        'offset': np.concatenate(event_offsets),
        'event_id': np.concatenate(event_id)
    })
    df['duration'] = df.offset - df.onset
    df = df.loc[:, ['onset', 'duration', 'event_id']].sort_values(by='onset')
    return df.set_index('onset')


def convert_stream(df, block, root, event_names=None):
    info_msg = ('Creating raw data for subject {}, session {}, task {}, '
                'run {}, channel {}, label {} from block {}, stream {}')
    info_msg = info_msg.format(df.subject, df.session, df.task, df.run,
                               df.channel, df.label, df.block, df.tdt_id)
    logging.info(info_msg)
    root = Path(root)
    if df.type == 'stream':
        data_fn = get_raw_fibre_path(root, df.subject, df.session, df.task,
                                     df.run, df.label, df.channel, 'npy')
        meta_fn = get_raw_fibre_path(root, df.subject, df.session, df.task,
                                     df.run, df.label, df.channel, 'json')
        data_fn.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            'fs': block.streams[df.tdt_id].fs,
            'start_time': block.streams[df.tdt_id].start_time,
        }
        np.save(data_fn, block.streams[df.tdt_id].data, allow_pickle=False)
        with open(meta_fn, 'w') as file:
            json.dump(meta, file, indent=4)
    elif df.type == 'epoc':
        fn = get_events_path(root, df.subject, df.session, df.task, df.run)
        fn.parent.mkdir(parents=True, exist_ok=True)
        events_df = get_epoch_df(block.epocs[df.tdt_id],
                                 event_names=event_names)
        events_df.to_csv(fn, sep=',', na_rep='n/a')


def convert_block(df, root, event_names=None):
    blocks = df.block.unique()
    if len(blocks) > 1:
        df.groupby('block').apply(convert_block, root, event_names)
    else:
        logging.info('Opening block {}'.format(blocks[0]))
        try:
            block = read_block(blocks[0])
            df.apply(convert_stream, block=block, root=root,
                     event_names=event_names, axis=1)
        except FileNotFoundError:
            logging.warn('Cannot open block at path {}'.format(blocks[0]))
