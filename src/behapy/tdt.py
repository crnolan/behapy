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


def load_session_tank_map(filename: str,
                          sourcedata_root: str = None) -> pd.DataFrame:
    """Loads a file mapping sessions to the tank files.

    Args:
        filename (str): path to the session mapping file
        sourcedata_root (str): path to which the entries in the sessions
                               file are relative; if None, the location of
                               the sessions file is used
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
    filename_parent = Path(filename).resolve().parent
    if sourcedata_root is None:
        sourcedata_root = filename_parent
    else:
        sourcedata_root = Path(sourcedata_root)
        if not sourcedata_root.is_absolute():
            sourcedata_root = (filename_parent / sourcedata_root).resolve()

    info.block = info.block.apply(lambda x: sourcedata_root / x)
    return info


def load_experiment_params(filename: str) -> Union[List[str], None]:
    with open(filename) as file:
        params = json.load(file)
    if 'event_names' not in params:
        params['event_names'] = None
        logging.warning('No event names found in experiment file, using '
                        'default names')
    if 'invert_events' not in params:
        params['invert_events'] = False
        logging.warning('Event polarity not defined, assuming 0 is off')
    return params


def get_epoch_df(epoch, event_names=None, invert_events=False):
    if event_names is None:
        event_names = [str(i) for i in range(8)]
    bits = [list('{:08b}'.format(x.astype(int))) for x in epoch.data]
    if invert_events:
        bits = ~np.array(bits).astype(bool)
        bits = bits.astype(int)
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


def convert_stream(df, block, root, event_names=None, invert_events=False):
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
                                 event_names=event_names,
                                 invert_events=invert_events)
        events_df.to_csv(fn, sep=',', na_rep='n/a')


def convert_block(df, root, event_names=None, invert_events=False):
    blocks = df.block.unique()
    if len(blocks) > 1:
        df.groupby('block').apply(convert_block, root, event_names,
                                  invert_events)
    else:
        logging.info('Opening block {}'.format(blocks[0]))
        try:
            block = read_block(blocks[0])
            df.apply(convert_stream, block=block, root=root,
                     event_names=event_names, invert_events=invert_events,
                     axis=1)
        except FileNotFoundError:
            logging.warn('Cannot open block at path {}'.format(blocks[0]))


def generate_session_map(root: str, files_from: str = None,
                         stream_map: dict = None,
                         epoc_map: dict = None) -> pd.DataFrame:
    """Generates a session map table.

    Given a root directory and optionally a relative path, generates a
    list of entries for data streams and events files from each TDT
    dataset.

    Args:
        root (str): path to the location that any files will be
                    referenced from
        files_from (str): path (relative to root) in which to search for
                          TDT files

    Returns:
        A pandas DataFrame with the following columns:
            block (str): path to the TDT block file, relative to root
            subject (str): subject ID
            session (str): session ID
            task (str): task ID
            run (str): run ID
            type (str): type of data (stream or epoc)
            tdt_id (str): TDT ID of the data
            channel (str): channel name
            label (str): label of the data
    """
    root = Path(root)
    if files_from is None:
        files_from = root
    else:
        files_from = root / files_from
    files_from = files_from.resolve()
    session_map = defaultdict(list)
    for tev_path in files_from.glob('**/*.tev'):
        block_path = tev_path.relative_to(root).parent
        block = read_block(tev_path.parent)
        streams = block.streams.keys() & stream_map.keys()
        logging.info(f'Ignoring streams {block.streams.keys() - streams}')
        epocs = block.epocs.keys() & epoc_map.keys()
        for stream in streams:
            session_map['block'].append(block_path)
            session_map['subject'].append(block.info.subject)
            session_map['session'].append(block.info.start_date.strftime('%Y%m%d_%H%M%S'))
            session_map['task'].append('task')
            session_map['run'].append('run')
            session_map['type'].append('stream')
            session_map['tdt_id'].append(stream)
            session_map['channel'].append(stream_map[stream][0])
            session_map['label'].append(stream_map[stream][1])
        for epoc in epocs:
            session_map['block'].append(block_path)
            session_map['subject'].append(block.info.subject)
            session_map['session'].append(block.info.start_date.strftime('%Y%m%d_%H%M%S'))
            session_map['task'].append('task')
            session_map['run'].append('run')
            session_map['type'].append('epoc')
            session_map['tdt_id'].append(epoc)
            session_map['channel'].append('events')
            session_map['label'].append(epoc_map[epoc])
    return pd.DataFrame(session_map)
