import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sig
from tdt import read_block
import json
from collections import defaultdict
from .pathutils import fibre_path, events_path

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


def get_epoch_df(epoch):
    df = pd.DataFrame({
        'onset': epoch.onset,
        'value': epoch.data
    })
    df['duration'] = pd.NA
    return df.loc[:, ['onset', 'duration', 'value']]


def convert_stream(df, block, out_path):
    info_msg = ('Creating raw data for subject {}, session {}, task {}, '
                'run {}, channel {}, label {} from block {}, stream {}')
    info_msg = info_msg.format(df.subject, df.session, df.task, df.run,
                               df.channel, df.label, df.block, df.tdt_id)
    logging.info(info_msg)
    out_path = Path(out_path)
    if df.type == 'stream':
        data_fn = fibre_path(out_path, df.subject, df.session, df.task, df.run,
                            df.label, df.channel, '.npy')
        meta_fn = fibre_path(out_path, df.subject, df.session, df.task, df.run,
                            df.label, df.channel, '.json')
        data_fn.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            'fs': block.streams[df.tdt_id].fs,
            'start_time': block.streams[df.tdt_id].start_time,
        }
        np.save(data_fn, block.streams[df.tdt_id].data, allow_pickle=False)
        with open(meta_fn, 'w') as file:
            json.dump(meta, file, indent=4)
    elif df.type == 'epoc':
        fn = events_path(out_path, df.subject, df.session, df.task, df.run)
        fn.parent.mkdir(parents=True, exist_ok=True)
        events_df = get_epoch_df(block.epocs[df.tdt_id])
        events_df.to_csv(fn, sep=',', na_rep='n/a')


def convert_block(df, raw_path):
    blocks = df.block.unique()
    if len(blocks) > 1:
        df.groupby('block').apply(convert_block, raw_path)
    else:
        logging.info('Opening block {}'.format(blocks[0]))
        try:
            block = read_block(blocks[0])
            df.apply(convert_stream, block=block, out_path=raw_path, axis=1)
        except FileNotFoundError as error:
            logging.warn('Cannot open block at path {}'.format(blocks[0]))
