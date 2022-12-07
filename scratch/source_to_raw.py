# %%
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sig
from tdt import read_block
import json
from collections import defaultdict

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# %%
def load_session_info(filename):
    dtypes = {
        'file': str,
        'subject': str,
        'session': str,
        'task': str,
        'run': str,
        'type': str,
        'channel': str,
        'name': str
    }
    index_names = ['subject', 'session', 'task', 'run', 'type', 'channel']
    info = pd.read_csv(filename, dtype=dtypes)
    return info


def convert_stream(df, block, raw_path):
    info_msg = ('Creating raw data for subject {}, session {}, task {}, '
                'run {}, channel {} from file {}, stream {}')
    info_msg = info_msg.format(df.subject, df.session, df.task, df.run,
                               df.channel, df.file, df.name)
    logging.info(info_msg)
    session_root = RAWPATH / 'sub-{:02d}/ses-{:02d}/'.format(
        df.subject, df.session)
    fn_fragment = 'sub-{:02d}_ses-{:02d}_task-{}'.format(
        df.subject, df.session, df.task
    )

    # Create folders if they don't yet exist
    (raw_path / 'fp').mkdir(parents=True, exist_ok=True)
    if df.type == 'stream':
        fn_stem = raw_path / 'fp' / (fn_fragment + '_' + df.channel)
        data_fn = str(fn_stem) + '.npy'
        meta_fn = str(fn_stem) + '.json'
        meta = {
            'fs': block.streams[df.name].fs,
            'start_time': block.streams[df.name].start_time,
        }
        np.save(data_fn, block.streams[df.name].data, allow_pickle=False)
        with open(meta_fn, 'w') as file:
            json.dump(meta, file, indent=4)
    elif df.type == 'epoc':
        fn = raw_path / (fn_fragment + '_' + df.channel + '.csv')
        events_df = get_epoch_df(block.epocs[df.name])
        events_df.to_csv(fn, sep=',', na_rep='n/a')


def convert_block(df, raw_path):
    fns = df.file.unique()
    if len(fns) > 1:
        df.groupby('file').apply(convert_block, raw_path)
    else:
        logging.info('Opening file {}'.format(fns[0]))
        block = read_block(SOURCEPATH / fns[0])
        df.apply(convert_stream, block=block, raw_path=raw_path, axis=1)


# %%
PROJECTPATH = Path('..')
SOURCEPATH = PROJECTPATH / 'sourcedata/'
RAWPATH = PROJECTPATH / 'rawdata/'


session_info = pd.read_csv(SOURCEPATH / 'session-map.csv')
convert_block(session_info, RAWPATH)

# %%

def get_epoch_df(epoch):
    df = pd.DataFrame({
        'onset': epoch.onset,
        'value': epoch.data
    })
    df['duration'] = pd.NA
    return df.loc[:, ['onset', 'duration', 'value']]


def save_data(data_list):
    for data, fn in data_list:
        np.save(fn ,data, allow_pickle=False)


def save_meta(meta_list):
    for meta, fn in meta_list:
        with open(fn, 'w') as file:
            json.dump(meta, file, indent=4)


# %%
for info in session_info.reset_index().itertuples():
    if info.animal_id == 0:
        continue
    try:
        block = read_block(SOURCEPATH / info.path)
        session_root = RAWPATH / 'sub-{:02d}/ses-{:02d}/'.format(
            info.animal_id, info.session_id)
        fn_fragment = 'sub-{:02d}_ses-{:02d}_task-{}-{:02d}'.format(
            info.animal_id, info.session_id, info.task, info.task_id
        )
        # Create folders if they don't yet exist
        (session_root / 'fp').mkdir(parents=True, exist_ok=True)

        L465_fn = session_root / 'fp' / (fn_fragment + '_L465.npy')
        meta_465_fn = session_root / 'fp' / (fn_fragment + '_L465.json')
        L405_fn = session_root / 'fp' / (fn_fragment + '_L405.npy')
        meta_405_fn = session_root / 'fp' / (fn_fragment + '_L405.json')
        event_fn = session_root / (fn_fragment + '_events.tsv')
        p = info.port
        data_465, meta_465 = downsample_stream(block, p+'65'+p, downsample)
        try:
            data_405, meta_405 = downsample_stream(block, p+'05'+p, downsample)
        except AttributeError:
            data_405, meta_405 = downsample_stream(block, p+'04'+p, downsample)
        save_data([(data_465, L465_fn), (data_405, L405_fn)])
        save_meta([(meta_465, meta_465_fn), (meta_405, meta_405_fn)])
        df = get_epoch_df(block.epocs['Prt'+p])
        df.to_csv(event_fn, sep='\t', na_rep='n/a')
    except FileNotFoundError as fe:
        pass


# %%
def generate_relative_filenames(animal_id, session_id, task, task_id):
    session_root = Path('sub-{:02d}/ses-{:02d}/'.format(
        animal_id, session_id))
    fn_fragment = 'sub-{:02d}_ses-{:02d}_task-{}-{:02d}'.format(
        animal_id, session_id, task, task_id
    )

    L465_fn = session_root / 'fp' / (fn_fragment + '_L465.npy')
    meta_465_fn = session_root / 'fp' / (fn_fragment + '_L465.json')
    L405_fn = session_root / 'fp' / (fn_fragment + '_L405.npy')
    meta_405_fn = session_root / 'fp' / (fn_fragment + '_L405.json')
    event_fn = session_root / (fn_fragment + '_events.tsv')
    return {
        '465': L465_fn,
        '465_meta': meta_465_fn,
        '405': L405_fn,
        '405_meta': meta_405_fn,
        'events': event_fn
    }
    

# %%
# Handle subject 8, task REV1-6 (two recordings for one session)
# DON'T RUN THIS: we do without
infos = session_info.query(
    '(animal_id == 8) & (task == "REV1") & (task_id == 6)')
streamdict = defaultdict(list)
for info in infos.reset_index().itertuples():
    block = read_block(SOURCEPATH / info.path)
    p = info.port
    data, meta = downsample_stream(block, p+'65'+p, downsample)
    streamdict['465'].append(data)
    streamdict['465_meta'].append(meta)
    data, meta = downsample_stream(block, p+'05'+p, downsample)
    streamdict['405'].append(data)
    streamdict['405_meta'].append(meta)
    streamdict['events'].append(get_epoch_df(block.epocs['Prt'+p]))
data_465 = np.concatenate(streamdict['465'])
meta_465 = streamdict['465_meta'][0]
data_405 = np.concatenate(streamdict['405'])
meta_405 = streamdict['405_meta'][0]
events_df = pd.concat(streamdict['events'], axis=0)

fn_dict = generate_relative_filenames(info.animal_id, info.session_id,
                                      info.task, info.task_id)
save_data([(data_465, RAWPATH / fn_dict['465']),
           (data_405, RAWPATH / fn_dict['405'])])
save_meta([(meta_465, RAWPATH / fn_dict['465_meta']),
           (meta_405, RAWPATH / fn_dict['405_meta'])])
events_df.to_csv(RAWPATH / fn_dict['events'], sep='\t', na_rep='n/a')

# %%
