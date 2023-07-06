from typing import Union
import logging
import argparse
import glob
import json
import numpy as np
import pandas as pd
import panel as pn
from . import medpc
from pathlib import Path
from .pathutils import list_raw, get_preprocessed_fibre_path
from .tdt import load_session_tank_map, load_event_names, convert_block
from .visuals import PreprocessDashboard
from . import fp


def tdt2bids(session_fn: str, experiment_fn: str, bids_root: str) -> None:
    """Convert TDT tanks into BIDS format.

    Args:
        session_fn: Map of the files to sessions
        experiment_fn:
        bids_root: Root path of the BIDS structure (data will be put in the
                   `rawdata` sub-folder of `bids_root`)
    """
    session_df = load_session_tank_map(session_fn)
    event_names = load_event_names(experiment_fn)
    convert_block(session_df, Path(bids_root), event_names)


def tdt2bids_command():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Convert TDT tanks into BIDS format'
    )
    parser.add_argument('session_fn', type=str,
                        help='path to CSV file with TDT session information')
    parser.add_argument('experiment_fn', type=str,
                        help='path to JSON file with experiment details')
    parser.add_argument('bids_root', type=str,
                        help='root path of the BIDS dataset (data will '
                             'be put in the rawdata sub-folder of bids_root)')
    args = parser.parse_args()
    tdt2bids(**vars(args))


def medpc2csv(source_pattern: str,
              output_path: str,
              config_fn: str) -> None:
    """Convert MedPC timestamp + event arrays to CSV

    Will produce two CSV files, one for the experimental info and one
    for the event array.

    Args:
        source_pattern: Glob path pattern for source files
        output_path: Path for the two output CSV files
        events_mapping_fn: Configuration file containing variable mapping
                           and the events mapping dict.
    """
    all_info = []
    all_events = []
    with open(config_fn) as file:
        config = json.load(file)
    config['event_map'] = {int(key): value
                           for key, value in config['event_map'].items()}
    for fn in glob.glob(source_pattern):
        variables = medpc.parse_file(fn)
        info = medpc.experiment_info(variables)
        events = medpc.get_events(variables[config['timestamp']],
                                  variables[config['event_index']],
                                  config['event_map'])
        events['subject'] = info['subject']
        events.set_index(['subject', 'timestamp'], inplace=True)
        all_info.append(info)
        all_events.append(events)

    if all_info:
        info_df = pd.DataFrame(all_info)
        info_df.set_index(['subject'], inplace=True)
    if all_events:
        events_df = pd.concat(all_events)

    if len(info_df['experiment'].unique()) > 1:
        exp_name = 'multi'
    else:
        exp_name = info_df['experiment'][0]
    info_df.to_csv(Path(output_path) / '{}_info.csv'.format(exp_name))
    events_df.to_csv(Path(output_path) / '{}_events.csv'.format(exp_name))


def medpc2csv_command():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Convert timestamp + event arrays to CSV'
    )
    parser.add_argument('source_pattern', type=str,
                        help='path pattern for source files')
    parser.add_argument('output_path', type=str,
                        help='folder for the two output CSV files')
    parser.add_argument('config_fn', type=str,
                        help='config file defining events variables')
    args = parser.parse_args()
    medpc2csv(**vars(args))


def preprocess_dash(bidsroot):
    bidsroot = Path(bidsroot)
    recordings = pd.DataFrame(list_raw(bidsroot))
    signals = recordings.loc[:, ['subject', 'session', 'task', 'run', 'label']].drop_duplicates()

    def get_recording(index):
        r = signals.iloc[index]
        signal = fp.load_signal(bidsroot, r.subject, r.session, r.task, r.run,
                                r.label, 'iso')
        return signal

    dash = PreprocessDashboard(signals, get_recording)
    pn.serve(dash.view(), port=8080)


def preprocess_dash_command():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Load the preprocessing dashboard'
    )
    parser.add_argument('bidsroot', type=str,
                        help='path to the BIDS root')
    args = parser.parse_args()
    preprocess_dash(**vars(args))


def preprocess(bidsroot):
    bidsroot = Path(bidsroot)
    recordings = pd.DataFrame(list_raw(bidsroot))
    signals = recordings.loc[:, ['subject', 'session', 'task', 'run', 'label']].drop_duplicates()

    def save_recording(row):
        return fp.preprocess(bidsroot, row.subject, row.session, row.task,
                             row.run, row.label)

    signals.apply(save_recording, axis=1)


def preprocess_command():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=('Preprocess the dataset from raw data using the rejected '
                     'intervals')
    )
    parser.add_argument('bidsroot', type=str,
                        help='path to the BIDS root')
    args = parser.parse_args()
    preprocess(**vars(args))