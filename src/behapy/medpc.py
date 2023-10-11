"""Convert the bizarre MedPC output to a sane raw data structure
"""
import warnings
from collections import namedtuple
from typing import Tuple, Union, Any
from datetime import datetime
from string import ascii_uppercase
from pathlib import Path
import pandas as pd
from .pathutils import get_events_path


def load_medpc_map(filename: str) -> pd.DataFrame:
    """Loads a file mapping sessions to the MedPC files.

    Args:
        filename (str): path to the session mapping file
    """
    dtypes = {
        'path': str,
        'subject': str,
        'session': str,
        'task': str,
        'run': str
    }
    info = pd.read_csv(filename, dtype=dtypes)
    resolved_path = Path(filename).resolve().parent
    info.block = info.block.apply(lambda x: resolved_path / x)
    return info


def experiment_info(variables: "dict[str, str]") -> pd.Series:
    """Parse the experiment infomation from variables.

    Args:
        variables: A set of MedPC variables extracted via `parse_file`.

    Returns:
        A `pd.Series` containing subject, experiment, group, box, start
        datetime and end datetime.
    """
    startstr = variables['Start Date'] + ' ' + variables['Start Time']
    endstr = variables['End Date'] + ' ' + variables['End Time']
    start = datetime.strptime(startstr, '%m/%d/%y %H:%M:%S')
    end = datetime.strptime(endstr, '%m/%d/%y %H:%M:%S')
    return pd.Series({
        'subject': variables['Subject'],
        'experiment': variables['Experiment'],
        'group': variables['Group'],
        'box': variables['Box'],
        'start': start,
        'end': end,
        'MSN': variables['MSN']})


def get_events(timestamps: "list[str]",
               event_idxs: "list[str]",
               event_map: "dict[int, str]" = None) -> pd.DataFrame:
    """Parse string-encoded timestamps and events.

    Args:
        timestamps: A list of strings of floats representing seconds as
            written by MedPC.
        event_idxs: A list of strings of floats representing event indices
            as written by MedPC.
        event_map: A map from event indices (as integers) to event codes.

    Returns:
        A `pd.DataFrame` of timestamps with the corresponding event code.
    """
    ts_prev = 0.0
    event_list = []
    valid_events = False
    for ts in timestamps:
        if float(ts) > 0.:
            valid_events = True
            break
    if not valid_events:
        warnings.warn('No valid events in list')
        return pd.DataFrame({'timestamp': [], 'event': []})
    for ts, event in zip(timestamps, event_idxs):
        if float(ts) - ts_prev < 0:
            break
        if event_map is not None:
            event_list.append((pd.Timedelta(float(ts), unit='s'),
                               event_map[int(float(event))]))
        else:
            event_list.append((pd.Timedelta(float(ts), unit='s'),
                               int(float(event))))
        ts_prev = float(ts)
    return pd.DataFrame(event_list,
                        columns=['timestamp', 'event'])


def parse_line(line: str, prev_token: str, prev_data: Any) -> Tuple[str, str]:
    if len(line.strip()) == 0:
        return None, None
    token, value = line.split(':', maxsplit=1)
    if token[0] == ' ':
        if not isinstance(prev_data, list):
            prev_data = []
        assert len(prev_data) == int(token), 'Unexpected length of array'
        prev_data.extend(value.split())
        return prev_token, prev_data
    else:
        return token, value.strip()


def parse_file(filename: str) -> dict:
    variables = {}
    token = ''
    data = ''
    with open(filename, 'r') as mpcfile:
        while (line := mpcfile.readline()):
            token, data = parse_line(line, token, data)
            if token is None:
                continue
            variables[token] = data
    return variables


def convert_file(path, subject, session, task, run, config, bids_root):
    variables = parse_file(path)
    events = get_events(variables[config['timestamp']],
                        variables[config['event_index']],
                        config['event_map'])
    outpath = get_events_path(bids_root, subject, session, task, run)
    events.to_csv(outpath)