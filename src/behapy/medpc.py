"""Convert the bizarre MedPC output to a sane raw data structure
"""
from collections import namedtuple
from typing import Tuple, Union, Any
from datetime import datetime
from string import ascii_uppercase
import pandas as pd


def experiment_info(variables: dict[str, str]) -> pd.Series:
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
        'Subject': variables['Subject'],
        'Experiment': variables['Experiment'],
        'Group': variables['Group'],
        'Box': variables['Box'],
        'Start': start,
        'End': end})


def get_events(timestamps: list[str],
               event_idxs: list[str],
               event_map: dict[int, str] = None) -> pd.DataFrame:
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
                        columns=['Timestamp', 'Event'])

    
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
