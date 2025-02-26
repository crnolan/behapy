"""Convert the bizarre MedPC output to a sane raw data structure
"""
import logging
from collections import namedtuple
from typing import Tuple, Union, Any
from datetime import datetime
from pathlib import Path
from string import ascii_uppercase
import pandas as pd
import re


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
               event_map: "dict[int, str]" = None,
               offset_map: "dict[int, str]" = None) -> pd.DataFrame:
    """Parse string-encoded timestamps and events.

    Args:
        timestamps: A list of strings of floats representing seconds as
            written by MedPC.
        event_idxs: A list of strings of floats representing event indices
            as written by MedPC.
        event_map: A map from event indices (as integers) to event codes.
        offset_map: A map from event indices (as integers) to event codes
            that serve as offsets to the same-labelled event codes in
            event_map.

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
        logging.warning('No valid events in list')
        return pd.DataFrame({'onset': [],
                             'duration': [],
                             'event_id': []}).set_index('onset')
    for ts, event in zip(timestamps, event_idxs):
        if float(ts) - ts_prev < 0:
            break
        event_list.append((pd.Timedelta(float(ts), unit='s'),
                        0.,
                        int(float(event))))
        ts_prev = float(ts)
    df = pd.DataFrame(event_list,
                    columns=['timestamp', 'duration', 'event_id'])
    if event_map is None:
        df = df.rename(columns={'timestamp': 'onset'}).set_index('onset')
    elif offset_map is not None:
        # For any events that have an offset in the offset_map, we want to
        # use the timestamp of that offset event to calculate the duration of
        # the relevant onset event.
        onsets = df.query(f'event_id not in {list(offset_map.keys())}').copy()
        offsets = df.query(f'event_id in {list(offset_map.keys())}').copy()
        onsets['event_id'] = onsets['event_id'].map(event_map).fillna(onsets['event_id'])
        offsets['event_id'] = offsets['event_id'].map(offset_map)
        onsets['event_num'] = onsets.groupby('event_id').cumcount()
        offsets['event_num'] = offsets.groupby('event_id').cumcount()
        offsets = (offsets.rename(columns={'timestamp': 'offset'})
                          .set_index(['event_id', 'event_num'])['offset'])
        onsets['duration'] = (onsets.join(offsets,
                                          on=['event_id', 'event_num'],
                                          how='left')
                                    .eval('offset - timestamp')
                                    .fillna(0))
        df = (onsets[['timestamp', 'duration', 'event_id']]
              .rename(columns={'timestamp': 'onset'})
              .set_index('onset'))
    else:
        # Otherwise just rename any events that are in the event_map
        df['event_id'] = df['event_id'].map(event_map).fillna(df['event_id'])
    return df


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


def generate_mapping(
        sourcepath: Union[str, Path],
        filename_re: str = r".*Subject (?P<subject>[^\.]+)\.txt",
        msn_re: str = None,
        subject: str = None,
        session: str = None,
        task: str = None,
        run: str = None,
        subject_map: dict[str, str] = {},
        session_map: dict[str, str] = {},
        task_map: dict[str, str] = {},
        run_map: dict[str, str] = {}
        ) -> pd.DataFrame:
    """Generate a mapping of MedPC backup files to raw events files.

    Args:
        sourcepath: The location to search for MedPC backup files.
        filename_re: A regular expression to extract variables from the
            filenames.
        msn_re: A regular expression to extract variables from the MSN
            field of the backup files.
        subject_map: A mapping of the subject field as extracted from
            any regular expression to the subject identifier.
        session_map: A mapping of the session field as extracted from
            any regular expression to the session identifier.
        task_map: A mapping of the task field as extracted from any
            regular expression to the task identifier.
        run_map: A mapping of the run field as extracted from any
            regular expression to the run identifier.

    Returns:
        A `pd.DataFrame` with a row for each data file and columns for
        subject, session, task and run.
    """
    sourcefiles = []
    sourcepath = Path(sourcepath)
    if not sourcepath.exists():
        logging.error(f'Provided source path {sourcepath} does not exist')
        return

    for fn in sourcepath.glob('**/Backup of *Subject *.txt'):
        # First match any regular expressions in the filename
        match = re.search(filename_re, str(fn))
        if not match:
            logging.warning(f'Bad template match for {fn.name}')
            continue
        groups = match.groupdict()
        if 'subject' in groups:
            subject = groups['subject']
        if 'session' in groups:
            session = groups['session']
        if 'task' in groups:
            task = groups['task']
        if 'run' in groups:
            run = groups['run']
        # Then match any regular expressions in the MSN field of the file
        if msn_re:
            variables = parse_file(fn)
            match = re.search(msn_re, variables['MSN'])
            if not match:
                logging.warning(f'Bad MSN match for {fn.name}')
                continue
            groups = match.groupdict()
            if 'subject' in groups:
                subject = groups['subject']
            if 'session' in groups:
                session = groups['session']
            if 'task' in groups:
                task = groups['task']
            if 'run' in groups:
                run = groups['run']
        if subject is None or session is None or task is None or run is None:
            logging.warning(f'Incomplete subject/session information for '
                            f'{fn.name}, skipping file')
            continue
        sourcefiles.append((fn, subject, session, task, run))
    df = pd.DataFrame(sourcefiles,
                      columns=['sourcefile', 'subject', 'session', 'task',
                               'run'])
    df['subject'] = df['subject'].replace(subject_map)
    df['session'] = df['session'].replace(session_map)
    df['task'] = df['task'].replace(task_map)
    df['run'] = df['run'].replace(run_map)
    return df


def events_to_bids(bidsroot: Union[Path, str],
                   sourcefile: Union[Path, str],
                   subject: str,
                   session: str,
                   task: str,
                   run: str,
                   timestamp_var: str,
                   event_var: str,
                   event_map: dict[int, str] = None,
                   postfix: str = 'events'):
    """Convert a raw MedPC file to BIDS format.

    The provided sourcefile will be converted into an events CSV file named
    `sub-{subject}_ses-{session}_task-{task}_run-{run}_events.csv` in the
    folder `$bidsroot/rawdata/sub-{subject}/ses-{session}`.

    Args:
        bidsroot: The root directory of the BIDS-ish dataset.
        sourcefile: The raw MedPC file to convert.
        subject: The subject identifier.
        session: The session identifier.
        task: The task identifier.
        run: The run identifier.
        timestamp_var: The name of the variable in the MedPC file that
            contains the events timestamps.
        event_var: The name of the variable in the MedPC file that
            contains the event indices.
        event_map: A dictionary mapping event indices to event names.
        postfix: A string to append to the filename before the extension.
    """
    variables = parse_file(sourcefile)
    events = get_events(variables[timestamp_var],
                        variables[event_var],
                        event_map)
    events_fn = (bidsroot
                 / f'rawdata/sub-{subject}/ses-{session}/'
                 / f'sub-{subject}_ses-{session}_task-{task}_run-{run}_{postfix}.csv')
    if events_fn.exists():
        logging.info(f'Events file already exists for {sourcefile}')
        return
    events_fn.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(events_fn, index=False)
