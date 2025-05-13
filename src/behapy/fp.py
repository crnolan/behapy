from typing import Tuple, Iterable
from pathlib import Path
import json
import logging
import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit
from collections import namedtuple
from typing import Iterable
import pandas as pd
import statsmodels.api as sm
from intervaltree import IntervalTree, Interval
from numpy.lib.stride_tricks import sliding_window_view
import bottleneck as bn
from .pathutils import get_raw_fibre_path, list_raw, \
    get_rejected_intervals_path, get_preprocessed_fibre_path
from .config import load_preprocess_config


logger = logging.getLogger(__name__)
Event = namedtuple('Event', ['name', 'fields', 'codes', 'onset', 'offset'])


def series_like(df: pd.Series,
                name: str,
                default: float = 0.) -> pd.Series:
    series = pd.Series(default, index=df.index, name=name)
    series.attrs = df.attrs.copy()
    _ = series.attrs.pop('artifact_channel', None)
    _ = series.attrs.pop('channels', None)
    _ = series.attrs.pop('iso_channel', None)
    _ = series.attrs.pop('channel', None)
    return series


def load_channel(root, subject, session, task, run, label, channel):
    data_fn = get_raw_fibre_path(root, subject, session, task, run, label,
                                 channel, 'npy')
    meta_fn = get_raw_fibre_path(root, subject, session, task, run, label,
                                 channel, 'json')
    with open(meta_fn) as file:
        meta = json.load(file)
    data = np.load(data_fn)
    return data, meta


def load_signals(root, subject, session, task, run, label,
                 artifact_channel=None, exclude_artifact_channel=True):
    """Load all raw signals for a given site.
    """
    root = Path(root).absolute()
    recordings = pd.DataFrame(
        list_raw(root, subject=subject, session=session, task=task,
                 run=run, label=label))
    subjects = recordings.loc[:, 'subject'].unique()
    sessions = recordings.loc[:, 'session'].unique()
    tasks = recordings.loc[:, 'task'].unique()
    labels = recordings.loc[:, 'label'].unique()
    if any([item.shape[0] != 1
            for item in [subjects, sessions, tasks, labels]]):
        msg = (f'Multiple signal names found for session '
               f'with subject {subject}, session {session}, task {task}, '
               f'run {run} and label {label}')
        logging.error(msg)
        raise ValueError(msg)

    # Load channels
    data = []
    t0 = None
    fs = None
    for r in recordings.itertuples():
        d, meta = load_channel(root=root,
                               subject=r.subject,
                               session=r.session,
                               task=r.task,
                               run=r.run,
                               label=r.label,
                               channel=r.channel)
        if fs is None:
            fs = meta['fs']
        if t0 is None:
            t0 = meta['start_time']
        if (fs != meta['fs']) or (t0 != meta['start_time']):
            msg = ('Unequal sample frequencies and/or start times '
                   'for subject {}, session {}, task {}, run {} and label {}')
            msg.format(subject, session, task, run, label)
            raise ValueError(msg)
        t = pd.Index(np.arange(d.shape[0]) / fs + t0, name='time')
        data.append(pd.Series(d, name=r.channel, index=t))

    signal = pd.concat(data, axis=1)
    signal.index.name = 'time'
    signal.attrs['root'] = root
    signal.attrs['fs'] = fs
    signal.attrs['start_time'] = t0
    signal.attrs['subject'] = subject
    signal.attrs['session'] = session
    signal.attrs['task'] = task
    signal.attrs['run'] = run
    signal.attrs['label'] = label
    channels = signal.columns.to_list()
    if artifact_channel is None:
        acd = set(['iso', 'isos', 'isosbestic'])
        artifact_channel = set(channels).intersection(acd)
        if len(artifact_channel) == 0:
            artifact_channel = next(
                ([s] for s in channels if s.endswith(('.N', '.n'))), [])
        if len(artifact_channel) == 0:
            logger.warning(f'No artifact channel found for subject {subject}, '
                           f'session {session}, task {task}, run {run} and '
                           f'label {label}, using first channel {channels[0]}')
            artifact_channel = channels[0]
        elif len(artifact_channel) == 1:
            artifact_channel = artifact_channel.pop()
        else:
            raise ValueError(
                f'Multiple default artifact channels found for subject '
                f'{subject}, session {session}, task {task}, run {run} and '
                f'label {label}: {artifact_channel}')

    signal.attrs['artifact_channel'] = artifact_channel
    if exclude_artifact_channel:
        signal.attrs['channels'] = [c for c in channels
                                      if c != artifact_channel]
    else:
        signal.attrs['channels'] = channels

    if len(signal.attrs['channels']) == 0:
        raise ValueError(f'No channels found for subject {subject}, '
                         f'session {session}, task {task}, run {run} '
                         f'and label {label}')
    return signal


def downsample(signal, factor=None):
    if factor is None:
        # Downsample to something reasonable
        factor = 1
        while signal.attrs['fs'] / factor > 20:
            factor *= 2
    ds = sig.decimate(signal.to_numpy(), factor, ftype='fir',
                      zero_phase=True, axis=0)
    df = pd.DataFrame(ds, index=signal.index[::factor], columns=signal.columns)
    df.attrs = signal.attrs
    df.attrs['fs'] = signal.attrs['fs'] / factor
    return df


def save_rejections(tree, root, subject, session, task, run, label):
    # Save the provided IntervalTree as a CSV
    fn = get_rejected_intervals_path(root, subject, session, task, run,
                                     label)
    tree.merge_overlaps()
    df = pd.DataFrame.from_records([(b[0], b[1]) for b in list(tree)],
                                   columns=['start_time', 'end_time'])
    fn.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fn, index=False)


def load_rejections(root, subject, session, task, run, label):
    # Load rejected intervals if present
    fn = get_rejected_intervals_path(root, subject, session, task, run,
                                     label)
    intervals = []
    if not fn.exists():
        return None
    # Load the CSV
    rej = pd.read_csv(fn)
    intervals = [(r.start_time, r.end_time) for r in rej.itertuples()]
    return IntervalTree.from_tuples(intervals)


def find_discontinuities(signal, mean_window=3, std_window=30, nstd_thresh=2):
    # This currently relies on the isobestic channel being valid.

    # How many samples to consider for the sliding mean
    n = int(signal.attrs['fs'] * mean_window)
    # Assume that the STD of the iso channel is constant. We can
    # then use the median of a sliding window STD as our
    # characteristic STD.
    std_n = int(signal.attrs['fs'] * std_window)
    if 'channels' not in signal.attrs:
        logger.info('Using original single-channel format')
        channels = [signal.attrs['channel']]
        artifact_channel = signal.attrs['iso_channel']
    else:
        channels = signal.attrs['channels']
        artifact_channel = signal.attrs['artifact_channel']
    data = signal[channels].to_numpy()
    data_rstds = bn.move_std(data, std_n, axis=0)
    data_thresh = np.nanmedian(data_rstds, axis=0)
    data_rmeans = bn.move_mean(np.pad(data, ((n, n), (0, 0)), 'edge'), n, axis=0)
    afct = signal[artifact_channel].to_numpy()
    afct_rstds = bn.move_std(afct, std_n, axis=0)
    afct_thresh = np.nanmedian(afct_rstds, axis=0)
    mean_thresh = afct_thresh * nstd_thresh
    afct_rmeans = bn.move_mean(np.pad(afct, ((n, n)), 'edge'), n, axis=0)
    d = (afct_rmeans[n:-n] - afct_rmeans[(n*2):])
    d_thresh = np.abs(d) > mean_thresh
    # Find the start and end of each mean shift
    mean_shift_bounds = np.diff(d_thresh.astype(int))
    try:
        # If the first bound is a falling edge, insert a rising edge
        if mean_shift_bounds[0] == -1:
            mean_shift_bounds[0] = 0
        elif mean_shift_bounds[mean_shift_bounds != 0][0] == -1:
            mean_shift_bounds[0] = 1
        # If the last bound is a rising edge, insert a falling edge
        if mean_shift_bounds[-1] == 1:
            mean_shift_bounds[-1] = 0
        elif mean_shift_bounds[mean_shift_bounds != 0][-1] == 1:
            mean_shift_bounds[-1] = -1
    except IndexError:
        pass
    # For each shift, adjust the bounds by searching from the opposite
    # bound and looking for the first time the signal (rather than the
    # mean) is within the threshold bounds. Use the real signal in this
    # case rather than the isosbestic channel.
    onsets = np.where(mean_shift_bounds == 1)[0]
    offsets = np.where(mean_shift_bounds == -1)[0]
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        k = np.argmax(np.abs(data[offset:onset:-1, :] - data_rmeans[[onset+n], :]) < data_thresh)
        if k > 0:
            onsets[i] = offset - k
        k = np.argmax(np.abs(data[onset:offset:1, :] - data_rmeans[[offset+n], :]) < data_thresh)
        if k > 0:
            offsets[i] = onset + k
    return [(onset, offset)
            for onset, offset in zip(onsets, offsets)
            if offset - onset > 0]


def find_disconnects(signal, zero_nstd_thresh=5, mean_window=3, std_window=30,
                     nstd_thresh=2):
    bounds = find_discontinuities(signal, mean_window=mean_window,
                                  std_window=std_window, nstd_thresh=nstd_thresh)
    if 'channels' not in signal.attrs:
        logger.info('Using original single-channel format')
        channels = [signal.attrs['channel']]
    else:
        channels = signal.attrs['channels']
    data = signal[channels].to_numpy()
    ts = signal.index.to_numpy()
    std_n = int(signal.attrs['fs'] * std_window)
    data_rstds = bn.move_std(data, std_n, axis=0)
    zero_thresh = np.nanmedian(data_rstds, axis=0) * zero_nstd_thresh
    dc_intervals = IntervalTree()
    bounds = [(0, 0)] + bounds + [(data.shape[0]-1, data.shape[0]-1)]
    for (on0, off0), (on1, off1) in zip(bounds[:-1], bounds[1:]):
        if np.any(np.mean(data[off0:on1, :], axis=0) < zero_thresh):
            dc_intervals.add(Interval(ts[on0], ts[off1]))
    dc_intervals.merge_overlaps()
    return dc_intervals


def intervals_to_mask(signal: pd.DataFrame, intervals: Interval) -> pd.Series:
    """Convert a list of intervals to a boolean mask.

    Args:
        signal: The timeseries over which to generate a mask.
        intervals: A list of intervals.

    Returns:
        A boolean mask with True for valid samples and False for rejected
        samples.
    """
    _intervals = intervals.copy()
    _intervals.merge_overlaps()
    interval_list = [(i[0], i[1]) for i in list(_intervals)]
    mask = pd.Series(True, index=signal.index)
    for start, end in interval_list:
        mask.loc[start:end] = False
    return mask


def reject(signal, intervals, fill=False):
    """ Filter the site data to remove the specified intervals. """
    mask = intervals_to_mask(signal, intervals)
    if fill:
        # Return a copy of the signal with the rejected intervals
        # replace with a linear interpolation between the endpoints.
        signal = signal.copy()
        signal.loc[~mask] = np.nan
        signal = signal.interpolate(method='linear', limit_direction='both')
        signal['mask'] = mask
        return signal
    else:
        return signal[mask].copy()


def save_rejected_intervals(signal: 'pd.DataFrame',
                            intervals: 'Sequence[Tuple[int, int]]'):
    path = get_rejected_intervals_path(signal.attrs['base'],
                                       signal.attrs['subject'],
                                       signal.attrs['session'],
                                       signal.attrs['task'],
                                       signal.attrs['run'],
                                       signal.attrs['label'])


def map_events(events: Iterable[Event]):
    """ Create a dict mapping event codes to the respective events. """
    return {key: event for event in events.values() for key in event.fields}


def invalidate_samples(df, start, end):
    """Invalidate samples """
    if start == None:
        start = df.index[0]
    if end == None:
        end = df.index[-1]
    if 'Valid' not in df:
        df['Valid'] = True
    df.loc[start:end, 'Valid'] = False
    return df


def smooth(data, numtaps=1001, cutoff=1):
    try:
        if smooth.numtaps != numtaps or smooth.cutoff != cutoff:
            raise AttributeError("Filter parameters changed")
        b = smooth.filter_b
    except AttributeError:
        b = sig.firwin(numtaps, cutoff=[cutoff], fs=data.attrs['fs'], pass_zero=True)
        smooth.filter_b = b
        smooth.numtaps = numtaps
        smooth.cutoff = cutoff
    # smoothed = series_like(data, 'smoothed')
    smoothed = data.copy()
    smoothed[:] = sig.filtfilt(b, 1, data.to_numpy(), axis=0).astype(np.float32)
    return smoothed


def detrend(data, numtaps=1001, cutoff=0.05):
    try:
        if detrend.numtaps != numtaps or detrend.cutoff != cutoff:
            raise AttributeError("Filter parameters changed")
        b = detrend.filter_b
    except AttributeError:
        b = sig.firwin(numtaps, cutoff=[cutoff], fs=data.attrs['fs'],
                       pass_zero=False)
        detrend.filter_b = b
        detrend.numtaps = numtaps
        detrend.cutoff = cutoff
    # detrended = series_like(data, 'detrended')
    detrended = data.copy()
    detrended[:] = sig.filtfilt(b, 1, data.to_numpy(), axis=0).astype(np.float32)
    return detrended


def exp_fit(data):
    def _exp_func(x, a1, b1, a2, b2, c):
        return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c
    _max = data.max()
    popt, pcov = curve_fit(_exp_func, data.index, data, maxfev=10000,
                           bounds=[(-_max, 0, -_max, 0, 0), (_max, np.inf, _max, np.inf, _max)])
    logger.info(f'popt: {popt}')
    fit = series_like(data, 'fit')
    fit[:] = _exp_func(data.index.to_numpy(), *popt)
    return fit


def full_fit(data):
    def _func(x, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, gamma):
        x0 = x[:(len(x) // 2)]
        x1 = x[(len(x) // 2):]
        return (a0 * np.exp(-b0 * x0) + a1 * np.exp(-b1 * x0)
                + gamma * (a2 * np.exp(-b2 * x)
                           + a3 * np.exp(-b3 * x)
                           + c1)
                + c0)
    M = data.max()
    popt, pcov = curve_fit(_func, data.index, data, maxfev=10000,
                           bounds=[(-M, -M, -M, -M, 0, 0, 0, 0, -np.inf, -np.inf, ),
                                   (_max, np.inf, _max, np.inf, _max)])
    logger.info(f'popt: {popt}')
    fit = series_like(data, 'fit')
    fit[:] = _func(data.index.to_numpy(), *popt)
    return fit


def debleach(data):
    """ Debleach the data by fitting an exponential and subtracting"""
    exp_func = lambda x, a, b, c: a * np.exp(-b * x) + c
    popt, pcov = curve_fit(exp_func, data.index, data, maxfev=10000)
    fit = series_like(data, 'fit')
    fit[:] = exp_func(data.index.to_numpy(), *popt)
    return (data - fit) / fit


def fit_debleached(data, control):
    data_lp = smooth(data)
    control_lp = smooth(control)
    ols_model = sm.OLS(data_lp, control_lp)
    return data_lp - ols_model.fit().fittedvalues,


def rlm(signal):
    """ Fit the site data to the isobestic channel using a robust regression.
    """
    if len(signal.attrs['channels']) > 1:
        raise ValueError('Only one channel is supported.')
    ch = signal.attrs['channels'][0]
    iso = signal.attrs['artifact_channel']
    df = signal.loc[signal['mask'], :]
    fitted = series_like(signal, 'fitted')
    fitted[signal['mask']] = sm.RLM(df[ch], df[iso]).fit().fittedvalues
    fitted.attrs['channels'] = ['fitted']
    dff = series_like(signal, ch)
    dff[signal['mask']] = (df[ch] - fitted[signal['mask']]) / fitted[signal['mask']]
    dff.attrs['channels'] = [ch]
    return dff.to_frame(), fitted


def ratiometric(positive, negative, mask):
    fit = positive.copy()
    fit[~mask] = positive[~mask] / negative[~mask]
    return fit


def normalise(signal, control, mask, fs, method='fit', detrend=True):
    # smoothed = smooth(control[~mask], fs=fs)
    smoothed = control[~mask]
    fit = np.polyfit(smoothed, signal[~mask], deg=1)
    # Subtract the smoothed projection - fitting the raw control
    # projection results in a greater variance in the fitted signal than
    # the actual signal
    signal_fit = signal.copy()
    signal_fit[~mask] = fit[0] * smoothed + fit[1]
    df = signal - signal_fit
    if detrend:
        # Filter with a highpass filter to remove remaining drift
        df = detrend(df, fs)

    if method == 'fit':
        dff = df / signal_fit
        return dff / dff.std()
    elif method == 'const':
        dff = df / np.mean(signal)
        return dff / dff.std()
    elif method == 'df':
        return df
    elif method == 'yfit':
        return signal_fit
    else:
        raise ValueError("Unrecognised normalisation method {}".format(method))


def preprocess(root, subject, session, task, run, label):
    config = load_preprocess_config(root)
    intervals = load_rejections(root, subject, session, task, run, label)
    # Check if the recording has rejections saved
    if intervals is None:
        logger.info(f'Recording for subject {subject}, '
                    f'session {session}, task {task}, '
                    f'run {run} and label {label} has no '
                    f'rejections file, skipping.')
        return False
    logger.info(f'Preprocessing subject {subject}, '
                f'session {session}, task {task}, '
                f'run {run}, label {label}...')
    recording = load_signals(root, subject, session, task, run, label, 'iso')
    recording = downsample(recording, 64)
    rej = reject(recording, intervals, fill=True)
    ch = recording.attrs['channels']
    # We were doing a robust regression, but the fit isn't good enough.
    # Let's just detrend and divide by the smoothed signal instead.
    # dff = fp.series_like(recording, name='dff')
    # dff.loc[rej.index] = fp.detrend(rej[ch])
    if config['method'] == 'rlm':
        dff, fitted = rlm(rej)
    else:
        dff = detrend(rej[ch], cutoff=config['detrend_cutoff'])
        dff = dff / smooth(rej[ch])
    # dff.name = 'dff'
    # dff = dff.to_frame()
    dff['mask'] = rej['mask']
    dff.attrs['root'] = str(dff.attrs['root'])
    data_fn = get_preprocessed_fibre_path(
        root, subject, session, task, run, label, 'parquet')
    meta_fn = get_preprocessed_fibre_path(
        root, subject, session, task, run, label, 'json')
    data_fn.parent.mkdir(parents=True, exist_ok=True)
    try:
        dff.to_parquet(data_fn, engine='pyarrow')
    except TypeError as e:
        logger.warning(f"Serialization error with pyarrow: {e}")
        dff.to_parquet(data_fn, engine='fastparquet')
    meta = dff.attrs
    meta['root'] = str(root)
    with open(meta_fn, 'w') as file:
        json.dump(meta, file)
    return True


def load_preprocessed_fibre(root, subject, session, task, run, label):
    root = Path(root)
    data_fn = get_preprocessed_fibre_path(
        root, subject, session, task, run, label, 'parquet')
    meta_fn = get_preprocessed_fibre_path(
        root, subject, session, task, run, label, 'json')
    if not data_fn.exists():
        return None
    data = pd.read_parquet(data_fn)
    if data.index.name != 'time':
        data.index = pd.to_timedelta(data.index, unit='s')
        data.index.name = 'time'
    if not pd.api.types.is_timedelta64_dtype(data.index):
        data.index = pd.to_timedelta(data.index, unit='s')
        data.index.name = 'time'
    with open(meta_fn, 'r') as file:
        meta = json.load(file)
    data.attrs.update(meta)
    return data


def epoch_events(data: Iterable[float],
                 events: Iterable[float],
                 window: Tuple[float, float],
                 baseline_window: Tuple[float, float],
                 fs: float, tstart: float = 0.,
                 method: str = 'z') -> Tuple[np.ndarray, np.ndarray]:
    """ Epoch data at supplied event times and optionally baseline.

    Args:
        data: An array of samples.
        events: An array containing event times in seconds (must be sorted).
        window: Start and end time of epoch in seconds, e.g. (-1.0, 1.0).
        baseline_window: Start and end time of baseline period for each
            epoch in seconds, if required, e.g. (-2.0, -1.0).
        fs: Sampling frequency of data,
        tstart: Time in seconds of the first sample of data,
        method: Baselining method, either z-scored ('z', default),
            baseline mean subtracted ('base') or unbaselined signal ('nobase').

    Returns:
        Array of (baselined) epoch values with a correponding timestamp array.
    """
    n = data.shape[0]
    ts = np.arange(n) / fs + tstart
    ixs = np.searchsorted(ts, events)
    window_ix = np.ceil(np.array(window) * fs).astype(int)
    if method != 'nobase':
        baseline_ix = np.ceil(np.array(baseline_window) * fs).astype(int)
    dslice = lambda b, w: data[slice(*(b + w))]
    if method == 'z':
        z = lambda d, b: (d - np.mean(b)) / np.std(b)
        epochs = np.array([z(dslice(ix, window_ix), dslice(ix, baseline_ix))
                           for ix in ixs])
    elif method == 'base':
        base = lambda d, b: d - np.mean(b)
        epochs = np.array([base(dslice(ix, window_ix), dslice(ix, baseline_ix))
                           for ix in ixs])
    elif method == 'nobase':
        epochs = np.array([dslice(ix, window_ix)
                           for ix in ixs])
    else:
        raise ValueError('Invalid epoching method {}'.format(method))

    return epochs, np.arange(*window_ix) / fs
