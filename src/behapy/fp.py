from copy import copy
from typing import Tuple, Iterable
from pathlib import Path
import json
import logging
import numpy as np
import scipy.signal as sig
from collections import namedtuple
from typing import Iterable
import pandas as pd
import statsmodels.api as sm
from intervaltree import IntervalTree, Interval
from numpy.lib.stride_tricks import sliding_window_view
import bottleneck as bn
from .pathutils import get_fibre_path, get_recordings, \
    get_rejected_intervals_path


Event = namedtuple('Event', ['name', 'fields', 'codes', 'onset', 'offset'])


def SessionMeta():

    def __init__(self, sub, ses, task, run, label, channel, 
                 fs=None, start_time=None):
        self.sub = sub
        self.ses = ses
        self.task = task
        self.label = label
        self.channel = channel
        self.meta = {
            'fs': fs,
            'start_time': start_time,
        }

    def from_json(filename):
        session_meta = SessionMeta()
        with open(filename) as file:
            session_meta.meta = json.load(file)
        return session_meta
    
    def to_json():
        pass


class SessionSite():

    def __init__(self):
        self.root = None
        self.subject = None
        self.session = None
        self.task = None
        self.runs = []
        self.label = None
        self.channels = []
        self.iso_index = None
        self.ts = None
        self.data = None
        self.rejections = None
        self.fs = None


    def __copy__(self):
        site = SessionSite()
        site.root = self.root
        site.subject = self.subject
        site.session = self.session
        site.task = self.task
        site.runs = self.runs
        site.label = self.label
        site.channels = copy(self.channels)
        site.iso_index = self.iso_index
        site.ts = self.ts
        site.data = self.data
        site.rejections = copy(self.rejections)
        site.fs = self.fs
        return site
    
    def __deepcopy__(self):
        site = copy(self)
        site.data = copy(self.data)
        return site

    def iso(self):
        if self.iso_index is None:
            raise ValueError('Isosbestic channel not defined')
        return self.data[:, self.iso_index]

    def signal(self):
        if self.data.shape[1] != 2:
            raise ValueError('Only one channel is supported.')
        return self.data[:, (self.iso_index + 1) % 2]
    
    def downsample(self, factor=None):
        if factor is None:
            # Downsample to something reasonable
            factor = 1
            while self.fs / factor > 20:
                factor *= 2
        data_ds = sig.decimate(self.data, factor, ftype='fir', zero_phase=True,
                               axis=0)
        ds = copy(self)
        ds.fs = self.fs / factor
        ds.ts = self.ts[::factor]
        ds.data = data_ds
        return ds
    
    def load(root, subject, session, label, iso_channel):
        root = Path(root)
        recordings = pd.DataFrame(get_recordings(root / 'rawdata',
                                                 subject, session, label))
        subjects = recordings.loc[:, 'subject'].unique()
        sessions = recordings.loc[:, 'session'].unique()
        tasks = recordings.loc[:, 'task'].unique()
        labels = recordings.loc[:, 'label'].unique()
        if any([item.shape[0] != 1
                for item in [subjects, sessions, tasks, labels]]):
            msg = ('Multiple session names found for session'
                   ' with subject {}, session {} and label {}')
            msg = msg.format(subject, session, label)
            logging.error(msg)
            raise ValueError(msg)

        # Load all channels
        data = []
        t0 = None
        fs = None
        channels = []
        runs = []
        for r in recordings.itertuples():
            d, meta = load_channel(base=root/'rawdata',
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
                msg = ('Unequal sample frequencies and/or start times'
                       ' for subject {}, session {} and label {}')
                msg.format(subject, session, label)
                raise ValueError(msg)
            data.append(d)
            runs.append(r.run)
            channels.append(r.channel)

        if (np.unique(runs).shape[0] != 1):
            msg = ('Multiple runs not yet implemented (subject {}, session {}'
                   ' and label {})').format(subject, session, label)
            raise NotImplementedError(msg)

        # Load rejected intervals if present
        rej_fn = get_rejected_intervals_path(root, subject, session, label)
        if rej_fn.exists():
            # Load the CSV
            rej = pd.read_csv(rej_fn)
            

        sl = SessionSite()
        sl.root = root
        sl.subject = subject
        sl.session = session
        sl.label = label
        sl.task = tasks[0]
        sl.runs = runs
        sl.channels = channels
        if iso_channel is not None:
            sl.iso_index = channels.index(iso_channel)
        sl.ts = np.arange(data[0].shape[0]) / fs + t0
        sl.data = np.vstack(data).T
        sl.fs = fs
        return sl


def load_channel(base, subject, session, task, run, label, channel):
    data_fn = get_fibre_path(base, subject, session, task, run, label, channel,
                             'npy')
    meta_fn = get_fibre_path(base, subject, session, task, run, label, channel,
                             'json')
    with open(meta_fn) as file:
        meta = json.load(file)
    data = np.load(data_fn)
    return data, meta


def load_channels(base, subject, session, task, run, label, channels,
                  downsample=None):
    fs = None
    t0 = None
    data = []
    for channel in channels:
        d, meta = load_channel(base, subject, session, task, run, label,
                               channel)
        if fs is None:
            fs = meta['fs']
        if t0 is None:
            t0 = meta['start_time']
        if (fs != meta['fs']) or (t0 != meta['start_time']):
            raise ValueError('Unequal sample frequencies and/or start times.')
        data.append(d)
    ts = np.arange(data[0].shape[0]) / fs + t0
    sdata = np.vstack(data).T
    if downsample is None:
        return sdata, ts
    else:
        ds = sig.decimate(sdata, downsample, ftype='fir', zero_phase=True,
                          axis=0)
        return ds, ts[::downsample]


def find_discontinuities(site, mean_window=3, std_window=30, nstd_thresh=2):
    # This currently relies on the isobestic channel being valid.

    # How many samples to consider for the sliding mean
    n = int(site.fs * mean_window)
    # Assume that the STD of the iso channel is constant. We can
    # then use the median of a sliding window STD as our
    # characteristic STD.
    std_n = int(site.fs * std_window)
    # iso_rstds = np.std(sliding_window_view(site.iso(), std_n), axis=-1)
    data = site.iso()
    iso_rstds = bn.move_std(data, std_n, axis=-1)
    thresh = np.median(iso_rstds[~np.isnan(iso_rstds)], axis=-1)
    mean_thresh = thresh * nstd_thresh
    # Calculate a sliding mean 
    # iso_rmeans = np.mean(sliding_window_view(np.pad(site.iso(), n, 'edge'), n), axis=-1)
    iso_rmeans = bn.move_mean(np.pad(data, n, 'edge'), n, axis=-1)
    d = (iso_rmeans[n:-n] - iso_rmeans[(n*2):])
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
    # mean) is within the threshold bounds.
    onsets = np.where(mean_shift_bounds == 1)[0]
    offsets = np.where(mean_shift_bounds == -1)[0]
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        k = np.argmax(np.abs(data[offset:onset:-1] - iso_rmeans[onset+n]) < thresh)
        if k > 0:
            onsets[i] = offset - k
        k = np.argmax(np.abs(data[onset:offset:1] - iso_rmeans[offset+n]) < thresh)
        if k > 0:
            offsets[i] = onset + k
    return list(zip(onsets, offsets))


def find_disconnects(site, zero_nstd_thresh=5, mean_window=3, std_window=30,
                     nstd_thresh=2):
    bounds = find_discontinuities(site, mean_window=mean_window,
                                  std_window=std_window, nstd_thresh=nstd_thresh)
    data = site.iso()
    std_n = int(site.fs * std_window)
    data_rstds = bn.move_std(data, std_n, axis=-1)
    data_rstds = data_rstds[~np.isnan(data_rstds)]
    zero_thresh = np.median(data_rstds, axis=-1) * zero_nstd_thresh
    dc_intervals = IntervalTree()
    bounds = [(0, 0)] + bounds + [(len(data)-1, len(data)-1)]
    for (on0, off0), (on1, off1) in zip(bounds[:-1], bounds[1:]):
        if np.mean(data[off0:on1]) < zero_thresh:
            dc_intervals.add(Interval(on0, off1))
    dc_intervals.merge_overlaps()
    return [(b[0], b[1]) for b in list(dc_intervals)]


def fit(site):
    """ Fit the site data to the isobestic channel. """
    if site.data.shape[1] != 2:
        raise ValueError('Only one channel is supported.')
    rlm_model = sm.RLM(site.signal(), site.iso())
    return rlm_model.fit()


def normalise_site(site):
    """ Normalise the site data to the isobestic channel. """
    iso = site.iso()
    
    for i, channel in enumerate(site.channels):
        if i != site.iso_index:
            site.data[channel] = site.data[channel] / iso
    return site

def save_rejected_intervals(site: 'SessionSite',
                            intervals: 'Sequence[Tuple[int, int]]'):
    path = get_rejected_intervals_path(site.root, site.subject, site.session,
                                       site.task, site.label)
    

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


def smooth(x, fs):
    try:
        b = smooth.filter_b
    except AttributeError:
        b = sig.firwin2(int(fs*8), freq=[0, 0.01, 0.05, fs/2],
                        gain=[1.0, 1.0, 0.0001, 0.0], fs=fs)
        smooth.filter_b = b
    xds = sig.decimate(x, 10, ftype='fir', zero_phase=True)
    xf = sig.filtfilt(b, 1, xds)
    return sig.resample(xf, x.shape[0])


def detrend_hp(x, fs):
    try:
        b = detrend_hp.filter_b
    except AttributeError:
        # b = sig.firwin2(int(fs*8), freq=[0, 0.1, 0.2, 0.4, fs/2],
        #                 gain=[0., 1e-7, 0.1, 1., 1.], fs=fs)
        b = sig.firwin2(1001, freq=[0, 0.05, 0.1, fs/2],
                        gain=[0., 0.001, 1., 1.], fs=fs)
        detrend_hp.filter_b = b
    return sig.filtfilt(b, 1, x)

    
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
        df = detrend_hp(df, fs)
        
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


def collect_tdt_events(raw, events):
    ts = np.array([])
    keys = np.array([])
    event_map = map_events(events)
    for key in raw.epocs.keys() & event_map.keys():
        ts = np.append(ts, raw.epocs[key].onset)
        keys = np.append(keys, [key] * len(raw.epocs[key].onset))
    order = np.argsort(ts)
    return ts[order], keys[order]


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
