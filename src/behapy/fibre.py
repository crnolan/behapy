from typing import Tuple, Iterable
import numpy as np
import scipy.signal as sig


def smooth(x, fs):
    try:
        b = smooth.filter_b
    except AttributeError:
        b = sig.firwin2(8192, freq=[0, 0.1, 0.2, fs/20],
                        gain=[1.0, 1.0, 0.0001, 0.0], fs=fs/10)
        smooth.filter_b = b
    xds = sig.decimate(x, 10, ftype='fir', zero_phase=True)
    xf = sig.filtfilt(b, 1, xds)
    return sig.resample(xf, x.shape[0])


def normalise(signal, control, mask, fs, method='fit'):
    smoothed = smooth(control[mask], fs=fs)
    fit = np.polyfit(smoothed, signal[mask], deg=1)
    signal_fit = fit[0] * control + fit[1]
    df = signal - signal_fit
    if method == 'fit':
        return df / signal_fit
    elif method == 'const':
        return df / np.mean(signal)
    elif method == 'df':
        return df
    else:
        raise ValueError("Unrecognised normalisation method {}".format(method))

    
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
