from typing import Tuple, Iterable, Union
from pathlib import Path
import json
import logging
import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit
from collections import namedtuple
import pandas as pd
import statsmodels.api as sm
from intervaltree import IntervalTree, Interval
from numpy.lib.stride_tricks import sliding_window_view
import bottleneck as bn
from .pathutils import (
    get_raw_fibre_path,
    list_raw,
    get_rejected_intervals_path,
    get_preprocessed_fibre_path,
)
from .config import load_preprocess_config


logger = logging.getLogger(__name__)
Event = namedtuple("Event", ["name", "fields", "codes", "onset", "offset"])


def series_like(
    df: Union[pd.Series, pd.DataFrame], name: str, default: float = 0.0
) -> pd.Series:
    series = pd.Series(default, index=df.index, name=name)
    series.attrs = df.attrs.copy()
    _ = series.attrs.pop("artifact_channel", None)
    _ = series.attrs.pop("channels", None)
    _ = series.attrs.pop("iso_channel", None)
    _ = series.attrs.pop("channel", None)
    _ = series.attrs.pop("types", None)
    _ = series.attrs.pop("references", None)
    return series


def load_channel(root, subject, session, task, run, label, channel):
    data_fn = get_raw_fibre_path(
        root, subject, session, task, run, label, channel, "npy"
    )
    meta_fn = get_raw_fibre_path(
        root, subject, session, task, run, label, channel, "json"
    )
    with open(meta_fn) as file:
        meta = json.load(file)
    data = np.load(data_fn)
    return data, meta


def load_signals(root, subject, session, task, run, label):
    """Load all raw signals for a given site."""
    root = Path(root).absolute()
    recordings = pd.DataFrame(
        list_raw(
            root, subject=subject, session=session, task=task, run=run, label=label
        )
    )
    subjects = recordings.loc[:, "subject"].unique()
    sessions = recordings.loc[:, "session"].unique()
    tasks = recordings.loc[:, "task"].unique()
    labels = recordings.loc[:, "label"].unique()
    if any([item.shape[0] != 1 for item in [subjects, sessions, tasks, labels]]):
        msg = (
            f"Multiple signal names found for session "
            f"with subject {subject}, session {session}, task {task}, "
            f"run {run} and label {label}"
        )
        logging.error(msg)
        raise ValueError(msg)

    # Load channels
    data = []
    t0 = None
    fs = None
    channel_types = {}
    references = {}
    for r in recordings.itertuples():
        d, meta = load_channel(
            root=root,
            subject=r.subject,
            session=r.session,
            task=r.task,
            run=r.run,
            label=r.label,
            channel=r.channel,
        )
        if fs is None:
            fs = meta["fs"]
        if t0 is None:
            t0 = meta["start_time"]
        if (fs != meta["fs"]) or (t0 != meta["start_time"]):
            msg = (
                "Unequal sample frequencies and/or start times "
                "for subject {}, session {}, task {}, run {} and label {}"
            )
            msg.format(subject, session, task, run, label)
            raise ValueError(msg)
        if "type" in meta:
            channel_types[r.channel] = meta["type"]
        elif r.channel in ["iso", "isos", "isosbestic"]:
            channel_types[r.channel] = "isosbestic"
        else:
            channel_types[r.channel] = "unknown"
        if (
            "reference" in meta
            and meta["reference"] is not None
            and meta["reference"] != ""
        ):
            references[r.channel] = meta["reference"]
        else:
            references[r.channel] = None
        t = pd.Index(np.arange(d.shape[0]) / fs + t0, name="time")
        data.append(pd.Series(d, name=r.channel, index=t))

    signal = pd.concat(data, axis=1)
    signal.index.name = "time"
    signal.attrs["root"] = root
    signal.attrs["fs"] = fs
    signal.attrs["start_time"] = t0
    signal.attrs["subject"] = subject
    signal.attrs["session"] = session
    signal.attrs["task"] = task
    signal.attrs["run"] = run
    signal.attrs["label"] = label
    signal.attrs["types"] = channel_types
    signal.attrs["references"] = references
    return signal


def downsample(signal, factor=None):
    if factor is None:
        # Downsample to something reasonable
        factor = 1
        while signal.attrs["fs"] / factor > 20:
            factor *= 2
    ds = sig.decimate(signal.to_numpy(), factor, ftype="fir", zero_phase=True, axis=0)
    df = pd.DataFrame(ds, index=signal.index[::factor], columns=signal.columns)
    df.attrs = signal.attrs
    df.attrs["fs"] = signal.attrs["fs"] / factor
    return df


def save_rejections(tree, root, subject, session, task, run, label):
    # Save the provided IntervalTree as a CSV
    fn = get_rejected_intervals_path(root, subject, session, task, run, label)
    tree.merge_overlaps()
    df = pd.DataFrame.from_records(
        [(b[0], b[1]) for b in list(tree)], columns=["start_time", "end_time"]
    )
    fn.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fn, index=False)


def load_rejections(root, subject, session, task, run, label):
    # Load rejected intervals if present
    fn = get_rejected_intervals_path(root, subject, session, task, run, label)
    intervals = []
    if not fn.exists():
        return None
    # Load the CSV
    rej = pd.read_csv(fn)
    intervals = [(r.start_time, r.end_time) for r in rej.itertuples()]
    return IntervalTree.from_tuples(intervals)


def find_discontinuities(signal, mean_window=3, std_window=30, nstd_thresh=2):
    # How many samples to consider for the sliding mean
    n = int(signal.attrs["fs"] * mean_window)
    std_n = int(signal.attrs["fs"] * std_window)
    data = signal.to_numpy()
    # Use the median of a sliding window STD as our characteristic STD.
    data_rstds = bn.move_std(data, std_n, axis=0)
    data_thresh = np.nanmedian(data_rstds, axis=0)
    data_rmeans = bn.move_mean(np.pad(data, ((n, n)), "edge"), n, axis=0)
    mean_thresh = data_thresh * nstd_thresh
    d = data_rmeans[n:-n] - data_rmeans[(n * 2) :]
    d_thresh = np.abs(d) > mean_thresh
    # Find the start and end of each mean shift
    mean_shift_bounds = np.diff(d_thresh.astype(int), axis=0)
    try:
        # If the first bound is a falling edge, insert a rising edge,
        # unless it is the first sample, in which case ignore it.
        if mean_shift_bounds[0] == -1:
            mean_shift_bounds[0] = 0
        elif mean_shift_bounds[mean_shift_bounds != 0][0] == -1:
            mean_shift_bounds[0] = 1
        # If the last bound is a rising edge, insert a falling edge
        if mean_shift_bounds[-1] == 1:
            mean_shift_bounds[-1] = 0
        elif mean_shift_bounds[mean_shift_bounds[:] != 0][-1] == 1:
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
        k = np.argmax(
            np.abs(data[offset:onset:-1] - data_rmeans[[onset + n]]) < data_thresh
        )
        if k > 0:
            onsets[i] = offset - k
        k = np.argmax(
            np.abs(data[onset:offset:1] - data_rmeans[[offset + n]]) < data_thresh
        )
        if k > 0:
            offsets[i] = onset + k
    return [
        (onset, offset) for onset, offset in zip(onsets, offsets) if offset - onset > 0
    ]


def find_disconnects(
    signal, zero_nstd_thresh=5, mean_window=3, std_window=30, nstd_thresh=2
):
    dc_intervals = IntervalTree()
    for ch in signal:
        bounds = find_discontinuities(
            signal[ch],
            mean_window=mean_window,
            std_window=std_window,
            nstd_thresh=nstd_thresh,
        )
        data = signal[ch].to_numpy()
        ts = signal[ch].index.to_numpy()
        std_n = int(signal.attrs["fs"] * std_window)
        data_rstds = bn.move_std(data, std_n, axis=0)
        zero_thresh = np.nanmedian(data_rstds, axis=0) * zero_nstd_thresh
        bounds = [(0, 0)] + bounds + [(data.shape[0] - 1, data.shape[0] - 1)]
        for (on0, off0), (on1, off1) in zip(bounds[:-1], bounds[1:]):
            if np.any(np.mean(data[off0:on1], axis=0) < zero_thresh):
                dc_intervals.add(Interval(ts[on0], ts[off1]))
    dc_intervals.merge_overlaps()
    return dc_intervals


def intervals_to_mask(signal: pd.DataFrame, intervals: IntervalTree) -> pd.Series:
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


def reject(signal: pd.DataFrame, intervals: IntervalTree) -> pd.DataFrame:
    """Filter the site data to remove the specified intervals.

    Rejected samples are replaced with NaN.

    Args:
        signal: The timeseries from which to remove or replace the
            supplied intervals.
        intervals: A list of intervals to reject.

    Returns:
        A copy of the signal with the specified intervals replaced with NaN.
    """
    mask = intervals_to_mask(signal, intervals)
    return signal.where(mask, np.nan)


def map_events(events: Iterable[Event]):
    """Create a dict mapping event codes to the respective events."""
    return {key: event for event in events.values() for key in event.fields}


def smooth(signal, params):
    column_index = pd.MultiIndex.from_product(
        [[signal.name], []], names=["channel", "sigtype"]
    )
    results = pd.DataFrame(index=signal.index, columns=column_index)
    smooth_params = params.get("smooth", {})
    numtaps = smooth_params.get("numtaps", 1001)
    cutoff = smooth_params.get("cutoff", 1.0)
    try:
        if smooth.numtaps != numtaps or smooth.cutoff != cutoff:
            raise AttributeError("Filter parameters changed")
        b = smooth.filter_b
    except AttributeError:
        b = sig.firwin(numtaps, cutoff=[cutoff], fs=signal.attrs["fs"], pass_zero=True)
        smooth.filter_b = b
        smooth.numtaps = numtaps
        smooth.cutoff = cutoff
    # smoothed = series_like(data, 'smoothed')
    results[signal.name, "flp"] = sig.filtfilt(
        b,
        1,
        signal.interpolate(method="linear", limit_direction="both").to_numpy(),
        axis=0,
    ).astype(np.float32)
    return results


def detrend(signal: pd.Series, params: dict = {}) -> pd.DataFrame:
    """Detrend the data using a FIR filter."""
    column_index = pd.MultiIndex.from_product(
        [[signal.name], []], names=["channel", "sigtype"]
    )
    results = pd.DataFrame(index=signal.index, columns=column_index)
    detrend_params = params.get("detrend", {})
    numtaps = detrend_params.get("numtaps", 1001)
    cutoff = detrend_params.get("cutoff", 0.05)
    scale = detrend_params.get("scale", True)
    try:
        if detrend.numtaps != numtaps or detrend.cutoff != cutoff:
            raise AttributeError("Filter parameters changed")
        b = detrend.filter_b
    except AttributeError:
        b = sig.firwin(numtaps, cutoff=[cutoff], fs=signal.attrs["fs"], pass_zero=False)
        detrend.filter_b = b
        detrend.numtaps = numtaps
        detrend.cutoff = cutoff
    results[signal.name, "fhp"] = sig.filtfilt(
        b,
        1,
        signal.interpolate(method="linear", limit_direction="both").to_numpy(),
        axis=0,
    ).astype(np.float32)
    smoothed = smooth(signal, params)
    results[smoothed.columns] = smoothed
    if scale:
        results[signal.name, "dff"] = (
            results[signal.name, "fhp"] / results[signal.name, "flp"]
        )
    else:
        results[signal.name, "dff"] = (
            results[signal.name, "fhp"] / results[signal.name, "flp"].mean()
        )
    return results


def sglexp(x, a1, b1, c):
    return a1 * np.exp(-b1 * x) + c


def dblexp(x, a1, a2, b1, b2, c):
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c


def exp_min_fit(signal, params):
    _params = params.get("exp_min_fit", {})
    minpoints = signal.cummin().drop_duplicates()
    M = signal.max()
    if _params.get("method", "single") == "double":
        popt, _ = curve_fit(
            dblexp,
            minpoints.index.to_numpy().T,
            minpoints.to_numpy().T,
            maxfev=10000,
            bounds=([0, 0, 0, 0, -M * 10], [M, M, 1, 1, M * 10]),
            nan_policy="omit",
            loss="soft_l1",
        )
        fit = dblexp(signal.index.to_numpy(), *popt)
    else:
        popt, _ = curve_fit(
            sglexp,
            minpoints.index.to_numpy().T,
            minpoints.to_numpy().T,
            maxfev=10000,
            bounds=([0, 0, -M * 10], [M, 1, M * 10]),
            nan_policy="omit",
            loss="soft_l1",
        )
        fit = sglexp(signal.index.to_numpy(), *popt)
    logger.debug(f"popt for channel {signal.name}: {popt}")
    return fit


def debleach(signal: pd.Series, params: dict = {}) -> pd.DataFrame:
    """Debleach the data using an exponential fit over minimums.

    Takes the cumulative minimum of the signal, finds the best fit using
    robust linear least squares, and calculates the dff by dividing the
    difference of the signal and the fit by the fit.
    """
    column_index = pd.MultiIndex.from_product(
        [[signal.name], ["fit", "dff"]], names=["channel", "sigtype"]
    )
    results = pd.DataFrame(index=signal.index, columns=column_index)
    fit = exp_min_fit(signal, params)
    dff = (signal - fit) / fit
    results[signal.name, "fit"] = fit
    results[signal.name, "dff"] = dff
    return results


def rlm(df, signal_name, control_name, params):
    """Fit the site data to the isobestic channel using a robust regression."""
    column_index = pd.MultiIndex.from_product(
        [[signal_name, control_name], ["fit", "dff"]], names=["channel", "sigtype"]
    )
    results = pd.DataFrame(index=df.index, columns=column_index)
    fit_signal = debleach(df[signal_name], params)
    fit_control = debleach(df[control_name], params)
    results[fit_signal.columns] = fit_signal
    results[fit_control.columns] = fit_control
    # for name in fit_signal:
    #     results[name] = fit_signal[name]
    # for ch in fit_control:
    #     results[control_name, ch] = fit_control[ch]
    filtered = results.dropna()
    results[signal_name, "control_fit"] = (
        sm.RLM(filtered[signal_name, "dff"], filtered[control_name, "dff"])
        .fit()
        .fittedvalues
    )
    results[signal_name, "dff_fit"] = (
        results[signal_name, "dff"] - results[signal_name, "control_fit"]
    )
    return results


def ratiometric(df, signal, reference, params):
    dff = detrend(df[[signal, reference]], params)
    df_filt = df.dropna()
    fitted = series_like(df, "fitted")
    fitted[df_filt.index] = sm.RLM(df_filt[signal], df_filt[control]).fit().fittedvalues
    dff = series_like(df, signal)
    dff[df_filt.index] = (
        df.loc[df_filt.index, signal] - fitted[df_filt.index]
    ) / fitted[df_filt.index]
    return dff.to_frame()


def get_signal_channels(channels: pd.DataFrame) -> list[str]:
    """Get the channel names of all signal channels.

    Args:
        channels: A DataFrame with channel names as columns and
            channel types in the 'types' attribute.

    Returns:
        A list of channel names of type 'signal'.
    """
    return [ch for ch in channels.columns if channels.attrs["types"][ch] == "signal"]


def is_reference_type(channel_type):
    """Check if a channel type is a known reference channel."""
    return channel_type in ["isosbestic", "control", "ratiometric"]


def get_normalisation_method(channel_type, methods):
    method = None
    if isinstance(methods, str):
        method = methods
    if isinstance(methods, dict):
        method = methods.get(channel_type, "detrend")
    if channel_type == "raw":
        if method is None:
            method = "detrend"
        if method not in ["detrend", "fit"]:
            logger.warning(
                f"Method {methods} invalid for raw channel type, " f"using 'detrend'."
            )
            method = "detrend"
        return method
    elif channel_type in ["isosbestic", "control"]:
        if method is None:
            method = "detrend"
        if method not in ["detrend", "fit", "rlm"]:
            logger.warning(
                f"Method {methods} invalid for {channel_type} channel type, "
                f"using 'detrend'."
            )
            method = "detrend"
        return method
    elif channel_type == "ratiometric":
        if method is None:
            method = "ratio"
        if method not in ["detrend", "fit", "ratio"]:
            logger.warning(
                f"Method {methods} invalid for ratiometric channel type, "
                f"using 'ratio'."
            )
            method = "ratio"
        return method
    else:
        raise ValueError(f"Unknown channel type {channel_type}")


def normalise(
    channels: pd.DataFrame,
    methods: Union[None, str, dict[str, str]] = None,
    params: dict = {},
) -> pd.DataFrame:

    channel_types = channels.attrs["types"]
    references = channels.attrs["references"]
    signals = [ch for ch in channels.columns if channel_types[ch] == "signal"]
    column_index = pd.MultiIndex.from_product(
        [signals, []], names=["channel", "sigtype"]
    )
    results = pd.DataFrame(index=channels.index, columns=column_index)
    results.attrs = channels.attrs.copy()
    results.attrs.pop("types", None)
    results.attrs.pop("references", None)
    for ch in signals:
        ref_ch = None
        if references[ch] is None:
            # Signal has no reference, use method for raw
            method = get_normalisation_method("raw", methods)
        else:
            if is_reference_type(channel_types[references[ch]]):
                ref_ch = references[ch]
                method = get_normalisation_method(channel_types[ref_ch], methods)
            else:
                raise ValueError(
                    f"Unknown reference channel type "
                    f"{channel_types[references[ch]]} for "
                    f"channel {ch}"
                )
        logger.info(
            f"Normalising channel {ch} using method {method}")
        if method == "detrend":
            dff = detrend(channels[ch], params)
            results[dff.columns] = dff
            results[ch, "z"] = (dff[ch, "dff"] - dff[ch, "dff"].mean()) / dff[
                ch, "dff"
            ].std()
        elif method == "fit":
            fit = debleach(channels[ch], params)
            results[fit.columns] = fit
            results[ch, "z"] = (fit[ch, "dff"] - fit[ch, "dff"].mean()) / fit[
                ch, "dff"
            ].std()
        elif method == "rlm":
            rlm_fit = rlm(channels, ch, ref_ch, params)
            results[rlm_fit.columns] = rlm_fit
            results[ch, "z"] = (
                rlm_fit[ch, "dff_fit"] - rlm_fit[ch, "dff_fit"].mean()
            ) / rlm_fit[ch, "dff_fit"].std()
        elif method == "ratio":
            dff = ratiometric(channels, ch, ref_ch, params)
            results[ch, "dff"] = dff
            results[ch, "z"] = (dff - dff.mean()) / dff.std()

    return results.sort_index(axis=1, level=0, sort_remaining=False)


def preprocess(root, subject, session, task, run, label):
    config = load_preprocess_config(root)
    intervals = load_rejections(root, subject, session, task, run, label)
    # Check if the recording has rejections saved
    if intervals is None:
        logger.info(
            f"Recording for subject {subject}, "
            f"session {session}, task {task}, "
            f"run {run} and label {label} has no "
            f"rejections file, skipping."
        )
        return False
    logger.info(
        f"Preprocessing subject {subject}, "
        f"session {session}, task {task}, "
        f"run {run}, label {label}..."
    )
    recording = load_signals(root, subject, session, task, run, label)
    recording = downsample(recording, 64)
    rej = reject(recording, intervals)
    #
    ch = recording.attrs["channels"]
    # We were doing a robust regression, but the fit isn't good enough.
    # Let's just detrend and divide by the smoothed signal instead.
    # dff = fp.series_like(recording, name='dff')
    # dff.loc[rej.index] = fp.detrend(rej[ch])
    if config["method"] == "rlm":
        dff, fitted = rlm(rej)
    else:
        dff = detrend(rej[ch], config)
    # dff.name = 'dff'
    # dff = dff.to_frame()
    dff["mask"] = rej["mask"]
    dff.attrs["root"] = str(dff.attrs["root"])
    data_fn = get_preprocessed_fibre_path(
        root, subject, session, task, run, label, "parquet"
    )
    meta_fn = get_preprocessed_fibre_path(
        root, subject, session, task, run, label, "json"
    )
    data_fn.parent.mkdir(parents=True, exist_ok=True)
    try:
        dff.to_parquet(data_fn, engine="pyarrow")
    except TypeError as e:
        logger.warning(f"Serialization error with pyarrow: {e}")
        dff.to_parquet(data_fn, engine="fastparquet")
    meta = dff.attrs
    meta["root"] = str(root)
    with open(meta_fn, "w") as file:
        json.dump(meta, file)
    return True


def load_preprocessed_fibre(root, subject, session, task, run, label):
    root = Path(root)
    data_fn = get_preprocessed_fibre_path(
        root, subject, session, task, run, label, "parquet"
    )
    meta_fn = get_preprocessed_fibre_path(
        root, subject, session, task, run, label, "json"
    )
    if not data_fn.exists():
        return None
    data = pd.read_parquet(data_fn)
    if data.index.name != "time":
        data.index = pd.to_timedelta(data.index, unit="s")
        data.index.name = "time"
    if not pd.api.types.is_timedelta64_dtype(data.index):
        data.index = pd.to_timedelta(data.index, unit="s")
        data.index.name = "time"
    with open(meta_fn, "r") as file:
        meta = json.load(file)
    data.attrs.update(meta)
    return data


def epoch_events(
    data: Iterable[float],
    events: Iterable[float],
    window: Tuple[float, float],
    baseline_window: Tuple[float, float],
    fs: float,
    tstart: float = 0.0,
    method: str = "z",
) -> Tuple[np.ndarray, np.ndarray]:
    """Epoch data at supplied event times and optionally baseline.

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
    if method != "nobase":
        baseline_ix = np.ceil(np.array(baseline_window) * fs).astype(int)
    dslice = lambda b, w: data[slice(*(b + w))]
    if method == "z":
        z = lambda d, b: (d - np.mean(b)) / np.std(b)
        epochs = np.array(
            [z(dslice(ix, window_ix), dslice(ix, baseline_ix)) for ix in ixs]
        )
    elif method == "base":
        base = lambda d, b: d - np.mean(b)
        epochs = np.array(
            [base(dslice(ix, window_ix), dslice(ix, baseline_ix)) for ix in ixs]
        )
    elif method == "nobase":
        epochs = np.array([dslice(ix, window_ix) for ix in ixs])
    else:
        raise ValueError("Invalid epoching method {}".format(method))

    return epochs, np.arange(*window_ix) / fs
