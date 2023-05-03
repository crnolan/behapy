# %%
from pathlib import Path
import pandas as pd
import numpy as np
import behapy.fp as fp
from behapy.pathutils import get_recordings, get_session_meta_path
import statsmodels.api as sm
import scipy.signal as sig
from intervaltree import IntervalTree, Interval
import holoviews as hv
from holoviews import opts
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread, rasterize
import panel as pn
import param
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
# pn.extension(comms='vscode')
import behapy.visuals as vis

# %%
BIDSROOT = Path('/scratch/cnolan/TB006')
RAWROOT = BIDSROOT / 'rawdata'
ANALROOT = BIDSROOT / 'derivatives/ds64'

# %%
recordings = pd.DataFrame(get_recordings(RAWROOT))
sites = recordings.loc[:, ['subject', 'session', 'label']].drop_duplicates()


def get_recording(index):
    r = sites.iloc[index]
    site = fp.SessionSite.load(BIDSROOT, r.subject, r.session, r.label, 'iso')
    ds_site = site.downsample(64)
    intervals = fp.find_disconnects(ds_site)
    return ds_site, intervals


def filter_fit(site, intervals):
    rej = fp.reject(site, intervals)
    fit = fp.fit(rej)
    rej.data[:, (rej.iso_index + 1) % 2] = fit.fittedvalues
    return rej


# %%
dash = vis.PreprocessDashboard(sites, get_recording, filter_fit)
# dash.view()
pn.serve(dash.view(), port=8080)

# %%
runs = recordings.loc[:, ['sub', 'ses', 'task', 'run', 'label']].drop_duplicates()
index_cols = ['sub', 'ses', 'task', 'run', 'label']
downsample_factor = 64


# %%
site = fp.SessionSite.load(BIDSROOT, '27', 'Retrain1.1', 'LDMS', 'iso')
site = site.downsample(64)
bounds = fp.find_discontinuities(site, mean_window=3, nstd_thresh=2)
# bounds = fp.find_disconnects(site)
iso_shade = datashade(hv.Curve((site.ts, site.iso())),
                      aggregator=ds.count(), cmap='blue')
start_lines = hv.Overlay([hv.VLine(site.ts[i]).opts(color='green') for i, j in bounds])
end_lines = hv.Overlay([hv.VLine(site.ts[j]).opts(color='red') for i, j in bounds])
(iso_shade.opts(width=800) * start_lines * end_lines).redim(
    x='time', y=hv.Dimension('F')).opts(height=300)

# %%


# %%
BIDSROOT = Path('..')
RAWROOT = BIDSROOT / 'rawdata'
dlight, dlight_meta = fp.load_channel(
    RAWROOT, '18', 'rr10_4', 'rr10', 1, 'RDMS', 'dLight')
iso, iso_meta = fp.load_channel(
    RAWROOT, '18', 'rr10_4', 'rr10', 1, 'RDMS', 'iso')
ts = np.arange(dlight.shape[0]) / dlight_meta['fs']

# %%
# downsample first
downsample_factor = 64
iso_ds = sig.decimate(iso, downsample_factor, ftype='fir', zero_phase=True)
dlight_ds = sig.decimate(dlight, downsample_factor, ftype='fir', zero_phase=True)
ts_ds = ts[::downsample_factor]

# %%
# pad
n = int(dlight_meta['fs'] / downsample_factor * 5)
ts_pad = np.concatenate([ts_ds[:n] - (ts_ds[n] - ts_ds[0]),
                        ts_ds,
                        ts_ds[-n:] + (ts_ds[-1] - ts_ds[-n-1])])
iso_pad = np.concatenate([np.repeat(iso_ds[0], n),
                          iso_ds,
                          np.repeat(iso_ds[-1], n)])
dlight_pad = np.concatenate([np.repeat(dlight_ds[0], n),
                             dlight_ds,
                             np.repeat(dlight_ds[-1], n)])

# %%
dlight_df = pd.Series(dlight_pad, index=ts_pad)
iso_df = pd.Series(iso_pad, index=ts_pad)

# %%
#
from numpy.lib.stride_tricks import sliding_window_view
std_n = int(dlight_meta['fs'] / downsample_factor * 30)
np.std(sliding_window_view(iso_pad, std_n), axis=-1).shape
iso_rmeans = iso_df.rolling(n).mean()
iso_rstds = iso_df.rolling(std_n).std()
d = iso_rmeans.diff(-n)
d_thresh = d.abs() > iso_rstds.median() * 3
d_peaks = sig.find_peaks(d.abs(), prominence=iso_rstds.median())[0]
d_thresh_peaks = d_thresh.iloc[d_peaks]
d_thresh_peaks[d_thresh_peaks].shape
d_shade = datashade(hv.Curve((d.index, d)),
                    aggregator=ds.count(), cmap='blue')
iso_shade = datashade(hv.Curve((ts_pad, iso_pad)),
                    aggregator=ds.count(), cmap='blue')
stds_shade = datashade(hv.Curve((ts_pad, iso_rstds)),
                    aggregator=ds.count(), cmap='blue')
thresh_lines = hv.Overlay([hv.VLine(t) for t in d_thresh_peaks[d_thresh_peaks].index])
(d_shade.opts(width=800) + stds_shade.opts(width=800) + iso_shade.opts(width=800) * thresh_lines).cols(1).redim(
    x='time', y=hv.Dimension('F')).opts(height=300)

# %%
# If the mean of any interval is near zero, mask out the samples in that
# interval, and for a second after the interval.
reject_intervals = IntervalTree()
# time for signal to stabilise after a disconnection (in seconds)
bounce_buffer = 2.0
# time to exclude before a disconnection because the disconnection time
# will be the centre of the drop
disconnect_buffer = 1.0

disconts = d_thresh_peaks[d_thresh_peaks].index.tolist()
disconts = [0] + disconts + [ts_ds[-1]]
for t0, t1 in zip(disconts[:-1], disconts[1:]):
    if iso_df.loc[t0:t1].mean() < iso_rstds.median() * 5:
        # If any true signal has a mean smaller than 5 times the median
        # std, we've got problems.
        reject_intervals.add(Interval(t0-disconnect_buffer, t1+bounce_buffer))

# %%
import holoviews.streams as streams

def reject_overlay(intervals, selected=[]):
    if intervals is None or intervals == []:
        return hv.Overlay([])
    colors = ['red' if x in selected else 'pink' for x in range(len(intervals))]
    return hv.Overlay([hv.VSpan(*(interval[0:2])).opts(color=c)
                       for interval, c in zip(intervals, colors)])

def record_reject(boundsx, x, y):
#     if boundsx is None:
#         return hv.Overlay([])
    if None not in [x, y]:
        reject_intervals.remove_overlap(x)
    if boundsx is not None and None not in boundsx:
        reject_intervals.add(Interval(*boundsx))
        reject_intervals.merge_overlaps()
    return reject_overlay(reject_intervals)

trace_shade = datashade(hv.Curve((iso_df.index, iso_df)), aggregator=ds.count()).redim(
    x='time', y=hv.Dimension('F')).opts(width=800, tools=['xbox_select, tap'])

reject_stream = streams.BoundsX(source=trace_shade, boundsx=None, transient=True)
select_stream = streams.DoubleTap(source=trace_shade, x=None, y=None, transient=True)
reject_dmap = hv.DynamicMap(record_reject, streams=[reject_stream, select_stream])
(trace_shade * reject_dmap).opts(height=300, responsive=True)

# %%
# Filter out rejected regions
# mask = pd.Series(False, index=ts)
filtered = pd.DataFrame(np.vstack([dlight, iso]).T, index=ts, columns=['dlight', 'iso'])
for i in reject_intervals:
    filtered.loc[i[0]:i[1]] = pd.NA
filtered.dropna(inplace=True)

# %%
rlm_model = sm.RLM(filtered.dlight, filtered.iso)
rlm_results = rlm_model.fit()
# fit_df = pd.Series(rlm_results.fittedvalues, index=ts)

# %%
dlight_shade = datashade(hv.Curve((filtered.index, filtered.dlight)),
                         aggregator=ds.count(), cmap='blue')
iso_shade = datashade(hv.Curve((filtered.index, filtered.iso)),
                      aggregator=ds.count(), cmap='red')
fit_shade = datashade(hv.Curve((filtered.index, rlm_results.fittedvalues)),
                      aggregator=ds.count(), cmap='green')
(dlight_shade * iso_shade * fit_shade).redim(
    x='time', y=hv.Dimension('F')).opts(width=800, height=300, tools=['xbox_select, tap'])

# %%
dff_shade = datashade(hv.Curve((filtered.index, (filtered.dlight-rlm_results.fittedvalues)/rlm_results.fittedvalues)),
                      aggregator=ds.count(), cmap='blue')
dff_shade.redim(
    x='time', y=hv.Dimension('F')).opts(width=800, height=300, tools=['xbox_select, tap'])

# %%
