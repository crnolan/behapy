# %%
from pathlib import Path
import pandas as pd
import numpy as np
import behapy.fp as fp
import statsmodels.api as sm
import scipy.signal as sig
from sklearn.cluster import OPTICS
import holoviews as hv
from holoviews import opts
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread, rasterize
hv.extension('bokeh')
import panel
panel.extension(comms='vscode')

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
dlight_df = pd.Series(dlight_ds, index=ts_ds)
iso_df = pd.Series(iso_ds, index=ts_ds)

# %%
#
# Extend timeseries to n length before and after to not lose any initial
# / ending discontinuities
n = 100
iso_rmeans = iso_df.rolling(n).mean()
iso_rstds = iso_df.rolling(n).std()
d = iso_rmeans.diff(-n)
d_thresh = d.abs() > iso_rstds.median()
d_peaks = sig.find_peaks(d.abs(), prominence=iso_rstds.median())[0]
d_thresh_peaks = d_thresh.iloc[d_peaks]
d_thresh_peaks[d_thresh_peaks].shape
d_shade = datashade(hv.Curve((d.index, d)),
                    aggregator=ds.count(), cmap='blue')
iso_shade = datashade(hv.Curve((ts_ds, iso_ds)),
                    aggregator=ds.count(), cmap='blue')
stds_shade = datashade(hv.Curve((ts_ds, iso_rstds)),
                    aggregator=ds.count(), cmap='blue')
thresh_lines = hv.Overlay([hv.VLine(t) for t in d_thresh_peaks[d_thresh_peaks].index])
(d_shade.opts(width=800) + stds_shade.opts(width=800) + iso_shade.opts(width=800) * thresh_lines).cols(1).redim(
    x='time', y=hv.Dimension('F')).opts(height=300)

# %%
clust = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.01)
clust.fit(np.vstack([ts_ds, iso_ds]).T)

# %%
rlm_model = sm.RLM(dlight_ds, iso_ds)
rlm_results = rlm_model.fit()
rlm_results.params
# fit_df = pd.Series(rlm_results.fittedvalues, index=ts)

# %%
dlight_shade = datashade(hv.Curve((ts_ds, dlight_ds)),
                         aggregator=ds.count(), cmap='blue')
iso_shade = datashade(hv.Curve((ts_ds, iso_ds)),
                      aggregator=ds.count(), cmap='red')
fit_shade = datashade(hv.Curve((ts_ds, rlm_results.fittedvalues)),
                      aggregator=ds.count(), cmap='green')
(dlight_shade * iso_shade * fit_shade).redim(
    x='time', y=hv.Dimension('F')).opts(width=800, height=300, tools=['xbox_select, tap'])

# %%
dff_shade = datashade(hv.Curve((ts_ds, (dlight-rlm_results.fittedvalues)/rlm_results.fittedvalues)),
                      aggregator=ds.count(), cmap='blue')
dff_shade.redim(
    x='time', y=hv.Dimension('F')).opts(width=800, height=300, tools=['xbox_select, tap'])

# %%
