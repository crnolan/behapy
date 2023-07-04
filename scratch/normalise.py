# %%
from pathlib import Path
import pandas as pd
import numpy as np
import behapy.fp as fp
from behapy.pathutils import list_raw, get_session_meta_path
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
pn.extension(comms='vscode')
import behapy.visuals as vis
from behapy.visuals import signal_shade
hv.extension('bokeh')

# %%
BIDSROOT = Path('/scratch/cnolan/TB006')
RAWROOT = BIDSROOT / 'rawdata'
ANALROOT = BIDSROOT / 'derivatives/ds64'

# %%
recordings = pd.DataFrame(list_raw(BIDSROOT))
sites = recordings.loc[:, ['subject', 'session', 'task', 'run', 'label']].drop_duplicates()


def get_recording(index):
    r = sites.iloc[index]
    signal = fp.load_signal(BIDSROOT, r.subject, r.session, r.task, r.run,
                            r.label, 'iso')
    return signal


# %%
data = get_recording(0).astype(np.float64)
data_ds = fp.downsample(data, 64)
discontig = fp.find_discontinuities(data_ds, mean_window=3, nstd_thresh=2)
bounds = fp.find_disconnects(data_ds)
data_reject = fp.reject(data_ds, bounds)

# %%
plot = (signal_shade(data_ds['dLight'], 'F', 'red') *
        signal_shade(data_ds['iso'], 'F', 'blue') *
        hv.Overlay([hv.VLine(x0/data_ds.attrs['fs']).opts(color='green') for x0, x1 in discontig]) *
        hv.Overlay([hv.VLine(x1/data_ds.attrs['fs']).opts(color='orange') for x0, x1 in discontig]))
plot.opts(opts.RGB(responsive=True, min_width=600, min_height=200))

# %%
detrended = fp.series_like(data_ds, 'detrended')
detrended[:] = fp.detrend_hp(fp.reject(data_ds['dLight'], bounds, fill=True), data_ds.attrs['fs'])
# detrended[:] = fp.reject(data_ds['dLight'], bounds, fill=True)

# %%
((signal_shade(data_ds['dLight'], 'F', 'red') * signal_shade(data_ds['iso'], 'F', 'blue')).opts(xaxis=None) +
 signal_shade(detrended, 'F-filt', 'green')).cols(1).opts(opts.RGB(responsive=True, min_width=600, min_height=200))

# %%
def get_fitted_value(iso, signal):
    reg = sm.OLS(iso, signal)
    fit = reg.fit()
    return fit.fittedvalues[fit.fittedvalues.shape[0]//2]

from numpy.lib.stride_tricks import sliding_window_view
fit = fp.series_like(data_ds, 'fit')
fit.iloc[24:-25] = [get_fitted_value(x[0, :], x[1, :])
                    for x in sliding_window_view(data_ds.to_numpy(), 50, axis=0)]

# %%
(signal_shade(data_ds['dLight'], 'F', 'red') *
 signal_shade(data_ds['iso'], 'F', 'blue') *
 signal_shade(fit, 'F', 'green')).opts(opts.RGB(responsive=True, min_width=600, min_height=200))

# %%
# How about an exponential curve fit to remove bleaching effects from
# both channels, then fit the iso to the dLight?
debleach_iso = fp.debleach(data_reject['iso'])
debleach_dlight = fp.debleach(data_reject['dLight'])
efit_iso = fp.series_like(data_reject, 'efit_iso')
efit_iso.loc[debleach_iso.index] = debleach_iso
efit_dlight = fp.series_like(data_reject, 'efit_dlight')
efit_dlight.loc[debleach_dlight.index] = debleach_dlight

efit_dlight_smooth = fp.smooth(efit_dlight)
efit_iso_smooth = fp.smooth(efit_iso)
fitfit = fp.series_like(data_reject, 'fitfit')
ols_model = sm.OLS(efit_dlight_smooth, efit_iso_smooth)
fitfit.loc[:] = efit_dlight - ols_model.fit().fittedvalues


# %%
raw_shade = (signal_shade(data_ds['dLight'], 'F', 'red') *
             signal_shade(fp.exp_fit(data_reject['iso']), 'F', 'red') *
             signal_shade(data_ds['iso'], 'F', 'blue') *
             signal_shade(fp.exp_fit(data_reject['dLight']), 'F', 'blue'))
fit_shade = (signal_shade(efit_iso, 'dF/F', 'blue') *
             signal_shade(efit_iso_smooth, 'dF/F', 'blue') *
             signal_shade(efit_dlight, 'dF/F', 'red') *
             signal_shade(efit_dlight_smooth, 'dF/F', 'red') *
             signal_shade(fitfit, 'dF/F', 'green'))
(raw_shade.opts(xaxis=None) + fit_shade).cols(1).opts(opts.RGB(responsive=True, min_width=600, min_height=300))

# %%
import matplotlib.pyplot as plt
b = sig.firwin(1001, cutoff=[0.05], fs=15, pass_zero=False)
w, h = sig.freqz(b)
fig = plt.figure()
plt.title('Digital filter frequency response')
ax1 = fig.add_subplot(111)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
plt.plot(w, angles, 'g')
plt.ylabel('Angle (radians)', color='g')
plt.grid()
plt.axis('tight')
plt.xlim([0, 0.6])
plt.show()
# %%
