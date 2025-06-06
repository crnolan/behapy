import logging
from typing import Union
from functools import partial, reduce
from intervaltree import Interval
import json
from . import fp
from .config import load_preprocess_config
import pandas as pd
import holoviews as hv
from holoviews import opts
import datashader as ds
from holoviews.operation.datashader import datashade
import holoviews.streams as streams
import panel as pn
import param

hv.extension("bokeh")
pn.extension("tabulator", "mathjax", loading_spinner="dots", comms="vscode")
# pn.extension('tabulator')

logger = logging.getLogger(__name__)


def channel_color(channel_name: str, color: Union[str, None]) -> str:
    if color is not None:
        return color
    if hasattr(channel_color, "colors"):
        if channel_name in channel_color.colors:
            color = channel_color.colors[channel_name]
        else:
            color = next(channel_color.color_iter)
            channel_color.colors[channel_name] = color
    else:
        channel_color.colors = {}
        channel_color.color_iter = iter(hv.Cycle("Colorblind").values)
        color = next(channel_color.color_iter)
        channel_color.colors[channel_name] = color
    return color


def channel_curve(
    channel: pd.Series,
    ydim=None,
    group=None,
    label=None,
    color=None,
    line_dash=None,
    ylabel=None,
) -> hv.Curve:
    """Create a holoviews Curve from a single channel of data.

    Args:
        channel (pd.Series): A pandas Series containing the channel
        data. The name of the series should be a 2-tuple of the form
        (channel_name, channel_type), with channel_type being one of the
        raw channel types or outputs of the normalisation steps."""
    if (
        channel.name is None
        or not isinstance(channel.name, tuple)
        or not len(channel.name) == 2
    ):
        raise ValueError(
            "Channel name must be a 2-tuple of the form (channel_name, channel_type)"
        )
    channel_name = channel.name[0]
    channel_type = channel.name[1]
    # x = pd.to_timedelta(channel.index, unit="s")
    x = channel.index.to_numpy()
    color = channel_color(channel_name, color)
    if line_dash is None:
        line_dash = (
            "dotted"
            if channel_type in ["fit", "raw_fit", "flp", "fhp", "dff_fit"]
            else "solid"
        )
    kdims = ["Time (s)"]
    group = channel_type if group is None else group
    if channel_type in ["raw", "signal", "isosbestic", "control", "ratiometric"]:
        ydim = "F" if ydim is None else ydim
        ylabel = r"$$F$$" if ylabel is None else ylabel
        label = f"{channel_name} ({channel_type})" if label is None else label
    elif channel_type in ["raw_fit"]:
        ydim = "F" if ydim is None else ydim
        ylabel = r"$$F$$" if ylabel is None else ylabel
        label = f"{channel_name} (exp fit)" if label is None else label
    elif channel_type in ["flp"]:
        ydim = "F" if ydim is None else ydim
        ylabel = r"$$F$$" if ylabel is None else ylabel
        label = f"{channel_name} (low-pass)" if label is None else label
    elif channel_type in ["fhp"]:
        ydim = "F" if ydim is None else ydim
        ylabel = r"$$F$$" if ylabel is None else ylabel
        label = f"{channel_name} (high-pass)" if label is None else label
    elif channel_type in ["dff"]:
        ydim = "dff" if ydim is None else ydim
        ylabel = r"$$\Delta F/F$$" if ylabel is None else ylabel
        label = f"{channel_name} (dF/F)" if label is None else label
    elif channel_type in ["dff_fit"]:
        ydim = "dff" if ydim is None else ydim
        ylabel = r"$$\Delta F/F$$" if ylabel is None else ylabel
        label = f"{channel_name} (best fit)" if label is None else label
    elif channel_type in ["simultaneous_fit"]:
        ydim = "F" if ydim is None else ydim
        ylabel = r"$$F$$" if ylabel is None else ylabel
        label = f"{channel_name} (best fit)" if label is None else label
    elif channel_type in ["dff_diff"]:
        ydim = "dff_diff" if ydim is None else ydim
        ylabel = r"$$\Delta({\Delta F}/F)$$" if ylabel is None else ylabel
        label = f"{channel_name} (residual)" if label is None else label
    elif channel_type in ["z"]:
        ydim = "z" if ydim is None else ydim
        ylabel = r"z" if ylabel is None else ylabel
        label = f"{channel_name} (z-score)" if label is None else label
    else:
        raise ValueError(
            f"Unknown channel type {channel_type} for channel {channel_name}."
        )
    # return ((channel_name, channel_type, ydim), hv.Curve(
    #     (x, channel.to_numpy()),
    #     kdims=kdims,
    #     vdims=[ydim],
    #     group=group,
    #     label=label,
    # ).opts(color=color, line_dash=line_dash, ylabel=ylabel))
    return hv.Curve(
        (x, channel.to_numpy()),
        kdims=kdims,
        vdims=[ydim],
        group=group,
        label=label,
    ).opts(color=color, line_dash=line_dash, ylabel=ylabel)


# def signal_shade(df, y_dim, cmap):
#     return datashade(signal_curve(df, y_dim=y_dim), aggregator=ds.count(), cmap=cmap)


def interval_overlay(intervals, selected=[]):
    if intervals is None or intervals == []:
        return hv.Overlay([])
    colors = ["red" if x in selected else "pink" for x in range(len(intervals))]
    return hv.Overlay(
        [
            hv.VSpan(*(interval[0:2])).opts(color=c)
            for interval, c in zip(intervals, colors)
        ]
    )


def record_intervals(bounds, x, y, intervals, interval_callback=None):
    if None not in [x, y]:
        intervals.remove_overlap(x)
        logger.debug(f"Intervals now {intervals}")
        if interval_callback is not None:
            interval_callback()
    if bounds is not None and None not in [bounds[0], bounds[2]]:
        intervals.add(Interval(bounds[0], bounds[2]))
        intervals.merge_overlaps()
        logger.debug(f"Intervals now {intervals}")
        if interval_callback is not None:
            interval_callback()
    return interval_overlay(intervals)


def interval_overlay_map(trace, intervals, interval_callback=None):
    interval_stream = streams.BoundsXY(source=trace, transient=False)
    select_stream = streams.DoubleTap(source=trace, x=None, y=None, transient=False)
    return hv.DynamicMap(
        partial(
            record_intervals, intervals=intervals, interval_callback=interval_callback
        ),
        streams=[interval_stream, select_stream],
    )


def rejection_shade(recording, intervals, interval_callback=None, y_dim="raw"):
    try:
        isoch = recording.attrs["artifact_channel"]
    except KeyError:
        isoch = recording.attrs["iso_channel"]
    iso_shade = datashade(
        signal_curve(recording[isoch], y_dim=y_dim), aggregator=ds.count(), cmap="blue"
    )
    iso_shade = iso_shade.opts(default_tools=[], tools=["xbox_select", "xwheel_zoom"])
    overlay = interval_overlay_map(iso_shade, intervals, interval_callback)
    # Add horizontal selection tool
    plot = iso_shade * overlay
    return plot


class PreprocessDashboard(param.Parameterized):
    selected_index = param.Integer(default=None, allow_None=True)
    metadata_table = param.DataFrame(
        pd.DataFrame(columns=["subject", "session", "task", "run", "label"])
    )
    interval_update = param.Integer(default=0, allow_None=False)
    zdff_update = param.Integer(default=0, allow_None=False)

    def __init__(self, recordings, data_func, bidsroot, **params):
        super().__init__(**params)
        self.metadata_table = recordings
        self.data_func = data_func
        self.bidsroot = bidsroot
        self.config = load_preprocess_config(self.bidsroot)
        self.recording = None
        self.intervals = None
        self.regression = None
        self.dff = None

    def on_selection_change(self, event):
        if event.new:
            selected_index = event.new[0]
            signal = self.data_func(selected_index)
            self.recording = fp.downsample(signal, 64)
            # Check whether there is an interval file
            sa = signal.attrs
            self.intervals = fp.load_rejections(
                sa["root"],
                sa["subject"],
                sa["session"],
                sa["task"],
                sa["run"],
                sa["label"],
            )
            if self.intervals is None:
                self.intervals = fp.find_disconnects(self.recording)
                self.update_intervals()
            self.selected_index = selected_index
        else:
            self.selected_index = None

    def on_reset_intervals(self, event):
        self.intervals = fp.find_disconnects(self.recording)
        self.interval_update += 1

    def create_reset_intervals_button(self):
        reset_intervals_btn = pn.widgets.Button(
            name="Reset rejects", button_type="primary"
        )
        reset_intervals_btn.on_click(self.on_reset_intervals)
        return reset_intervals_btn

    def create_tabulator_widget(self):
        # Create a Tabulator widget with the metadata_table DataFrame
        tabulator_widget = pn.widgets.Tabulator(self.metadata_table)

        # Attach the callback function to the 'selection' parameter of
        # the Tabulator widget
        tabulator_widget.param.watch(self.on_selection_change, "selection")

        return tabulator_widget

    def update_intervals(self):
        if self.recording is None:
            return
        ra = self.recording.attrs
        fp.save_rejections(
            self.intervals,
            ra["root"],
            ra["subject"],
            ra["session"],
            ra["task"],
            ra["run"],
            ra["label"],
        )
        self.interval_update += 1

    @param.depends("selected_index", "interval_update", watch=True)
    def update_zdff(self):
        rej = fp.reject(self.recording, self.intervals)
        dff = fp.normalise(rej, params=self.config)
        self.dff = dff
        self.zdff_update += 1

    @param.depends("zdff_update")
    def plot_all(self):
        if self.recording is None or self.dff is None:
            return
        tools = ["xbox_select"]
        signals = fp.get_signal_channels(self.recording)
        raw_curves = []
        df_curves = []
        dff_curves = []
        relative_curves = []
        z_curves = []
        channel_opts = []
        references = self.recording.attrs["references"]
        channel_types = self.recording.attrs["types"]
        recs = self.recording.copy()
        recs.columns = pd.MultiIndex.from_tuples(
            [(ch, channel_types[ch]) for ch in recs.columns],
            names=["channel_name", "channel_type"],
        )
        for ch in recs:
            raw_curves.append(channel_curve(recs[ch]))
        for ch in self.dff:
            curve = channel_curve(self.dff[ch])
            if ch[1] in [
                "raw",
                "signal",
                "isosbestic",
                "control",
                "ratiometric",
                "raw_fit",
                "flp",
            ]:
                raw_curves.append(curve)
            elif ch[1] in ["fhp"]:
                df_curves.append(curve)
            elif ch[1] in ["dff", "dff_fit"]:
                dff_curves.append(curve)
            # elif ch[1] in ["simultaneous_fit"]:
            #     relative_curves.append(curve)
            elif ch[1] in ["dff_diff"]:
                relative_curves.append(curve)
            elif ch[1] in ["z"]:
                z_curves.append(curve)

        iom = interval_overlay_map(raw_curves[0], self.intervals, self.update_intervals)
        plots = [(hv.Overlay(raw_curves) * iom).opts(xaxis=None)]
        if len(df_curves) > 0:
            plots.append(hv.Overlay(df_curves).opts(xaxis=None))
        if len(dff_curves) > 0:
            plots.append(hv.Overlay(dff_curves).opts(xaxis=None))
        if len(relative_curves) > 0:
            plots.append(hv.Overlay(relative_curves).opts(xaxis=None))
        plots.append(hv.Overlay(z_curves))
        plot = hv.Layout(plots).opts(
            opts.Curve(
                alpha=0.8,
                responsive=True,
                min_width=600,
                min_height=200,
                tools=tools,
            )
        )
        return plot.cols(1)

    def view(self):
        return pn.Row(
            pn.Column(
                self.create_tabulator_widget(),
                self.create_reset_intervals_button(),
                sizing_mode="stretch_height",
                min_height=600,
            ),
            pn.Column(self.plot_all, sizing_mode="stretch_both"),
            styles=dict(background="WhiteSmoke"),
            sizing_mode="stretch_both",
        )
