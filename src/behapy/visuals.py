import logging
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
pn.extension("tabulator", "mathjax", comms="vscode")
# pn.extension('tabulator')


def signal_curve(df, y_dim):
    return hv.Curve((df.index.to_numpy(), df), "time", y_dim)
    # return hv.Curve((pd.to_timedelta(df.index, unit='s'), df))


def signal_shade(df, y_dim, cmap):
    return datashade(signal_curve(df, y_dim=y_dim), aggregator=ds.count(), cmap=cmap)


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
        logging.debug(f"Intervals now {intervals}")
        if interval_callback is not None:
            interval_callback()
    if bounds is not None and None not in [bounds[0], bounds[2]]:
        intervals.add(Interval(bounds[0], bounds[2]))
        intervals.merge_overlaps()
        logging.debug(f"Intervals now {intervals}")
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

    def build_curve_opts(self, label, color):
        return [
            opts.Curve(
                "signal." + label,
                line_dash="solid",
                color=color,
            ),
            opts.Curve(
                "isosbestic." + label,
                line_dash="dashed",
                color=color,
            ),
            opts.Curve(
                "ratio." + label,
                line_dash="dashed",
                color=color,
            ),
            opts.Curve(
                "control." + label,
                line_dash="dashed",
                color=color,
            ),
            opts.Curve(
                "dff." + label,
                line_dash="solid",
                color=color,
            ),
            opts.Curve(
                "rawfit." + label,
                line_dash="dotted",
                color=color,
            ),
            opts.Curve(
                "flp." + label,
                line_dash="dotted",
                color=color,
            ),
            opts.Curve(
                "fhp." + label,
                line_dash="dashed",
                color=color,
            ),
            opts.Curve(
                "control_fit." + label,
                line_dash="dashed",
                color=color,
            ),
            opts.Curve(
                "dff_fit." + label,
                line_dash="solid",
                color=color,
            ),
            opts.Curve(
                "zdff." + label,
                ylabel=r"$$\Delta F/F$$ (z-scored)",
                color=color,
            )
        ]

    @param.depends("zdff_update")
    def plot_all(self):
        if self.recording is None or self.dff is None:
            return
        tools = ["xbox_select"]
        signals = fp.get_signal_channels(self.recording)
        raw_curves = []
        dff_curves = []
        relative_curves = []
        z_curves = []
        channel_opts = []
        dff_name = r"$$\Delta F/F$$"
        color_iter = iter(hv.Cycle("Colorblind").values)
        references = self.recording.attrs["references"]
        channel_types = self.recording.attrs["types"]
        for ch in signals:
            raw_curves.append(
                hv.Curve(
                    (self.recording.index.to_numpy(), self.recording[ch]),
                    kdims=["Time"],
                    vdims=["Raw"],
                    group="signal",
                    label=ch,
                )
            )
            if "fit" in self.dff[ch]:
                raw_curves.append(
                    hv.Curve(
                        (self.dff.index.to_numpy(), self.dff[ch, "fit"]),
                        kdims=["Time"],
                        vdims=["Raw fit"],
                        group="rawfit",
                        label=ch,
                    )
                )
            if "flp" in self.dff[ch]:
                raw_curves.append(
                    hv.Curve(
                        (self.dff.index.to_numpy(), self.dff[ch, "flp"]),
                        kdims=["Time"],
                        vdims=["Low pass filtered"],
                        group="flp",
                        label=ch,
                    )
                )
            if "dff" in self.dff[ch]:
                dff_curves.append(
                    hv.Curve(
                        (self.dff.index.to_numpy(), self.dff[ch, "dff"]),
                        kdims=["Time"],
                        vdims=["dff"],
                        group="dff",
                        label=ch,
                    )
                )
            if "fhp" in self.dff[ch]:
                dff_curves.append(
                    hv.Curve(
                        (self.dff.index.to_numpy(), self.dff[ch, "fhp"]),
                        kdims=["Time"],
                        vdims=["High pass filtered"],
                        group="fhp",
                        label=ch,
                    )
                )
            if references[ch] is not None:
                if fp.is_reference_type(channel_types[references[ch]]):
                    raw_curves.append(
                        hv.Curve(
                            (
                                self.recording[references[ch]].index.to_numpy(),
                                self.recording[references[ch]],
                            ),
                            kdims=["Time"],
                            vdims=["Raw"],
                            group=channel_types[references[ch]],
                            label=references[ch],
                        )
                    )
                    if references[ch] in self.dff.columns.get_level_values(0):
                        if "fit" in self.dff[references[ch]]:
                            raw_curves.append(
                                hv.Curve(
                                    (self.dff.index.to_numpy(), self.dff[references[ch], "fit"]),
                                    kdims=["Time"],
                                    vdims=["Raw fit"],
                                    group="rawfit",
                                    label=references[ch],
                                )
                            )
                        if "dff" in self.dff[references[ch]]:
                            dff_curves.append(
                                hv.Curve(
                                    (self.dff.index.to_numpy(), self.dff[references[ch], "dff"]),
                                    kdims=["Time"],
                                    vdims=["dff"],
                                    group="dff",
                                    label=references[ch],
                                )
                            )
                else:
                    logging.error(
                        f"Unknown reference channel type "
                        f"{channel_types[references[ch]]} for "
                        f"channel {ch}"
                    )
            if "control_fit" in self.dff[ch]:
                relative_curves.append(
                    hv.Curve(
                        (self.dff.index.to_numpy(), self.dff[ch, "dff"]),
                        kdims=["Time"],
                        vdims=["dff"],
                        group="dff",
                        label=ch,
                    )
                )
                relative_curves.append(
                    hv.Curve(
                        (self.dff.index.to_numpy(), self.dff[ch, "control_fit"]),
                        kdims=["Time"],
                        vdims=["dff"],
                        group="control_fit",
                        label=ch,
                    )
                )
            if "dff_fit" in self.dff[ch]:
                relative_curves.append(
                    hv.Curve(
                        (self.dff.index.to_numpy(), self.dff[ch, "dff_fit"]),
                        kdims=["Time"],
                        vdims=["dff"],
                        group="dff_fit",
                        label=ch,
                    )
                )
            z_curves.append(
                hv.Curve(
                    (self.dff.index.to_numpy(), self.dff[ch, "z"]),
                    kdims=["Time"],
                    vdims=["zdff"],
                    group="zdff",
                    label=ch,
                )
            )
            channel_opts.extend(self.build_curve_opts(ch, next(color_iter)))
            if references[ch] is not None:
                channel_opts.extend(self.build_curve_opts(references[ch], next(color_iter)))

        iom = interval_overlay_map(raw_curves[0], self.intervals, self.update_intervals)
        plots = [(hv.Overlay(raw_curves) * iom).opts(xaxis=None)]
        if len(dff_curves) > 0:
            plots.append(hv.Overlay(dff_curves).opts(xaxis=None))
        if len(relative_curves) > 0:
            plots.append(hv.Overlay(relative_curves).opts(xaxis=None))
        plots.append(hv.Overlay(z_curves))
        plot = hv.Layout(plots).opts(
            channel_opts
            + [
                opts.Curve(
                    responsive=True,
                    min_width=600,
                    min_height=300,
                    tools=tools,
                ),
            ]
        )
        # raw_plot = reduce(lambda a, b: a * b, [iso_shade, overlay] + sig_shades)
        # raw_plot = raw_plot.opts(xaxis=None)
        # plot = raw_plot + hv.Layout(dff_shades)
        # plot = plot.opts(
        #     opts.RGB(responsive=True, min_width=600, min_height=300, tools=tools)
        # )
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
