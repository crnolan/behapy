from typing import Tuple, Iterable
from collections import namedtuple
import numpy as np
import scipy.signal as sig


Event = namedtuple('Event', ['name', 'fields', 'codes', 'onset', 'offset'])


def map_events(events: Iterable[Event]):
    """ Create a dict mapping event codes to the respective events. """
    return {key: event for event in events.values() for key in event.fields}


def collect_tdt_events(raw, events):
    ts = np.array([])
    keys = np.array([])
    event_map = map_events(events)
    for key in raw.epocs.keys() & event_map.keys():
        ts = np.append(ts, raw.epocs[key].onset)
        keys = np.append(keys, [key] * len(raw.epocs[key].onset))
    order = np.argsort(ts)
    return ts[order], keys[order]

