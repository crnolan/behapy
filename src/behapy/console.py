import logging
import argparse
from pathlib import Path
from .tdt import load_session_tank_map, convert_block


def tdt2bids(session_fn: str, bids_root: str) -> None:
    """Convert TDT tanks into BIDS format.

    Args:
        session_fn: Map of the files to sessions
        bids_root: Root path of the BIDS structure (data will be put in the
                   `rawdata` sub-folder of `bids_root`)
    """
    session_df = load_session_tank_map(session_fn)
    convert_block(session_df, Path(bids_root) / 'rawdata')


def tdt2bids_command():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Convert TDT tanks into BIDS format'
    )
    parser.add_argument('session_fn', type=str,
                        help='path to file with TDT session information')
    parser.add_argument('bids_root', type=str,
                        help='root path of the BIDS dataset (data will '
                             'be put in the rawdata sub-folder of bids_root)')
    args = parser.parse_args()
    tdt2bids(**vars(args))

