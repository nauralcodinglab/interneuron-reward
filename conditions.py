import os
from enum import Enum, unique
from dataclasses import dataclass

from scipy.io import loadmat


@unique
class CellType(Enum):
    pyr = 1
    som = 2
    pv = 3
    vip = 4

    def as_path(self) -> str:
        """Get CellType as a string component of a file path."""
        if self == CellType.pyr:
            path = 'PN'
        elif self == CellType.som:
            path = 'SOM'
        elif self == CellType.pv:
            path = 'PV'
        elif self == CellType.vip:
            path = 'VIP'
        else:
            raise ValueError('No path for CellType {}'.format(self))

        return path


class Day:
    def __init__(self, day: int):
        self._day = day

    def __hash__(self):
        return self._day

    def as_path(self) -> str:
        """Get Day as a string component of a file path."""
        return 'Day {}'.format(self._day)


DAYS = (Day(day) for day in [1, 3, 5, 7])


@unique
class RawDataType(Enum):
    fluo = 1
    timestamp = 2

    def as_path(self) -> str:
        """Get RawDataType as a file extension."""
        if self == RawDataType.fluo:
            path = "df.mat"
        elif self == RawDataType.timestamp:
            path = "ws.mat"
        else:
            raise ValueError('No path for RawDataType {}'.format(self))

        return path


@dataclass
class RawDataFile:
    data: dict
    mouseid: str
    celltype: CellType
    datatype: RawDataType
    day: Day


def __is_valid_mouse_path(path: str) -> bool:
    if os.path.isdir(path):
        return True
    else:
        return False


def walk_raw_data_from(path_to_root: str) -> RawDataFile:
    """Iterate over raw data files.

    Walk a directory hierarchy with raw calcium imaging and timestamp data
    starting from argument `path_to_root`.

    """
    # Implementation note: raw data is organized hierarchically
    # 1. Cell type
    # 2. Mouse
    # 3. Day
    # 4. Dataset type (fluorescence, trial timestamps, etc)
    # The nested loops below walk the hierarchy and raise a FileNotFoundError
    # if an expected condition (eg day) is missing
    for celltype in CellType:
        path_to_celltype = os.path.join(path_to_root, celltype.as_path())
        for mouseid in os.listdir(path_to_celltype):

            path_to_mouse = os.path.join(path_to_celltype, mouseid)
            if not __is_valid_mouse_path(path_to_mouse):
                continue

            for day in DAYS:
                path_to_day = os.path.join(path_to_mouse, day.as_path())
                for dtype in RawDataType:

                    # Try to find a file with extension matching dtype
                    matching_fname = None
                    for fname in os.listdir(path_to_day):
                        if fname.startswith(mouseid) and fname.endswith(
                            dtype.as_path()
                        ):
                            matching_fname = fname
                    if matching_fname is None:
                        raise FileNotFoundError(
                            "Could not find {} file in {}".format(
                                dtype, path_to_day
                            )
                        )

                    yield RawDataFile(
                        loadmat(os.path.join(path_to_day, matching_fname)),
                        mouseid,
                        celltype,
                        dtype,
                        day,
                    )
