import os
from enum import Enum, unique
from dataclasses import dataclass

from scipy.io import loadmat
import pandas as pd

import conditions as cond


@unique
class RawDataType(Enum):
    fluo = 1
    time_stamp = 2

    def as_path(self) -> str:
        """Get RawDataType as a file extension."""
        if self == RawDataType.fluo:
            path = "df.mat"
        elif self == RawDataType.time_stamp:
            path = "ws.mat"
        else:
            raise ValueError('No path for RawDataType {}'.format(self))

        return path


@dataclass
class RawDataSpec:
    """Location and kind of raw data."""

    path: str
    mouse_id: str
    cell_type: cond.CellType
    data_type: RawDataType
    day: cond.Day

    def get_data(self) -> dict:
        return loadmat(self.path)


def __is_valid_mouse_path(path: str) -> bool:
    if os.path.isdir(path):
        return True
    else:
        return False


def walk_raw_data_from(path_to_root: str) -> RawDataSpec:
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
    for celltype in cond.CellType:
        path_to_celltype = os.path.join(path_to_root, celltype.as_path())
        for mouseid in os.listdir(path_to_celltype):

            path_to_mouse = os.path.join(path_to_celltype, mouseid)
            if not __is_valid_mouse_path(path_to_mouse):
                continue

            for day in cond.DAYS:
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

                    yield RawDataSpec(
                        os.path.join(path_to_day, matching_fname),
                        mouseid,
                        celltype,
                        dtype,
                        day,
                    )


class TrialTimetable(pd.DataFrame):
    baseline_duration = 2.0  # Time from trial start to tone start
    post_reward_duration = 10.0  # Time from reward start to trial end

    def __init__(self, spec: RawDataSpec):
        """Construct a TrialTimetable from a RawDataSpec."""
        if not issubclass(type(spec), RawDataSpec):
            raise TypeError(
                "Expected argument to be a RawDataSpec, got {} instead".format(
                    type(spec)
                )
            )
        if spec.datatype != RawDataType.time_stamp:
            raise TypeError(
                "TrialTimetables can only be constructed from RawDataFiles"
                " containing timestamps."
            )

        raw_data = spec.get_data()

        # Initialize dataframe
        ## Copy some columns from raw_data
        content = {
            'tone_start': raw_data.data['tone_start'],
            'tone_end': raw_data.data['tone_end'],
            'reward_start': raw_data.data[
                'wt_start'
            ],  # TODO double check with Candice
        }
        ## Add computed columns
        content['tone_duration'] = content['tone_end'] - content['tone_start']
        content['trial_start'] = content['tone_start'] - self.baseline_duration
        content['trial_end'] = (
            content['water_start'] + self.post_reward_duration
        )
        content['trial_duration'] = (
            content['trial_end'] - content['trial_start']
        )
        content['reward_delivered'] = [
            True for i in len(content['trial_start'])
        ]

        # TODO: extract catch trial data

        super(TrialTimetable, self).__init__(content)

        # Initialize meta attributes
        self['mouse_id'] = raw_data.mouse_id
        self['cell_type'] = raw_data.cell_type



def walk_trial_timetables(path_to_root: str):
    for raw_data_spec in walk_raw_data_from(path_to_root):
        if raw_data_spec.data_type == RawDataType.time_stamp:
            yield TrialTimetable(raw_data_spec)
        else:
            continue

def walk_fluo_datasets(path_to_root: str):
    for raw_data_spec in walk_raw_data_from(path_to_root):
        if raw_data_spec.data_type == RawDataType.fluo:
            pass


def walk_datasets_by_type(path_to_root: str, type_: RawDataType):
    for raw_data_spec in walk_raw_dat_from(path_to_root):
        if raw_data_spec.data_type == type_:

            if raw_data_spec.data_type == RawDataType.time_stamp:
                yield TrialTimetable(raw_data_spec)
            else:
                raise NotImplementedError

