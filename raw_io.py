import os
from enum import Enum, unique
from dataclasses import dataclass
from copy import copy
import warnings

from scipy.io import loadmat
import numpy as np
import pandas as pd
import h5py

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


class RawDataSpec:
    """Location and kind of raw data."""

    def __init__(self, mouse_id: str, cell_type: cond.CellType, day: cond.Day):
        self.mouse_id = mouse_id
        self.cell_type = cell_type
        self.day = day

        self.__dataset_paths = []
        self.__dataset_types = []

    def add_dataset_path(self, type_: RawDataType, path: str):
        assert len(self.__dataset_paths) == len(self.__dataset_types)

        if not isinstance(type_, RawDataType):
            raise TypeError('Argument `type_` must be a RawDataType')
        if type_ in self.__dataset_types:
            raise ValueError(
                'RawDataSpec instance already has a {} dataset'.format(type_)
            )

        self.__dataset_paths.append(os.path.abspath(path))
        self.__dataset_types.append(type_)

    def get_dataset_by_type(self, type_: RawDataType) -> dict:
        assert len(self.__dataset_paths) == len(self.__dataset_types)

        if not isinstance(type_, RawDataType):
            raise TypeError('Argument `type_` must be a RawDataType')

        # Search for path associated with the dataset type `type_` and load it
        for i, attached_dataset_type in enumerate(self.__dataset_types):
            if attached_dataset_type == type_:
                if type_ == RawDataType.fluo:
                    return loadmat(self.__dataset_paths[i])['df']
                else:
                    return loadmat(self.__dataset_paths[i])

        raise ValueError(
            'RawDataSpec instance does not have a {} dataset'.format(type_)
        )


def __is_valid_mouse_path(path: str) -> bool:
    if os.path.isdir(path):
        return True
    else:
        return False


def __find_filename_by_dataset_type(type_: RawDataType, dir_to_search: str):
    """Look for a file containing a given RawDataType."""
    # Try to find a file with extension matching dtype
    matching_fnames = []
    for fname in os.listdir(dir_to_search):
        if fname.endswith(type_.as_path()) and not fname.startswith('.'):
            matching_fnames.append(fname)

    if len(matching_fnames) == 0:
        raise FileNotFoundError(
            "Could not find {} file in {}".format(type_, dir_to_search)
        )
    elif len(matching_fnames) > 1:
        raise RuntimeError(
            "Found multiple possible {} files in {}: {}".format(
                type_, dir_to_search, matching_fnames
            )
        )
    else:
        return matching_fnames[0]


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

                dataset_spec = RawDataSpec(mouseid, celltype, day)

                for dtype in RawDataType:
                    fname = __find_filename_by_dataset_type(dtype, path_to_day)
                    dataset_spec.add_dataset_path(
                        dtype, os.path.join(path_to_day, fname)
                    )

                yield dataset_spec


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

        raw_data = spec.get_dataset_by_type(RawDataType.time_stamp)

        # Initialize dataframe
        ## Copy some columns from raw_data
        content = {
            'tone_start': raw_data['tone_start'].flatten(),
            'tone_end': raw_data['tone_start'].flatten() + 1.0,
            'reward_start': raw_data['wt_start'].flatten(),
        }

        del raw_data

        ## Add computed columns
        content['tone_duration'] = content['tone_end'] - content['tone_start']
        content['trial_start'] = content['tone_start'] - self.baseline_duration
        content['trial_end'] = (
            content['reward_start'] + self.post_reward_duration
        )
        content['trial_duration'] = (
            content['trial_end'] - content['trial_start']
        )
        content['reward_delivered'] = [
            True for i in range(len(content['trial_start']))
        ]

        # TODO: extract catch trial data

        super(TrialTimetable, self).__init__(content)

        # Initialize meta attributes
        self['mouse_id'] = str(spec.mouse_id)
        self['cell_type'] = str(spec.cell_type)
        self['day'] = str(spec.day)


class Fluorescence:
    def __init__(self, spec: RawDataSpec):
        self.fluo = spec.get_dataset_by_type(RawDataType.fluo)
        self.frame_rate = 30.0

    @property
    def num_frames(self):
        return self.fluo.shape[-1]

    @property
    def duration(self):
        return self.num_frames / self.frame_rate

    @property
    def time(self):
        return np.arange(0, self.num_frames - 0.5) / self.frame_rate

    def get_time_slice(self, start, stop=None):
        fluo_copy = copy(self)
        start_ind = np.argmin(np.abs(self.time - start))
        if stop is None:
            time_slice = self.fluo[..., start_ind][..., np.newaxis]
        else:
            stop_ind = np.argmin(np.abs(self.time - stop))
            if stop_ind == start_ind:
                time_slice = self.fluo[..., start_ind][..., np.newaxis]
            elif stop_ind > start_ind:
                time_slice = self.fluo[..., start_ind:stop_ind]
            else:
                raise ValueError(
                    'Expected stop ({}) >= start ({}), or `stop=None`'.format(
                        stop, start
                    )
                )

        fluo_copy.fluo = time_slice
        return fluo_copy


class LongFluorescence:
    """Trial-by-trial fluorescence stored in long data format.

    Attributes
    ----------
    fluo: 2D array
        Trial by trial fluorescence. Rows are trials/neurons and columns are
        timesteps.
    trial_num: vector
    cell_num: vector

    """

    _dtypes = {
        'trial_num': np.uint32,
        'cell_num': np.uint32,
        'fluo': np.float32,
    }

    def __init__(self, trial_num, cell_num, fluo_matrix):
        """Initialize fluorescence data in long format.

        `trial_num` and `cell_num` are broadcasted to match `fluo_matrix` if
        needed.

        """
        assert np.ndim(fluo_matrix) == 2

        self.frame_rate = 30.0

        self.fluo = np.asarray(fluo_matrix, dtype=self._dtypes['fluo'])
        self.trial_num = np.broadcast_to(
            trial_num, self.fluo.shape[0],
        ).astype(self._dtypes['trial_num'])
        self.cell_num = np.asarray(cell_num, self.fluo.shape[0],).astype(
            dtype=self._dtypes['cell_num']
        )

    def append(self, trial_num, cell_num, fluo_matrix):
        """Append data in long format.

        `trial_num` and `cell_num` are broadcasted to match `fluo_matrix` if
        needed.

        """
        assert np.ndim(fluo_matrix) == 2

        fluo_numrows = np.shape(fluo_matrix)[0]

        self.trial_num = np.concatenate(
            [
                self.trial_num,
                np.broadcast_to(trial_num, fluo_numrows).astype(
                    self._dtypes['trial_num']
                ),
            ]
        )
        self.cell_num = np.concatenate(
            [
                self.cell_num,
                np.broadcast_to(cell_num, fluo_numrows).astype(
                    self._dtypes['cell_num']
                ),
            ]
        )
        self.fluo = np.concatenate(
            [self.fluo, np.asarray(fluo_matrix, dtype=self._dtypes['fluo'])],
            axis=0,
        )

    def save(self, fname):
        """Save fluorescence data in long format to an HDF5 file."""
        with h5py.File(fname, 'w') as f:
            f.attrs['frame_rate'] = self.frame_rate
            f.create_dataset(
                'trial_num',
                data=self.trial_num,
                dtype=self._dtypes['trial_num'],
            )
            f.create_dataset(
                'cell_num', data=self.cell_num, dtype=self._dtypes['cell_num']
            )
            f.create_dataset(
                'fluo', data=self.fluo, dtype=self._dtypes['fluo']
            )
            f.close()

    @staticmethod
    def load(fname):
        """Load fluorescence data in long format from an HDF5 file."""
        with h5py.File(fname, 'r') as f:
            long_fluo = LongFluorescence(
                f['trial_num'][:], f['cell_num'][:], f['fluo'][...]
            )
            long_fluo.frame_rate = f.attrs['frame_rate']
            f.close()

        return long_fluo


class Session:
    """A session is composed of trials.

    Dimensionality [trials, neurons, timesteps]

    """

    def __init__(self, spec: RawDataSpec):
        self.trial_time_table = TrialTimetable(spec)
        self.fluo = Fluorescence(spec)
        self._stack_fluo_by_trials()

    def _stack_fluo_by_trials(self):
        fluo_traces = []
        num_frames = []
        for start, stop in zip(
            self.trial_time_table['trial_start'],
            self.trial_time_table['trial_end'],
        ):
            trial_slice = self.fluo.get_time_slice(start, stop)
            fluo_traces.append(trial_slice)
            num_frames.append(trial_slice.num_frames)
            print(trial_slice.num_frames)

        num_frames = np.array(num_frames)

        if (
            np.abs(num_frames.min() - num_frames.max()) / num_frames.max()
            > 0.05
        ):
            warnings.warn(
                "More than 5 pct difference between "
                "shortest and longest trial: {}, {}".format(
                    num_frames.min(), num_frames.max()
                )
            )

        min_num_frames = num_frames.min()
        fluo_arrays = []
        for tr in fluo_traces:
            fluo_arrays.append(tr.fluo[:, :min_num_frames])

        stacked_fluo = np.array(fluo_arrays)
        self.fluo.fluo = stacked_fluo

        assert (
            self.fluo.fluo.shape[0] == self.trial_time_table.shape[0]
        ), '{} not equal to {}'.format(
            self.fluo.fluo.shape[0], self.trial_time_table.shape[0]
        )
        print(self.fluo.num_frames)


def walk_sessions(path_to_root: str):
    for raw_data_spec in walk_raw_data_from(path_to_root):
        yield Session(raw_data_spec)
