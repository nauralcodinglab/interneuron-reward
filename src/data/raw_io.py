import os
from enum import Enum, unique
from dataclasses import dataclass
from copy import copy
import warnings

from scipy.io import loadmat
import numpy as np
import pandas as pd
import h5py
import h5table

from . import conditions as cond


@unique
class RawDataType(Enum):
    fluo = 1
    time_stamp = 2

    def as_path(self) -> str:
        """Get RawDataType as a file extension."""
        if self == RawDataType.fluo:
            path = "df.mat"
        elif self == RawDataType.time_stamp:
            path = "ws_trimmed.mat"
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


class DataCorruptionError(ValueError):
    pass


class DataCorruptionWarning(Warning):
    pass


class TrialTimetable(pd.DataFrame):
    baseline_duration = 2.0  # Time from trial start to tone start
    post_reward_duration = 10.0  # Time from reward start to trial end
    __h5_dataset_name = 'trial_timetable'

    def __init__(self, *args, **kwargs):
        """Initialize a new TrialTimetable."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_spec(spec: RawDataSpec):
        """Construct a TrialTimetable from a RawDataSpec."""
        if not issubclass(type(spec), RawDataSpec):
            raise TypeError(
                "Expected argument to be a RawDataSpec, got {} instead".format(
                    type(spec)
                )
            )

        raw_data = spec.get_dataset_by_type(RawDataType.time_stamp)

        # Check data integrity.
        if np.shape(raw_data['tone_start']) != np.shape(raw_data['wt_start']):
            raise DataCorruptionError(
                'Could not load timetable dataset from mouse {} '
                'celltype {} on day {} due to different shapes'
                ' of `tone_start` {} and `wt_start` {}'.format(
                    spec.mouse_id,
                    spec.cell_type,
                    spec.day,
                    np.shape(raw_data['tone_start']),
                    np.shape(raw_data['wt_start']),
                )
            )
        if any(raw_data['tone_start'] > raw_data['wt_start']):
            warnings.warn(
                DataCorruptionWarning(
                    'Found {} trials with `tone_start` after `wt_start`'
                    ' in mouse {} celltype {} on day {}'.format(
                        sum(raw_data['tone_start'] > raw_data['wt_start']),
                        spec.mouse_id,
                        spec.cell_type,
                        spec.day,
                    )
                )
            )

        # Initialize dataframe
        ## Copy some columns from raw_data
        TONE_DURATION = 1.0
        content = {
            'tone_start': raw_data['tone_start'].flatten(),
            'tone_end': raw_data['tone_start'].flatten() + TONE_DURATION,
            'reward_start': raw_data['wt_start'].flatten(),
        }

        del raw_data

        ## Add computed columns
        content['tone_duration'] = content['tone_end'] - content['tone_start']
        content['trial_start'] = (
            content['tone_start'] - TrialTimetable.baseline_duration
        )
        content['trial_end'] = (
            content['reward_start'] + TrialTimetable.post_reward_duration
        )
        content['trial_duration'] = (
            content['trial_end'] - content['trial_start']
        )
        content['reward_delivered'] = [
            True for i in range(len(content['trial_start']))
        ]

        # TODO: extract catch trial data

        trial_timetable = TrialTimetable(content)
        trial_timetable['trial_num'] = [
            i for i in range(trial_timetable.shape[0])
        ]

        # Initialize meta attributes
        trial_timetable['mouse_id'] = str(spec.mouse_id)
        trial_timetable['cell_type'] = str(spec.cell_type)
        trial_timetable['day'] = str(spec.day)

        # See function for definition of an invalid trial.
        trial_timetable._drop_invalid_trials_inplace()

        return trial_timetable

    def _drop_invalid_trials_inplace(self):
        negative_duration = self['trial_duration'] < 0.0
        if sum(negative_duration) > 0:
            warnings.warn(
                'Dropping {} trials with duration < 0'.format(
                    sum(negative_duration)
                )
            )

        self.drop(index=self.index[negative_duration], inplace=True)

    def _drop_out_of_range_trials_inplace(self, fluo):
        """Drop trials that are out of time bounds inplace."""
        in_time_bounds = (self['trial_start'] > 0.0) & (
            self['trial_end'] < fluo.duration
        )
        out_of_bounds_trials = self.index[~in_time_bounds]
        self.drop(index=out_of_bounds_trials, inplace=True)

    def save(self, fname_or_group):
        """Save TrialTimetable to HDF5.

        Arguments
        ---------
        fname_or_group : str or h5py.Group

        """
        if isinstance(fname_or_group, str):
            with h5py.File(fname_or_group, 'a') as f:
                self._save_to_h5pygroup(f)
        elif isinstance(fname_or_group, h5py.Group):
            self._save_to_h5pygroup(fname_or_group)
        else:
            raise TypeError(
                'Expected argument `fname_or_group` to be a str filename '
                'or h5py.Group instance, got {} of type {} instead'.format(
                    fname_or_group, type(fname_or_group)
                )
            )

    def _save_to_h5pygroup(self, group: h5py.Group):
        h5table.save_dataframe(group, self.__h5_dataset_name, self)
        group[self.__h5_dataset_name].attrs[
            'baseline_duration'
        ] = self.baseline_duration
        group[self.__h5_dataset_name].attrs[
            'post_reward_duration'
        ] = self.post_reward_duration

    @staticmethod
    def load(fname_or_group):
        """Load TrialTimetable from HDF5.

        Arguments
        ---------
        fname_or_group : str or h5py.Group

        """
        if isinstance(fname_or_group, str):
            with h5py.File(fname_or_group, 'r') as f:
                trial_timetable = TrialTimetable._load_from_h5pygroup(f)
        elif isinstance(fname_or_group, h5py.Group):
            trial_timetable = TrialTimetable._load_from_h5pygroup(
                fname_or_group
            )
        else:
            raise TypeError(
                'Expected argument `fname_or_group` to be a str filename '
                'or h5py.Group instance, got {} of type {} instead'.format(
                    fname_or_group, type(fname_or_group)
                )
            )

        return trial_timetable

    @staticmethod
    def _load_from_h5pygroup(group: h5py.Group):
        trial_timetable = TrialTimetable(
            h5table.load_dataframe(group, TrialTimetable.__h5_dataset_name)
        )
        trial_timetable.baseline_duration = group[
            TrialTimetable.__h5_dataset_name
        ].attrs['baseline_duration']
        trial_timetable.post_reward_duration = group[
            TrialTimetable.__h5_dataset_name
        ].attrs['post_reward_duration']

        return trial_timetable


class Fluorescence:
    def __init__(self):
        self.frame_rate = 30.0
        self.fluo = None
        self.is_z_score = False
        try:
            self.trial_num = None
            self.cell_num = None
        except AttributeError:
            pass

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


class RawFluorescence(Fluorescence):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_spec(spec: RawDataSpec):
        fluo = RawFluorescence()
        fluo.fluo = spec.get_dataset_by_type(RawDataType.fluo)
        return fluo

    def normalize(self):
        """Transform cell-by-cell fluorescence signal into Z-score."""
        if not self.is_z_score:
            self.fluo -= self.fluo.mean(axis=1)[:, np.newaxis]
            self.fluo /= self.fluo.std(axis=1)[:, np.newaxis]
            self.is_z_score = True
        else:
            raise RuntimeError(
                'This fluorescence signal has already been'
                ' transformed into a Z-score.'
            )

    def to_deep(self, trial_timetable: TrialTimetable):
        fluo_traces = []
        num_frames = []
        for start, stop in zip(
            trial_timetable['trial_start'], trial_timetable['trial_end'],
        ):
            trial_slice = self.get_time_slice(start, stop)
            fluo_traces.append(trial_slice)
            num_frames.append(trial_slice.num_frames)

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

        assert (
            stacked_fluo.shape[0] == trial_timetable.shape[0]
        ), '{} not equal to {}'.format(
            stacked_fluo.shape[0], trial_timetable.shape[0]
        )

        deep_fluo = DeepFluorescence(
            trial_timetable['trial_num'],
            np.arange(stacked_fluo.shape[1]),
            stacked_fluo,
        )
        deep_fluo.is_z_score = self.is_z_score

        return deep_fluo


class TrialFluorescence(Fluorescence):
    _dtypes = {
        'trial_num': np.uint32,
        'cell_num': np.uint32,
        'fluo': np.float32,
    }

    def __init__(self):
        super().__init__()

    def save(self, name_or_group):
        """Save fluorescence data to an HDF5 file or group.

        Arguments
        ---------
        name_or_group: str or h5py.Group
            Name of file in which to save dataset, or an h5py.Group to save it
            in.

        Raises
        ------
        TypeError if `name_or_group` is not a str or h5py.Group.

        """
        if isinstance(name_or_group, str):
            with h5py.File(name_or_group, 'w') as f:
                self._save_to_h5pygroup(f)
                f.close()
        elif isinstance(name_or_group, h5py.Group):
            self._save_to_h5pygroup(name_or_group)
        else:
            raise TypeError(
                'Expected argument `name_or_group` to be a str filename '
                'or h5py.Group instance, got {} of type {} instead'.format(
                    name_or_group, type(name_or_group)
                )
            )

    def _save_to_h5pygroup(self, group: h5py.Group):
        if isinstance(self, DeepFluorescence):
            group.attrs['__type'] = 'DeepFluorescence'
        elif isinstance(self, LongFluorescence):
            group.attrs['__type'] = 'LongFluorescence'
        else:
            raise NotImplementedError(
                'Method not implemented for {}'.format(type(self))
            )

        group.attrs['frame_rate'] = self.frame_rate
        group.attrs['is_z_score'] = self.is_z_score
        group.create_dataset(
            'trial_num', data=self.trial_num, dtype=self._dtypes['trial_num'],
        )
        group.create_dataset(
            'cell_num', data=self.cell_num, dtype=self._dtypes['cell_num']
        )
        group.create_dataset(
            'fluo', data=self.fluo, dtype=self._dtypes['fluo']
        )

    @staticmethod
    def load(name_or_group):
        """Load fluorescence data from an HDF5 file or group."""
        if isinstance(name_or_group, str):
            with h5py.File(name_or_group, 'r') as f:
                fluo = TrialFluorescence._load_from_h5pygroup(f)
                f.close()
        elif isinstance(name_or_group, h5py.Group):
            fluo = TrialFluorescence._load_from_h5pygroup(name_or_group)
        else:
            raise TypeError(
                'Expected argument `name_or_group` to be a str filename '
                'or h5py.Group instance, got {} of type {} instead'.format(
                    name_or_group, type(name_or_group)
                )
            )

        return fluo

    @staticmethod
    def _load_from_h5pygroup(group):
        if group.attrs['__type'] == 'DeepFluorescence':
            fluo = DeepFluorescence(
                group['trial_num'][:], group['cell_num'][:], group['fluo'][...]
            )
        elif group.attrs['__type'] == 'LongFluorescence':
            fluo = LongFluorescence(
                group['trial_num'][:], group['cell_num'][:], group['fluo'][...]
            )
        else:
            raise NotImplementedError(
                'Method not implemented for {}'.format(group.attrs['__type'])
            )
        fluo.frame_rate = group.attrs['frame_rate']
        fluo.is_z_score = group.attrs['is_z_score']

        return fluo


class DeepFluorescence(TrialFluorescence):
    """Trial-by-trial fluorescence stored as a multidimensional array."""

    def __init__(self, trial_num, cell_num, fluo_arr):
        assert np.ndim(fluo_arr) == 3

        super().__init__()

        self.fluo = np.asarray(fluo_arr, dtype=self._dtypes['fluo'])
        self.trial_num = np.broadcast_to(
            trial_num, self.fluo.shape[0],
        ).astype(self._dtypes['trial_num'])
        self.cell_num = np.broadcast_to(cell_num, self.fluo.shape[1],).astype(
            dtype=self._dtypes['cell_num']
        )

        assert self.fluo.shape == (
            self.num_trials,
            self.num_cells,
            self.num_frames,
        )

    @property
    def num_trials(self):
        return len(self.trial_num)

    @property
    def num_cells(self):
        return len(self.cell_num)

    def trial_mean(self):
        assert self.fluo.shape == (
            self.num_trials,
            self.num_cells,
            self.num_frames,
        )

        mean_fluo = LongFluorescence(0, self.cell_num, self.fluo.mean(axis=0))
        mean_fluo.is_z_score = self.is_z_score

        return mean_fluo

    def to_long(self):
        """Get data in LongFluorescence format."""
        assert self.fluo.shape == (
            self.num_trials,
            self.num_cells,
            self.num_frames,
        )
        trial_num = np.broadcast_to(
            self.trial_num, (self.num_cells, self.num_trials)
        ).T.flatten()
        cell_num = np.broadcast_to(
            self.cell_num, (self.num_trials, self.num_cells)
        ).flatten()
        assert len(trial_num) == len(cell_num)
        fluo = self.fluo.reshape((-1, self.num_frames))
        assert fluo.shape[0] == len(trial_num)

        long_fluo = LongFluorescence(trial_num, cell_num, fluo)
        long_fluo.is_z_score = self.is_z_score

        return long_fluo


class ShapeWarning(Warning):
    pass


class ShapeError(ValueError):
    pass


class LongFluorescence(TrialFluorescence):
    """Trial-by-trial fluorescence stored in long data format.

    Attributes
    ----------
    fluo: 2D array
        Trial by trial fluorescence. Rows are trials/neurons and columns are
        timesteps.
    trial_num: vector
    cell_num: vector

    """

    def __init__(self, trial_num, cell_num, fluo_matrix):
        """Initialize fluorescence data in long format.

        `trial_num` and `cell_num` are broadcasted to match `fluo_matrix` if
        needed.

        """
        assert np.ndim(fluo_matrix) == 2

        super().__init__()

        self.fluo = np.asarray(fluo_matrix, dtype=self._dtypes['fluo'])
        self._meta = pd.DataFrame(
            {
                'trial_num': np.broadcast_to(trial_num, self.num_rows).astype(
                    self._dtypes['trial_num']
                ),
                'cell_num': np.broadcast_to(cell_num, self.num_rows).astype(
                    dtype=self._dtypes['cell_num']
                ),
            }
        )

    @property
    def trial_num(self):
        return self._meta['trial_num']

    @property
    def cell_num(self):
        return self._meta['cell_num']

    def set_meta_attr(self, name, value):
        """Set value of a meta attribute by name, with broadcasting."""
        if name in self._meta:
            warnings.warn(
                'Meta-attribute {} already exists and '
                'will be overwritten'.format(name)
            )

        self._meta[name] = value

    def get_meta_attr(self, name):
        if name not in self._meta:
            raise ValueError(
                'Instance does not have a meta-attribute `{}`'.format(name)
            )

        return self._meta[name]

    @property
    def num_rows(self):
        return self.fluo.shape[0]

    def append(self, other):
        """Append data in long format in-place.

        Arguments
        ---------
        other : LongFluorescence
            Fluorescence object to append.

        Returns
        -------
        None.

        """
        assert np.ndim(other.fluo) == np.ndim(self.fluo)
        if self.is_z_score != other.is_z_score:
            raise ValueError(
                'Expected self and other `is_z_score` attributes to'
                ' match, got {} and {} instead.'.format(
                    self.is_z_score, other.is_z_score
                )
            )
        if self.num_frames != other.num_frames:
            warnings.warn(
                ShapeWarning(
                    '`num_frames` in self ({}) and other ({}) do not match, {} '
                    'excess frames will be trimmed from the end of {}'.format(
                        self.num_frames,
                        other.num_frames,
                        np.abs(self.num_frames - other.num_frames),
                        'self'
                        if self.num_frames > other.num_frames
                        else 'other',
                    )
                )
            )

        try:
            new_meta = self._meta.append(other._meta, True, False)
            new_fluo = np.concatenate(
                [
                    self.fluo[:, : min(self.num_frames, other.num_frames)],
                    np.asarray(other.fluo, dtype=self._dtypes['fluo'])[
                        :, : min(self.num_frames, other.num_frames)
                    ],
                ],
                axis=0,
            )

            assert (
                new_meta.shape[0] == new_fluo.shape[0]
            ), 'Number of rows will not match.'

            # Save the new attributes if an error hasn't already occurred.
            self._meta = new_meta
            self.fluo = new_fluo

        except AssertionError:
            raise ShapeError(
                'Cannot append {} object with {} rows of fluorescence and '
                '{} rows of metadata'.format(
                    type(other), other.fluo.shape[0], other._meta.shape[0]
                )
            )

    def remove_nan(self):
        """Remove rows with NaN fluorescence.

        Returns
        -------
        Number of rows removed.

        """
        nan_entries = np.isnan(self.fluo)
        if np.any(nan_entries):

            # Print a warning message
            all_nan_rows = np.all(nan_entries, axis=1)
            any_nan_rows = np.any(nan_entries, axis=1)
            if np.array_equal(all_nan_rows, any_nan_rows):
                warnings.warn(
                    'Removing {} all-nan rows'.format(np.sum(all_nan_rows))
                )
            else:
                warnings.warn(
                    'Removing {} rows with nans, {} of which are all-nan'.format(
                        np.sum(any_nan_rows), np.sum(all_nan_rows)
                    )
                )

            # Remove entries with nans
            self._meta = self._meta.loc[~any_nan_rows, :]
            self.fluo = self.fluo[~any_nan_rows, :]

            num_entries_removed = np.sum(any_nan_rows)

        else:
            num_entries_removed = 0

        return num_entries_removed


class SessionTrials:
    """Trials from an imaging session.

    Attributes
    ----------
    trial_timetable : TrialTimetable
    fluo : TrialFluorescence
    cell_type
    mouse_id
    day

    """

    def __init__(
        self,
        trial_timetable: TrialTimetable,
        fluo: TrialFluorescence,
        cell_type,
        mouse_id,
        day,
    ):
        self.trial_timetable = trial_timetable
        self.fluo = fluo
        self.cell_type = cell_type
        self.mouse_id = mouse_id
        self.day = day

    @staticmethod
    def from_spec(spec: RawDataSpec):
        """Create a Session from a RawDataSpec.

        Fluorescence is transformed into a Z-score.

        """
        trial_timetable = TrialTimetable.from_spec(spec)
        raw_fluo = RawFluorescence.from_spec(spec)

        # Transform raw fluorescence into z-score
        raw_fluo.normalize()

        # Some trials might go past the end of the fluorescence recording.
        # We should drop them since they can't be analyzed.
        trial_timetable._drop_out_of_range_trials_inplace(raw_fluo)

        # Cut the raw fluorescence into trials based on the trial_timetable
        # (works better if trials are all around the same length, since trials
        # are truncated to the length of the shortest trial)
        deep_fluo = raw_fluo.to_deep(trial_timetable)

        # Construct and return SessionTrials object.
        sess = SessionTrials(
            trial_timetable, deep_fluo, spec.cell_type, spec.mouse_id, spec.day
        )
        return sess

    def save(self, fname_or_group):
        """Save a Session to HDF5."""
        if isinstance(fname_or_group, str):
            with h5py.File(fname_or_group, 'a') as f:
                self._save_to_h5pygroup(f)
                f.close()
        elif isinstance(fname_or_group, h5py.Group):
            self._save_to_h5pygroup(fname_or_group)
        else:
            raise TypeError(
                'Expected argument `fname_or_group` to be a str filename '
                'or h5py.Group instance, got {} of type {} instead'.format(
                    fname_or_group, type(fname_or_group)
                )
            )

    def _save_to_h5pygroup(self, group):
        group.attrs['cell_type'] = str(self.cell_type)
        group.attrs['mouse_id'] = str(self.mouse_id)
        group.attrs['day'] = str(self.day)
        self.trial_timetable.save(group)
        self.fluo.save(group)

    @staticmethod
    def load(fname_or_group):
        """Load a Session from HDF5."""
        if isinstance(fname_or_group, str):
            with h5py.File(fname_or_group, 'r') as f:
                sess = SessionTrials._load_from_h5pygroup(f)
                f.close()
        elif isinstance(fname_or_group, h5py.Group):
            sess = SessionTrials._load_from_h5pygroup(fname_or_group)
        else:
            raise TypeError(
                'Expected argument `fname_or_group` to be a str filename '
                'or h5py.Group instance, got {} of type {} instead'.format(
                    fname_or_group, type(fname_or_group)
                )
            )

        return sess

    @staticmethod
    def _load_from_h5pygroup(group):
        trial_timetable = TrialTimetable.load(group)
        fluo = TrialFluorescence.load(group)
        sess = SessionTrials(
            trial_timetable,
            fluo,
            cond.CellType.from_str(group.attrs['cell_type']),
            group.attrs['mouse_id'],
            cond.Day.from_str(group.attrs['day']),
        )
        return sess


def walk_sessions(path_to_root: str):
    for raw_data_spec in walk_raw_data_from(path_to_root):
        yield SessionTrials.from_spec(raw_data_spec)
