import os

import h5py
import numpy as np

from . import conditions as cond
from . import raw_io as rio


def load_sessions_by_day(
    raw_data_path, day: cond.Day, catch_behaviour='exclude'
):
    sessions = {}
    with h5py.File(os.path.join(raw_data_path, str(day) + '.h5'), 'r') as f:
        for cell_type in cond.CellType:
            sessions[cell_type] = []
            for mouse in f[cell_type.as_path()].keys():
                sess_trials = rio.SessionTrials.load(
                    f[cell_type.as_path()][mouse]
                )

                if catch_behaviour == 'include':
                    pass
                else:
                    if catch_behaviour == 'exclude':
                        selection_mask = sess_trials.trial_timetable[
                            'reward_delivered'
                        ]
                    elif catch_behaviour == 'select':
                        selection_mask = ~sess_trials.trial_timetable[
                            'reward_delivered'
                        ]
                    else:
                        raise ValueError(
                            'Expected `catch_behaviour` to be one of '
                            '`include`, `exclude`, or `select`, '
                            'got {} instead.'.format(catch_behaviour)
                        )

                    sess_trials.trial_timetable.drop(
                        index=sess_trials.trial_timetable.index[~selection_mask], inplace=True
                    )
                    sess_trials.fluo.fluo = sess_trials.fluo.fluo[
                        selection_mask, ...
                    ]
                    sess_trials.fluo.trial_num = sess_trials.fluo.trial_num[
                        selection_mask
                    ]

                assert (
                    sess_trials.trial_timetable.shape[0]
                    == sess_trials.fluo.num_trials
                )
                assert np.array_equal(
                    sess_trials.trial_timetable['trial_num']
                    .to_numpy()
                    .flatten(),
                    sess_trials.fluo.trial_num,
                )

                sessions[cell_type].append(
                    sess_trials
                )

        f.close()

    return sessions
