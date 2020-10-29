#!/usr/bin/env python
"""Load raw data into SQL database."""

import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from lib.data import raw_io as rio
from lib.data import tables as tab
from lib.data import conditions as cond

RAW_DATA_PATH = Path().joinpath('..', 'interneuron-reward-data', 'raw')

# Connect to SQL database via sqlalchemy session
engine = sa.create_engine(os.environ['SQLALCHEMY_ENGINE_URL'])
Session = sessionmaker(bind=engine)
sa_session = Session()

# Create database tables if they don't already exist.
tab.Base.metadata.create_all(engine)

# Walk sessions, adding them to the database
for sess in rio.walk_sessions(RAW_DATA_PATH):
    print('Saving {} {} {}'.format(sess.cell_type, sess.day, sess.mouse_id))

    # Add mouse if it doesn't already exist
    mouse_exists = (
        sa_session.query(tab.Mouse).filter_by(id=sess.mouse_id).first()
    )
    if not mouse_exists:
        sa_session.add(tab.Mouse(id=sess.mouse_id, cell_type=sess.cell_type))
        sa_session.commit()

    # Add this session's trials
    trial_records = []
    for _, trial in sess.trial_timetable.iterrows():
        if trial['reward_delivered']:
            trial_kind = cond.TrialKind.non_catch
        else:
            trial_kind = cond.TrialKind.catch

        trial_records.append(
            tab.Trial(
                mouse_id=sess.mouse_id,
                day=int(sess.day),
                trial_kind=trial_kind,
                start_time=trial['trial_start'],
                stop_time=trial['trial_end'],
            )
        )
    sa_session.add_all(trial_records)
    sa_session.commit()

    # Add cells from today if they don't already exist
    for i, cell_num in enumerate(sess.fluo.cell_num):

        # Get the record for this cell, creating it if it doesn't exist
        cell_record = (
            sa_session.query(tab.Cell)
            .filter_by(mouse_id=sess.mouse_id, within_mouse_id=cell_num)
            .first()
        )
        if not cell_record:
            cell_record = tab.Cell(
                mouse_id=sess.mouse_id, within_mouse_id=cell_num
            )
            sa_session.add(cell_record)
            sa_session.commit()

        # Add trial-by-trial traces
        for j, trial_record in enumerate(trial_records):
            trace_record = tab.Trace(
                trial_id=trial_record.id,
                cell_id=cell_record.id,
                trace=sess.fluo.fluo[j, i, :],
            )
            sa_session.add(trace_record)

        # Add trial average traces
        all_trial_average_trace = tab.TrialAverageTrace(
            cell_id=cell_record.id,
            trial_kind=cond.TrialKind.all,
            day=int(sess.day),
            num_trials=sess.fluo.num_trials,
            trace=sess.fluo.fluo[:, i, :].mean(axis=0)
        )
        catch_mask = ~sess.trial_timetable['reward_delivered'].to_numpy()
        catch_trial_average_trace = tab.TrialAverageTrace(
            cell_id=cell_record.id,
            trial_kind=cond.TrialKind.catch,
            day=int(sess.day),
            num_trials=catch_mask.sum(),
            trace=sess.fluo.fluo[:, i, :][catch_mask, :].mean(axis=0)
        )
        non_catch_trial_average_trace = tab.TrialAverageTrace(
            cell_id=cell_record.id,
            trial_kind=cond.TrialKind.non_catch,
            day=int(sess.day),
            num_trials=(~catch_mask).sum(),
            trace=sess.fluo.fluo[:, i, :][~catch_mask, :].mean(axis=0)
        )
        sa_session.add_all([all_trial_average_trace, catch_trial_average_trace, non_catch_trial_average_trace])

    sa_session.commit()
