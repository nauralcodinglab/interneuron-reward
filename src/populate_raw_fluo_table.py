#!/usr/bin/env python
"""Load raw flourescence data into existing SQL database."""

import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from lib.data import raw_io as rio
from lib.data import tables as tab

RAW_DATA_PATH = Path().joinpath('..', 'interneuron-reward-data', 'raw')

# Connect to SQL database via sqlalchemy session
engine = sa.create_engine(os.environ['SQLALCHEMY_ENGINE_URL'])
Session = sessionmaker(bind=engine)
sa_session = Session()

# Create database tables if they don't already exist.
tab.Base.metadata.create_all(engine)

# Walk imaging sessions and save fluorescence traces from each.
for spec in rio.walk_raw_data_from(RAW_DATA_PATH):
    print(
        f'Saving full session traces for {spec.mouse_id} {spec.cell_type} {spec.day}'
    )

    # Load the fluorescence traces for this session. Do not Z-score.
    raw_fluo = rio.RawFluorescence.from_spec(spec)

    # Store the fluorescence trace for each cell.
    for i, cell_num in enumerate(raw_fluo.cell_num):
        cell_record = (
            sa_session.query(tab.Cell)
            .filter_by(mouse_id=spec.mouse_id, within_mouse_id=cell_num)
            .first()
        )

        sessiontrace_record = tab.SessionTrace(
            cell_id=cell_record.id, day=int(spec.day), trace=raw_fluo.fluo[i, :]
        )
        sa_session.add(sessiontrace_record)

    sa_session.commit()
