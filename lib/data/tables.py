import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects import mysql
import numpy as np

from . import conditions as cond

Base = declarative_base()


class BaseNumpyVector:
    def process_bind_param(self, value, dialect):
        assert np.ndim(value) == 1
        return np.asarray(value, dtype=np.float32).tostring()

    def process_result_value(self, value, dialect):
        return np.frombuffer(value, dtype=np.float32)


class NumpyVector(sa.types.TypeDecorator, BaseNumpyVector):
    impl = sa.LargeBinary

class BigNumpyVector(sa.types.TypeDecorator, BaseNumpyVector):
    impl = mysql.MEDIUMBLOB


class TrialAverageTrace(Base):
    """Fluorescence trace for one cell averaged over many trials."""

    __tablename__ = 'TrialAverageTraces'

    cell_id = sa.Column(
        sa.Integer(), sa.ForeignKey('Cells.id'), primary_key=True
    )
    cell = relationship('Cell', back_populates='trial_average_traces')

    trial_kind = sa.Column(sa.Enum(cond.TrialKind), primary_key=True)
    day = sa.Column(sa.Integer(), primary_key=True)

    num_trials = sa.Column(sa.Integer())

    trace = sa.Column(NumpyVector)

    def __repr__(self):
        return (
            f'<Average trace from Cell {self.cell_id}, {self.trial_kind} '
            'on day {self.day}>'
        )


class Trace(Base):
    """Fluorescence trace from a single cell during a single trial."""

    __tablename__ = 'Traces'

    cell_id = sa.Column(
        sa.Integer(), sa.ForeignKey('Cells.id'), primary_key=True
    )
    cell = relationship('Cell', back_populates='traces')

    trial_id = sa.Column(
        sa.Integer(), sa.ForeignKey('Trials.id'), primary_key=True
    )
    trial = relationship('Trial', back_populates='traces')

    trace = sa.Column(NumpyVector)

    def __repr__(self):
        return f'<Trace from Cell {self.cell_id}, Trial {self.trial_id}>'


class SessionTrace(Base):
    """Fluorescence trace from a single cell during an entire session.

    Not Z-scored.

    """

    __tablename__ = 'SessionTraces'

    cell_id = sa.Column(
        sa.Integer(), sa.ForeignKey('Cells.id'), primary_key=True
    )
    cell = relationship('Cell', back_populates='session_trace')

    day = sa.Column(sa.Integer(), primary_key=True)

    trace = sa.Column(BigNumpyVector)

    def __repr__(self):
        return f'<SessionTrace from Cell {self.cell_id}>'


class Trial(Base):
    __tablename__ = 'Trials'

    id = sa.Column(sa.Integer(), primary_key=True, autoincrement=True)
    mouse_id = sa.Column(
        sa.String(5), sa.ForeignKey('Mice.id'), nullable=False
    )
    mouse = relationship('Mouse', back_populates='trials')

    day = sa.Column(sa.Integer(), nullable=False)
    trial_kind = sa.Column(sa.Enum(cond.TrialKind), nullable=False)

    start_time = sa.Column(sa.Float())
    stop_time = sa.Column(sa.Float())

    traces = relationship('Trace', back_populates='trial')

    def __repr__(self):
        return f'<Trial {self.id}>'

    def __str__(self):
        if self.catch:
            prefix = 'catch'
        else:
            prefix = 'non-catch'

        return (
            f'Trial {self.id}: {prefix} trial in mouse {self.mouse_id} '
            f'({self.mouse.cell_type}) on day {self.day}'
        )


class Cell(Base):
    """A single cell or ROI, possibly tracked over multiple days."""

    __tablename__ = 'Cells'

    id = sa.Column(sa.Integer(), autoincrement=True, primary_key=True)
    within_mouse_id = sa.Column(sa.Integer())
    mouse_id = sa.Column(sa.String(5), sa.ForeignKey('Mice.id'))
    mouse = relationship('Mouse', back_populates='cells')

    traces = relationship('Trace', back_populates='cell')
    trial_average_traces = relationship(
        'TrialAverageTrace', back_populates='cell'
    )
    session_trace = relationship('SessionTrace', back_populates='cell')

    def __repr__(self):
        return f'<Cell {self.id}>'

    def __str__(self):
        return (
            f'Cell {self.id}: {self.mouse.cell_type} cell from mouse '
            f'{self.mouse.id}'
        )


class Mouse(Base):
    __tablename__ = 'Mice'

    id = sa.Column(sa.String(5), primary_key=True)
    cell_type = sa.Column(sa.Enum(cond.CellType), nullable=False)

    cells = relationship('Cell', back_populates='mouse')
    trials = relationship('Trial', back_populates='mouse')

    def __repr__(self):
        return f'<Mouse {self.id}>'

    def __str__(self):
        return f'Mouse {self.id}: {self.cell_type}'
