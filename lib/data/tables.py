import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import numpy as np

from . import conditions as cond

Base = declarative_base()


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

    # Fluorescence traces will be stored as a blob and converted to numpy
    # via a property
    _trace_blob = sa.Column(sa.LargeBinary(600 * 4))
    _trace_blob_dtype = np.float32

    @property
    def trace(self):
        return np.frombuffer(self._trace_blob, dtype=self._trace_blob_dtype)

    @trace.setter
    def trace(self, value):
        self._trace_blob = np.asarray(
            value, dtype=self._trace_blob_dtype
        ).tostring()

    def __repr__(self):
        return f'<Trace from Cell {self.cell_id}, Trial {self.trial_id}>'


class Trial(Base):
    __tablename__ = 'Trials'

    id = sa.Column(sa.Integer(), primary_key=True, autoincrement=True)
    mouse_id = sa.Column(
        sa.String(5), sa.ForeignKey('Mice.id'), nullable=False
    )
    mouse = relationship('Mouse', back_populates='trials')

    day = sa.Column(sa.Integer(), nullable=False)
    catch = sa.Column(sa.Boolean(), nullable=False)

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
