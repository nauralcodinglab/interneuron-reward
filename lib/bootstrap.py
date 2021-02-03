from typing import Callable, Iterable, List, Dict

import numpy as np

from .data import tables as tab
from .data import conditions as cond


TRIAL_LENGTH_IN_FRAMES = 390


def get_mouseid_sampler(
    session, cell_type: cond.CellType,
) -> Callable[[], Iterable[str]]:
    """Get a function that returns a bootstrap sample of mouse ids when called."""
    mouse_records = (
        session.query(tab.Mouse)
        .filter(
            tab.Mouse.cell_type == cell_type,
            tab.Mouse.id != 'CL199',  # Excluded due to fluorescence drift
        )
        .all()
    )
    mouse_ids = [m.id for m in mouse_records]

    def sample() -> Iterable[str]:
        f"""Get a bootstrap sample of {cell_type} mouse ids."""
        return np.random.choice(mouse_ids, len(mouse_ids), replace=True)

    return sample


def get_mouseids_by_celltype(session, cell_type: cond.CellType):
    mouse_records = (
        session.query(tab.Mouse).filter(tab.Mouse.cell_type == cell_type).all()
    )
    return [m.id for m in mouse_records]


def get_day_one_cell_ids(session, mouse_id: str) -> Iterable[str]:
    """Get ids of cells that were added on day 1."""
    this_mouse_day_one_cells = (
        session.query(tab.Cell)
            .join(tab.SessionTrace, tab.SessionTrace.cell_id == tab.Cell.id)
            .filter(tab.SessionTrace.day == 1, tab.Cell.mouse_id == mouse_id)
            .all()
    )
    assert np.all([c.mouse_id == mouse_id for c in this_mouse_day_one_cells])

    cell_ids = [c.id for c in this_mouse_day_one_cells]
    return cell_ids


def get_cellid_sampler(session, mouseid: str) -> Callable[[], np.ndarray]:
    cell_ids = get_day_one_cell_ids(session, mouseid)

    def sample() -> List[int]:
        f"""Sample cell ids from mouse {mouseid} with replacement."""
        return np.random.choice(cell_ids, len(cell_ids), replace=True).tolist()

    return sample


def get_averagetrace_given_cellid_sampler(
    session, cellid_sampler: Callable, mouse_id: str, day: int
) -> Callable[[], np.ndarray]:
    num_trials_today = (
        session.query(tab.Trial)
        .filter(tab.Trial.day == day, tab.Trial.mouse_id == mouse_id)
        .count()
    )
    num_frames_today = len(
        session.query(tab.SessionTrace)
            .filter(
                tab.SessionTrace.cell_id
                == get_day_one_cell_ids(session, mouse_id)[0],
                tab.SessionTrace.day == day,
            )
            .first()
            .trace
    )

    session_traces: Dict[str, np.ndarray] = _cache_session_traces(session, mouse_id, day)

    def sample() -> np.ndarray:
        """Randomly sample trial times and cells and return the trial averages.

        Randomly samples cells using cellid_sampler. If the output of
        cellid_sampler is deterministic, the cells are not randomly sampled.

        Returns
        -------
        np.ndarray
            Matrix of trial-averaged fluorescence measurements. Rows are cells
            and columns are timesteps. Not necessarily normalized.

        """
        cellids = cellid_sampler()
        trial_start_inds = np.random.randint(
            0,
            num_frames_today - TRIAL_LENGTH_IN_FRAMES,
            size=(num_trials_today),
        )

        average_traces = np.zeros((len(cellids), TRIAL_LENGTH_IN_FRAMES))
        for i, cellid in enumerate(cellids):
            for start_ind in trial_start_inds:
                average_traces[i, :] += session_traces[cellid][
                    start_ind : (start_ind + TRIAL_LENGTH_IN_FRAMES)
                ]

        average_traces /= num_trials_today

        return average_traces

    return sample


def _cache_session_traces(session, mouse_id: str, day: int) -> Dict[str, np.ndarray]:
    # Get session trace records for all cells from mouse on day.
    day_one_cell_ids = get_day_one_cell_ids(session, mouse_id)
    session_records: Dict[str, tab.SessionTrace] = {
        cid: (
            session.query(tab.SessionTrace)
                .filter(tab.SessionTrace.cell_id == cid, tab.SessionTrace.day == day)
                .first()
        )
        for cid in day_one_cell_ids
    }

    # Check that records have been loaded correctly.
    for rec in session_records.values():
        assert rec.cell.mouse_id == mouse_id
        assert rec.day == day

    # Extract traces into a dict and return it
    session_traces: Dict[str, np.ndarray] = {
        cid: rec.trace for cid, rec in session_records.items()
    }

    return session_traces
