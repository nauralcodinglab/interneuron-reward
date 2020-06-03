import os
from enum import Enum, unique


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
