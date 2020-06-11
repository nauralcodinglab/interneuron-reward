import os
import re
from enum import Enum, unique


@unique
class CellType(Enum):
    pyr = 1
    som = 2
    pv = 3
    vip = 4

    def __str__(self):
        if self == CellType.pyr:
            string = 'pyr'
        elif self == CellType.som:
            string = 'som'
        elif self == CellType.pv:
            string = 'pv'
        elif self == CellType.vip:
            string = 'vip'
        else:
            raise ValueError('Not defined for CellType {}'.format(self))

        return string

    @staticmethod
    def from_str(str_):
        if str_ == 'pyr':
            cell_type = CellType.pyr
        elif str_ == 'som':
            cell_type = CellType.som
        elif str_ == 'pv':
            cell_type = CellType.pv
        elif str_ == 'vip':
            cell_type = CellType.vip
        else:
            raise ValueError('No CellType corresponding to {}'.format(str_))

        return cell_type

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

    @staticmethod
    def from_path(path_):
        if path_ == 'PN':
            cell_type = CellType.pyr
        elif path_ == 'SOM':
            cell_type = CellType.som
        elif path_ == 'PV':
            cell_type = CellType.pv
        elif path_ == 'VIP':
            cell_type = CellType.vip
        else:
            raise ValueError('No CellType corresponding to {}'.format(path_))

        return cell_type


class Day:
    def __init__(self, day: int):
        self._day = day

    def __hash__(self):
        return self._day

    def __str__(self):
        return 'day_{}'.format(self._day)

    @staticmethod
    def from_str(str_):
        regex_matches = re.search(r'day_(\d+)', str(str_))
        if regex_matches is None:
            raise ValueError(
                'Could not coerce {} to Day. (A Day str should'
                ' look like `day_x`, where x is an integer.)'.format(str_)
            )

        return Day(int(regex_matches[1]))

    def as_path(self) -> str:
        """Get Day as a string component of a file path."""
        return 'Day {}'.format(self._day)

    @staticmethod
    def from_path(path_):
        regex_matches = re.search(r'Day (\d+)', str(path_))
        if regex_matches is None:
            raise ValueError(
                'Could not coerce {} to Day. (A Day path should'
                ' look like `Day x`, where x is an integer.)'.format(path_)
            )

        return Day(int(regex_matches[1]))


DAYS = (Day(1), Day(3), Day(5), Day(7))
