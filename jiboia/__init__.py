from dataclasses import dataclass, field
from typing import Literal

from .boolean.boolean_utils import BooleanUtils
from .utils.csv_utils import CsvUtils
from .utils.chunk_utils import (
    chunk_df,
    chunk_iterate,
    chunk_iterate_index,
)
from .dataframe.df_utils import DfUtils
from .datetime.datetime_utils import DateTimeUtils
from .null.null_utils import NullUtils
from .number.number_utils import NumberUtils
from .string.string_utils import StringUtils
from .time.time_utils import TimeUtils


@dataclass
class JiboiaConfig:
    inplace: bool = False
    show_log: bool = True
    chunk_size: int = 500_000
    match_min_rate: int = 0
    null_values: list[str] = field(default_factory=list)
    to_case: None | Literal["lower", "upper"] = None
    to_ASCII: bool = False
    bool_number: bool = False
    create_category: bool = True

    def reset(self) -> None:
        """Volta todos os campos para os valores padrão, mantendo o mesmo objeto."""
        self.__init__()


class Jiboia:
    @staticmethod
    def config(
        *,
        inplace: bool = False,
        show_log: bool = True,
        chunk_size: int = 500_000,
        match_min_rate: int = 0,
        null_values: list[str] | None = None,
        to_case: None | Literal["lower", "upper"] = None,
        to_ASCII: bool = False,
        bool_number: bool = False,
        create_category: bool = True,
    ) -> None:
        config.inplace = inplace
        config.show_log = show_log
        config.chunk_size = chunk_size
        config.match_min_rate = match_min_rate
        config.null_values = list(null_values or [])
        config.to_case = to_case
        config.to_ASCII = to_ASCII
        config.bool_number = bool_number
        config.create_category = create_category

    @staticmethod
    def reset_config() -> None:
        """Reseta as configurações da Jiboia para os valores padrão."""
        config.reset()


jiboia = Jiboia()
config = JiboiaConfig()

bool = BooleanUtils()
csv = CsvUtils()
dt = DateTimeUtils()
df = DfUtils()
null = NullUtils()
num = NumberUtils()
str = StringUtils()
time = TimeUtils()

__all__ = [
    "jiboia",
    "Jiboia",
    "config",
    "bool",
    "csv",
    "dt",
    "df",
    "null",
    "num",
    "str",
    "time",
]
