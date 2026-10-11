from typing import Callable

import pandas as pd

from .regex_pattern import (
    regex_pattern_time_utc,
    regex_pattern_time_amp_pm,
    regex_pattern_time_hh_mm,
    regex_pattern_time_hh_mm_ss,
    regex_pattern_time_hh_mm_ss_n,
    regex_pattern_timedelta,
)
from ..utils.log_utils import print_log
from ..string.string_utils import StringUtils
from ..utils.str_utils import combine_regex
from ..utils.validation_utils import (
    PandasSupportedDtypes,
    is_valid_to_normalize,
)


class TimeUtils:
    @staticmethod
    def _apply_on_match(
        dataframe: pd.DataFrame,
        column_name: str,
        pattern: str,
        transform: Callable[[pd.Series], pd.Series] | None = None,
        chunk_size: int = 500_000,
    ) -> None:
        """
        Processa a coluna em chunks, alterando o dataframe inplace.

        - transform informado: aplica `transform` apenas aos valores que casam com `pattern`.
        - transform None: valores que NÃO casam com `pattern` viram nulos.
        """
        total_rows: int = len(dataframe)
        parts: list[pd.Series] = []

        for start_index in range(0, total_rows, chunk_size):
            end_index: int = min(start_index + chunk_size, total_rows)
            chunk: pd.Series = dataframe[column_name].iloc[start_index:end_index]

            # na=False: nulos contam como "não casa" (no pandas, sem isso o mask teria NaN)
            mask: pd.Series = chunk.str.match(pattern, na=False)

            if transform is None:
                parts.append(chunk.where(mask))
            else:
                parts.append(chunk.where(~mask, transform(chunk)))

        result: pd.Series = pd.concat(parts)
        result.index = dataframe.index  # atribuição por posição
        dataframe[column_name] = result

    @staticmethod
    def normalize(
        dataframe: pd.DataFrame,
        column_name: str,
        match_min_rate: int = 50,
        inplace: bool = False,
        chunk_size: int = 500_000,
        show_log: bool = True,
    ) -> bool | pd.DataFrame:
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        is_time: bool = TimeUtils.is_time(
            series=dataframe[column_name],
            match_min_rate=match_min_rate,
            chunk_size=chunk_size,
        )
        if not is_time:
            return False

        # UTC -> hh:mm:00
        pattern_time_utc: str = combine_regex(regex_pattern_time_utc)
        if StringUtils.match(
            dataframe[column_name],
            regex=pattern_time_utc,
            match_min_rate=0,
            chunk_size=chunk_size,
        ):
            TimeUtils._apply_on_match(
                dataframe,
                column_name,
                pattern_time_utc,
                transform=lambda s: (
                    s.str.replace(" ", "", regex=False)
                    .str.replace("UTC", "", regex=False)
                    .str.slice(0, 2)
                    + ":"
                    + s.str.slice(2, 4)
                    + ":00"
                ),
                chunk_size=chunk_size,
            )

        # hh:mm -> hh:mm:00
        pattern_time_hh_mm: str = combine_regex(regex_pattern_time_hh_mm)
        if StringUtils.match(
            dataframe[column_name],
            regex=pattern_time_hh_mm,
            match_min_rate=0,
            chunk_size=chunk_size,
        ):
            TimeUtils._apply_on_match(
                dataframe,
                column_name,
                pattern_time_hh_mm,
                transform=lambda s: s + ":00",
                chunk_size=chunk_size,
            )

        # hh:mm:ss.s -> hh:mm:ss
        pattern_time_hh_mm_ss_n: str = combine_regex(regex_pattern_time_hh_mm_ss_n)
        if StringUtils.match(
            dataframe[column_name],
            regex=pattern_time_hh_mm_ss_n,
            match_min_rate=0,
            chunk_size=chunk_size,
        ):
            TimeUtils._apply_on_match(
                dataframe,
                column_name,
                pattern_time_hh_mm_ss_n,
                transform=lambda s: s.str.slice(0, 8),
                chunk_size=chunk_size,
            )

        # Valores inválidos são convertidos em nulos
        pattern_time_hh_mm_ss: str = combine_regex(regex_pattern_time_hh_mm_ss)
        if StringUtils.match(
            dataframe[column_name],
            regex=pattern_time_hh_mm_ss,
            match_min_rate=0,
            chunk_size=chunk_size,
        ):
            TimeUtils._apply_on_match(
                dataframe,
                column_name,
                pattern_time_hh_mm_ss,
                transform=None,
                chunk_size=chunk_size,
            )

        # str -> timedelta (tempo desde 00:00:00)
        parsed: pd.Series = pd.to_datetime(
            dataframe[column_name], format="%H:%M:%S", errors="coerce"
        )
        dataframe[column_name] = parsed - parsed.dt.normalize()

        print_log(
            column_name=column_name,
            column_type=str(dataframe[column_name].dtype),
            show_log=show_log,
        )

        if not inplace:
            return dataframe

        return True

    @staticmethod
    def is_time(
        series: pd.Series,
        match_min_rate: int | None = 50,
        chunk_size: int = 500_000,
    ) -> bool:
        is_valid: bool = is_valid_to_normalize(
            series=series,
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        combined_regex: str = combine_regex(
            regex_pattern_time_utc
            + regex_pattern_time_hh_mm
            + regex_pattern_time_hh_mm_ss
            + regex_pattern_time_hh_mm_ss_n
        )

        return bool(
            StringUtils.match(
                series=series,
                regex=combined_regex,
                match_min_rate=match_min_rate,
                chunk_size=chunk_size,
            )
        )

    @staticmethod
    def is_time_am_pm(
        series: pd.Series,
        match_min_rate: int | None = 50,
        chunk_size: int = 500_000,
    ) -> bool:
        is_valid: bool = is_valid_to_normalize(
            series=series,
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        combined_regex: str = combine_regex(regex_pattern_time_amp_pm)

        return bool(
            StringUtils.match(
                series=series,
                regex=combined_regex,
                match_min_rate=match_min_rate,
                chunk_size=chunk_size,
            )
        )

    @staticmethod
    def is_unique_timedelta_format(
        series: pd.Series,
        chunk_size: int = 500_000,
    ) -> bool:
        is_valid: bool = is_valid_to_normalize(
            series=series,
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        timedelta_types_found: int = 0

        for pattern in regex_pattern_timedelta:
            if StringUtils.match(
                series=series,
                regex=pattern["regex"],
                match_min_rate=0,
                chunk_size=chunk_size,
            ):
                timedelta_types_found += 1

        return timedelta_types_found == 1
