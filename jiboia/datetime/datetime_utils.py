import pandas as pd

from .regex_pattern import (
    regex_pattern_date,
    regex_pattern_bad_date,
    regex_pattern_datetime_all,
)
from ..utils.log_utils import print_log
from ..string.string_utils import StringUtils
from ..utils.str_utils import combine_regex
from ..utils.validation_utils import (
    PandasSupportedDtypes,
    is_valid_to_normalize,
)


class DateTimeUtils:
    @staticmethod
    def normalize(
        dataframe: pd.DataFrame,
        column_name: str,
        match_min_rate: int = 50,
        inplace: bool = False,
        chunk_size: int = 500_000,
        show_log: bool = True,
    ) -> bool | pd.DataFrame:
        return DateTimeUtils.to_datetime(
            dataframe=dataframe,
            column_name=column_name,
            match_min_rate=match_min_rate,
            inplace=inplace,
            chunk_size=chunk_size,
            show_log=show_log,
        )

    @staticmethod
    def to_datetime(
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

        is_date: bool = DateTimeUtils.is_date(
            series=dataframe[column_name],
            match_min_rate=match_min_rate,
            chunk_size=chunk_size,
        )
        if not is_date:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        is_unique_datetime_pattern: bool = DateTimeUtils.is_unique_datetime_pattern(
            series=dataframe[column_name],
            chunk_size=chunk_size,
        )

        if is_unique_datetime_pattern:
            is_normalized = DateTimeUtils.normalize_unique_pattern(
                dataframe=dataframe,
                column_name=column_name,
                inplace=True,
                chunk_size=chunk_size,
            )
            if not is_normalized:
                return False

            print_log(
                column_name=column_name,
                column_type=str(dataframe[column_name].dtype),
                show_log=show_log,
            )

            return True if inplace else dataframe

        DateTimeUtils.normalize_date_delimiters(
            dataframe=dataframe,
            column_name=column_name,
            inplace=True,
            chunk_size=chunk_size,
        )

        # Datas com um único dígito: 1/1/2010
        has_bad_date_format: bool = StringUtils.match(
            series=dataframe[column_name],
            regex=combine_regex(regex_pattern_bad_date),
            chunk_size=chunk_size,
        )
        if has_bad_date_format:
            DateTimeUtils.fix_digit_shape(
                dataframe, column_name, inplace=True, chunk_size=chunk_size
            )

        is_normalized = DateTimeUtils.normalize_mixed_dates(
            dataframe=dataframe,
            column_name=column_name,
            inplace=True,
            chunk_size=chunk_size,
        )
        if not is_normalized:
            return False

        print_log(
            column_name=column_name,
            column_type=str(dataframe[column_name].dtype),
            show_log=show_log,
        )

        return True if inplace else dataframe

    @staticmethod
    def is_date(
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
            regex_pattern_date + regex_pattern_bad_date + regex_pattern_datetime_all
        )

        return bool(
            StringUtils.match(
                series=series,
                regex=combined_regex,
                match_min_rate=0 if match_min_rate is None else match_min_rate,
                chunk_size=chunk_size,
            )
        )

    @staticmethod
    def is_unique_datetime_pattern(
        series: pd.Series,
        chunk_size: int = 500_000,
    ) -> bool:
        is_valid: bool = is_valid_to_normalize(
            series=series,
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        datetime_types_found: int = 0

        for pattern in regex_pattern_datetime_all:
            if StringUtils.match(
                series=series,
                regex=pattern["regex"],
                match_min_rate=0,
                chunk_size=chunk_size,
            ):
                datetime_types_found += 1

        return datetime_types_found == 1

    @staticmethod
    def normalize_unique_pattern(
        dataframe: pd.DataFrame,
        column_name: str,
        inplace: bool = False,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        is_unique_datetime_pattern: bool = DateTimeUtils.is_unique_datetime_pattern(
            series=dataframe[column_name],
            chunk_size=chunk_size,
        )
        if not is_unique_datetime_pattern:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        combined_regex: str = combine_regex(regex_pattern_datetime_all)

        # Valores que não casam com o padrão viram nulo
        StringUtils._apply_chunks(
            dataframe,
            column_name,
            lambda s: s.where(s.str.match(combined_regex, na=False)),
            chunk_size,
        )

        # utc=True evita colunas "object" quando há offsets diferentes;
        # depois remove o timezone, mantendo datetime64 "ingênuo" como no cudf
        converted: pd.Series = pd.to_datetime(
            dataframe[column_name], errors="coerce", utc=True
        ).dt.tz_localize(None)

        dataframe[column_name] = converted.astype("datetime64[ns]")

        return True if inplace else dataframe

    @staticmethod
    def normalize_date_delimiters(
        dataframe: pd.DataFrame,
        column_name: str,
        inplace: bool = False,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        StringUtils._apply_chunks(
            dataframe,
            column_name,
            # um único regex no lugar de 4 replaces encadeados
            lambda s: s.str.replace(r"[/ _.]", "-", regex=True),
            chunk_size,
        )

        return True if inplace else dataframe

    @staticmethod
    def fix_digit_shape(
        dataframe: pd.DataFrame,
        column_name: str,
        inplace: bool = False,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        def _fix(chunk: pd.Series) -> pd.Series:
            # d-mm-yy e d-m-yy -> 0d-mm-yy e 0d-m-yy
            chunk = chunk.str.replace(
                r"^(?:\d[^\w\d]\d{1,2}[^\w\d]\d{2,4})$",
                r"0\g<0>",
                regex=True,
            )
            # dd-m-yy -> dd-0m-yy
            # (no re do Python, \g<N> separa o grupo do "0", sem o truque do "|")
            return chunk.str.replace(
                r"^(\d{2}[^\w\d])(\d[^\w\d]\d{2,4})$",
                r"\g<1>0\g<2>",
                regex=True,
            )

        StringUtils._apply_chunks(dataframe, column_name, _fix, chunk_size)

        return True if inplace else dataframe

    @staticmethod
    def combine_date_time(
        dataframe: pd.DataFrame,
        column_name: str,
        time_column_name: str,
        inplace: bool = False,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        """
        Soma o horário de uma coluna timedelta a uma coluna datetime.
        """
        is_valid_datetime: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.datetime_types,
        )
        is_valid_timedelta: bool = is_valid_to_normalize(
            series=dataframe[time_column_name],
            valid_types=PandasSupportedDtypes.timedelta_types,
        )
        if not is_valid_datetime or not is_valid_timedelta:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        dataframe[column_name] = dataframe[column_name] + dataframe[time_column_name]

        return True if inplace else dataframe

    @staticmethod
    def normalize_mixed_dates(
        dataframe: pd.DataFrame,
        column_name: str,
        inplace: bool = False,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        def _parse(chunk: pd.Series) -> pd.Series:
            result: pd.Series = pd.Series(
                pd.NaT, index=chunk.index, dtype="datetime64[ns]"
            )
            for pattern in regex_pattern_date:
                mask: pd.Series = chunk.str.match(pattern["regex"], na=False)
                if not mask.any():
                    continue
                parsed: pd.Series = pd.to_datetime(
                    chunk.where(mask), format=pattern["format"], errors="coerce"
                )
                # primeiro padrão que casar vence; não sobrescreve o já convertido
                result = result.fillna(parsed)
            return result

        StringUtils._apply_chunks(dataframe, column_name, _parse, chunk_size)

        dataframe[column_name] = dataframe[column_name].astype("datetime64[s]")

        return True if inplace else dataframe
