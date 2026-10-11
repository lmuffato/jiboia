import pandas as pd

from .regex_pattern import (
    regex_pattern_boolean,
    regex_pattern_boolean_raw,
    regex_pattern_boolean_numeric_raw,
)
from ..utils.chunk_utils import chunk_iterate
from ..utils.log_utils import print_log
from ..string.string_utils import StringUtils
from ..utils.str_utils import combine_regex
from ..utils.validation_utils import (
    PandasSupportedDtypes,
    is_valid_to_normalize,
)


class BooleanUtils:
    @staticmethod
    def normalize(
        dataframe: pd.DataFrame,
        column_name: str,
        bool_number: bool = False,
        match_min_rate: int = 50,
        inplace: bool = False,
        show_log: bool = True,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        valid_types: list[str] = (
            PandasSupportedDtypes.str_types + PandasSupportedDtypes.numeric_types
        )

        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=valid_types,
        )
        if not is_valid:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        has_normalized: bool = False

        if bool_number:
            has_normalized = BooleanUtils.from_binary_num(
                dataframe=dataframe,
                column_name=column_name,
                inplace=True,
                show_log=show_log,
                chunk_size=chunk_size,
            ) or BooleanUtils.from_binary_str(
                dataframe=dataframe,
                column_name=column_name,
                inplace=True,
                show_log=show_log,
                chunk_size=chunk_size,
            )

        if not has_normalized:
            has_normalized = BooleanUtils.from_bool_str(
                dataframe=dataframe,
                column_name=column_name,
                match_min_rate=match_min_rate,
                inplace=True,
                show_log=show_log,
                chunk_size=chunk_size,
            )

        if has_normalized:
            return True if inplace else dataframe

        return False

    @staticmethod
    def from_binary_str(
        dataframe: pd.DataFrame,
        column_name: str,
        inplace: bool = False,
        show_log: bool = True,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        is_binary_str: bool = BooleanUtils.is_binary_str(
            series=dataframe[column_name],
            chunk_size=chunk_size,
        )
        if not is_binary_str:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        mapping_dict: dict = {}
        for pattern in regex_pattern_boolean_numeric_raw:
            for fmt in pattern["format"]:
                mapping_dict[fmt] = pattern["pattern"]

        dataframe[column_name] = (
            dataframe[column_name]
            .astype("string")
            .str.lower()
            .map(mapping_dict)
            .astype("boolean")
        )

        print_log(
            column_name=column_name,
            column_type=str(dataframe[column_name].dtype),
            show_log=show_log,
        )

        return True if inplace else dataframe

    @staticmethod
    def from_binary_num(
        dataframe: pd.DataFrame,
        column_name: str,
        inplace: bool = False,
        show_log: bool = True,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.numeric_types,
        )
        if not is_valid:
            return False

        is_binary_num: bool = BooleanUtils.is_binary_num(
            series=dataframe[column_name],
            chunk_size=chunk_size,
        )
        if not is_binary_num:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        col: pd.Series = dataframe[column_name]

        # 1 -> True, 0 -> False, nulo continua nulo
        dataframe[column_name] = (col == 1).astype("boolean").mask(col.isna())

        print_log(
            column_name=column_name,
            column_type=str(dataframe[column_name].dtype),
            show_log=show_log,
        )

        return True if inplace else dataframe

    @staticmethod
    def from_bool_str(
        dataframe: pd.DataFrame,
        column_name: str,
        match_min_rate: int = 50,
        inplace: bool = False,
        show_log: bool = True,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        is_bool: bool = BooleanUtils.is_bool(
            series=dataframe[column_name],
            match_min_rate=match_min_rate,
            chunk_size=chunk_size,
        )
        if not is_bool:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        mapping_dict: dict = {}
        for pattern in regex_pattern_boolean:
            for fmt in pattern["format"]:
                mapping_dict[fmt] = pattern["pattern"]

        StringUtils._apply_chunks(
            dataframe,
            column_name,
            lambda s: s.astype("string").str.lower().map(mapping_dict),
            chunk_size,
        )

        dataframe[column_name] = dataframe[column_name].astype("boolean")

        print_log(
            column_name=column_name,
            column_type=str(dataframe[column_name].dtype),
            show_log=show_log,
        )

        return True if inplace else dataframe

    @staticmethod
    def is_binary_str(
        series: pd.Series,
        chunk_size: int = 500_000,
    ) -> bool:
        is_valid: bool = is_valid_to_normalize(
            series=series,
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        for chunk in chunk_iterate(series, chunk_size):
            if not chunk.dropna().str.match(r"^[01]$", na=False).all():
                return False

        return True

    @staticmethod
    def is_binary_num(
        series: pd.Series,
        chunk_size: int = 500_000,
    ) -> bool:
        is_valid: bool = is_valid_to_normalize(
            series=series,
            valid_types=PandasSupportedDtypes.numeric_types,
        )
        if not is_valid:
            return False

        for chunk in chunk_iterate(series, chunk_size):
            if not chunk.dropna().isin([0, 1]).all():
                return False

        return True

    @staticmethod
    def is_bool(
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

        combined_regex: str = combine_regex(regex_pattern_boolean_raw)

        return bool(
            StringUtils.match(
                series=series,
                regex=combined_regex,
                match_min_rate=0 if match_min_rate is None else match_min_rate,
                chunk_size=chunk_size,
            )
        )
