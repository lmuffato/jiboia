from ..utils.log_utils import print_normalize_type_log
from ..utils.validation_utils import (
    PandasSupportedDtypes,
    is_valid_to_normalize,
)
from ..string.string_utils import StringUtils
from .regex_pattern import RAW_INVALID_LOWERCASE_VALUES

import pandas as pd


class NullUtils:
    @staticmethod
    def normalize(
        dataframe: pd.DataFrame,
        column_name: str,
        null_values: list[str] | None = None,
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

        null_values = null_values or []

        all_null_values: set[str] = {v.lower() for v in null_values} | set(
            RAW_INVALID_LOWERCASE_VALUES
        )

        if not inplace:
            dataframe = dataframe.copy()

        def _nullify(chunk: pd.Series) -> pd.Series:
            # nulos viram NaN no str.lower() e isin() devolve False para eles
            mask: pd.Series = chunk.str.lower().isin(all_null_values)
            return chunk.mask(mask)

        StringUtils._apply_chunks(dataframe, column_name, _nullify, chunk_size)

        print_normalize_type_log(
            column_name=column_name,
            value_original="null",
            value_final="<NA>",
            show_log=show_log,
        )

        if not inplace:
            return dataframe

        return True

    @staticmethod
    def get_default_nulls() -> list[str]:
        # Devolve uma cópia para que quem chama não altere a lista do módulo
        return list(RAW_INVALID_LOWERCASE_VALUES)
