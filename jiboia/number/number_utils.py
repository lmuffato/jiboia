from typing import Literal

import numpy as np
import pandas as pd

from ..utils.log_utils import print_log
from ..string.string_utils import StringUtils
from ..utils.str_utils import combine_regex
from ..utils.validation_utils import (
    PandasSupportedDtypes,
    is_valid_to_normalize,
)
from .regex_pattern import (
    regex_pattern_bad_formatted_number,
    regex_pattern_valid_number,
    regex_pattern_list,
)


_SIGNED_INT_TYPES: list[str] = ["int8", "int16", "int32", "int64"]


def _downcast_integer(col: pd.Series) -> pd.Series:
    """
    Converte para o menor inteiro com sinal que comporta os valores.

    Se houver nulos, usa os dtypes nullable do pandas (Int8, Int16, ...), pois os
    dtypes numpy (int8, ...) não suportam nulo. Sem nulos, usa os dtypes numpy.
    """
    non_null: pd.Series = col.dropna()
    if non_null.empty:
        return col

    min_value = non_null.min()
    max_value = non_null.max()
    has_nulls: bool = bool(col.isna().any())

    for type_name in _SIGNED_INT_TYPES:
        info = np.iinfo(type_name)
        if info.min <= min_value and max_value <= info.max:
            target: str = type_name.capitalize() if has_nulls else type_name
            return col.astype(target)

    return col  # não cabe em int64 (ex.: uint64 grande): mantém o dtype


class NumberUtils:
    @staticmethod
    def normalize(
        dataframe: pd.DataFrame,
        column_name: str,
        match_min_rate: int | None = 50,
        inplace: bool = False,
        chunk_size: int = 500_000,
        show_log: bool = True,
    ) -> bool | pd.DataFrame:
        """
        Converte uma coluna de um DataFrame pandas para o tipo numérico mais apropriado.

        - Se a coluna for string/object, tenta convertê-la para numérica.
        - Se a coluna for float, verifica se pode ser inteira sem perda de dados.
        - Faz downcast para o menor tipo inteiro possível.

        Args:
            dataframe: DataFrame a ser modificado.
            column_name: Nome da coluna a ser normalizada.
            match_min_rate: Percentual mínimo (0-100) de valores não nulos que devem
                virar número para que uma coluna de string seja convertida.
            inplace: Se True, modifica o original e retorna True. Se False, retorna uma cópia.
            chunk_size: Tamanho dos chunks usados nas verificações de regex.
            show_log: Se True, imprime a coluna convertida e o tipo final.

        Returns:
            True (inplace), o DataFrame convertido (não inplace) ou False se a
            coluna não for elegível / não atingir `match_min_rate`.
        """
        match_min_rate = 0 if match_min_rate is None else match_min_rate

        valid_types: list[str] = (
            PandasSupportedDtypes.str_types + PandasSupportedDtypes.numeric_types
        )

        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=valid_types,
        )
        if not is_valid:
            return False

        original_dtype: str = str(dataframe[column_name].dtype)
        is_str_column: bool = original_dtype in PandasSupportedDtypes.str_types

        is_number_in_str: bool = is_str_column and NumberUtils.is_number_in_str(
            series=dataframe[column_name],
            match_min_rate=match_min_rate,
            chunk_size=chunk_size,
        )
        is_number: bool = is_number_in_str or (
            original_dtype in PandasSupportedDtypes.numeric_types
        )
        if not is_number:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        if is_str_column:
            NumberUtils.fix_decimal(
                dataframe=dataframe,
                column_name=column_name,
                chunk_size=chunk_size,
                inplace=True,
            )

        col: pd.Series = dataframe[column_name]

        if is_str_column:
            numeric_col: pd.Series = pd.to_numeric(col, errors="coerce")
            non_null_before: int = int(col.notna().sum())
            non_null_after: int = int(numeric_col.notna().sum())

            if non_null_before == 0:
                return False

            if round((non_null_after / non_null_before) * 100) < match_min_rate:
                return False

            col = numeric_col

        if pd.api.types.is_float_dtype(col.dtype):
            non_null: pd.Series = col.dropna().astype("float64")
            is_integer_column: bool = bool(
                np.isfinite(non_null).all() and (non_null == non_null.round(0)).all()
            )

            if is_integer_column:
                dataframe[column_name] = _downcast_integer(col)
            else:
                # Sem downcast para float32: perderia precisão
                dataframe[column_name] = col

            print_log(
                column_name=column_name,
                column_type=str(dataframe[column_name].dtype),
                show_log=show_log,
            )

        elif pd.api.types.is_integer_dtype(col.dtype):
            dataframe[column_name] = _downcast_integer(col)
            print_log(
                column_name=column_name,
                column_type=str(dataframe[column_name].dtype),
                show_log=show_log,
            )

        if not inplace:
            return dataframe

        return True

    @staticmethod
    def fix_decimal(
        dataframe: pd.DataFrame,
        column_name: str,
        chunk_size: int = 500_000,
        inplace: bool = False,
    ) -> bool | pd.DataFrame:
        """
        Converte strings numéricas com separador de milhar e decimal
        ('1.234,56', '0,99', ...) para um formato compatível com float
        ('1234.56', '0.99'), processando em chunks.

        - '.' é o separador de milhar
        - ',' é o separador decimal

        Parâmetros
        ----------
        dataframe : pd.DataFrame
        column_name : str
        chunk_size : int, default=500_000
            Número máximo de linhas processadas por vez.
        inplace : bool, default=False
            Se True, altera a coluna do DataFrame original e retorna True.
        """
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        pattern: str = combine_regex(regex_pattern_bad_formatted_number)

        has_bad_formatted_number: bool = StringUtils.match(
            series=dataframe[column_name],
            regex=pattern,
            match_min_rate=0,
            chunk_size=chunk_size,
        )
        if not has_bad_formatted_number:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        def _fix(chunk: pd.Series) -> pd.Series:
            mask: pd.Series = chunk.str.match(pattern, na=False)
            fixed: pd.Series = (
                chunk.str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            return chunk.where(~mask, fixed)

        StringUtils._apply_chunks(dataframe, column_name, _fix, chunk_size)

        if not inplace:
            return dataframe

        return True

    @staticmethod
    def is_number_in_str(
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
            regex_pattern_valid_number + regex_pattern_bad_formatted_number
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
    def has_list(
        series: pd.Series,
        chunk_size: int = 500_000,
    ) -> bool:
        is_valid: bool = is_valid_to_normalize(
            series=series,
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        combined_regex: str = combine_regex(regex_pattern_list)

        return bool(
            StringUtils.match(
                series=series,
                regex=combined_regex,
                match_min_rate=0,
                chunk_size=chunk_size,
            )
        )
