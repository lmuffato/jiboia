from typing import Callable, Literal

import pandas as pd

from ..boolean.regex_pattern import (
    regex_pattern_boolean_raw,
    regex_pattern_boolean_numeric_raw,
)
from ..utils.chunk_utils import chunk_iterate
from ..datetime.regex_pattern import (
    regex_pattern_date,
    regex_pattern_datetime_all,
)
from ..utils.log_utils import (
    print_normalize_space_log,
    print_normalize_string_log,
    print_to_category_log,
)
from ..number.regex_pattern import (
    regex_pattern_bad_formatted_number,
    regex_pattern_valid_number,
)
from ..time.regex_pattern import (
    regex_pattern_time_utc,
    regex_pattern_time_hh_mm,
    regex_pattern_time_hh_mm_ss,
    regex_pattern_time_hh_mm_ss_n,
)
from ..utils.str_utils import combine_regex
from ..utils.validation_utils import (
    PandasSupportedDtypes,
    is_valid_to_normalize,
)


# Tabela de remoção de acentos (equivale aos .str.replace em cadeia do original)
_ASCII_TABLE: dict[int, str] = str.maketrans(
    {
        **{c: "a" for c in "áàâãä"}, **{c: "A" for c in "ÁÀÂÃÄ"},
        **{c: "e" for c in "éèêë"},  **{c: "E" for c in "ÉÈÊË"},
        **{c: "i" for c in "íìîï"},  **{c: "I" for c in "ÍÌÎÏ"},
        **{c: "o" for c in "óòôõö"}, **{c: "O" for c in "ÓÒÔÕÖ"},
        **{c: "u" for c in "úùûü"},  **{c: "U" for c in "ÚÙÛÜ"},
        "ç": "c", "Ç": "C",
    }
)


def _match_mask(series: pd.Series, regex: str) -> pd.Series:
    """str.match com nulos tratados como False (no pandas o padrão seria NaN)."""
    return series.str.match(regex, na=False)


class StringUtils:
    @staticmethod
    def _apply_chunks(
        dataframe: pd.DataFrame,
        column_name: str,
        func: Callable[[pd.Series], pd.Series],
        chunk_size: int,
    ) -> None:
        """Aplica `func` à coluna em chunks e grava o resultado inplace."""
        parts: list[pd.Series] = [
            func(chunk) for chunk in chunk_iterate(dataframe[column_name], chunk_size)
        ]
        if not parts:
            return
        result: pd.Series = pd.concat(parts)
        result.index = dataframe.index  # atribuição por posição
        dataframe[column_name] = result

    @staticmethod
    def normalize(
        dataframe: pd.DataFrame,
        column_name: str,
        to_case: None | Literal["lower", "upper"] = None,
        to_ASCII: bool = False,
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

        StringUtils.normalize_spaces(
            dataframe=dataframe,
            column_name=column_name,
            inplace=True,
            chunk_size=chunk_size,
            show_log=show_log,
        )

        StringUtils.normalize_str(
            dataframe=dataframe,
            column_name=column_name,
            to_case=to_case,
            to_ASCII=to_ASCII,
            inplace=True,
            chunk_size=chunk_size,
            show_log=show_log,
        )

        if not inplace:
            return dataframe
        return True

    @staticmethod
    def normalize_spaces(
        dataframe: pd.DataFrame,
        column_name: str,
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

        StringUtils._apply_chunks(
            dataframe,
            column_name,
            # colapsa espaços múltiplos e remove espaços no início/fim
            lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip(),
            chunk_size,
        )

        print_normalize_space_log(column_name=column_name, show_log=show_log)

        if not inplace:
            return dataframe
        return True

    @staticmethod
    def normalize_str(
        dataframe: pd.DataFrame,
        column_name: str | None = None,
        to_case: None | Literal["lower", "upper"] = None,
        to_ASCII: bool = False,
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

        if not to_ASCII:
            if to_case == "lower":
                dataframe[column_name] = dataframe[column_name].str.lower()
            elif to_case == "upper":
                dataframe[column_name] = dataframe[column_name].str.upper()

            if not inplace:
                return dataframe
            return True

        def _transform(chunk: pd.Series) -> pd.Series:
            if to_case == "lower":
                chunk = chunk.str.lower()
            elif to_case == "upper":
                chunk = chunk.str.upper()
            return chunk.str.translate(_ASCII_TABLE)

        StringUtils._apply_chunks(dataframe, column_name, _transform, chunk_size)

        print_normalize_string_log(
            column_name=column_name,
            show_log=show_log,
            to_case=to_case,
            to_ASCII=to_ASCII,
        )

        if not inplace:
            return dataframe
        return True

    @staticmethod
    def to_category(
        dataframe: pd.DataFrame,
        column_name: str | None = None,
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

        is_str: bool = StringUtils.is_str(
            series=dataframe[column_name],
            chunk_size=chunk_size,
        )
        if not is_str:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        # Categorias do pandas não aceitam nulo, então ele fica fora da lista
        unique_values: pd.Series = dataframe[column_name].dropna().drop_duplicates()
        non_null_values: int = int(dataframe[column_name].notna().sum())

        # Os valores únicos precisam ser até 50% dos dados válidos
        can_be_category: bool = (non_null_values // 2) >= len(unique_values)
        if not can_be_category:
            return False

        categorical_dtype = pd.CategoricalDtype(
            categories=unique_values.sort_values().tolist(),
            ordered=True,
        )
        dataframe[column_name] = dataframe[column_name].astype(categorical_dtype)

        print_to_category_log(column_name=column_name, show_log=show_log)

        if not inplace:
            return dataframe
        return True

    @staticmethod
    def is_str(
        series: pd.Series,
        chunk_size: int = 500_000,
    ) -> bool:
        is_valid: bool = is_valid_to_normalize(
            series=series,
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        patterns: list[str] = [
            combine_regex(regex_pattern_boolean_raw + regex_pattern_boolean_numeric_raw),
            combine_regex(regex_pattern_date + regex_pattern_datetime_all),
            combine_regex(regex_pattern_bad_formatted_number + regex_pattern_valid_number),
            combine_regex(
                regex_pattern_time_utc
                + regex_pattern_time_hh_mm
                + regex_pattern_time_hh_mm_ss
                + regex_pattern_time_hh_mm_ss_n
            ),
        ]

        # Se casar com qualquer padrão (bool, data, número, hora), não é texto puro
        for pattern in patterns:
            if StringUtils.match(series=series, regex=pattern, chunk_size=chunk_size):
                return False

        return True

    @staticmethod
    def normalize_unique_values(
        dataframe: pd.DataFrame,
        column_name: str,
        mapping_dict: dict[str, list[str]],
        null_values: list[str] | None = None,
        inplace: bool = False,
    ) -> bool | pd.DataFrame:
        """
        Normaliza e limpa uma coluna de um DataFrame.

        Args:
            dataframe: DataFrame a ser modificado.
            column_name: Coluna a ser normalizada.
            mapping_dict: Dicionário no formato {'novo_valor': ['antigo1', 'antigo2']}.
            null_values: Lista opcional de valores a serem convertidos para nulo.
            inplace: Se True, altera o original e retorna True.
        """
        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        replace_map: dict[str, str] = {
            old_value: new_value
            for new_value, old_values_list in mapping_dict.items()
            for old_value in old_values_list
        }

        # Substitui os valores antigos pela chave do dict
        dataframe[column_name] = dataframe[column_name].replace(replace_map)

        # Converte valores em nulo (mask evita o replace(lista, None) do pandas)
        if null_values:
            dataframe[column_name] = dataframe[column_name].mask(
                dataframe[column_name].isin(null_values)
            )

        if not inplace:
            return dataframe
        return True

    # MATCHING FUNCTIONS
    @staticmethod
    def match(
        series: pd.Series,
        regex: str,
        match_min_rate: int = 0,
        chunk_size: int = 500_000,
    ) -> bool:
        """
        Verifica se uma Series de strings satisfaz uma correspondência por regex,
        considerando um percentual mínimo de acerto.

        Parâmetros
        ----------
        series : pd.Series
            Série com os textos a avaliar.
        regex : str
            Expressão regular usada na correspondência.
        match_min_rate : int, padrão=0
            Percentual mínimo (0-100) das linhas **não nulas** que devem casar:

            - 0     -> True se pelo menos **uma** linha casar (equivale a any()).
            - 1-99  -> True se pelo menos `match_min_rate%` das linhas não nulas casarem.
            - 100   -> True apenas se **todas** as linhas não nulas casarem.

            O valor é limitado ao intervalo [0, 100].
        chunk_size : int, padrão=500_000
            Linhas processadas por vez.

        Retorno
        -------
        bool
        """
        match_min_rate = max(0, min(100, int(match_min_rate)))

        total_not_null_rows: int = int(series.notna().sum())

        if match_min_rate == 0:
            for chunk in chunk_iterate(series, chunk_size):
                if _match_mask(chunk, regex).any():
                    return True
            return False

        if match_min_rate < 100:
            match_min: int = (total_not_null_rows * match_min_rate) // 100
            total_match: int = 0

            for chunk in chunk_iterate(series, chunk_size):
                total_match += int(_match_mask(chunk, regex).sum())

                if total_match >= match_min:
                    return True
            return False

        # match_min_rate == 100
        total_match = 0
        for chunk in chunk_iterate(series, chunk_size):
            total_match += int(_match_mask(chunk, regex).sum())

        return total_match == total_not_null_rows

    @staticmethod
    def match_count(
        series: pd.Series,
        pattern: str,
        chunk_size: int = 500_000,
        match_min_rate: int = 0,
    ) -> int:
        """
        Retorna o número de ocorrências de um padrão.
        Se `match_min_rate` (1 a 99) for informado, a busca para ao atingir
        essa porcentagem do total de linhas.
        """
        total_rows: int = len(series)
        total_match: int = 0

        use_limit: bool = 0 < match_min_rate < 100
        match_min: int = (total_rows * match_min_rate) // 100 if use_limit else 0

        for chunk in chunk_iterate(series, chunk_size):
            total_match += int(_match_mask(chunk, pattern).sum())

            if use_limit and total_match >= match_min:
                return total_match

        return total_match

    @staticmethod
    def match_infer(
        series: pd.Series,
        regex_patterns: list[dict[str, str]],
        chunk_size: int = 500_000,
    ) -> list[dict[str, str]]:
        """
        Retorna o número de ocorrências para uma lista de padrões
        (preenche a chave "frequency" de cada item).
        """
        for pattern in regex_patterns:
            pattern["frequency"] = 0

        for chunk in chunk_iterate(series, chunk_size):
            for pattern in regex_patterns:
                pattern["frequency"] += int(_match_mask(chunk, pattern["regex"]).sum())

        return regex_patterns

    @staticmethod
    def sanitize_unique(
        dataframe: pd.DataFrame,
        column_name: str,
        sanitize_values: dict[str, list[str]] | None = None,
        null_values: list[str] | None = None,
        inplace: bool = True,
    ) -> bool | pd.DataFrame:
        sanitize_values = sanitize_values or {}
        null_values = null_values or []

        is_valid: bool = is_valid_to_normalize(
            series=dataframe[column_name],
            valid_types=PandasSupportedDtypes.str_types,
        )
        if not is_valid:
            return False

        if not inplace:
            dataframe = dataframe.copy()

        if null_values:
            dataframe[column_name] = dataframe[column_name].mask(
                dataframe[column_name].isin(null_values)
            )

        for target_value, values_to_replace in sanitize_values.items():
            dataframe[column_name] = dataframe[column_name].replace(
                values_to_replace, target_value
            )

        if not inplace:
            return dataframe
        return True
