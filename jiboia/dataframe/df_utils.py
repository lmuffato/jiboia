from typing import Literal

import pandas as pd

from ..boolean.boolean_utils import BooleanUtils
from ..null.null_utils import NullUtils
from ..number.number_utils import NumberUtils
from ..string.string_utils import StringUtils
from ..time.time_utils import TimeUtils
from ..datetime.datetime_utils import DateTimeUtils
from ..utils.log_utils import (
    print_drop_column_log,
    print_text_green,
    print_text_yellow,
    print_normalize_df_space_log,
    print_normalize_df_string_log,
)


class DfUtils:
    @staticmethod
    def normalize(
        dataframe: pd.DataFrame,
        match_min_rate: int = 50,
        null_values: list[str] | None = None,
        to_case: None | Literal["lower", "upper"] = None,
        to_ASCII: bool = False,
        bool_number: bool = False,
        create_category: bool = False,
        drop_columns: list[str] | None = None,
        inplace: bool = False,
        show_log: bool = True,
        chunk_size: int = 500_000,
    ) -> bool | pd.DataFrame:
        """
        Normaliza todas as colunas de um DataFrame pandas aplicando múltiplas
        transformações automáticas em valores de string, nulos, numéricos, booleanos,
        temporais e categóricos.

        Operações aplicadas por coluna, nesta ordem:
        1. StringUtils.normalize -> espaços múltiplos e strings (case e ASCII).
        2. NullUtils.normalize -> padronização e substituição de valores nulos.
        3. NumberUtils.normalize -> strings numéricas para o menor tipo numérico possível.
        4. BooleanUtils.normalize -> strings booleanas para tipo boolean.
        5. TimeUtils.normalize -> parsing de valores de tempo (HH:MM:SS).
        6. DateTimeUtils.normalize -> parsing de valores datetime.
        7. StringUtils.to_category (opcional) -> conversão da coluna em categoria.

        Parâmetros
        ----------
        dataframe : pd.DataFrame
            DataFrame de entrada a ser normalizado.
        match_min_rate : int, default=50
            Percentual mínimo de correspondência para converter a coluna (0-100).
        null_values : list[str] | None
            Valores considerados nulos além dos padrões (ex.: ["NA", "null"]).
        to_case : {'lower', 'upper'} | None
            Converte textos para minúsculo ou maiúsculo.
        to_ASCII : bool, default=False
            Converte caracteres acentuados para ASCII puro.
        create_category : bool, default=False
            Converte colunas de texto em categorias.
        drop_columns : list[str] | None
            Colunas a remover antes da normalização.
        inplace : bool, default=False
            Se True, altera o DataFrame original e retorna True.
        show_log : bool, default=True
            Imprime logs de cada etapa.
        chunk_size : int, default=500_000
            Tamanho dos chunks para processar séries grandes.

        Retorna
        -------
        bool | pd.DataFrame
            True se `inplace=True`; caso contrário, um novo DataFrame normalizado.
        """
        null_values = null_values or []
        drop_columns = drop_columns or []

        if not inplace:
            dataframe = dataframe.copy()

        if drop_columns:
            DfUtils.drop_columns(
                dataframe=dataframe,
                drop_columns=drop_columns,
                inplace=True,
                show_log=show_log,
            )

        column_names: list[str] = list(dataframe.columns)

        for column_name in column_names:
            StringUtils.normalize(
                dataframe=dataframe,
                column_name=column_name,
                to_case=to_case,
                to_ASCII=to_ASCII,
                inplace=True,
                show_log=False,
            )

            NullUtils.normalize(
                dataframe=dataframe,
                column_name=column_name,
                null_values=null_values,
                inplace=True,
                show_log=False,
                chunk_size=chunk_size,
            )

        print_normalize_df_space_log(show_log=show_log)
        print_normalize_df_string_log(
            to_case=to_case,
            to_ASCII=to_ASCII,
            show_log=show_log,
        )

        for column_name in column_names:
            NumberUtils.normalize(
                dataframe=dataframe,
                column_name=column_name,
                match_min_rate=match_min_rate,
                inplace=True,
                chunk_size=chunk_size,
                show_log=show_log,
            )

            BooleanUtils.normalize(
                dataframe=dataframe,
                bool_number=bool_number,
                column_name=column_name,
                match_min_rate=match_min_rate,
                inplace=True,
                show_log=show_log,
            )

            TimeUtils.normalize(
                dataframe=dataframe,
                column_name=column_name,
                match_min_rate=match_min_rate,
                inplace=True,
                show_log=show_log,
            )

            DateTimeUtils.normalize(
                dataframe=dataframe,
                column_name=column_name,
                match_min_rate=match_min_rate,
                inplace=True,
                chunk_size=chunk_size,
                show_log=show_log,
            )

            if create_category:
                StringUtils.to_category(
                    dataframe=dataframe,
                    column_name=column_name,
                    inplace=True,
                    chunk_size=chunk_size,
                    show_log=show_log,
                )

        if not inplace:
            return dataframe

        return True

    @staticmethod
    def drop_columns(
        dataframe: pd.DataFrame,
        drop_columns: list[str],
        inplace: bool = False,
        show_log: bool = True,
    ) -> bool | pd.DataFrame:
        if not inplace:
            dataframe = dataframe.copy()

        columns_to_delete: list[str] = [
            column_name for column_name in drop_columns if column_name in dataframe.columns
        ]

        if columns_to_delete:
            dataframe.drop(columns=columns_to_delete, inplace=True)
            print_drop_column_log(
                show_log=show_log,
                columns_to_delete=columns_to_delete,
            )

        if not inplace:
            return dataframe

        return True

    @staticmethod
    def df_size_info(dataframe: pd.DataFrame, print_info: bool = False) -> dict[str, float | int]:
        rows: int = dataframe.shape[0]
        columns: int = dataframe.shape[1]
        ram_size_mb: float = round(
            dataframe.memory_usage(index=True, deep=True).sum() / (1024 * 1024), 2
        )

        df_info: dict[str, float | int] = {
            "rows": rows,
            "columns": columns,
            "RAM size Mb": ram_size_mb,
        }

        if print_info:
            print(
                print_text_green("Done!"),
                "rows:",
                print_text_yellow(rows),
                "columns:",
                print_text_yellow(columns),
                "RAM size Mb:",
                print_text_yellow(ram_size_mb),
            )

        return df_info

    @staticmethod
    def frequency(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Retorna um DataFrame com a frequência de cada valor único de uma coluna,
        ordenado do maior para o menor.

        Args:
            dataframe (pd.DataFrame): DataFrame de entrada.
            column_name (str): Nome da coluna para calcular a frequência.

        Returns:
            pd.DataFrame: Colunas [column_name, 'frequency'].
        """
        df_frequency = dataframe[column_name].value_counts().reset_index()
        df_frequency.columns = [column_name, "frequency"]

        return df_frequency
