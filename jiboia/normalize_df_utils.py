from .analysis_utils import AnalysisUtils
from .normalize_date_utils import NormalizeDateUtils
from .normalize_null_utils import NormalizeNullUtils
from .normalize_int_utils import NormalizeIntUtils
from .normalize_float_utils import NormalizeFloatUtils
from .normalize_string_utils import NormalizeStringUtils
from typing import Optional
import pandas as pd


class NormalizeDfUtils:

    @staticmethod
    def normalize_df(
        current_df: pd.DataFrame,
        null_values: list[any] = [],
        remove_accents: bool = True,
        keep_alphanum_space: bool = False,
        exclude_zero: bool = False,
        exclude_negative: bool = False
    ) -> None:
        columns: list[str] = current_df.columns.tolist()
        print(f"column names: {columns}")

        # Normaliza todas as strigs do current_df
        NormalizeStringUtils.normalize_string_by_columns(
            current_df=current_df,
            columns=columns,
            remove_accents=remove_accents,
            keep_alphanum_space=keep_alphanum_space
        )
        print("normalized: string")

        # Normaliza todas as representaçoes de valores ausentes convertendo em nan
        NormalizeNullUtils.normalize_null_by_columns(
            current_df=current_df,
            columns=columns,
            null_values=null_values
        )
        print("normalized: null")

        current_df_types: pd.DataFrame = AnalysisUtils.infer_column_types(current_df)
        print(f"column types: {current_df_types}")

        columns_int: list[str] = (
            current_df_types[current_df_types['type'] == 'int']['column_name'].tolist()
        )

        columns_float: list[str] = (
            current_df_types[current_df_types['type'] == 'float']['column_name'].tolist()
        )

        columns_time: list[str] = (
            current_df_types[current_df_types['type'] == 'time']['column_name'].tolist()
        )

        columns_date: list[str] = (
            current_df_types[current_df_types['type'] == 'date']['column_name'].tolist()
        )

        columns_bool: list[str] = (
            current_df_types[current_df_types['type'] == 'bool']['column_name'].tolist()
        )

        columns_datetime: list[str] = (
            current_df_types[current_df_types['type'] == 'datetime']['column_name'].tolist()
        )

        columns_none: list[str] = (
            current_df_types[current_df_types['type'] == 'None']['column_name'].tolist()
        )
        print(columns_none)

        NormalizeIntUtils.normalize_int_by_columns(
            current_df=current_df,
            columns=columns_int,
            exclude_zero=exclude_zero,
            exclude_negative=exclude_zero
        )
        print("normalized: int")

        NormalizeFloatUtils.normalize_float_by_columns(
            current_df=current_df,
            columns=columns_float,
            exclude_zero=exclude_zero,
            exclude_negative=exclude_zero
        )
        print("normalized: float")

        NormalizeDateUtils.normalize_str_date_to_date_standard_str_by_columns(
            current_df=current_df,
            columns=columns_date
        )
        print("normalized: date")

        NormalizeDateUtils.normalize_str_time_to_hh_mm_ss_by_columns(
            current_df=current_df,
            columns=columns_time
        )
        print("normalized: time")

        for null_column in columns_none:
            current_df.drop(null_column, axis=1, inplace=True)
        print("deleted columns: NoneType")

        # for nome_coluna in columns_str:
        #     df[nome_coluna] = df[nome_coluna].astype('string')

        print("normalized: finish")

