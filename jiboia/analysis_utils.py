from csv import Dialect
from datetime import datetime, time
from pandas.api.types import CategoricalDtype
from typing import Literal, Optional
from .regex_patterns import (
    DATE_REGEX_PATTERNS,
    TIME_REGEX_PATTERNS,
    DATE_TIME_REGEX_PATTERNS,
)
import csv
import numpy as np
import pandas as pd
import re


class AnalysisUtils:
    data_types = Literal['int', 'float', 'bool', 'date', 'time', 'datetime', 'str']

    @staticmethod
    def _number_cleaned(value: object) -> str:

        value_str: str = str(value).strip()

        if '.' in value_str and ',' in value_str:
            # formato 1000,00 -> 1000.00
            value_str: str = value_str.replace('.', '').replace(',', '.')

        if ',' in value_str:
            # formato 1.000,00 -> 1000.00
            value_str: str = value_str.replace(',', '.')

        return value_str


    @staticmethod
    def _is_zero(value: object) -> bool:
        cleaned_number_str: str = AnalysisUtils._number_cleaned(value)

        try:
            float_value: float = float(cleaned_number_str)

            if float_value == 0.0:
                return True
            else:
                return False
        except ValueError:
            return False


    @staticmethod
    def _is_int(value: object) -> bool:
        cleaned_number_str: str = AnalysisUtils._number_cleaned(value)

        try:
            float_value: float = float(cleaned_number_str)

            if float_value.is_integer():
                return True
            else:
                return False
        except ValueError:
            return False


    @staticmethod
    def _is_float(value: object) -> bool:
        '''
        Retorna True se o valor puder ser interpretado como float,
        incluindo formatos com separadores de milhar e vírgula como separador decimal.
        '''
        cleaned_number_str: str = AnalysisUtils._number_cleaned(value)

        try:
            float_value: float = float(cleaned_number_str)

            if not float_value.is_integer():
                return True
            else:
                return False
        except ValueError:
            return False


    @staticmethod
    def _is_bool(value: object) -> bool:
        """
        Retorna True se o valor puder ser interpretado como booleano.
        Considera 'true', 'false', '1', '0' (case-insensitive), além de bool nativo.
        """
        value_str = str(value).strip().lower()
        return value_str in {'True', 'False', 'true', 'false', }


    @staticmethod
    def _is_date(value: object) -> bool:
        value_str: str = str(value)

        for pattern, _ in DATE_REGEX_PATTERNS:
            if re.fullmatch(pattern, value_str):
                return True

        return False


    def _is_time(value: object) -> bool:
        value_str = str(value)

        for pattern, _ in TIME_REGEX_PATTERNS:
            if re.fullmatch(pattern, value_str):
                return True

        return False


    @staticmethod
    def _is_datetime(value: object) -> bool:
        """
        Verifica se o valor corresponde a um dos formatos datetime definidos em DATE_TIME_REGEX_PATTERNS.
        """

        value_str = str(value)

        for pattern, date_format in DATE_TIME_REGEX_PATTERNS:
            if re.fullmatch(pattern, value_str):
                try:
                    datetime.strptime(value_str, date_format)
                    return True
                except ValueError:
                    continue

        return False


    @staticmethod
    def _detect_type(value: object) -> Optional[Literal['int', 'float', 'bool', 'date', 'time', 'str', 'zero']]:
        if pd.isna(value):
            return None
        
        elif AnalysisUtils._is_zero(value):
            return 'zero'

        elif AnalysisUtils._is_int(value):
            return 'int'

        elif AnalysisUtils._is_float(value):
            return 'float'
        
        elif AnalysisUtils._is_time(value):
            return 'time'
        
        elif AnalysisUtils._is_date(value):
            return 'date'

        elif AnalysisUtils._is_bool(value):
            return 'bool'
        
        elif AnalysisUtils._is_datetime(value):
            return 'datetime'

        else:
            return 'str'


    @staticmethod
    def get_dominant_type(
        non_null_rows_sample: pd.Series,
        without_zero: bool = False
    ) -> Literal['int', 'float', 'bool', 'date', 'time', 'datetime', 'str', 'None', 'zero']:

        type_frequency: pd.Series = (
            non_null_rows_sample
            .apply(AnalysisUtils._detect_type)
            .value_counts()
        )

        if without_zero and 'zero' in type_frequency:
            type_frequency: pd.Series = type_frequency.drop(labels='zero')

        # Verifica a frequência de "float"
        float_count = type_frequency.get('float', 0)
        total_count = len(non_null_rows_sample)
        
        # TODO: Rever isso para analisar toda a coluna em caso de float e int
        # Se pelo menos 1% dos valores forem float, considera o tipo dominante como float
        if float_count / total_count >= 0.1:
            dominant_type = 'float'
        else:
            # Caso contrário, pega o tipo mais frequente
            dominant_type = type_frequency.idxmax()

        return dominant_type


    @staticmethod
    def sample_size(
        current_df: pd.DataFrame,
        rows_percent: int = 2
    ) -> int:
        if not (0 <= rows_percent <= 100):
            raise ValueError('rows_percent deve estar entre 0 e 100')

        row_count: int = len(current_df)
        sample_size: int = max(1, int((rows_percent / 100) * row_count))

        return sample_size


    @staticmethod
    def infer_column_types(
        current_df: pd.DataFrame,
        rows_percent: int = 2,
    ) -> pd.DataFrame:

        sample_size: int = AnalysisUtils.sample_size(current_df, rows_percent)

        columns: list[str] = current_df.columns.tolist()

        inferred_types: list[str] = []

        for column in columns:

            non_null_rows_sample: pd.Series = (
                current_df[column]
                .dropna()
                .head(sample_size)
            )

            if non_null_rows_sample.empty:
                dominant_type: str = 'None'
            else:
                dominant_type: str = AnalysisUtils.get_dominant_type(non_null_rows_sample)
                
                # Reanalisa as colunas numéricas caso haja dúvida
                if dominant_type == 'zero':
                    full_non_null_column: pd.Series = current_df[column].dropna()
                    dominant_type = AnalysisUtils.get_dominant_type(
                        non_null_rows_sample=full_non_null_column,
                        without_zero=True
                    )

            inferred_types.append(dominant_type)

        df_types: pd.DataFrame = pd.DataFrame({
            'column_name': columns,
            'type': inferred_types
        })
        print(df_types)

        return df_types


    @staticmethod
    def get_sep_delimiter(
        file_path: str,
        encoding: Optional[str] = 'latin1',
        skiprows: Optional[int] = 0
    ) -> str:
        sample_characters: int = 2048

        with open(file_path, 'r', encoding=encoding) as file:
            for _ in range(skiprows):
                next(file)  # pula as linhas iniciais

            sample: str = file.read(sample_characters)
            sniffer: csv.Sniffer = csv.Sniffer()
            dialect: Dialect = sniffer.sniff(sample)

        # with open(file_path, 'r', encoding=encoding) as file:
        #     sample: str = file.read(sample_characters)
        #     sniffer: csv.Sniffer = csv.Sniffer()
        #     dialect: Dialect = sniffer.sniff(sample)
        
        print(f'sep_delimiter: "{dialect.delimiter}"')
        return dialect.delimiter


    @staticmethod
    def get_frequency_by_column(current_df: pd.DataFrame, column: str) -> pd.DataFrame:
        '''
        Gera um current_df com a frequencia de valores de uma coluna
        '''
        current_df_frequency: pd.DataFrame = (
            current_df[column]
            .value_counts()
            .reset_index()
        )
    
        current_df_frequency.columns = [column, 'frequency']
    
        current_df_frequency: pd.DataFrame = (
            current_df_frequency
            .sort_values(
                by='frequency',
                ascending=False
            )
        )
    
        return current_df_frequency


    @staticmethod
    def get_unique_values(current_df: pd.DataFrame, column: str) -> pd.DataFrame:
        '''
        Retorna um dataframe com os valores únicos de uma coluna
        '''
        df_unique = pd.DataFrame({
            column: current_df[column].dropna().unique()
        })

        df_unique = df_unique.sort_values(
            by=column,
            ascending=True
        ).reset_index(drop=True)

        return df_unique
    
    @staticmethod
    def df_size_info(current_df: pd.DataFrame) -> None:
        return {
            "rows": current_df.shape[0],
            "columns": current_df.shape[1],
            "RAM MB": round(current_df.memory_usage(index=True, deep=True).sum() / (1024 * 1024), 2)
        }