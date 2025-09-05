from datetime import datetime, time, date
from pandas.api.types import CategoricalDtype
from typing import Optional
from .regex_patterns import (
    DATE_REGEX_PATTERNS,
    TIME_REGEX_PATTERNS,
    DATE_TIME_REGEX_PATTERNS,
)
import numpy as np
import pandas as pd
import re


class NormalizeDateUtils:
    # Converter um valor de data string em datetime
    @staticmethod
    def _convert_date_string_to_datetime(date_str: str) -> Optional[datetime]:
        '''
        Converte a data de um dataframe em um único formato datetime
        '''

        for pattern, date_format in DATE_REGEX_PATTERNS:
            if re.match(pattern, date_str):
                try:
                    return datetime.strptime(date_str, date_format)
                except ValueError:
                    continue

        return None
    

    # Converter uma data string para string no formato 'yyyy-mm-dd'
    @staticmethod
    def _convert_str_date_to_date_standard_str(date_str: str) -> Optional[str]:
        '''
        Converte uma data em diferentes formatos para uma string no formato yyyy-mm-dd.
        Retorna None se não conseguir converter.
        '''
        for pattern, date_format in DATE_REGEX_PATTERNS:
            if re.match(pattern, date_str):
                try:
                    dt = datetime.strptime(date_str, date_format)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue

        return None


    # Converter todas as linhas de um current_df em datetime
    @staticmethod
    def normalize_str_date_to_date_standard_str_by_columns(
        current_df: pd.DataFrame,
        columns: list[str]
    ) -> None:
        '''
        Converte todas as linhas de uma coluna de data datetime
        '''

        for column in columns:
            if column not in current_df.columns:
                continue
            current_df[column] = (
                current_df[column]
                .apply(lambda row: NormalizeDateUtils._convert_str_date_to_date_standard_str(row))
            )


    # Converter todas as linhas de um current_df em datetime
    @staticmethod
    def normalize_str_date_to_datetime_by_columns(
        current_df: pd.DataFrame,
        columns: list[str]
    ) -> None:
        '''
        Converte todas as linhas de uma coluna de data datetime
        '''

        for column in columns:
            if column not in current_df.columns:
                continue
            current_df[column] = (
                current_df[column]
                .apply(lambda row: NormalizeDateUtils._convert_date_string_to_datetime(row))
            )


    @staticmethod
    def create_column_datetime_from_date(
        current_df: pd.DataFrame,
        date_column: str = 'DATE',
        new_column: str = 'DATETIME',
        before_column: Optional[str] = None
    ) -> None:
        '''
        Cria uma nova coluna `DATETIME` a partir de uma coluna de data
        em formato string "yyyy-mm-dd" ou datetime.date/datetime64.

        Se `before_column` for fornecida e existir no DataFrame,
        a nova coluna será inserida antes dela.
        '''

        def parse_date(row: pd.Series) -> Optional[pd.Timestamp]:
            date_val = row.get(date_column)

            if pd.isna(date_val):
                return np.nan

            try:
                if isinstance(date_val, str):
                    return datetime.strptime(date_val, '%Y-%m-%d')
                elif isinstance(date_val, (datetime, pd.Timestamp)):
                    return pd.to_datetime(date_val)
                elif isinstance(date_val, date):
                    return datetime.combine(date_val, time(0, 0))
                else:
                    return np.nan
            except Exception:
                return np.nan

        result_series: pd.Series = current_df.apply(parse_date, axis=1)

        if before_column and before_column in current_df.columns:
            insert_position: int = current_df.columns.get_loc(before_column)
            current_df.insert(loc=insert_position, column=new_column, value=result_series)
        else:
            current_df[new_column] = result_series



    # TODO: Ver melhor isso
    @staticmethod
    def create_column_datetime_from_date_and_time(
        current_df: pd.DataFrame,
        date_column: str = 'DATE',
        time_column: str = 'TIME',
        new_column: str = 'DATETIME',
        before_column: Optional[str] = None
    ) -> None:
        '''
        Cria uma nova coluna datetime combinando:
        - Uma coluna de data (no formato "yyyy-mm-dd" como string ou datetime)
        - Uma coluna de hora (no formato "HH:MM:SS")

        Se `before_column` for fornecida e existir no DataFrame,
        a nova coluna será inserida antes dela.
        '''

        def combine(row: pd.Series) -> pd.Timestamp:
            date_val = row.get(date_column)
            time_val = row.get(time_column)

            if pd.isna(date_val) or pd.isna(time_val):
                return np.nan

            try:
                if isinstance(date_val, str):
                    date_val = datetime.strptime(date_val, '%Y-%m-%d').date()

                if isinstance(time_val, str):
                    time_parts = [int(part) for part in time_val.strip().split(':')]
                    while len(time_parts) < 3:
                        time_parts.append(0)
                    hour_val = time(*time_parts[:3])
                else:
                    return np.nan

                return datetime.combine(date_val, hour_val)
            except Exception:
                return np.nan

        result_series = current_df.apply(combine, axis=1)

        if before_column and before_column in current_df.columns:
            insert_position = current_df.columns.get_loc(before_column)
            current_df.insert(insert_position, new_column, result_series)
        else:
            current_df[new_column] = result_series



    @staticmethod
    def _normalize_str_time_to_hh_mm_ss(time_str: str) -> Optional[str]:
        '''
        Normaliza uma string de hora para o formato hh:mm:00
        '''
        if not isinstance(time_str, str) or not time_str.strip():
            return np.nan

        time_str = time_str.strip()

        try:
            if re.match(r'^\d{1,2}$', time_str):
                return f'{int(time_str):02d}:00:00'

            if re.match(r'^\d{1,2}:\d{2}$', time_str):
                dt = datetime.strptime(time_str, '%H:%M')
                return dt.strftime('%H:%M:00')

            if re.match(r'^\d{1,2}:\d{2}:\d{2}(\.\d+)?$', time_str):
                dt = datetime.strptime(time_str.split('.')[0], '%H:%M:%S')
                return dt.strftime('%H:%M:%S')
            
            if re.match(r'^\d{4}\sUTC$', time_str):
                dt = datetime.strptime(time_str, '%H%M UTC')
                return dt.strftime('%H:%M:00')
        except ValueError:
            pass

        return np.nan


    @staticmethod
    def normalize_str_time_to_hh_mm_ss_by_columns(
        current_df: pd.DataFrame,
        columns: list[str]
    ) -> None:
        '''
        Aplica a normalização do formato de hora hh:mm:00 em múltiplas colunas do DataFrame.
        Substitui valores inválidos por NaN.
        '''
        for column in columns:
            if column in current_df.columns:
                current_df[column] = current_df[column].apply(
                    NormalizeDateUtils._normalize_str_time_to_hh_mm_ss
                )


    # @staticmethod
    # def create_column_hour_string_by_column(
    #     current_df: pd.DataFrame,
    #     column: Optional[str] = 'DATE',
    #     before_column: Optional[str] = None
    # ) -> None:
    #     '''
    #     Cria uma coluna `HORA` no formato HH:00 a partir de uma coluna datetime.
    #     Se `before_column` for fornecida e existir no DataFrame, a nova coluna será inserida antes dela.
    #     '''

    #     hora_series = current_df[column].dt.hour.apply(lambda h: f'{h:02d}:00')

    #     if before_column and before_column in current_df.columns:
    #         insert_position = current_df.columns.get_loc(before_column)
    #         current_df.insert(insert_position, 'HORA', hora_series)
    #     else:
    #         current_df['HORA'] = hora_series


    def create_column_hora(
        current_df: pd.DataFrame,
        datetime_column: Optional[str] = 'DATETIME',
        before_column: Optional[str] = None
    ) -> None:
        '''
        Cria uma nova coluna HORA no formato "HH:00" (ex: 00:00, 01:00, ..., 23:00)
        a partir de uma coluna datetime. A coluna é transformada em categoria ordenada.
        '''

        horas_ordem: list[str] = [f'{h:02d}:00' for h in range(24)]
        categoria_horas = CategoricalDtype(categories=horas_ordem, ordered=True)

        result_series = current_df[datetime_column].dt.hour.map(lambda h: f'{h:02d}:00')

        if before_column and before_column in current_df.columns:
            insert_position = current_df.columns.get_loc(before_column)
            current_df.insert(insert_position, 'HORA', result_series.astype(categoria_horas))
        else:
            current_df['HORA'] = result_series.astype(categoria_horas)


    def create_column_ano(
        current_df: pd.DataFrame,
        column: Optional[str]
    ) -> None:
        '''
        Cria uma coluna `ANO` com o ano extraído de uma coluna datetime
        '''
        if not column:
            column = 'DATE'

        current_df['ANO'] = current_df[column].dt.year


    @staticmethod
    def create_column_mes_int(current_df: pd.DataFrame, column: Optional[str]) -> None:
        '''
        Cria uma coluna `MES_INT` com o mês extraído de uma coluna datetime.
        '''
        if not column:
            column = 'DATE'

        current_df['MES_INT'] = current_df[column].dt.month

    
    @staticmethod
    def create_column_dia_semana(
        current_df: pd.DataFrame,
        datetime_column: Optional[str] = 'DATETIME',
        before_column: Optional[str] = None
    ) -> None:
        '''
        Cria uma nova coluna DIA_SEMANA com o nome do dia da semana a partir de uma coluna datetime,
        e a transforma em categoria ordenada (SEGUNDA a domingo).
        '''
        ordem_dias: list[str] = [
            'SEGUNDA', 'TERÇA', 'QUARTA', 'QUINTA',
            'SEXTA', 'SABADO', 'DOMINGO'
        ]
        categoria_dias = CategoricalDtype(categories=ordem_dias, ordered=True)

        result_series = current_df[datetime_column].dt.dayofweek.map(lambda i: ordem_dias[i])

        if before_column and before_column in current_df.columns:
            insert_position = current_df.columns.get_loc(before_column)
            current_df.insert(insert_position, 'DIA_SEMANA', result_series.astype(categoria_dias))
        else:
            current_df['DIA_SEMANA'] = result_series.astype(categoria_dias)


    @staticmethod
    def create_column_mes(
        current_df: pd.DataFrame,
        datetime_column: Optional[str] = 'DATETIME',
        before_column: Optional[str] = None
    ) -> None:
        '''
        Cria uma nova coluna MES com o nome do dia da semana a partir de uma coluna datetime,
        e a transforma em categoria ordenada (SEGUNDA a domingo).
        '''
        if not datetime_column:
            datetime_column: str = 'DATE'

        meses_ordem: list[str] = [
            'JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN',
            'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ'
        ]

        categoria_dias = CategoricalDtype(categories=meses_ordem, ordered=True)

        result_series = current_df[datetime_column].dt.month.map(lambda i: meses_ordem[i - 1])

        if before_column and before_column in current_df.columns:
            insert_position = current_df.columns.get_loc(before_column)
            current_df.insert(insert_position, 'MES', result_series.astype(categoria_dias))
        else:
            current_df['MES'] = result_series.astype(categoria_dias)


    # @staticmethod
    # def create_column_hour_category(
    #     current_df: pd.DataFrame,
    #     time_column: str,
    #     before_column: Optional[str] = None
    # ) -> None:
    #     '''
    #     Cria uma nova coluna "HORA" com a hora extraída (00 a 23) da coluna time_column,
    #     no formato string pyarrow categorizada e ordenada.
    #     '''

    #     # Validação básica
    #     if time_column not in current_df.columns:
    #         raise ValueError(f"Coluna '{time_column}' não encontrada no DataFrame")

    #     # Lista de categorias string zero-padded "00" a "23"
    #     horas_ordem: list[str] = [f"{h:02d}" for h in range(24)]

    #     # Define categoria ordenada
    #     categoria_horas: CategoricalDtype = CategoricalDtype(categories=horas_ordem, ordered=True)

    #     # Extrai substring hh da coluna string[pyarrow] via pyarrow compute
    #     substr_array = pc.utf8_slice(current_df[time_column].array, 0, 2)

    #     # Converte para pd.Series com dtype string[pyarrow]
    #     horas_pyarrow: pd.Series = pd.Series(substr_array).astype("string[pyarrow]")

    #     # Converte para categoria ordenada (pandas Categorical)
    #     horas_categ = horas_pyarrow.astype(categoria_horas)

    #     # Insere a coluna no DataFrame no local correto
    #     if before_column and before_column in current_df.columns:
    #         pos = current_df.columns.get_loc(before_column)
    #         current_df.insert(pos, 'HORA', horas_categ)
    #     else:
    #         current_df['HORA'] = horas_categ