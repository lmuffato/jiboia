import functools
from typing import Callable, Generator

import numpy as np
import pandas as pd


def combine_regex(regex_patterns: list[dict[str, str]]) -> str:
    return "|".join(pattern["regex"] for pattern in regex_patterns)


def chunk_df(chunk_size: int = 500_000):
    """
    Decorador para processar uma coluna de um DataFrame em chunks.
    Modifica o DataFrame inplace.

    Divide o DataFrame em pedaços, aplica a função decorada a cada um e
    devolve o resultado da coluna `column_name` remontado no DataFrame original.

    Modo de usar:
    1. Como decorador:
       @chunk_df(chunk_size=100_000)
       def funcao(dataframe: pd.DataFrame, column_name: str, ...):
           ...
           return dataframe

    2. Na chamada de uma função já declarada:
       funcao_decorada = chunk_df(chunk_size=100_000)(funcao)
       funcao_decorada(df, "coluna", ...)

    Parâmetros:
    - chunk_size (int): Tamanho de cada pedaço a ser processado.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(dataframe: pd.DataFrame, column_name: str, *args, **kwargs):
            if len(dataframe) <= chunk_size:
                return func(dataframe, column_name, *args, **kwargs)

            processed_parts: list[pd.Series] = []

            for start in range(0, len(dataframe), chunk_size):
                end = min(start + chunk_size, len(dataframe))
                # .copy() evita SettingWithCopyWarning se a função alterar o chunk
                chunk = dataframe.iloc[start:end].copy()

                processed_chunk = func(chunk, column_name, *args, **kwargs)
                processed_parts.append(processed_chunk[column_name])

            result = pd.concat(processed_parts)
            # Atribui por posição (e não por label), seguro mesmo com índice duplicado
            result.index = dataframe.index
            dataframe[column_name] = result

            return dataframe
        return wrapper
    return decorator


def chunk_iterate(
    series: pd.Series,
    chunk_size: int = 500_000,
) -> Generator[pd.Series, None, None]:
    total_rows: int = len(series)
    for start_index in range(0, total_rows, chunk_size):
        end_index: int = min(start_index + chunk_size, total_rows)
        yield series.iloc[start_index:end_index]


def chunk_iterate_index(
    dataframe: pd.DataFrame,
    column_name: str,
    chunk_size: int = 500_000,
) -> Generator[np.ndarray, None, None]:
    """
    Itera sobre uma coluna retornando blocos de índices (posições) de tamanho `chunk_size`.

    Args:
        dataframe (pd.DataFrame): O DataFrame.
        column_name (str): Nome da coluna.
        chunk_size (int): O tamanho do bloco.

    Yields:
        np.ndarray: Array de posições do bloco atual.
    """
    total_rows = len(dataframe[column_name])

    for start_index in range(0, total_rows, chunk_size):
        end_index = min(start_index + chunk_size, total_rows)
        yield np.arange(start_index, end_index)


def get_index_samples(
    series: pd.Series,
    n_parts: int = 100,
    n_samples: int = 10,
) -> np.ndarray:
    series_size = len(series)

    if (n_parts * n_samples) >= series_size:
        raise ValueError(
            "The total number of samples requested exceeds or equals the series size. "
            "Please provide a smaller value for n_parts or n_samples."
        )

    if series_size // n_parts == 0:
        raise ValueError(
            "The number of parts is greater than the series size. "
            "Please provide a smaller value for n_parts."
        )

    # Passo entre as partes
    step_pass = series_size // n_parts

    # Índices iniciais de cada bloco (ex: 0, 1000, 2000, ...)
    start_indices = np.arange(n_parts) * step_pass

    # Offsets dentro de cada bloco (ex: 0, 1, 2, ... n_samples-1)
    sample_offsets = np.arange(n_samples)

    all_indices = (start_indices[:, None] + sample_offsets).flatten()

    # Garante que os índices não ultrapassem o tamanho da Series
    return all_indices[all_indices < series_size]


def series_samples(
    series: pd.Series,
    n_parts: int = 100,
    n_samples: int = 10,
) -> pd.Series:
    index_samples: np.ndarray = get_index_samples(
        series=series,
        n_parts=n_parts,
        n_samples=n_samples,
    )

    return series.iloc[index_samples]
