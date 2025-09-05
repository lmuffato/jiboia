from .regex_patterns import (
    RAW_INVALID_VALUES
)
import numpy as np
import pandas as pd


class NormalizeNullUtils:
    @staticmethod
    def _is_nan_value(value: str, raw_invalid_values: list) -> bool:

        if pd.isna(value) or value in raw_invalid_values:
            return True
        else:
            return False

    @staticmethod
    def normalize_null_by_columns(
        current_df: pd.DataFrame,
        columns: list[str],
        # null_values: list[str] = ['-9999', -9999]
        null_values: list[str] = []
    ) -> None:
        
        raw_invalid_values: list = RAW_INVALID_VALUES.copy()

        if null_values:
            raw_invalid_values.extend(null_values)

        for column in columns:
            if column not in current_df.columns:
                continue

            # Maior overhead, reescreve linha com valor original se não estiver na regra
            current_df[column] = current_df[column].apply(
                lambda row: pd.NA if NormalizeNullUtils._is_nan_value(row, raw_invalid_values) else row
            )

            # Menor overhead, não reescreve linha que não está na regra
            # mask: pd.Series = current_df[column].apply(NormalizeNullUtils._is_nan_value)
            # current_df.loc[mask, column] = np.nan
