from typing import Optional
import numpy as np
import pandas as pd


class NormalizeIntUtils:
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
    def _normalize_int(
        value: object,
        exclude_zero: bool = False,
        exclude_negative: bool = False
    ) -> Optional[int]:
        cleaned_number_str: str = NormalizeIntUtils._number_cleaned(value)

        try:
            value: int = int(float(cleaned_number_str))

            if (exclude_zero and value == 0) or (exclude_negative and value < 0):
                return pd.NA

            return value
        except (ValueError, TypeError):
            return pd.NA


    @staticmethod
    def normalize_int_by_columns(
        current_df: pd.DataFrame,
        columns: list[str],
        exclude_zero: bool = False,
        exclude_negative: bool = False
    ) -> None:
           
        for column in columns:
            if column in current_df.columns:
                current_df[column] = (
                    current_df[column]
                    .apply(
                        lambda row: NormalizeIntUtils._normalize_int(row, exclude_zero, exclude_negative)
                    )
                )
