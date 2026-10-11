from typing import ClassVar

import pandas as pd


class PandasSupportedDtypes:
    str_types: ClassVar[list[str]] = ["object", "string"]
    int_types: ClassVar[list[str]] = ["int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"]
    uint_types: ClassVar[list[str]] = ["uint8", "uint16", "uint32", "uint64", "UInt8", "UInt16", "UInt32", "UInt64"]
    float_types: ClassVar[list[str]] = ["float32", "float64", "Float32", "Float64"]
    decimal_types: ClassVar[list[str]] = ["decimal"]  # ver observação abaixo
    numeric_types: ClassVar[list[str]] = int_types + uint_types + float_types + decimal_types
    bool_types: ClassVar[list[str]] = ["bool", "boolean"]
    datetime_types: ClassVar[list[str]] = ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"]
    timedelta_types: ClassVar[list[str]] = ["timedelta64[s]", "timedelta64[ms]", "timedelta64[us]", "timedelta64[ns]"]
    category_types: ClassVar[list[str]] = ["category"]
    struct_types: ClassVar[list[str]] = ["struct"]  # ver observação abaixo


def is_valid_to_normalize(
    series: pd.Series,
    valid_types: list[str] | None = None,
    invalid_types: list[str] | None = None,
) -> bool:
    valid_types = valid_types or []
    invalid_types = invalid_types or []

    series_type: str = str(series.dtype)

    is_empty: bool = series.isna().all()

    if is_empty:
        return False

    is_valid_type: bool = True
    is_invalid_type: bool = True

    if valid_types:
        is_valid_type = series_type in valid_types

    if invalid_types:
        is_invalid_type = series_type not in invalid_types

    return is_valid_type and is_invalid_type
