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


import chardet


class FileAnalysisUtils:
    @staticmethod
    def get_csv_sep_delimiter(
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

        with open(file_path, "rb") as f:
            rawdata = f.read(100000)  # lê parte do arquivo
            result = chardet.detect(rawdata)
            print(result)

        print(f'sep_delimiter: "{dialect.delimiter}"')
        return dialect.delimiter



