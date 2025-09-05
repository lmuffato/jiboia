from .analysis_utils import AnalysisUtils
from .normalize_date_utils import NormalizeDateUtils
from .normalize_null_utils import NormalizeNullUtils
from .normalize_int_utils import NormalizeIntUtils
from .normalize_float_utils import NormalizeFloatUtils
from .normalize_string_utils import NormalizeStringUtils
from .normalize_df_utils import NormalizeDfUtils
from .file_analysis_utils import FileAnalysisUtils
from .regex_patterns import (
    RegexPatternsCombine,
    RAW_INVALID_VALUES,
    DATE_REGEX_PATTERNS,
    TIME_REGEX_PATTERNS,
    DATE_TIME_REGEX_PATTERNS,
    RAW_INVALID_VALUES
)


class Jiboia:

    # DATAFRAME
    normalize_df = staticmethod(NormalizeDfUtils.normalize_df)

    # TYPES:
    infer_column_types = staticmethod(AnalysisUtils.infer_column_types)

    # STRING
    normalize_string_by_columns = staticmethod(NormalizeStringUtils.normalize_string_by_columns)
    # normalize_category = staticmethod(NormalizeStringUtils.normalize_category)
    # normalize_unique_values = staticmethod(NormalizeStringUtils.normalize_unique_values)
    
    # NULL
    normalize_null_by_columns = staticmethod(NormalizeNullUtils.normalize_null_by_columns)

    # BOOL
    # normalize_bool = staticmethod(GPUNormalizeBooleanUtils.normalize_bool)
    
    # NUMBER
    normalize_int_by_columns = staticmethod(NormalizeIntUtils.normalize_int_by_columns)
    normalize_float_by_columns = staticmethod(NormalizeFloatUtils.normalize_float_by_columns)


    # DATE
    normalize_str_date_to_date_standard_str_by_columns = staticmethod(NormalizeDateUtils.normalize_str_date_to_date_standard_str_by_columns)
    # normalize_datetime = staticmethod(NormalizeDateUtils.normalize_datetime)
    # is_date_column_by_samples = staticmethod(NormalizeDateUtils.is_date_column_by_samples)
    # is_date = staticmethod(NormalizeDateUtils.is_date)
    # to_date_iso = staticmethod(NormalizeDateUtils.to_date_iso)
    # to_datetime = staticmethod(NormalizeDateUtils.to_datetime)


    # TIME
    normalize_str_time_to_hh_mm_ss_by_columns = staticmethod(NormalizeDateUtils.normalize_str_time_to_hh_mm_ss_by_columns)
    
    # normalize_time = staticmethod(GPUNormalizeTimeUtils.normalize_time)
    # is_time = staticmethod(GPUNormalizeTimeUtils.is_time)

    # # TIMEDELTA
    # normalize_timedelta = staticmethod(GPUNormalizeTimeUtils.normalize_timedelta)

    # # DATETIME
    # create_datetime_column_from_date_and_time = staticmethod(GPUNormalizeDateTimeUtils.create_datetime_column_from_date_and_time)
    # create_day_week_column = staticmethod(GPUNormalizeDateTimeUtils.create_day_week_column)
    # create_month_br_category_column = staticmethod(GPUNormalizeDateTimeUtils.create_month_br_category_column)
    # create_day_week_br_category_column = staticmethod(GPUNormalizeDateTimeUtils.create_day_week_br_category_column)
    # create_hour_category_column = staticmethod(GPUNormalizeDateTimeUtils.create_hour_category_column)

    # ANALYSIS
    # drop_columns = staticmethod(AnalysisUtils.drop_columns)
    # cudf_size_info = staticmethod(AnalysisUtils.cudf_size_info)
    # pandasdf_size_info = staticmethod(AnalysisUtils.pandasdf_size_info)
    # get_gpu_memory_info = staticmethod(AnalysisUtils.get_gpu_memory_info)
    # is_vram_use_limit = staticmethod(AnalysisUtils.is_vram_use_limit)
    # get_index_samples = staticmethod(AnalysisUtils.get_index_samples)
    # infer_by_sample = staticmethod(AnalysisUtils.infer_by_sample)

    # # FILE
    get_csv_sep_delimiter = staticmethod(FileAnalysisUtils.get_csv_sep_delimiter)
    # get_csv_info = staticmethod(FileAnalysisUtils.get_csv_info)
    # convert_csv_to_utf8 = staticmethod(FileAnalysisUtils.convert_csv_to_utf8)
    # convert_all_csvs_to_utf8 = staticmethod(FileAnalysisUtils.convert_all_csvs_to_utf8)
    # read_csv_files = staticmethod(FileAnalysisUtils.read_csv_files)

# instância única para ser usada como namespace
jiboia = Jiboia()

__all__ = ["jiboia", "Jiboia"]
