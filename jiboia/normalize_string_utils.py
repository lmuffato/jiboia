import pandas as pd
import re
import unicodedata


class NormalizeStringUtils:
    # Função interna que normaliza uma string
    @staticmethod
    def _normalize_string(
        text: str,
        remove_accents: bool = False,
        keep_alphanum_space: bool = False
    ) -> str:
        '''
        Normaliza string:
        - Remove acentos (opcional)
        - Remove caracteres que não são letras, números ou espaço (opcional)
        - Remove espaços extras
        - Converte para UPPERCASE
        '''
        if not isinstance(text, str):
            return text

        if remove_accents:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))

        if keep_alphanum_space:
            text = re.sub(r'[^A-Za-z0-9 ]+', '', text)

        text = re.sub(r'\s+', ' ', text).strip()

        return text.upper()


    @staticmethod
    def normalize_string_by_columns(
        current_df: pd.DataFrame,
        columns: list[str],
        remove_accents: bool = False,
        keep_alphanum_space: bool = False
    ) -> None:
        '''
        Aplica a normalização de string nas colunas especificadas.
        '''
        for column in columns:
            if column not in current_df.columns:
                continue
            current_df[column] = current_df[column].apply(
                lambda row: NormalizeStringUtils._normalize_string(
                    row,
                    remove_accents=remove_accents,
                    keep_alphanum_space=keep_alphanum_space
                )
            )


    @staticmethod
    def normalize_unique_values_in_column(
        current_df: pd.DataFrame, 
        column: str, 
        target_value: str, 
        values_to_replace: list[str]
    ) -> None:
        '''
        Substitui valores específicos de uma coluna por um valor alvo.
        
        Parâmetros:
        - current_df: DataFrame contendo os dados.
        - column: Nome da coluna onde a substituição ocorrerá.
        - target_value: O valor que substituirá os valores encontrados.
        - values_to_replace: Lista de valores que serão substituídos pelo valor alvo.
        '''
        # Substitui os valores encontrados pela coluna `column` que estão na lista `values_to_replace` com `target_value`
        current_df[column].replace(values_to_replace, target_value, inplace=True)
