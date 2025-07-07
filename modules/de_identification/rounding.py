import pandas as pd
import numpy as np
from typing import Literal, Union


class RoundingProcessor:
    """숫자 데이터의 라운딩 처리를 담당하는 클래스"""
    
    @staticmethod
    def round_column(
        df: pd.DataFrame,
        column_name: str,
        rounding_type: Literal['floor', 'ceil', 'round'],
        decimal_places: int = None,
        integer_place: int = None
    ) -> pd.Series:
        """
        데이터프레임의 특정 컬럼에 라운딩 적용
        
        Args:
            df: 처리할 데이터프레임
            column_name: 처리할 컬럼명
            rounding_type: 'floor' (내림), 'ceil' (올림), 'round' (반올림)
            decimal_places: 소수점 자리수 (1, 2, 3...)
            integer_place: 정수 자리수 (10, 100, 1000...)
            
        Returns:
            처리된 컬럼 데이터
        """
        if column_name not in df.columns:
            raise ValueError(f"컬럼 '{column_name}'을 찾을 수 없습니다.")
        
        column_data = df[column_name].copy()
        
        # 숫자가 아닌 데이터는 그대로 반환
        if not pd.api.types.is_numeric_dtype(column_data):
            raise ValueError(f"컬럼 '{column_name}'은(는) 숫자형 데이터가 아닙니다.")
        
        # 소수점 자리 처리
        if decimal_places is not None:
            if rounding_type == 'floor':
                return np.floor(column_data * (10 ** decimal_places)) / (10 ** decimal_places)
            elif rounding_type == 'ceil':
                return np.ceil(column_data * (10 ** decimal_places)) / (10 ** decimal_places)
            else:  # round
                return np.round(column_data, decimal_places)
        
        # 정수 자리 처리
        elif integer_place is not None:
            if rounding_type == 'floor':
                return np.floor(column_data / integer_place) * integer_place
            elif rounding_type == 'ceil':
                return np.ceil(column_data / integer_place) * integer_place
            else:  # round
                return np.round(column_data / integer_place) * integer_place
        
        else:
            raise ValueError("decimal_places 또는 integer_place 중 하나는 지정해야 합니다.")
    
    @staticmethod
    def get_preview(
        df: pd.DataFrame,
        column_name: str,
        rounding_type: Literal['floor', 'ceil', 'round'],
        decimal_places: int = None,
        integer_place: int = None,
        sample_size: int = 5
    ) -> pd.DataFrame:
        """
        라운딩 결과 미리보기
        
        Returns:
            원본값과 결과값을 보여주는 데이터프레임
        """
        # 유효한 숫자 데이터만 샘플링
        valid_data = df[df[column_name].notna()][column_name]
        
        if len(valid_data) == 0:
            return pd.DataFrame({"원본값": [], "결과값": []})
        
        # 샘플 데이터 선택
        sample_indices = valid_data.sample(min(sample_size, len(valid_data))).index
        original_values = df.loc[sample_indices, column_name]
        
        # 라운딩 적용
        rounded_values = RoundingProcessor.round_column(
            df.loc[sample_indices],
            column_name,
            rounding_type,
            decimal_places,
            integer_place
        )
        
        return pd.DataFrame({
            "원본값": original_values.values,
            "결과값": rounded_values.values
        })