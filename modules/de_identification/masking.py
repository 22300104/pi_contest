import pandas as pd
import numpy as np
import re
from typing import Literal, Union, List, Dict


class MaskingProcessor:
    """데이터 마스킹 처리를 담당하는 클래스"""
    
    @staticmethod
    def mask_column(
        df: pd.DataFrame,
        column_name: str,
        masking_type: str,
        **kwargs
    ) -> pd.Series:
        """
        데이터프레임의 특정 컬럼에 마스킹 적용
        
        Args:
            df: 처리할 데이터프레임
            column_name: 처리할 컬럼명
            masking_type: 마스킹 유형
            **kwargs: 마스킹 타입별 추가 파라미터
            
        Returns:
            마스킹된 컬럼 데이터
        """
        if column_name not in df.columns:
            raise ValueError(f"컬럼 '{column_name}'을 찾을 수 없습니다.")
        
        column_data = df[column_name].copy()
        
        # 문자열로 변환 (숫자도 마스킹 가능하도록)
        str_column = column_data.astype(str)
        
        # 마스킹 타입별 처리
        if masking_type == "basic":
            return MaskingProcessor._basic_masking(str_column, **kwargs)
        elif masking_type == "position":
            return MaskingProcessor._position_masking(str_column, **kwargs)
        elif masking_type == "pattern":
            return MaskingProcessor._pattern_masking(str_column, **kwargs)
        elif masking_type == "condition":
            return MaskingProcessor._condition_masking(str_column, **kwargs)
        elif masking_type == "special":
            return MaskingProcessor._special_masking(str_column, **kwargs)
        else:
            raise ValueError(f"알 수 없는 마스킹 타입: {masking_type}")
    
    @staticmethod
    def _basic_masking(
        series: pd.Series,
        direction: str = "front",
        length: int = 3,
        mask_char: str = "*",
        min_preserve: int = 1
    ) -> pd.Series:
        """기본 마스킹 (앞/뒤에서 N글자)"""
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            val_len = len(val_str)
            
            # 최소 보존 길이 체크
            if val_len <= min_preserve:
                return val_str
            
            # 마스킹 가능한 최대 길이
            max_mask_len = val_len - min_preserve
            actual_mask_len = min(length, max_mask_len)
            
            if direction == "front":
                return mask_char * actual_mask_len + val_str[actual_mask_len:]
            else:  # back
                return val_str[:-actual_mask_len] + mask_char * actual_mask_len
        
        return series.apply(mask_value)
    
    @staticmethod
    def _position_masking(
        series: pd.Series,
        mask_type: str,
        mask_char: str = "*",
        **params
    ) -> pd.Series:
        """위치 기반 마스킹"""
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            val_len = len(val_str)
            
            if mask_type == "first_n":
                n = params.get('n', 3)
                n = min(n, val_len)
                return mask_char * n + val_str[n:]
                
            elif mask_type == "last_n":
                n = params.get('n', 4)
                n = min(n, val_len)
                return val_str[:-n] + mask_char * n
                
            elif mask_type == "range":
                start = params.get('start', 2) - 1  # 1-based to 0-based
                end = params.get('end', 5)
                start = max(0, min(start, val_len))
                end = min(end, val_len)
                
                if start < end:
                    return val_str[:start] + mask_char * (end - start) + val_str[end:]
                return val_str
                
            elif mask_type == "middle":
                preserve = params.get('preserve', 2)
                if val_len <= preserve * 2:
                    return val_str
                return val_str[:preserve] + mask_char * (val_len - preserve * 2) + val_str[-preserve:]
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _pattern_masking(
        series: pd.Series,
        pattern_type: str,
        mask_char: str = "*",
        **params
    ) -> pd.Series:
        """패턴 기반 마스킹"""
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if pattern_type == "after_delimiter":
                delimiter = params.get('delimiter', '-')
                position = params.get('position', 'first')  # first or last
                
                parts = val_str.split(delimiter)
                if len(parts) <= 1:
                    return val_str
                
                if position == 'first':
                    # 첫 번째 구분자 이후 마스킹
                    masked_parts = [parts[0]] + [mask_char * len(part) for part in parts[1:]]
                else:  # last
                    # 마지막 부분만 마스킹
                    masked_parts = parts[:-1] + [mask_char * len(parts[-1])]
                
                return delimiter.join(masked_parts)
                
            elif pattern_type == "before_char":
                char = params.get('char', '@')
                preserve = params.get('preserve', 1)
                
                if char in val_str:
                    idx = val_str.index(char)
                    if idx > preserve:
                        return val_str[:preserve] + mask_char * (idx - preserve) + val_str[idx:]
                
                return val_str
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _condition_masking(
        series: pd.Series,
        condition_type: str,
        mask_char: str = "*",
        **params
    ) -> pd.Series:
        """조건부 마스킹"""
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            val_len = len(val_str)
            
            if condition_type == "by_length":
                # 길이별 다른 마스킹 규칙
                if val_len <= 3:
                    # 3글자 이하: 마지막 1글자만
                    return val_str[:-1] + mask_char if val_len > 1 else val_str
                elif val_len <= 6:
                    # 4-6글자: 뒤 2글자
                    return val_str[:-2] + mask_char * 2
                else:
                    # 7글자 이상: 가운데 마스킹
                    preserve = 2
                    return val_str[:preserve] + mask_char * (val_len - preserve * 2) + val_str[-preserve:]
                    
            elif condition_type == "percentage":
                percent = params.get('percent', 50) / 100
                position = params.get('position', 'back')
                
                mask_len = int(val_len * percent)
                if mask_len == 0:
                    return val_str
                
                if position == 'front':
                    return mask_char * mask_len + val_str[mask_len:]
                elif position == 'back':
                    return val_str[:-mask_len] + mask_char * mask_len
                else:  # distributed
                    # 고르게 분산
                    result = list(val_str)
                    indices = np.linspace(0, val_len-1, mask_len, dtype=int)
                    for idx in indices:
                        result[idx] = mask_char
                    return ''.join(result)
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _special_masking(
        series: pd.Series,
        special_type: str,
        mask_char: str = "*",
        **params
    ) -> pd.Series:
        """특수 마스킹"""
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if special_type == "numbers_only":
                # 숫자만 마스킹
                return re.sub(r'\d', mask_char, val_str)
                
            elif special_type == "letters_only":
                # 문자만 마스킹 (한글, 영문)
                return re.sub(r'[가-힣a-zA-Z]', mask_char, val_str)
                
            elif special_type == "keep_format":
                # 형식 유지 (구분자는 그대로)
                # 예: 1234-5678 → ****-****
                return re.sub(r'[^-\s./()]', mask_char, val_str)
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def get_preview(
        df: pd.DataFrame,
        column_name: str,
        masking_type: str,
        sample_size: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        """마스킹 결과 미리보기"""
        # 다양한 길이의 샘플 선택
        valid_data = df[df[column_name].notna()][column_name]
        
        if len(valid_data) == 0:
            return pd.DataFrame({"원본": [], "마스킹 결과": []})
        
        # 길이별로 다양한 샘플 선택
        str_data = valid_data.astype(str)
        lengths = str_data.str.len()
        
        # 다양한 길이의 샘플 추출
        samples = []
        for length in lengths.unique()[:sample_size]:
            length_samples = str_data[lengths == length].head(1)
            samples.extend(length_samples.tolist())
        
        # 부족하면 랜덤 샘플 추가
        if len(samples) < sample_size:
            additional = valid_data.sample(min(sample_size - len(samples), len(valid_data)))
            samples.extend(additional.tolist())
        
        # 샘플 데이터프레임 생성
        sample_df = pd.DataFrame({column_name: samples[:sample_size]})
        
        # 마스킹 적용
        masked_values = MaskingProcessor.mask_column(
            sample_df,
            column_name,
            masking_type,
            **kwargs
        )
        
        return pd.DataFrame({
            "원본": samples[:sample_size],
            "마스킹 결과": masked_values.values
        })