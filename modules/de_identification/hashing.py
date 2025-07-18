# modules/de_identification/hashing.py

import pandas as pd
import hashlib
import base64
import secrets
from typing import List, Dict, Optional, Literal, Union
import streamlit as st

class HashingProcessor:
    """일방향 해시 처리 클래스"""
    
    # 지원하는 해시 알고리즘
    ALGORITHMS = {
        'sha256': {'name': 'SHA-256', 'secure': True, 'length': 64},
        'sha512': {'name': 'SHA-512', 'secure': True, 'length': 128},
        'blake2b': {'name': 'Blake2b', 'secure': True, 'length': 128},
        'md5': {'name': 'MD5', 'secure': False, 'length': 32}
    }
    
    @staticmethod
    def generate_salt(length: int = 32) -> str:
        """암호학적으로 안전한 랜덤 Salt 생성"""
        return secrets.token_hex(length)
    
    @staticmethod
    def hash_value(
        value: Union[str, int, float],
        algorithm: str = 'sha256',
        salt: Optional[str] = None,
        encoding: str = 'utf-8'
    ) -> str:
        """단일 값을 해시"""
        # 값을 문자열로 변환
        str_value = str(value)
        
        # Salt 추가
        if salt:
            str_value = salt + str_value
        
        # 해시 객체 생성
        if algorithm == 'blake2b':
            hash_obj = hashlib.blake2b(str_value.encode(encoding))
        else:
            hash_obj = hashlib.new(algorithm, str_value.encode(encoding))
        
        return hash_obj.hexdigest()
    
    @classmethod
    def hash_column(
        cls,
        df: pd.DataFrame,
        column: str,
        algorithm: str = 'sha256',
        salt_type: Literal['none', 'global', 'individual'] = 'none',
        salt_value: Optional[str] = None,
        output_format: Literal['full', 'short', 'base64'] = 'full',
        short_length: int = 8
    ) -> pd.Series:
        """컬럼 전체를 해시 처리"""
        
        series = df[column].copy()
        
        # Salt 처리
        if salt_type == 'global':
            # 전역 Salt: 모든 값에 동일한 Salt 사용
            if not salt_value:
                salt_value = cls.generate_salt()
            
            hashed_series = series.apply(
                lambda x: cls.hash_value(x, algorithm, salt_value) if pd.notna(x) else None
            )
            
        elif salt_type == 'individual':
            # 개별 Salt: 각 행마다 다른 Salt (실제로는 잘 안 씀)
            hashed_series = series.apply(
                lambda x: cls.hash_value(x, algorithm, cls.generate_salt()) if pd.notna(x) else None
            )
            
        else:  # salt_type == 'none'
            # Salt 없음
            hashed_series = series.apply(
                lambda x: cls.hash_value(x, algorithm) if pd.notna(x) else None
            )
        
        # 출력 형식 처리
        if output_format == 'short':
            hashed_series = hashed_series.apply(
                lambda x: x[:short_length] if x else None
            )
        elif output_format == 'base64':
            hashed_series = hashed_series.apply(
                lambda x: base64.b64encode(bytes.fromhex(x)).decode('ascii')[:short_length] if x else None
            )
        
        return hashed_series
    
    @classmethod
    def hash_combined(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        algorithm: str = 'sha256',
        separator: str = '|',
        salt_type: Literal['none', 'global'] = 'none',
        salt_value: Optional[str] = None,
        output_format: Literal['full', 'short', 'base64'] = 'full',
        short_length: int = 8
    ) -> pd.Series:
        """여러 컬럼을 조합하여 해시"""
        
        # 컬럼들을 조합
        combined = df[columns].apply(
            lambda row: separator.join(str(val) for val in row if pd.notna(val)),
            axis=1
        )
        
        # Salt 처리
        if salt_type == 'global' and not salt_value:
            salt_value = cls.generate_salt()
        
        # 해시 적용
        hashed_series = combined.apply(
            lambda x: cls.hash_value(x, algorithm, salt_value if salt_type == 'global' else None) if x else None
        )
        
        # 출력 형식 처리
        if output_format == 'short':
            hashed_series = hashed_series.apply(
                lambda x: x[:short_length] if x else None
            )
        elif output_format == 'base64':
            hashed_series = hashed_series.apply(
                lambda x: base64.b64encode(bytes.fromhex(x)).decode('ascii')[:short_length] if x else None
            )
        
        return hashed_series
    
    @classmethod
    def get_preview(
        cls,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        hash_type: Literal['single', 'combined'] = 'single',
        sample_size: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        """해시 처리 미리보기"""
        
        # 샘플 추출
        sample_df = df.head(sample_size).copy()
        
        if hash_type == 'single' and isinstance(columns, str):
            # 단일 컬럼 해시
            original_col = f"{columns}_원본"
            hashed_col = f"{columns}_해시"
            
            preview_df = pd.DataFrame({
                original_col: sample_df[columns],
                hashed_col: cls.hash_column(sample_df, columns, **kwargs)
            })
            
        else:  # combined
            # 복수 컬럼 조합 해시
            if isinstance(columns, str):
                columns = [columns]
            
            preview_df = sample_df[columns].copy()
            preview_df['조합_해시'] = cls.hash_combined(sample_df, columns, **kwargs)
        
        return preview_df
    
    @classmethod
    def get_statistics(
        cls,
        df: pd.DataFrame,
        column: str
    ) -> Dict:
        """해시 처리 전 통계 정보"""
        
        series = df[column]
        
        return {
            'total_rows': len(series),
            'unique_values': series.nunique(),
            'null_values': series.isna().sum(),
            'duplicate_rate': 1 - (series.nunique() / len(series.dropna())),
            'sample_values': series.dropna().head(10).tolist()
        }