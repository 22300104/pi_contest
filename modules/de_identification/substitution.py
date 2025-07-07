import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Union


class SubstitutionProcessor:
    """데이터 치환 처리를 담당하는 클래스"""
    
    @staticmethod
    def substitute_column(
        df: pd.DataFrame,
        column_name: str,
        substitution_type: str,
        **kwargs
    ) -> pd.Series:
        """
        데이터프레임의 특정 컬럼에 치환 적용
        
        Args:
            df: 처리할 데이터프레임
            column_name: 처리할 컬럼명
            substitution_type: 치환 유형 (numeric, categorical, pattern)
            **kwargs: 치환 타입별 추가 파라미터
            
        Returns:
            치환된 컬럼 데이터
        """
        if column_name not in df.columns:
            raise ValueError(f"컬럼 '{column_name}'을 찾을 수 없습니다.")
        
        column_data = df[column_name].copy()
        
        # NULL 값 인덱스 저장
        null_mask = column_data.isna()
        
        # 치환 타입별 처리
        if substitution_type == "numeric":
            result = SubstitutionProcessor._numeric_substitution(column_data, **kwargs)
        elif substitution_type == "categorical":
            result = SubstitutionProcessor._categorical_substitution(column_data, **kwargs)
        elif substitution_type == "pattern":
            result = SubstitutionProcessor._pattern_substitution(column_data, **kwargs)
        else:
            raise ValueError(f"알 수 없는 치환 타입: {substitution_type}")
        
        # NULL 값 복원 (옵션에 따라)
        if kwargs.get('preserve_null', True):
            result[null_mask] = np.nan
        
        return result
    
    @staticmethod
    def _numeric_substitution(
        series: pd.Series,
        method: str = "manual",
        **params
    ) -> pd.Series:
        """숫자형 데이터 구간 치환"""
        
        if method == "manual":
            return SubstitutionProcessor._manual_interval_substitution(series, **params)
        elif method == "equal":
            return SubstitutionProcessor._equal_interval_substitution(series, **params)
        elif method == "quantile":
            return SubstitutionProcessor._quantile_interval_substitution(series, **params)
        elif method == "std":
            return SubstitutionProcessor._std_interval_substitution(series, **params)
        else:
            raise ValueError(f"알 수 없는 구간 설정 방식: {method}")
    
    @staticmethod
    def _manual_interval_substitution(
        series: pd.Series,
        intervals: List[Dict[str, Any]],
        **params
    ) -> pd.Series:
        """수동 구간 설정 치환
        
        intervals: [
            {'min': 0, 'max': 30, 'value': '낮음', 'include_min': True, 'include_max': False},
            {'min': 30, 'max': 60, 'value': '중간', 'include_min': True, 'include_max': False},
            {'min': 60, 'max': 100, 'value': '높음', 'include_min': True, 'include_max': True}
        ]
        """
        result = series.copy()
        
        # 숫자형으로 변환 시도
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
        except:
            raise ValueError("숫자형으로 변환할 수 없는 데이터가 포함되어 있습니다.")
        
        # 각 구간에 대해 치환 적용
        for interval in intervals:
            min_val = interval['min']
            max_val = interval['max']
            replace_val = interval['value']
            include_min = interval.get('include_min', True)
            include_max = interval.get('include_max', False)
            
            # 조건 생성
            if include_min and include_max:
                mask = (numeric_series >= min_val) & (numeric_series <= max_val)
            elif include_min and not include_max:
                mask = (numeric_series >= min_val) & (numeric_series < max_val)
            elif not include_min and include_max:
                mask = (numeric_series > min_val) & (numeric_series <= max_val)
            else:
                mask = (numeric_series > min_val) & (numeric_series < max_val)
            
            # 치환 적용
            result[mask] = replace_val
        
        return result
    
    @staticmethod
    def _equal_interval_substitution(
        series: pd.Series,
        n_intervals: int = 5,
        labels: List[str] = None,
        **params
    ) -> pd.Series:
        """등간격 구간 치환"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
        except:
            raise ValueError("숫자형으로 변환할 수 없는 데이터가 포함되어 있습니다.")
        
        # 유효한 값만 사용하여 최소/최대 계산
        valid_values = numeric_series.dropna()
        if len(valid_values) == 0:
            return series
        
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        # 등간격으로 구간 생성
        bins = np.linspace(min_val, max_val, n_intervals + 1)
        
        # 라벨 생성
        if labels is None:
            labels = [f"구간{i+1}" for i in range(n_intervals)]
        elif len(labels) != n_intervals:
            raise ValueError(f"라벨 개수({len(labels)})가 구간 개수({n_intervals})와 일치하지 않습니다.")
        
        # pd.cut을 사용하여 구간 분할
        result = pd.cut(numeric_series, bins=bins, labels=labels, include_lowest=True)
        
        # 문자열로 변환
        return result.astype(str)
    
    @staticmethod
    def _quantile_interval_substitution(
        series: pd.Series,
        n_quantiles: int = 4,
        labels: List[str] = None,
        **params
    ) -> pd.Series:
        """분위수 기반 구간 치환"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
        except:
            raise ValueError("숫자형으로 변환할 수 없는 데이터가 포함되어 있습니다.")
        
        # 라벨 생성
        if labels is None:
            if n_quantiles == 4:
                labels = ["Q1", "Q2", "Q3", "Q4"]
            elif n_quantiles == 10:
                labels = [f"D{i+1}" for i in range(n_quantiles)]
            else:
                labels = [f"분위{i+1}" for i in range(n_quantiles)]
        elif len(labels) != n_quantiles:
            raise ValueError(f"라벨 개수({len(labels)})가 분위수({n_quantiles})와 일치하지 않습니다.")
        
        # pd.qcut을 사용하여 분위수 분할
        try:
            result = pd.qcut(numeric_series.dropna(), q=n_quantiles, labels=labels, duplicates='drop')
            # 원래 인덱스에 맞춰 결과 재배치
            final_result = pd.Series(index=series.index, dtype='object')
            final_result[numeric_series.notna()] = result
            return final_result.astype(str)
        except:
            # 고유값이 너무 적은 경우 등간격으로 대체
            return SubstitutionProcessor._equal_interval_substitution(
                series, n_intervals=n_quantiles, labels=labels, **params
            )
    
    @staticmethod
    def _std_interval_substitution(
        series: pd.Series,
        n_std: float = 1.0,
        labels: List[str] = None,
        **params
    ) -> pd.Series:
        """표준편차 기반 구간 치환"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
        except:
            raise ValueError("숫자형으로 변환할 수 없는 데이터가 포함되어 있습니다.")
        
        # 통계량 계산
        mean = numeric_series.mean()
        std = numeric_series.std()
        
        # 기본 라벨
        if labels is None:
            labels = ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]
        
        # 구간 생성 (평균 ± n*표준편차)
        bins = [
            -np.inf,
            mean - 2 * n_std * std,
            mean - n_std * std,
            mean + n_std * std,
            mean + 2 * n_std * std,
            np.inf
        ]
        
        # pd.cut을 사용하여 구간 분할
        result = pd.cut(numeric_series, bins=bins, labels=labels)
        
        return result.astype(str)
    
    @staticmethod
    def _categorical_substitution(
        series: pd.Series,
        mappings: Dict[Any, Any],
        default_value: Any = None,
        **params
    ) -> pd.Series:
        """범주형 데이터 치환
        
        Args:
            mappings: 치환 매핑 딕셔너리 {'원본값': '치환값'}
            default_value: 매핑에 없는 값의 기본 치환값
        """
        result = series.copy()
        
        # 매핑 적용
        for old_value, new_value in mappings.items():
            mask = result == old_value
            result[mask] = new_value
        
        # 기본값 처리
        if default_value is not None:
            # 매핑되지 않은 값들에 기본값 적용
            mapped_values = set(mappings.keys())
            unmapped_mask = ~result.isin(mapped_values)
            result[unmapped_mask] = default_value
        
        return result
    
    @staticmethod
    def _pattern_substitution(
        series: pd.Series,
        patterns: List[Dict[str, str]],
        **params
    ) -> pd.Series:
        """패턴 기반 치환
        
        patterns: [
            {'pattern': '서울|경기|인천', 'value': '수도권'},
            {'pattern': '부산|대구|울산|경남|경북', 'value': '영남권'}
        ]
        """
        result = series.astype(str).copy()
        
        # 각 패턴에 대해 치환 적용
        for pattern_dict in patterns:
            pattern = pattern_dict['pattern']
            replace_value = pattern_dict['value']
            
            try:
                # 패턴에 매칭되는 값 찾기
                mask = result.str.contains(pattern, regex=True, na=False)
                result[mask] = replace_value
            except re.error:
                raise ValueError(f"잘못된 정규식 패턴: {pattern}")
        
        return result
    
    @staticmethod
    def get_statistics(
        df: pd.DataFrame,
        column_name: str,
        substitution_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """치환 전후 통계 정보 생성"""
        original_series = df[column_name]
        
        # 치환 적용
        substituted_series = SubstitutionProcessor.substitute_column(
            df, column_name, substitution_type, **kwargs
        )
        
        stats = {
            'original': {
                'unique_count': original_series.nunique(),
                'null_count': original_series.isna().sum(),
                'value_counts': original_series.value_counts().to_dict()
            },
            'substituted': {
                'unique_count': substituted_series.nunique(),
                'null_count': substituted_series.isna().sum(),
                'value_counts': substituted_series.value_counts().to_dict()
            }
        }
        
        # 숫자형 데이터인 경우 추가 통계
        if pd.api.types.is_numeric_dtype(original_series):
            stats['original']['mean'] = original_series.mean()
            stats['original']['std'] = original_series.std()
            stats['original']['min'] = original_series.min()
            stats['original']['max'] = original_series.max()
        
        return stats
    
    @staticmethod
    def get_preview(
        df: pd.DataFrame,
        column_name: str,
        substitution_type: str,
        sample_size: int = 10,
        **kwargs
    ) -> pd.DataFrame:
        """치환 결과 미리보기"""
        try:
            # 유효한 데이터 선택
            valid_data = df[df[column_name].notna()][column_name]
            
            if len(valid_data) == 0:
                return pd.DataFrame({"원본": [], "결과": []})
            
            # 다양한 샘플 선택
            if substitution_type == "categorical":
                # 범주형: 각 고유값에서 샘플
                unique_values = valid_data.unique()
                samples = []
                
                for value in unique_values[:sample_size]:
                    value_data = valid_data[valid_data == value]
                    if len(value_data) > 0:
                        samples.append(value_data.iloc[0])
                
                # 부족하면 랜덤 샘플 추가
                if len(samples) < sample_size:
                    remaining = min(sample_size - len(samples), len(valid_data))
                    if remaining > 0:
                        additional = valid_data.sample(remaining, random_state=42)
                        samples.extend(additional.tolist())
            else:
                # 숫자형: 분포를 고려한 샘플
                try:
                    numeric_data = pd.to_numeric(valid_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        # 분위수 기반 샘플링
                        quantiles = np.linspace(0, 1, min(sample_size, len(numeric_data)))
                        samples = [numeric_data.quantile(q) for q in quantiles]
                    else:
                        samples = valid_data.sample(min(sample_size, len(valid_data)), random_state=42).tolist()
                except:
                    samples = valid_data.sample(min(sample_size, len(valid_data)), random_state=42).tolist()
            
            # 샘플 개수 조정
            samples = samples[:sample_size]
            
            # 샘플 데이터프레임 생성
            sample_df = pd.DataFrame({column_name: samples})
            
            # 치환 적용
            try:
                substituted_values = SubstitutionProcessor.substitute_column(
                    sample_df,
                    column_name,
                    substitution_type,
                    **kwargs
                )
                
                # 치환 전후 비교 테이블 생성
                preview_df = pd.DataFrame({
                    "원본": samples,
                    "결과": substituted_values.values
                })
                
                # 중복 제거 (선택적)
                if kwargs.get('remove_duplicates', True):
                    preview_df = preview_df.drop_duplicates()
                
                return preview_df
                
            except Exception as e:
                # 처리 중 오류 발생 시
                return pd.DataFrame({
                    "원본": samples[:min(3, len(samples))],
                    "결과": ["오류: " + str(e)] * min(3, len(samples))
                })
                
        except Exception as e:
            # 전체 오류 발생 시
            return pd.DataFrame({
                "원본": ["오류 발생"],
                "결과": [str(e)]
            })