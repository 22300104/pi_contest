import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import openpyxl
import re

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size
        # 일반적인 null 표현들
        self.common_null_values = [
            '-', '--', '---', 'N/A', 'n/a', 'NA', 'na', 'null', 'NULL', 
            'None', 'none', 'NaN', 'nan', ' ', '  ', '#N/A', '#N/A N/A',
            '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND',
            '1.#QNAN', 'N.A.', 'n.a.', 'missing', 'Missing', 'MISSING',
            '.', '..', '...', '?', '??', 'unknown', 'Unknown', 'UNKNOWN',
            '없음', '해당없음', '미상', '알수없음', '모름', '빈칸', '공란'
        ]
    
    def safe_load_data(self, file, file_info: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """안전한 데이터 로드 (에러 로그 포함)"""
        warnings = []
        
        try:
            file.seek(0)
            
            if file_info['type'] == 'csv':
                # 다양한 구분자 시도
                for sep in [',', '\t', ';', '|']:
                    try:
                        file.seek(0)
                        df = pd.read_csv(
                            file, 
                            encoding=file_info['encoding'],
                            sep=sep,
                            na_values=self.common_null_values,
                            keep_default_na=True,
                            low_memory=False,
                            warn_bad_lines=True,
                            error_bad_lines=False  # 잘못된 줄은 건너뛰기
                        )
                        if len(df.columns) > 1:  # 성공적으로 분리됨
                            warnings.append(f"구분자 '{sep}' 사용")
                            break
                    except:
                        continue
                else:
                    # 기본값으로 시도
                    file.seek(0)
                    df = pd.read_csv(file, encoding=file_info['encoding'])
            else:
                df = pd.read_excel(file, engine='openpyxl', na_values=self.common_null_values)
            
            # 데이터 검증
            if df.empty:
                warnings.append("⚠️ 빈 데이터프레임")
            
            if df.shape[1] == 1:
                warnings.append("⚠️ 단일 열만 감지됨. 구분자를 확인하세요.")
            
            # 열 이름 정리
            df.columns = [self.clean_column_name(col) for col in df.columns]
            
            return df, warnings
            
        except Exception as e:
            warnings.append(f"❌ 로드 실패: {str(e)}")
            return None, warnings
    
    def clean_column_name(self, name: str) -> str:
        """열 이름 안전하게 정리"""
        if pd.isna(name):
            return "Unnamed"
        
        # 문자열로 변환
        name = str(name)
        
        # 특수문자 제거 (한글, 영문, 숫자, 언더스코어만 유지)
        name = re.sub(r'[^가-힣a-zA-Z0-9_\s]', '', name)
        
        # 공백을 언더스코어로
        name = name.strip().replace(' ', '_')
        
        # 빈 문자열이면 기본값
        if not name:
            name = "Column"
        
        return name
    
    def detect_data_issues(self, series: pd.Series) -> Dict[str, Any]:
        """데이터의 잠재적 문제점 감지"""
        issues = {
            'warnings': [],
            'suggestions': [],
            'data_quality': {}
        }
        
        # 1. 극단적인 값 체크
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                mean = non_null.mean()
                std = non_null.std()
                
                # 극단값 체크 (평균에서 6시그마 이상)
                if std > 0:
                    extreme_values = non_null[(np.abs(non_null - mean) > 6 * std)]
                    if len(extreme_values) > 0:
                        issues['warnings'].append(f"극단값 {len(extreme_values)}개 감지")
                        issues['suggestions'].append("이상치 제거 또는 로그 변환 고려")
        
        # 2. 일관성 없는 데이터 형식
        if series.dtype == 'object':
            # 날짜 형식 혼재 체크
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
                r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일'  # 한글 날짜
            ]
            
            pattern_counts = {}
            for pattern in date_patterns:
                matches = series.astype(str).str.match(pattern).sum()
                if matches > 0:
                    pattern_counts[pattern] = matches
            
            if len(pattern_counts) > 1:
                issues['warnings'].append("여러 날짜 형식 감지")
                issues['suggestions'].append("날짜 형식 통일 필요")
        
        # 3. 인코딩 문제
        if series.dtype == 'object':
            encoding_issues = series.astype(str).str.contains(r'[�?]+').sum()
            if encoding_issues > 0:
                issues['warnings'].append(f"인코딩 문제 의심 ({encoding_issues}개)")
                issues['suggestions'].append("인코딩 재확인 필요")
        
        # 4. 중복값
        duplicates = series.duplicated().sum()
        if duplicates > len(series) * 0.5:
            issues['warnings'].append(f"높은 중복률 ({duplicates/len(series)*100:.1f}%)")
        
        # 5. 데이터 품질 점수
        issues['data_quality'] = {
            '완전성': f"{(1 - series.isna().sum() / len(series)) * 100:.1f}%",
            '고유성': f"{series.nunique() / len(series) * 100:.1f}%",
            '일관성': "확인 필요" if len(issues['warnings']) > 0 else "양호"
        }
        
        return issues
    
    def safe_type_conversion(self, series: pd.Series, target_type: str) -> Tuple[pd.Series, Dict[str, Any]]:
        """안전한 타입 변환"""
        result = {
            'success': False,
            'converted_series': series,
            'conversion_stats': {},
            'errors': []
        }
        
        try:
            if target_type == 'numeric':
                # 천 단위 구분 기호 제거
                if series.dtype == 'object':
                    series_clean = series.str.replace(',', '').str.replace(' ', '')
                else:
                    series_clean = series
                
                # 숫자 변환
                converted = pd.to_numeric(series_clean, errors='coerce')
                
                result['conversion_stats'] = {
                    'original_non_null': series.notna().sum(),
                    'converted_count': converted.notna().sum(),
                    'conversion_failures': converted.isna().sum() - series.isna().sum()
                }
                
                result['converted_series'] = converted
                result['success'] = True
                
            elif target_type == 'datetime':
                # 여러 날짜 형식 시도
                for date_format in [None, '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y년 %m월 %d일']:
                    try:
                        if date_format:
                            converted = pd.to_datetime(series, format=date_format, errors='coerce')
                        else:
                            converted = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
                        
                        success_rate = converted.notna().sum() / series.notna().sum()
                        if success_rate > 0.5:  # 50% 이상 성공
                            result['converted_series'] = converted
                            result['success'] = True
                            result['conversion_stats']['format'] = date_format or 'auto'
                            break
                    except:
                        continue
                        
        except Exception as e:
            result['errors'].append(str(e))
        
        return result['converted_series'], result