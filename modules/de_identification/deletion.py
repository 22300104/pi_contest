import pandas as pd
import numpy as np
import re
from typing import Literal, Union, List, Dict, Any, Tuple


class DeletionProcessor:
    """데이터 부분 삭제 처리를 담당하는 클래스 (최적화 버전)"""
    
    @staticmethod
    def delete_column(
        df: pd.DataFrame,
        column_name: str,
        deletion_type: str,
        **kwargs
    ) -> pd.Series:
        """
        데이터프레임의 특정 컬럼에 부분 삭제 적용
        
        Args:
            df: 처리할 데이터프레임
            column_name: 처리할 컬럼명
            deletion_type: 삭제 유형
            **kwargs: 삭제 타입별 추가 파라미터
            
        Returns:
            삭제 처리된 컬럼 데이터
        """
        if column_name not in df.columns:
            raise ValueError(f"컬럼 '{column_name}'을 찾을 수 없습니다.")
        
        column_data = df[column_name].copy()
        
        # NULL 값 인덱스 저장 (성능 최적화)
        null_mask = column_data.isna()
        
        # 문자열로 변환
        str_column = column_data.astype(str)
        
        # 삭제 타입별 처리
        if deletion_type == "delimiter":
            result = DeletionProcessor._delimiter_deletion(str_column, **kwargs)
        elif deletion_type == "position":
            result = DeletionProcessor._position_deletion(str_column, **kwargs)
        elif deletion_type == "condition":
            result = DeletionProcessor._condition_deletion(str_column, **kwargs)
        elif deletion_type == "smart":
            result = DeletionProcessor._smart_deletion(str_column, **kwargs)
        else:
            raise ValueError(f"알 수 없는 삭제 타입: {deletion_type}")
        
        # NULL 값 복원
        result[null_mask] = np.nan
        
        return result
    
    @staticmethod
    def _delimiter_deletion(
        series: pd.Series,
        delimiter: str = "-",
        keep_position: str = "left",
        occurrence: str = "first",
        keep_count: int = 1
    ) -> pd.Series:
        """구분자 기반 삭제"""
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            # 구분자가 없으면 원본 반환
            if delimiter not in val_str:
                return val_str
            
            parts = val_str.split(delimiter)
            
            if occurrence == "all":
                # 모든 구분자 기준
                if keep_position == "left":
                    return delimiter.join(parts[:keep_count])
                elif keep_position == "right":
                    return delimiter.join(parts[-keep_count:])
                elif keep_position == "middle":
                    if len(parts) <= 2:
                        return val_str
                    start = (len(parts) - keep_count) // 2
                    end = start + keep_count
                    return delimiter.join(parts[start:end])
            else:
                # 첫 번째 또는 마지막 구분자만
                if occurrence == "first":
                    if keep_position == "left":
                        return parts[0]
                    else:  # right
                        return delimiter.join(parts[1:])
                else:  # last
                    if keep_position == "left":
                        return delimiter.join(parts[:-1])
                    else:  # right
                        return parts[-1]
            
            return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _position_deletion(
        series: pd.Series,
        unit: str = "character",
        mode: str = "simple",
        **params
    ) -> pd.Series:
        """위치/범위 기반 삭제 (통합 버전)"""
        
        # 단위별 처리 함수 선택
        if unit == "character":
            return DeletionProcessor._position_char_deletion(series, mode, **params)
        else:  # word
            return DeletionProcessor._position_word_deletion(series, mode, **params)
    
    @staticmethod
    def _position_char_deletion(series: pd.Series, mode: str, **params) -> pd.Series:
        """글자 단위 위치 삭제"""
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            val_len = len(val_str)
            
            if mode == "simple":
                # 단순 앞/뒤/양쪽 삭제
                position = params.get('position', 'front')
                count = params.get('count', 3)
                preserve_minimum = params.get('preserve_minimum', 1)
                
                if val_len <= preserve_minimum:
                    return val_str
                
                max_delete = val_len - preserve_minimum
                actual_delete = min(count, max_delete)
                
                if position == "front":
                    return val_str[actual_delete:]
                elif position == "back":
                    return val_str[:-actual_delete]
                elif position == "both":
                    delete_each = actual_delete // 2
                    return val_str[delete_each:-delete_each] if delete_each > 0 else val_str
                    
            elif mode == "specific":
                # 특정 위치 삭제/유지
                positions = params.get('positions', [])
                operation = params.get('operation', 'delete')
                
                if not positions:
                    return val_str
                
                result_chars = []
                for i, char in enumerate(val_str):
                    pos = i + 1  # 1-based
                    if operation == 'delete':
                        if pos not in positions:
                            result_chars.append(char)
                    else:  # keep
                        if pos in positions:
                            result_chars.append(char)
                
                return ''.join(result_chars)
                
            elif mode == "range":
                # 범위 삭제/유지
                ranges = params.get('ranges', [])
                operation = params.get('operation', 'delete')
                
                if not ranges:
                    return val_str
                
                result_chars = []
                for i, char in enumerate(val_str):
                    pos = i + 1  # 1-based
                    in_range = any(start <= pos <= end for start, end in ranges)
                    
                    if operation == 'delete':
                        if not in_range:
                            result_chars.append(char)
                    else:  # keep
                        if in_range:
                            result_chars.append(char)
                
                return ''.join(result_chars)
                
            elif mode == "interval":
                # 간격 삭제
                interval_type = params.get('interval_type', 'every_n')
                
                if interval_type == 'every_n':
                    n = params.get('n', 2)
                    offset = params.get('offset', 0)
                    result_chars = []
                    
                    for i, char in enumerate(val_str):
                        if (i - offset) % n != 0:
                            result_chars.append(char)
                    
                    return ''.join(result_chars)
                    
                elif interval_type == 'odd_even':
                    keep_odd = params.get('keep_odd', True)
                    result_chars = []
                    
                    for i, char in enumerate(val_str):
                        if (i % 2 == 0) == keep_odd:
                            result_chars.append(char)
                    
                    return ''.join(result_chars)
            
            return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _position_word_deletion(series: pd.Series, mode: str, **params) -> pd.Series:
        """단어 단위 위치 삭제"""
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            words = val_str.split()
            
            if len(words) == 0:
                return val_str
            
            if mode == "simple":
                # 단순 앞/뒤/양쪽 삭제
                position = params.get('position', 'front')
                count = params.get('count', 1)
                preserve_minimum = params.get('preserve_minimum', 1)
                
                if len(words) <= preserve_minimum:
                    return val_str
                
                max_delete = len(words) - preserve_minimum
                actual_delete = min(count, max_delete)
                
                if position == "front":
                    return ' '.join(words[actual_delete:])
                elif position == "back":
                    return ' '.join(words[:-actual_delete])
                elif position == "both":
                    front_delete = params.get('front_count', actual_delete // 2)
                    back_delete = params.get('back_count', actual_delete // 2)
                    
                    if len(words) <= front_delete + back_delete:
                        return ''
                    
                    if back_delete > 0:
                        return ' '.join(words[front_delete:-back_delete])
                    else:
                        return ' '.join(words[front_delete:])
                        
            elif mode == "specific":
                # 특정 위치 삭제/유지
                positions = params.get('positions', [])
                operation = params.get('operation', 'delete')
                
                if not positions:
                    return val_str
                
                result_words = []
                for i, word in enumerate(words):
                    pos = i + 1  # 1-based
                    if operation == 'delete':
                        if pos not in positions:
                            result_words.append(word)
                    else:  # keep
                        if pos in positions:
                            result_words.append(word)
                
                return ' '.join(result_words)
                
            elif mode == "range":
                # 범위 삭제/유지
                ranges = params.get('ranges', [])
                operation = params.get('operation', 'delete')
                
                if not ranges:
                    return val_str
                
                result_words = []
                for i, word in enumerate(words):
                    pos = i + 1  # 1-based
                    in_range = any(start <= pos <= end for start, end in ranges)
                    
                    if operation == 'delete':
                        if not in_range:
                            result_words.append(word)
                    else:  # keep
                        if in_range:
                            result_words.append(word)
                
                return ' '.join(result_words)
                
            elif mode == "interval":
                # 간격 삭제
                interval_type = params.get('interval_type', 'every_n')
                
                if interval_type == 'every_n':
                    n = params.get('n', 2)
                    offset = params.get('offset', 0)
                    result_words = []
                    
                    for i, word in enumerate(words):
                        if (i - offset) % n != 0:
                            result_words.append(word)
                    
                    return ' '.join(result_words)
                    
                elif interval_type == 'odd_even':
                    keep_odd = params.get('keep_odd', True)
                    result_words = []
                    
                    for i, word in enumerate(words):
                        if (i % 2 == 0) == keep_odd:
                            result_words.append(word)
                    
                    return ' '.join(result_words)
                    
            elif mode == "important":
                # 중요 단어만 유지
                keep_count = params.get('keep_count', 2)
                criteria = params.get('criteria', 'first')
                
                if criteria == 'first':
                    return ' '.join(words[:keep_count])
                elif criteria == 'last':
                    return ' '.join(words[-keep_count:])
                elif criteria == 'longest':
                    # 가장 긴 단어들 유지
                    sorted_words = sorted(enumerate(words), key=lambda x: len(x[1]), reverse=True)
                    keep_indices = sorted([idx for idx, _ in sorted_words[:keep_count]])
                    return ' '.join([words[i] for i in keep_indices])
            
            return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _condition_deletion(
        series: pd.Series,
        condition_type: str,
        **params
    ) -> pd.Series:
        """조건 기반 삭제"""
        
        if condition_type == "length":
            return DeletionProcessor._length_condition_deletion(series, **params)
        elif condition_type == "pattern":
            return DeletionProcessor._pattern_condition_deletion(series, **params)
        elif condition_type == "char_type":
            return DeletionProcessor._char_type_deletion(series, **params)
        elif condition_type == "dictionary":
            return DeletionProcessor._dictionary_deletion(series, **params)
        else:
            raise ValueError(f"알 수 없는 조건 타입: {condition_type}")
    
    @staticmethod
    def _length_condition_deletion(series: pd.Series, **params) -> pd.Series:
        """길이 조건 삭제"""
        unit = params.get('unit', 'word')
        min_length = params.get('min_length', 0)
        max_length = params.get('max_length', float('inf'))
        
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if unit == 'word':
                words = val_str.split()
                keep_words = []
                
                for word in words:
                    word_len = len(word)
                    if word_len < min_length or word_len > max_length:
                        keep_words.append(word)
                
                return ' '.join(keep_words)
            else:  # character - 전체 문자열 길이 기준
                str_len = len(val_str)
                if min_length <= str_len <= max_length:
                    return ''  # 조건에 맞으면 전체 삭제
                return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _pattern_condition_deletion(series: pd.Series, **params) -> pd.Series:
        """패턴 조건 삭제"""
        pattern = params.get('pattern', r'\d+')
        delete_matched = params.get('delete_matched', True)
        max_deletions = params.get('max_deletions', -1)
        unit = params.get('unit', 'match')  # match or word
        
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            try:
                if unit == 'match':
                    # 매칭된 부분만 처리
                    if delete_matched:
                        if max_deletions == -1:
                            return re.sub(pattern, '', val_str)
                        else:
                            return re.sub(pattern, '', val_str, count=max_deletions)
                    else:
                        # 매칭되지 않은 부분 삭제
                        matches = re.findall(pattern, val_str)
                        return ''.join(matches[:max_deletions] if max_deletions > 0 else matches)
                else:  # word
                    # 단어 단위로 패턴 확인
                    words = val_str.split()
                    result_words = []
                    
                    for word in words:
                        if delete_matched:
                            if not re.match(pattern, word):
                                result_words.append(word)
                        else:
                            if re.match(pattern, word):
                                result_words.append(word)
                    
                    return ' '.join(result_words)
            except:
                return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _char_type_deletion(series: pd.Series, **params) -> pd.Series:
        """문자 타입별 삭제"""
        char_type = params.get('char_type', 'digits')
        
        # 미리 컴파일된 패턴 (성능 최적화)
        patterns = {
            'digits': r'\d',
            'letters': r'[가-힣a-zA-Z]',
            'korean': r'[가-힣]',
            'english': r'[a-zA-Z]',
            'special': r'[^\w\s가-힣]',
            'whitespace': r'\s+'
        }
        
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if char_type in patterns:
                return re.sub(patterns[char_type], '', val_str)
            elif char_type == 'custom':
                chars_to_remove = params.get('characters', '')
                for char in chars_to_remove:
                    val_str = val_str.replace(char, '')
                return val_str
            elif char_type == 'except':
                keep_pattern = params.get('keep_pattern', r'[가-힣a-zA-Z0-9\s]')
                return ''.join(re.findall(keep_pattern, val_str))
            
            return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _dictionary_deletion(series: pd.Series, **params) -> pd.Series:
        """사전 기반 삭제"""
        dictionary = params.get('dictionary', [])
        case_sensitive = params.get('case_sensitive', False)
        unit = params.get('unit', 'word')
        
        if not dictionary:
            return series
        
        # 성능 최적화: set으로 변환
        if not case_sensitive:
            dictionary_set = set(word.lower() for word in dictionary)
        else:
            dictionary_set = set(dictionary)
        
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if unit == 'word':
                words = val_str.split()
                result_words = []
                
                for word in words:
                    check_word = word if case_sensitive else word.lower()
                    if check_word not in dictionary_set:
                        result_words.append(word)
                
                return ' '.join(result_words)
            else:  # exact match
                check_val = val_str if case_sensitive else val_str.lower()
                if check_val in dictionary_set:
                    return ''
                return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _smart_deletion(
        series: pd.Series,
        smart_type: str,
        **params
    ) -> pd.Series:
        """스마트 삭제 (데이터 타입 자동 인식)"""
        
        if smart_type == "auto_detect":
            return DeletionProcessor._auto_detect_deletion(series, **params)
        elif smart_type == "personal_info":
            return DeletionProcessor._personal_info_deletion(series, **params)
        elif smart_type == "redundant":
            return DeletionProcessor._redundant_deletion(series, **params)
        else:
            raise ValueError(f"알 수 없는 스마트 타입: {smart_type}")
    
    @staticmethod
    def _auto_detect_deletion(series: pd.Series, **params) -> pd.Series:
        """자동 감지 삭제"""
        # 미리 컴파일된 패턴들 (성능 최적화)
        email_pattern = re.compile(r'@[\w\.-]+')
        phone_pattern = re.compile(r'^[\d\-\s\(\)]+$')
        rrn_pattern = re.compile(r'^\d{6}[\-\s]?\d{7}$')
        date_pattern = re.compile(r'^\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2}')
        url_pattern = re.compile(r'https?://[^\s]+')
        
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            # 이메일
            if '@' in val_str and '.' in val_str:
                return email_pattern.sub('', val_str)
            
            # 전화번호
            elif phone_pattern.match(val_str) and len(re.findall(r'\d', val_str)) >= 7:
                digits = re.findall(r'\d', val_str)
                if len(digits) >= 11:  # 휴대폰
                    return ''.join(digits[:3]) + '-****-****'
                else:  # 일반 전화
                    return ''.join(digits[:3]) + '-****'
            
            # 주민번호
            elif rrn_pattern.match(val_str):
                return val_str[:6]
            
            # URL
            elif val_str.startswith(('http://', 'https://', 'www.')):
                return url_pattern.sub('', val_str)
            
            # 날짜
            elif date_pattern.match(val_str):
                return val_str[:4]  # 연도만
            
            # 주소
            elif any(keyword in val_str for keyword in ['시', '구', '동', '로', '길']):
                tokens = val_str.split()
                if len(tokens) > 2:
                    return ' '.join(tokens[:2])
            
            return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _personal_info_deletion(series: pd.Series, **params) -> pd.Series:
        """개인정보 수준별 삭제"""
        level = params.get('level', 'medium')
        
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if level == 'low':
                # 최소 삭제 - 민감한 부분만
                if '@' in val_str:  # 이메일
                    parts = val_str.split('@')
                    if len(parts[0]) > 3:
                        return parts[0][:3] + '***@' + parts[1]
                elif re.match(r'^\d{6}', val_str):  # 생년월일 형태
                    return val_str[:4] + '**'
                    
            elif level == 'medium':
                # 중간 수준
                if '@' in val_str:
                    return val_str.split('@')[0]
                elif re.match(r'^[\d\-]+$', val_str) and len(val_str) >= 10:
                    return val_str[:3] + '***'
                elif len(val_str) > 5:
                    return val_str[:3] + '***'
                    
            elif level == 'high':
                # 최대 삭제
                if '@' in val_str or re.match(r'^[\d\-]+$', val_str):
                    return '***'
                elif len(val_str) > 2:
                    return val_str[:2] + '***'
            
            return val_str
        
        return series.apply(delete_value)
    
    @staticmethod
    def _redundant_deletion(series: pd.Series, **params) -> pd.Series:
        """중복/불필요 정보 삭제"""
        remove_parentheses = params.get('remove_parentheses', True)
        remove_duplicates = params.get('remove_duplicates', True)
        trim_special = params.get('trim_special', True)
        
        def delete_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            # 연속된 동일 문자 제거
            if remove_duplicates:
                val_str = re.sub(r'(.)\1{2,}', r'\1\1', val_str)
            
            # 괄호 안 내용 제거
            if remove_parentheses:
                val_str = re.sub(r'\([^)]*\)', '', val_str)
                val_str = re.sub(r'\[[^\]]*\]', '', val_str)
                val_str = re.sub(r'\{[^}]*\}', '', val_str)
            
            # 앞뒤 불필요한 문자 제거
            if trim_special:
                val_str = val_str.strip(' -_.,;:!?')
            
            # 중복 공백 제거
            val_str = re.sub(r'\s+', ' ', val_str).strip()
            
            return val_str
        
        return series.apply(delete_value)
    
    # DeletionProcessor의 get_preview 메서드
    @staticmethod
    def get_preview(
        df: pd.DataFrame,
        column_name: str,
        deletion_type: str,
        sample_size: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        """삭제 결과 미리보기 (단순화 버전)"""
        try:
            # 유효한 데이터 선택
            valid_data = df[df[column_name].notna()][column_name]
            
            if len(valid_data) == 0:
                return pd.DataFrame({"원본": [], "결과": []})
            
            # 다양한 샘플 선택
            str_data = valid_data.astype(str)
            
            # 길이별로 다양한 샘플 선택
            lengths = str_data.str.len()
            unique_lengths = lengths.unique()
            
            samples = []
            
            # 다양한 길이의 샘플 추출
            for length in sorted(unique_lengths)[:sample_size]:
                length_samples = str_data[lengths == length].head(1)
                samples.extend(length_samples.tolist())
            
            # 부족하면 랜덤 샘플 추가
            if len(samples) < sample_size:
                remaining = min(sample_size - len(samples), len(valid_data))
                if remaining > 0:
                    additional = valid_data.sample(remaining, random_state=42)
                    samples.extend(additional.tolist())
            
            # 샘플 개수 조정
            samples = samples[:sample_size]
            
            # 샘플 데이터프레임 생성
            sample_df = pd.DataFrame({column_name: samples})
            
            # 삭제 적용
            try:
                deleted_values = DeletionProcessor.delete_column(
                    sample_df,
                    column_name,
                    deletion_type,
                    **kwargs
                )
                
                return pd.DataFrame({
                    "원본": samples,
                    "결과": deleted_values.values
                })
                
            except Exception as e:
                # 처리 중 오류 발생 시
                return pd.DataFrame({
                    "원본": samples,
                    "결과": ["오류: " + str(e)] * len(samples)
                })
                
        except Exception as e:
            # 전체 오류 발생 시
            return pd.DataFrame({
                "원본": ["오류 발생"],
                "결과": [str(e)]
            })
    
    