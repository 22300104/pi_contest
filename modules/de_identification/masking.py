import pandas as pd
import numpy as np
import re
from typing import Literal, Union, List, Dict, Any, Tuple


class MaskingProcessor:
    """데이터 마스킹 처리를 담당하는 클래스 (삭제와 동일한 구조)"""
    
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
        
        # NULL 값 인덱스 저장 (성능 최적화)
        null_mask = column_data.isna()
        
        # 문자열로 변환
        str_column = column_data.astype(str)
        
        # 마스킹 타입별 처리 (삭제와 동일한 구조)
        if masking_type == "delimiter":
            result = MaskingProcessor._delimiter_masking(str_column, **kwargs)
        elif masking_type == "position":
            result = MaskingProcessor._position_masking(str_column, **kwargs)
        elif masking_type == "condition":
            result = MaskingProcessor._condition_masking(str_column, **kwargs)
        elif masking_type == "smart":
            result = MaskingProcessor._smart_masking(str_column, **kwargs)
        else:
            raise ValueError(f"알 수 없는 마스킹 타입: {masking_type}")
        
        # NULL 값 복원
        result[null_mask] = np.nan
        
        return result
    
    @staticmethod
    def _delimiter_masking(
        series: pd.Series,
        delimiter: str = "-",
        keep_position: str = "left",
        occurrence: str = "first",
        keep_count: int = 1,
        mask_char: str = "*"
    ) -> pd.Series:
        """구분자 기반 마스킹"""
        def mask_value(val):
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
                    # 왼쪽 keep_count개 유지, 나머지 마스킹
                    masked_parts = parts[:keep_count] + [mask_char * len(part) for part in parts[keep_count:]]
                elif keep_position == "right":
                    # 오른쪽 keep_count개 유지, 나머지 마스킹
                    masked_parts = [mask_char * len(part) for part in parts[:-keep_count]] + parts[-keep_count:]
                elif keep_position == "middle":
                    if len(parts) <= 2:
                        return val_str
                    start = (len(parts) - keep_count) // 2
                    end = start + keep_count
                    masked_parts = []
                    for i, part in enumerate(parts):
                        if start <= i < end:
                            masked_parts.append(part)
                        else:
                            masked_parts.append(mask_char * len(part))
            else:
                # 첫 번째 또는 마지막 구분자만
                if occurrence == "first":
                    if keep_position == "left":
                        masked_parts = [parts[0]] + [mask_char * len(part) for part in parts[1:]]
                    else:  # right
                        masked_parts = [mask_char * len(parts[0])] + parts[1:]
                else:  # last
                    if keep_position == "left":
                        masked_parts = parts[:-1] + [mask_char * len(parts[-1])]
                    else:  # right
                        masked_parts = [mask_char * len(part) for part in parts[:-1]] + [parts[-1]]
            
            return delimiter.join(masked_parts)
        
        return series.apply(mask_value)
    
    @staticmethod
    def _position_masking(
        series: pd.Series,
        unit: str = "character",
        mode: str = "simple",
        mask_char: str = "*",
        **params
    ) -> pd.Series:
        """위치/범위 기반 마스킹 (통합 버전)"""
        
        # 단위별 처리 함수 선택
        if unit == "character":
            return MaskingProcessor._position_char_masking(series, mode, mask_char, **params)
        else:  # word
            return MaskingProcessor._position_word_masking(series, mode, mask_char, **params)
    
    @staticmethod
    def _position_char_masking(series: pd.Series, mode: str, mask_char: str, **params) -> pd.Series:
        """글자 단위 위치 마스킹"""
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            val_len = len(val_str)
            
            if mode == "simple":
                # 단순 앞/뒤/양쪽 마스킹
                position = params.get('position', 'front')
                count = params.get('count', 3)
                preserve_minimum = params.get('preserve_minimum', 1)
                
                if val_len <= preserve_minimum:
                    return val_str
                
                max_mask = val_len - preserve_minimum
                actual_mask = min(count, max_mask)
                
                if position == "front":
                    return mask_char * actual_mask + val_str[actual_mask:]
                elif position == "back":
                    return val_str[:-actual_mask] + mask_char * actual_mask
                elif position == "both":
                    mask_each = actual_mask // 2
                    if mask_each > 0:
                        return mask_char * mask_each + val_str[mask_each:-mask_each] + mask_char * mask_each
                    return val_str
                    
            elif mode == "specific":
                # 특정 위치 마스킹/보존
                positions = params.get('positions', [])
                operation = params.get('operation', 'mask')
                
                if not positions:
                    return val_str
                
                result_chars = []
                for i, char in enumerate(val_str):
                    pos = i + 1  # 1-based
                    if operation == 'mask':
                        if pos in positions:
                            result_chars.append(mask_char)
                        else:
                            result_chars.append(char)
                    else:  # preserve
                        if pos not in positions:
                            result_chars.append(mask_char)
                        else:
                            result_chars.append(char)
                
                return ''.join(result_chars)
                
            elif mode == "range":
                # 범위 마스킹/보존
                ranges = params.get('ranges', [])
                operation = params.get('operation', 'mask')
                
                if not ranges:
                    return val_str
                
                result_chars = []
                for i, char in enumerate(val_str):
                    pos = i + 1  # 1-based
                    in_range = any(start <= pos <= end for start, end in ranges)
                    
                    if operation == 'mask':
                        if in_range:
                            result_chars.append(mask_char)
                        else:
                            result_chars.append(char)
                    else:  # preserve
                        if not in_range:
                            result_chars.append(mask_char)
                        else:
                            result_chars.append(char)
                
                return ''.join(result_chars)
                
            elif mode == "interval":
                # 간격 마스킹
                interval_type = params.get('interval_type', 'every_n')
                
                if interval_type == 'every_n':
                    n = params.get('n', 2)
                    offset = params.get('offset', 0)
                    result_chars = []
                    
                    for i, char in enumerate(val_str):
                        if (i - offset) % n == 0:
                            result_chars.append(mask_char)
                        else:
                            result_chars.append(char)
                    
                    return ''.join(result_chars)
                    
                elif interval_type == 'odd_even':
                    keep_odd = params.get('keep_odd', True)
                    result_chars = []
                    
                    for i, char in enumerate(val_str):
                        if (i % 2 == 0) == keep_odd:
                            result_chars.append(char)
                        else:
                            result_chars.append(mask_char)
                    
                    return ''.join(result_chars)
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _position_word_masking(series: pd.Series, mode: str, mask_char: str, **params) -> pd.Series:
        """단어 단위 위치 마스킹"""
        # 마스킹 스타일 함수
        def apply_word_mask(word, style):
            if style == 'full':
                return mask_char * len(word)
            elif style == 'partial_front':
                if len(word) <= 2:
                    return word
                mask_len = len(word) // 2
                return mask_char * mask_len + word[mask_len:]
            elif style == 'partial_back':
                if len(word) <= 2:
                    return word
                mask_len = len(word) // 2
                return word[:-mask_len] + mask_char * mask_len
            elif style == 'keep_first':
                if len(word) <= 1:
                    return word
                return word[0] + mask_char * (len(word) - 1)
            elif style == 'keep_last':
                if len(word) <= 1:
                    return word
                return mask_char * (len(word) - 1) + word[-1]
            elif style == 'keep_edges':
                if len(word) <= 2:
                    return word
                return word[0] + mask_char * (len(word) - 2) + word[-1]
            else:
                return word
        
        mask_style = params.get('mask_style', 'full')
        
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            words = val_str.split()
            
            if len(words) == 0:
                return val_str
            
            if mode == "simple":
                # 단순 앞/뒤/양쪽 마스킹
                position = params.get('position', 'front')
                count = params.get('count', 1)
                preserve_minimum = params.get('preserve_minimum', 1)
                
                if len(words) <= preserve_minimum:
                    return val_str
                
                max_mask = len(words) - preserve_minimum
                actual_mask = min(count, max_mask)
                
                masked_words = words.copy()
                
                if position == "front":
                    for i in range(actual_mask):
                        masked_words[i] = apply_word_mask(words[i], mask_style)
                elif position == "back":
                    for i in range(len(words) - actual_mask, len(words)):
                        masked_words[i] = apply_word_mask(words[i], mask_style)
                elif position == "both":
                    front_count = params.get('front_count', actual_mask // 2)
                    back_count = params.get('back_count', actual_mask // 2)
                    
                    for i in range(min(front_count, len(words))):
                        masked_words[i] = apply_word_mask(words[i], mask_style)
                    for i in range(max(0, len(words) - back_count), len(words)):
                        masked_words[i] = apply_word_mask(words[i], mask_style)
                
                return ' '.join(masked_words)
                        
            elif mode == "specific":
                # 특정 위치 마스킹/보존
                positions = params.get('positions', [])
                operation = params.get('operation', 'mask')
                
                if not positions:
                    return val_str
                
                masked_words = []
                for i, word in enumerate(words):
                    pos = i + 1  # 1-based
                    if operation == 'mask':
                        if pos in positions:
                            masked_words.append(apply_word_mask(word, mask_style))
                        else:
                            masked_words.append(word)
                    else:  # preserve
                        if pos not in positions:
                            masked_words.append(apply_word_mask(word, mask_style))
                        else:
                            masked_words.append(word)
                
                return ' '.join(masked_words)
                
            elif mode == "range":
                # 범위 마스킹/보존
                ranges = params.get('ranges', [])
                operation = params.get('operation', 'mask')
                
                if not ranges:
                    return val_str
                
                masked_words = []
                for i, word in enumerate(words):
                    pos = i + 1  # 1-based
                    in_range = any(start <= pos <= end for start, end in ranges)
                    
                    if operation == 'mask':
                        if in_range:
                            masked_words.append(apply_word_mask(word, mask_style))
                        else:
                            masked_words.append(word)
                    else:  # preserve
                        if not in_range:
                            masked_words.append(apply_word_mask(word, mask_style))
                        else:
                            masked_words.append(word)
                
                return ' '.join(masked_words)
                
            elif mode == "interval":
                # 간격 마스킹
                interval_type = params.get('interval_type', 'every_n')
                
                masked_words = []
                if interval_type == 'every_n':
                    n = params.get('n', 2)
                    offset = params.get('offset', 0)
                    
                    for i, word in enumerate(words):
                        if (i - offset) % n == 0:
                            masked_words.append(apply_word_mask(word, mask_style))
                        else:
                            masked_words.append(word)
                    
                elif interval_type == 'odd_even':
                    keep_odd = params.get('keep_odd', True)
                    
                    for i, word in enumerate(words):
                        if (i % 2 == 0) == keep_odd:
                            masked_words.append(word)
                        else:
                            masked_words.append(apply_word_mask(word, mask_style))
                
                return ' '.join(masked_words)
                
            elif mode == "important":
                # 중요 단어만 유지 (나머지 마스킹)
                keep_count = params.get('keep_count', 2)
                criteria = params.get('criteria', 'first')
                
                masked_words = []
                if criteria == 'first':
                    for i, word in enumerate(words):
                        if i < keep_count:
                            masked_words.append(word)
                        else:
                            masked_words.append(apply_word_mask(word, mask_style))
                elif criteria == 'last':
                    for i, word in enumerate(words):
                        if i >= len(words) - keep_count:
                            masked_words.append(word)
                        else:
                            masked_words.append(apply_word_mask(word, mask_style))
                elif criteria == 'longest':
                    # 가장 긴 단어들 유지
                    word_lengths = [(i, len(word)) for i, word in enumerate(words)]
                    word_lengths.sort(key=lambda x: x[1], reverse=True)
                    keep_indices = set(idx for idx, _ in word_lengths[:keep_count])
                    
                    for i, word in enumerate(words):
                        if i in keep_indices:
                            masked_words.append(word)
                        else:
                            masked_words.append(apply_word_mask(word, mask_style))
                
                return ' '.join(masked_words)
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _condition_masking(
        series: pd.Series,
        condition_type: str,
        mask_char: str = "*",
        **params
    ) -> pd.Series:
        """조건 기반 마스킹"""
        
        if condition_type == "length":
            return MaskingProcessor._length_condition_masking(series, mask_char, **params)
        elif condition_type == "pattern":
            return MaskingProcessor._pattern_condition_masking(series, mask_char, **params)
        elif condition_type == "char_type":
            return MaskingProcessor._char_type_masking(series, mask_char, **params)
        elif condition_type == "dictionary":
            return MaskingProcessor._dictionary_masking(series, mask_char, **params)
        else:
            raise ValueError(f"알 수 없는 조건 타입: {condition_type}")
    
    @staticmethod
    def _length_condition_masking(series: pd.Series, mask_char: str, **params) -> pd.Series:
        """길이 조건 마스킹"""
        unit = params.get('unit', 'word')
        min_length = params.get('min_length', 0)
        max_length = params.get('max_length', float('inf'))
        
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if unit == 'word':
                words = val_str.split()
                masked_words = []
                
                for word in words:
                    word_len = len(word)
                    if min_length <= word_len <= max_length:
                        # 조건에 맞는 단어 마스킹
                        masked_words.append(mask_char * word_len)
                    else:
                        masked_words.append(word)
                
                return ' '.join(masked_words)
            else:  # character - 전체 문자열 길이 기준
                str_len = len(val_str)
                if min_length <= str_len <= max_length:
                    # 조건에 맞으면 전체 마스킹
                    return mask_char * str_len
                return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _pattern_condition_masking(series: pd.Series, mask_char: str, **params) -> pd.Series:
        """패턴 조건 마스킹"""
        pattern = params.get('pattern', r'\d+')
        mask_matched = params.get('mask_matched', True)
        max_masks = params.get('max_masks', -1)
        unit = params.get('unit', 'match')  # match or word
        
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            try:
                if unit == 'match':
                    # 매칭된 부분만 처리
                    if mask_matched:
                        # 매칭된 부분을 마스킹
                        if max_masks == -1:
                            return re.sub(pattern, lambda m: mask_char * len(m.group()), val_str)
                        else:
                            count = 0
                            def replace_func(match):
                                nonlocal count
                                if count < max_masks:
                                    count += 1
                                    return mask_char * len(match.group())
                                return match.group()
                            return re.sub(pattern, replace_func, val_str)
                    else:
                        # 매칭되지 않은 부분 마스킹 (복잡하므로 간단히 구현)
                        matches = list(re.finditer(pattern, val_str))
                        if not matches:
                            return mask_char * len(val_str)
                        
                        result = []
                        last_end = 0
                        
                        for match in matches[:max_masks] if max_masks > 0 else matches:
                            # 매치 이전 부분 마스킹
                            if match.start() > last_end:
                                result.append(mask_char * (match.start() - last_end))
                            # 매치 부분은 유지
                            result.append(match.group())
                            last_end = match.end()
                        
                        # 마지막 매치 이후 부분 마스킹
                        if last_end < len(val_str):
                            result.append(mask_char * (len(val_str) - last_end))
                        
                        return ''.join(result)
                else:  # word
                    # 단어 단위로 패턴 확인
                    words = val_str.split()
                    masked_words = []
                    
                    for word in words:
                        if mask_matched:
                            if re.match(pattern, word):
                                masked_words.append(mask_char * len(word))
                            else:
                                masked_words.append(word)
                        else:
                            if not re.match(pattern, word):
                                masked_words.append(mask_char * len(word))
                            else:
                                masked_words.append(word)
                    
                    return ' '.join(masked_words)
            except:
                return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _char_type_masking(series: pd.Series, mask_char: str, **params) -> pd.Series:
        """문자 타입별 마스킹"""
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
        
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if char_type in patterns:
                return re.sub(patterns[char_type], mask_char, val_str)
            elif char_type == 'custom':
                chars_to_mask = params.get('characters', '')
                for char in chars_to_mask:
                    val_str = val_str.replace(char, mask_char)
                return val_str
            elif char_type == 'except':
                keep_pattern = params.get('keep_pattern', r'[가-힣a-zA-Z0-9\s]')
                # 지정된 패턴 외의 문자를 마스킹
                result = []
                for char in val_str:
                    if re.match(keep_pattern, char):
                        result.append(char)
                    else:
                        result.append(mask_char)
                return ''.join(result)
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _dictionary_masking(series: pd.Series, mask_char: str, **params) -> pd.Series:
        """사전 기반 마스킹"""
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
        
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if unit == 'word':
                words = val_str.split()
                masked_words = []
                
                for word in words:
                    check_word = word if case_sensitive else word.lower()
                    if check_word in dictionary_set:
                        masked_words.append(mask_char * len(word))
                    else:
                        masked_words.append(word)
                
                return ' '.join(masked_words)
            else:  # exact match
                check_val = val_str if case_sensitive else val_str.lower()
                if check_val in dictionary_set:
                    return mask_char * len(val_str)
                return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _smart_masking(
        series: pd.Series,
        smart_type: str,
        mask_char: str = "*",
        **params
    ) -> pd.Series:
        """스마트 마스킹 (데이터 타입 자동 인식)"""
        
        if smart_type == "auto_detect":
            return MaskingProcessor._auto_detect_masking(series, mask_char, **params)
        elif smart_type == "personal_info":
            return MaskingProcessor._personal_info_masking(series, mask_char, **params)
        elif smart_type == "redundant":
            return MaskingProcessor._redundant_masking(series, mask_char, **params)
        else:
            raise ValueError(f"알 수 없는 스마트 타입: {smart_type}")
    
    @staticmethod
    def _auto_detect_masking(series: pd.Series, mask_char: str, **params) -> pd.Series:
        """자동 감지 마스킹"""
        # 미리 컴파일된 패턴들 (성능 최적화)
        email_pattern = re.compile(r'([^@]+)@(.+)')
        phone_pattern = re.compile(r'^[\d\-\s\(\)]+$')
        rrn_pattern = re.compile(r'^(\d{6})[\-\s]?(\d{7})$')
        date_pattern = re.compile(r'^(\d{4})[\-/\.](\d{1,2})[\-/\.](\d{1,2})')
        url_pattern = re.compile(r'(https?://[^/]+)(.*)')
        
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            # 이메일
            email_match = email_pattern.match(val_str)
            if email_match:
                local, domain = email_match.groups()
                if len(local) > 0:
                    masked_local = local[0] + mask_char * (len(local) - 1)
                    return f"{masked_local}@{domain}"
            
            # 전화번호
            elif phone_pattern.match(val_str) and len(re.findall(r'\d', val_str)) >= 7:
                digits = re.findall(r'\d', val_str)
                if len(digits) >= 11:  # 휴대폰
                    # 형식 유지하며 마스킹
                    return re.sub(r'\d', mask_char, val_str)
                else:  # 일반 전화
                    # 뒷자리만 마스킹
                    return re.sub(r'\d{4}$', mask_char * 4, val_str)
            
            # 주민번호
            rrn_match = rrn_pattern.match(val_str)
            if rrn_match:
                birth, back = rrn_match.groups()
                if '-' in val_str or ' ' in val_str:
                    sep = '-' if '-' in val_str else ' '
                    return birth + sep + back[0] + mask_char * 6
                else:
                    return birth + back[0] + mask_char * 6
            
            # URL
            url_match = url_pattern.match(val_str)
            if url_match:
                domain, path = url_match.groups()
                return domain + mask_char * min(len(path), 10)
            
            # 날짜
            date_match = date_pattern.match(val_str)
            if date_match:
                year = date_match.group(1)
                return year + mask_char * (len(val_str) - 4)
            
            # 주소
            elif any(keyword in val_str for keyword in ['시', '구', '동', '로', '길']):
                tokens = val_str.split()
                if len(tokens) > 2:
                    # 상세 주소 마스킹
                    masked_tokens = tokens[:2]
                    for token in tokens[2:]:
                        if re.search(r'\d', token):  # 숫자가 포함된 경우
                            masked_tokens.append(mask_char * len(token))
                        else:
                            masked_tokens.append(token)
                    return ' '.join(masked_tokens)
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _personal_info_masking(series: pd.Series, mask_char: str, **params) -> pd.Series:
        """개인정보 수준별 마스킹"""
        level = params.get('level', 'medium')
        
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            if level == 'low':
                # 최소 마스킹 - 민감한 부분만
                if '@' in val_str:  # 이메일
                    parts = val_str.split('@')
                    if len(parts[0]) > 3:
                        return parts[0][:3] + mask_char * (len(parts[0]) - 3) + '@' + parts[1]
                    else:
                        return parts[0][0] + mask_char * (len(parts[0]) - 1) + '@' + parts[1]
                elif re.match(r'^\d{6}', val_str):  # 생년월일 형태
                    return val_str[:4] + mask_char * 2 + val_str[6:]
                elif len(val_str) > 5:
                    # 일반 텍스트: 뒷부분만 마스킹
                    mask_len = len(val_str) // 3
                    return val_str[:-mask_len] + mask_char * mask_len
                    
            elif level == 'medium':
                # 중간 수준
                if '@' in val_str:
                    parts = val_str.split('@')
                    return parts[0][0] + mask_char * (len(parts[0]) - 1) + '@' + parts[1]
                elif re.match(r'^[\d\-]+$', val_str) and len(val_str) >= 10:
                    # 전화번호나 번호 형태
                    return val_str[:3] + mask_char * (len(val_str) - 7) + val_str[-4:]
                elif len(val_str) > 3:
                    # 일반 텍스트: 중간 부분 마스킹
                    preserve = max(1, len(val_str) // 4)
                    return val_str[:preserve] + mask_char * (len(val_str) - preserve * 2) + val_str[-preserve:]
                    
            elif level == 'high':
                # 최대 마스킹
                if '@' in val_str:
                    # 이메일: 첫 글자와 도메인 일부만
                    parts = val_str.split('@')
                    domain_parts = parts[1].split('.')
                    return parts[0][0] + mask_char * (len(parts[0]) - 1) + '@' + mask_char * len(domain_parts[0]) + '.' + '.'.join(domain_parts[1:])
                elif re.match(r'^[\d\-]+$', val_str):
                    # 숫자: 거의 전체 마스킹
                    return mask_char * (len(val_str) - 2) + val_str[-2:]
                elif len(val_str) > 2:
                    # 일반 텍스트: 첫 글자만
                    return val_str[0] + mask_char * (len(val_str) - 1)
            
            return val_str
        
        return series.apply(mask_value)
    
    @staticmethod
    def _redundant_masking(series: pd.Series, mask_char: str, **params) -> pd.Series:
        """중복/불필요 정보 마스킹"""
        mask_parentheses = params.get('mask_parentheses', True)
        mask_duplicates = params.get('mask_duplicates', True)
        mask_special = params.get('mask_special', True)
        
        def mask_value(val):
            if pd.isna(val) or val == 'nan':
                return val
            
            val_str = str(val)
            
            # 괄호 안 내용 마스킹
            if mask_parentheses:
                # 괄호와 내용을 마스킹 문자로 대체
                val_str = re.sub(r'\([^)]*\)', lambda m: '(' + mask_char * (len(m.group()) - 2) + ')', val_str)
                val_str = re.sub(r'\[[^\]]*\]', lambda m: '[' + mask_char * (len(m.group()) - 2) + ']', val_str)
                val_str = re.sub(r'\{[^}]*\}', lambda m: '{' + mask_char * (len(m.group()) - 2) + '}', val_str)
            
            # 연속된 동일 문자 마스킹
            if mask_duplicates:
                # 3번 이상 연속된 문자를 2개 + 마스킹으로
                val_str = re.sub(r'(.)\1{2,}', lambda m: m.group()[0:2] + mask_char * (len(m.group()) - 2), val_str)
            
            # 특수문자 마스킹
            if mask_special:
                # 앞뒤 특수문자만 마스킹
                # 앞쪽 특수문자
                match = re.match(r'^([^\w\s가-힣]+)', val_str)
                if match:
                    val_str = mask_char * len(match.group()) + val_str[len(match.group()):]
                
                # 뒤쪽 특수문자
                match = re.search(r'([^\w\s가-힣]+)$', val_str)
                if match:
                    val_str = val_str[:-len(match.group())] + mask_char * len(match.group())
            
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
        """마스킹 결과 미리보기 (단순화 버전)"""
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
            
            # 마스킹 적용
            try:
                masked_values = MaskingProcessor.mask_column(
                    sample_df,
                    column_name,
                    masking_type,
                    **kwargs
                )
                
                return pd.DataFrame({
                    "원본": samples,
                    "결과": masked_values.values
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