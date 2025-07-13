# utility_metrics.py (알고리즘 준수 개선 버전)

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime
import warnings


class UtilityMetrics:
    """데이터 유용성 평가 지표 계산 클래스 - 알고리즘 준수 버전"""
    
    def __init__(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame):
        """
        Args:
            original_df: 원본 데이터프레임
            anonymized_df: 비식별화된 데이터프레임
        """
        self.original_df = original_df
        self.anonymized_df = anonymized_df
        self.results_cache = {}
        
        # 데이터 검증
        self._validate_dataframes()
    
    def _validate_dataframes(self):
        """데이터프레임 검증"""
        if len(self.original_df) == 0 or len(self.anonymized_df) == 0:
            raise ValueError("빈 데이터프레임은 평가할 수 없습니다.")
            
        # 공통 컬럼 확인
        common_columns = set(self.original_df.columns) & set(self.anonymized_df.columns)
        if len(common_columns) == 0:
            raise ValueError("원본과 비식별 데이터에 공통 컬럼이 없습니다.")
    
    def get_applicable_metrics(self, columns: List[str]) -> Dict[str, bool]:
        """선택된 컬럼들에 대해 적용 가능한 평가 지표 반환"""
        applicable = {
            'U1': False, 'U2': False, 'U3': False, 'U4': False, 'U5': False,
            'U6': False, 'U7': False, 'U8': False, 'U9': True  # U9는 항상 가능
        }
        
        if not columns:
            return applicable
        
        # 컬럼 타입 확인
        numeric_cols = []
        for col in columns:
            if col in self.original_df.columns:
                if pd.api.types.is_numeric_dtype(self.original_df[col]):
                    numeric_cols.append(col)
                elif pd.api.types.is_datetime64_any_dtype(self.original_df[col]):
                    numeric_cols.append(col)  # 날짜도 수치로 변환 가능
        
        # 수치형 컬럼이 있으면
        if numeric_cols:
            applicable['U1'] = True  # 평균값 차이
            applicable['U3'] = True  # 코사인 유사도
            applicable['U4'] = True  # 정규화 유클리디안
            applicable['U5'] = True  # 표준화 유클리디안
            applicable['U8'] = True  # 비균일 엔트로피
            
            # 2개 이상이면 상관관계도 가능
            if len(numeric_cols) >= 2:
                applicable['U2'] = True
        
        # U6, U7은 k-익명성 관련 (준식별자가 있어야 함)
        # 일단 모든 컬럼에 대해 가능하다고 표시
        applicable['U6'] = True
        applicable['U7'] = True
        applicable['U8'] = True  # 모든 타입에 적용 가능
        
        return applicable
    
    # ===== U1: Mean Attribute (MA) =====
    def calculate_u1_ma(self, columns: List[str]) -> Dict[str, Any]:
        """
        U1: 평균값 차이 계산
        알고리즘: |AVG_X - AVG_Y| 값들의 총합
        """
        results = {
            'metric': 'U1',
            'name': '평균값 차이 (MA)',
            'column_results': {},
            'total_score': 0.0,
            'status': 'success'
        }
        
        total_diff = 0.0
        valid_cols = 0
        
        for col in columns:
            try:
                # 수치형으로 변환 시도
                if pd.api.types.is_datetime64_any_dtype(self.original_df[col]):
                    # 날짜는 timestamp로 변환 (알고리즘 명시)
                    orig_values = pd.to_datetime(self.original_df[col]).astype(np.int64) / 10**9
                    anon_values = pd.to_datetime(self.anonymized_df[col]).astype(np.int64) / 10**9
                else:
                    orig_values = pd.to_numeric(self.original_df[col], errors='coerce')
                    anon_values = pd.to_numeric(self.anonymized_df[col], errors='coerce')
                
                # NaN 제거 후 평균 계산
                orig_mean = orig_values.dropna().mean()
                anon_mean = anon_values.dropna().mean()
                
                # |AVG_X - AVG_Y| 계산
                diff = abs(orig_mean - anon_mean)
                
                results['column_results'][col] = {
                    'original_mean': orig_mean,
                    'anonymized_mean': anon_mean,
                    'difference': diff
                }
                
                total_diff += diff
                valid_cols += 1
                
            except Exception as e:
                results['column_results'][col] = {
                    'error': str(e)
                }
        
        if valid_cols > 0:
            # 알고리즘: j>=2인 경우 총합을 최종 점수로
            results['total_score'] = total_diff
        else:
            results['status'] = 'error'
            results['error'] = '계산 가능한 컬럼이 없습니다.'
        
        return results
    
    # ===== U2: Mean Correlation (MC) =====
    def calculate_u2_mc(self, columns: List[str]) -> Dict[str, Any]:
        """
        U2: 상관관계 보존도 계산
        알고리즘: |cor(Xi,Xj) - cor(Yi,Yj)|의 합 / sa2
        여기서 sa2 = C(속성수, 2) = n*(n-1)/2
        """
        results = {
            'metric': 'U2',
            'name': '상관관계 보존 (MC)',
            'pair_results': {},
            'total_score': 0.0,
            'status': 'success'
        }
        
        if len(columns) < 2:
            results['status'] = 'error'
            results['error'] = '최소 2개 이상의 컬럼이 필요합니다.'
            return results
        
        # 수치형 컬럼만 필터링
        numeric_cols = []
        for col in columns:
            try:
                pd.to_numeric(self.original_df[col], errors='coerce')
                numeric_cols.append(col)
            except:
                pass
        
        if len(numeric_cols) < 2:
            results['status'] = 'error'
            results['error'] = '수치형 컬럼이 2개 이상 필요합니다.'
            return results
        
        # 상관계수 계산
        orig_num = self.original_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        anon_num = self.anonymized_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        orig_corr = orig_num.corr()
        anon_corr = anon_num.corr()
        
        total_diff = 0.0
        pair_count = 0
        
        # 모든 컬럼 쌍에 대해 계산
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                
                orig_val = orig_corr.loc[col1, col2]
                anon_val = anon_corr.loc[col1, col2]
                
                # |cor(Xi,Xj) - cor(Yi,Yj)| 계산
                diff = abs(orig_val - anon_val)
                
                results['pair_results'][f"{col1}-{col2}"] = {
                    'original_corr': orig_val,
                    'anonymized_corr': anon_val,
                    'difference': diff
                }
                
                total_diff += diff
                pair_count += 1
        
        # sa2 = C(n,2) = n*(n-1)/2
        sa2 = len(numeric_cols) * (len(numeric_cols) - 1) / 2
        
        # 알고리즘: 전체 합을 sa2로 나눔
        results['total_score'] = total_diff / sa2 if sa2 > 0 else 0
        results['sa2'] = sa2
        results['total_difference'] = total_diff
        
        return results
    
    # ===== U3: Cosine Similarity (CS) =====
    def calculate_u3_cs(self, columns: List[str]) -> Dict[str, Any]:
        """
        U3: 코사인 유사도 계산
        알고리즘: 각 속성별 코사인 유사도를 계산하고 평균
        """
        results = {
            'metric': 'U3',
            'name': '코사인 유사도 (CS)',
            'column_results': {},
            'average_score': 0.0,
            'status': 'success'
        }
        
        valid_scores = []
        
        for col in columns:
            try:
                # 수치형으로 변환
                orig_values = pd.to_numeric(self.original_df[col], errors='coerce').dropna()
                anon_values = pd.to_numeric(self.anonymized_df[col], errors='coerce').dropna()
                
                # 길이 맞추기 (인덱스 기준)
                common_idx = orig_values.index.intersection(anon_values.index)
                orig_vec = orig_values.loc[common_idx].values
                anon_vec = anon_values.loc[common_idx].values
                
                # 코사인 유사도 계산: Xi·Yi / (||Xi|| * ||Yi||)
                dot_product = np.dot(orig_vec, anon_vec)
                norm_orig = np.linalg.norm(orig_vec)
                norm_anon = np.linalg.norm(anon_vec)
                
                if norm_orig > 0 and norm_anon > 0:
                    cosine_sim = dot_product / (norm_orig * norm_anon)
                else:
                    cosine_sim = 0.0
                
                results['column_results'][col] = {
                    'cosine_similarity': cosine_sim
                }
                valid_scores.append(cosine_sim)
                
            except Exception as e:
                results['column_results'][col] = {
                    'error': str(e)
                }
        
        if valid_scores:
            # 알고리즘: 여러 속성인 경우 평균
            results['average_score'] = np.mean(valid_scores)
        else:
            results['status'] = 'error'
            results['error'] = '계산 가능한 컬럼이 없습니다.'
        
        return results
    
    # ===== U4: Normalized Euclidian Distance (NED_SSE) =====
    def calculate_u4_ned(self, columns: List[str]) -> Dict[str, Any]:
        """
        U4: 정규화 유클리디안 거리 계산
        알고리즘: SSE 값들의 총합 (정규화는 알고리즘에 명시되지 않음)
        """
        results = {
            'metric': 'U4',
            'name': '정규화 유클리디안 거리 (NED_SSE)',
            'column_results': {},
            'total_score': 0.0,
            'status': 'success'
        }
        
        total_sse = 0.0
        
        for col in columns:
            try:
                # 수치형으로 변환
                orig_values = pd.to_numeric(self.original_df[col], errors='coerce')
                anon_values = pd.to_numeric(self.anonymized_df[col], errors='coerce')
                
                # 인덱스가 같은 레코드만 비교
                sse = 0.0
                count = 0
                
                for idx in self.original_df.index:
                    if idx in self.anonymized_df.index:
                        orig_val = orig_values.loc[idx]
                        anon_val = anon_values.loc[idx]
                        
                        if not pd.isna(orig_val) and not pd.isna(anon_val):
                            # SSE = Σ(Xi - Yi)²
                            sse += (orig_val - anon_val) ** 2
                            count += 1
                
                results['column_results'][col] = {
                    'sse': sse,
                    'normalized_sse': sse,  # 알고리즘에는 정규화 언급 없음
                    'record_count': count
                }
                
                total_sse += sse
                
            except Exception as e:
                results['column_results'][col] = {
                    'error': str(e)
                }
        
        # 알고리즘: SSE들의 총합이 최종 점수
        results['total_score'] = total_sse
        
        return results
    
    # ===== U5: Standardized Euclidian Distance (SED_SSE) =====
    def calculate_u5_sed(self, columns: List[str]) -> Dict[str, Any]:
        """
        U5: 표준화 유클리디안 거리 계산
        알고리즘: Σ((Xij - Yij) / σj)² 계산
        범주화된 경우 원본값과 가장 먼 값으로 대체
        """
        results = {
            'metric': 'U5',
            'name': '표준화 유클리디안 거리 (SED_SSE)',
            'column_results': {},
            'total_score': 0.0,
            'status': 'success'
        }
        
        total_sse = 0.0
        
        for col in columns:
            try:
                # 수치형으로 변환
                orig_values = pd.to_numeric(self.original_df[col], errors='coerce')
                anon_values = pd.to_numeric(self.anonymized_df[col], errors='coerce')
                
                # 표준편차 계산
                std_dev = orig_values.std()
                
                if std_dev > 0:
                    sse = 0.0
                    count = 0
                    
                    for idx in self.original_df.index:
                        if idx in self.anonymized_df.index:
                            orig_val = orig_values.loc[idx]
                            anon_val = anon_values.loc[idx]
                            
                            if not pd.isna(orig_val):
                                # 범주화된 경우 처리
                                if pd.isna(anon_val) or isinstance(self.anonymized_df[col].iloc[idx], str):
                                    # 알고리즘: "원본값과 가장 먼 값"으로 대체
                                    # 데이터 범위에서 가장 먼 값 계산
                                    data_min = orig_values.min()
                                    data_max = orig_values.max()
                                    
                                    # 원본값에서 더 먼 경계값 선택
                                    if abs(orig_val - data_min) > abs(orig_val - data_max):
                                        anon_val = data_min
                                    else:
                                        anon_val = data_max
                                
                                # ((Xij - Yij) / σj)²
                                sse += ((orig_val - anon_val) / std_dev) ** 2
                                count += 1
                    
                    results['column_results'][col] = {
                        'sse': sse,
                        'std_dev': std_dev,
                        'record_count': count
                    }
                    
                    total_sse += sse
                else:
                    results['column_results'][col] = {
                        'error': '표준편차가 0입니다.'
                    }
                
            except Exception as e:
                results['column_results'][col] = {
                    'error': str(e)
                }
        
        results['total_score'] = total_sse
        
        return results
    
    # ===== U6: Mean Distribution ECM (MD_ECM) =====
    def calculate_u6_md_ecm(self, quasi_identifiers: List[str], sensitive_attr: str) -> Dict[str, Any]:
        """
        U6: 동질집합 분산 계산
        알고리즘: 각 동질집합의 민감속성 분산을 계산하고 평균
        """
        results = {
            'metric': 'U6',
            'name': '동질집합 분산 (MD_ECM)',
            'total_score': 0.0,
            'status': 'success'
        }
        
        try:
            # 동질집합 생성
            if quasi_identifiers:
                ec_groups = self.anonymized_df.groupby(quasi_identifiers)
            else:
                results['status'] = 'error'
                results['error'] = '준식별자가 지정되지 않았습니다.'
                return results
            
            # 민감속성이 수치형인지 확인
            if sensitive_attr not in self.anonymized_df.columns:
                results['status'] = 'error'
                results['error'] = f'민감속성 {sensitive_attr}이 없습니다.'
                return results
            
            # 각 동질집합의 분산 계산
            variances = []
            ec_details = []
            
            for name, group in ec_groups:
                try:
                    # 수치형으로 변환
                    values = pd.to_numeric(group[sensitive_attr], errors='coerce').dropna()
                    
                    if len(values) > 1:
                        # 분산 계산
                        variance = values.var()
                        variances.append(variance)
                        
                        ec_details.append({
                            'ec_id': str(name),
                            'size': len(group),
                            'variance': variance
                        })
                except:
                    pass
            
            if variances:
                # 알고리즘: 전체 동질집합의 분산 평균
                results['total_score'] = np.mean(variances)
                results['ec_count'] = len(variances)
                results['ec_details'] = ec_details[:10]  # 상위 10개만
            else:
                results['status'] = 'error'
                results['error'] = '유효한 동질집합이 없습니다.'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    # ===== U7: Normalized Average ECSM (NA_ECSM) =====
    def calculate_u7_na_ecsm(self, quasi_identifiers: List[str]) -> Dict[str, Any]:
        """
        U7: 정규화 동질집합 크기 계산
        알고리즘: (N/N_EC)/k
        - N: 전체 레코드 수
        - N_EC: 동질집합 수
        - k: k-익명성의 대표 k값 (최소값)
        """
        results = {
            'metric': 'U7',
            'name': '정규화 집합크기 (NA_ECSM)',
            'total_score': 0.0,
            'status': 'success'
        }
        
        try:
            if quasi_identifiers:
                # 동질집합 생성
                ec_groups = self.anonymized_df.groupby(quasi_identifiers)
                ec_sizes = [len(group) for _, group in ec_groups]
                
                # 알고리즘 변수
                n = len(self.anonymized_df)  # 전체 레코드 수
                n_ec = len(ec_sizes)  # 동질집합 수
                k = min(ec_sizes) if ec_sizes else 1  # 대표 k값 (최소값)
                
                # (N/N_EC)/k 계산
                if n_ec > 0 and k > 0:
                    results['total_score'] = (n / n_ec) / k
                    results['details'] = {
                        'total_records': n,
                        'ec_count': n_ec,
                        'min_k': k,
                        'avg_ec_size': n / n_ec if n_ec > 0 else 0
                    }
                else:
                    results['status'] = 'error'
                    results['error'] = '동질집합 계산 오류'
            else:
                results['status'] = 'error'
                results['error'] = '준식별자가 지정되지 않았습니다.'
                
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    # ===== U8: Non-uniform Entropy Metric (NUEM) =====
    def calculate_u8_nuem(self, columns: List[str]) -> Dict[str, Any]:
        """
        U8: 비균일 엔트로피 계산
        알고리즘: 수식 (10)에 따른 복잡한 엔트로피 계산
        간소화된 버전: 변경된 레코드의 엔트로피 측정
        """
        results = {
            'metric': 'U8',
            'name': '비균일 엔트로피 (NUEM)',
            'total_score': 0.0,
            'status': 'success'
        }
        
        try:
            n = len(self.anonymized_df)
            j = len(columns)
            
            if n == 0 or j == 0:
                results['status'] = 'error'
                results['error'] = '데이터가 없습니다.'
                return results
            
            # 각 속성별 변경 정도 측정
            total_entropy = 0.0
            
            for col in columns:
                if col in self.original_df.columns and col in self.anonymized_df.columns:
                    # 원본과 비식별 데이터의 값 비교
                    changed_count = 0
                    
                    for idx in self.original_df.index:
                        if idx in self.anonymized_df.index:
                            orig_val = self.original_df.loc[idx, col]
                            anon_val = self.anonymized_df.loc[idx, col]
                            
                            # 값이 다른 경우 카운트
                            try:
                                if pd.isna(orig_val) and pd.isna(anon_val):
                                    continue
                                elif orig_val != anon_val:
                                    changed_count += 1
                            except:
                                # 비교가 어려운 경우 변경된 것으로 간주
                                changed_count += 1
                    
                    # 변경 비율 기반 엔트로피 계산
                    if changed_count > 0:
                        p_changed = changed_count / n
                        p_unchanged = 1 - p_changed
                        
                        # 엔트로피 계산 (0이 아닌 경우만)
                        entropy = 0
                        if p_changed > 0:
                            entropy -= p_changed * np.log2(p_changed)
                        if p_unchanged > 0:
                            entropy -= p_unchanged * np.log2(p_unchanged)
                        
                        total_entropy += entropy
            
            # 정규화 (속성 수로 나눔)
            results['total_score'] = total_entropy / j if j > 0 else 0
            results['details'] = {
                'total_records': n,
                'total_attributes': j,
                'raw_entropy': total_entropy
            }
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    # ===== U9: Anonymisation Ratio (AR) =====
    def calculate_u9_ar(self) -> Dict[str, Any]:
        """
        U9: 익명화율 계산
        알고리즘: (J/I) * 100
        - I: 원본 데이터셋 레코드 수
        - J: 비식별 데이터셋 레코드 수
        """
        results = {
            'metric': 'U9',
            'name': '익명화율 (AR)',
            'total_score': 0.0,
            'status': 'success'
        }
        
        try:
            i = len(self.original_df)  # 원본 레코드 수
            j = len(self.anonymized_df)  # 비식별 레코드 수
            
            if i > 0:
                # AR = (J/I) * 100
                ar = (j / i) * 100
                results['total_score'] = ar
                results['details'] = {
                    'original_records': i,
                    'anonymized_records': j,
                    'ratio_percent': ar
                }
            else:
                results['status'] = 'error'
                results['error'] = '원본 데이터가 없습니다.'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    # ===== 통합 평가 함수 =====
    def evaluate_all(self, 
                     numeric_columns: List[str] = None,
                     quasi_identifiers: List[str] = None,
                     sensitive_attribute: str = None) -> Dict[str, Any]:
        """모든 적용 가능한 지표에 대해 평가 수행"""
        
        results = {}
        
        # 자동으로 컬럼 타입 감지
        if numeric_columns is None:
            numeric_columns = self.original_df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
        
        # U1-U5, U8: 수치형 컬럼 필요
        if numeric_columns:
            results['U1'] = self.calculate_u1_ma(numeric_columns)
            
            if len(numeric_columns) >= 2:
                results['U2'] = self.calculate_u2_mc(numeric_columns)
            
            results['U3'] = self.calculate_u3_cs(numeric_columns)
            results['U4'] = self.calculate_u4_ned(numeric_columns)
            results['U5'] = self.calculate_u5_sed(numeric_columns)
            results['U8'] = self.calculate_u8_nuem(numeric_columns)
        
        # U6-U7: 준식별자 필요
        if quasi_identifiers:
            if sensitive_attribute and sensitive_attribute in self.anonymized_df.columns:
                results['U6'] = self.calculate_u6_md_ecm(quasi_identifiers, sensitive_attribute)
            
            results['U7'] = self.calculate_u7_na_ecsm(quasi_identifiers)
        
        # U9: 항상 계산 가능
        results['U9'] = self.calculate_u9_ar()
        
        return results