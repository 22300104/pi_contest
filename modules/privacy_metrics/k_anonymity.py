import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter


class KAnonymityAnalyzer:
    """k-익명성 분석을 위한 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 분석할 데이터프레임
        """
        self.df = df
        self.results_cache = {}
    
    def calculate_k_anonymity(
        self, 
        quasi_identifiers: List[str],
        sensitive_attribute: Optional[str] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        주어진 준식별자에 대한 k-익명성 계산
        
        Args:
            quasi_identifiers: 준식별자 컬럼 리스트
            sensitive_attribute: 민감 속성 (선택사항, l-다양성 계산용)
            
        Returns:
            k_value: 최소 k값
            statistics: 상세 통계 정보
        """
        # 캐시 키 생성
        cache_key = tuple(sorted(quasi_identifiers))
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # 준식별자 조합별 그룹 생성
        grouped = self.df.groupby(quasi_identifiers).size().reset_index(name='group_size')
        
        # k값 계산 (가장 작은 그룹의 크기)
        k_value = int(grouped['group_size'].min())
        
        # 통계 정보 계산
        statistics = {
            'min_k': k_value,
            'max_k': int(grouped['group_size'].max()),
            'avg_k': float(grouped['group_size'].mean()),
            'median_k': int(grouped['group_size'].median()),
            'std_k': float(grouped['group_size'].std()),
            'total_groups': len(grouped),
            'k_distribution': self._get_k_distribution(grouped),
            'risk_analysis': self._analyze_risk_records(grouped, quasi_identifiers),
        }
        
        # l-다양성 계산 (민감 속성이 제공된 경우)
        if sensitive_attribute and sensitive_attribute in self.df.columns:
            statistics['l_diversity'] = self._calculate_l_diversity(
                quasi_identifiers, sensitive_attribute
            )
        
        # 결과 캐싱
        self.results_cache[cache_key] = (k_value, statistics)
        
        return k_value, statistics
    
    def _get_k_distribution(self, grouped: pd.DataFrame) -> Dict[int, int]:
        """k값 분포 계산"""
        k_counts = grouped['group_size'].value_counts().sort_index()
        
        # k값을 구간으로 묶어서 표시 (너무 많은 경우)
        if len(k_counts) > 50:
            # 1, 2, 3, 4, 5, 6-10, 11-20, 21-50, 51-100, 100+
            distribution = {}
            for k, count in k_counts.items():
                if k <= 5:
                    distribution[k] = distribution.get(k, 0) + count
                elif k <= 10:
                    distribution['6-10'] = distribution.get('6-10', 0) + count
                elif k <= 20:
                    distribution['11-20'] = distribution.get('11-20', 0) + count
                elif k <= 50:
                    distribution['21-50'] = distribution.get('21-50', 0) + count
                elif k <= 100:
                    distribution['51-100'] = distribution.get('51-100', 0) + count
                else:
                    distribution['100+'] = distribution.get('100+', 0) + count
            return distribution
        else:
            return k_counts.to_dict()
    
    def _analyze_risk_records(
        self, 
        grouped: pd.DataFrame, 
        quasi_identifiers: List[str],
        k_threshold: int = 5
    ) -> Dict[str, Any]:
        """위험 레코드 분석"""
        # k값이 임계값 미만인 그룹
        risk_groups = grouped[grouped['group_size'] < k_threshold].copy()
        
        # 위험 레코드 수 계산
        risk_record_count = risk_groups['group_size'].sum()
        
        # 위험 그룹의 상세 정보
        risk_details = []
        if len(risk_groups) > 0:
            # 각 위험 그룹에 대한 정보
            for _, group in risk_groups.iterrows():
                qi_values = {qi: group[qi] for qi in quasi_identifiers}
                risk_details.append({
                    'quasi_identifiers': qi_values,
                    'group_size': group['group_size'],
                    'risk_level': self._calculate_risk_level(group['group_size'])
                })
        
        # 위험도별 분포
        risk_distribution = Counter([d['risk_level'] for d in risk_details])
        
        return {
            'risk_records': int(risk_record_count),
            'risk_groups': len(risk_groups),
            'risk_ratio': float(risk_record_count / len(self.df)) if len(self.df) > 0 else 0,
            'risk_distribution': dict(risk_distribution),
            'risk_details': risk_details[:100],  # 상위 100개만
            'threshold': k_threshold
        }
    
    def _calculate_risk_level(self, k: int) -> str:
        """k값에 따른 위험 수준 계산"""
        if k == 1:
            return "매우 높음"
        elif k == 2:
            return "높음"
        elif k <= 4:
            return "중간"
        else:
            return "낮음"
    
    def _calculate_l_diversity(
        self, 
        quasi_identifiers: List[str], 
        sensitive_attribute: str
    ) -> Dict[str, Any]:
        """l-다양성 계산"""
        # 준식별자 그룹별 민감 속성의 다양성 계산
        diversity_results = []
        
        for group_keys, group_df in self.df.groupby(quasi_identifiers):
            sensitive_values = group_df[sensitive_attribute].value_counts()
            l_value = len(sensitive_values)
            entropy = -sum((count/len(group_df)) * np.log2(count/len(group_df)) 
                          for count in sensitive_values.values)
            
            diversity_results.append({
                'group_size': len(group_df),
                'l_value': l_value,
                'entropy': entropy
            })
        
        diversity_df = pd.DataFrame(diversity_results)
        
        return {
            'min_l': int(diversity_df['l_value'].min()),
            'avg_l': float(diversity_df['l_value'].mean()),
            'min_entropy': float(diversity_df['entropy'].min()),
            'avg_entropy': float(diversity_df['entropy'].mean())
        }
    
    def find_optimal_generalization(
        self,
        quasi_identifiers: List[str],
        target_k: int = 5,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        목표 k값을 달성하기 위한 최적 일반화 수준 찾기
        
        Args:
            quasi_identifiers: 준식별자 리스트
            target_k: 목표 k값
            max_iterations: 최대 반복 횟수
            
        Returns:
            최적화 결과 및 권장사항
        """
        # 이 기능은 추후 구현
        # 일반화 계층을 정의하고 반복적으로 적용하여 최적점 찾기
        return {
            'status': 'not_implemented',
            'message': '이 기능은 추후 구현 예정입니다.'
        }
    
    def generate_report(
        self,
        quasi_identifiers: List[str],
        format: str = 'dict'
    ) -> Any:
        """
        k-익명성 분석 보고서 생성
        
        Args:
            quasi_identifiers: 준식별자 리스트
            format: 출력 형식 ('dict', 'html', 'pdf')
            
        Returns:
            보고서 (형식에 따라 다름)
        """
        k_value, statistics = self.calculate_k_anonymity(quasi_identifiers)
        
        report = {
            'summary': {
                'dataset_size': len(self.df),
                'quasi_identifiers': quasi_identifiers,
                'k_anonymity': k_value,
                'privacy_level': self._get_privacy_level(k_value),
                'risk_assessment': statistics['risk_analysis']
            },
            'statistics': statistics,
            'recommendations': self._generate_recommendations(k_value, statistics)
        }
        
        if format == 'dict':
            return report
        elif format == 'html':
            # HTML 보고서 생성 (추후 구현)
            return self._generate_html_report(report)
        else:
            return report
    
    def _get_privacy_level(self, k: int) -> str:
        """k값에 따른 프라이버시 수준 평가"""
        if k >= 100:
            return "매우 높음"
        elif k >= 50:
            return "높음"
        elif k >= 10:
            return "중간"
        elif k >= 5:
            return "낮음"
        else:
            return "매우 낮음"
    
    def _generate_recommendations(self, k_value: int, statistics: Dict) -> List[str]:
        """k-익명성 결과에 따른 권장사항 생성"""
        recommendations = []
        
        if k_value < 5:
            recommendations.append(
                "k값이 5 미만입니다. 추가적인 일반화나 억제를 적용하여 "
                "프라이버시 보호 수준을 높이는 것을 권장합니다."
            )
        
        risk_ratio = statistics['risk_analysis']['risk_ratio']
        if risk_ratio > 0.1:
            recommendations.append(
                f"전체 레코드의 {risk_ratio*100:.1f}%가 위험 수준입니다. "
                "준식별자를 재검토하거나 더 강한 비식별화를 적용하세요."
            )
        
        if statistics['avg_k'] / statistics['min_k'] > 10:
            recommendations.append(
                "k값의 분포가 매우 불균형합니다. "
                "특정 그룹에 대한 추가적인 처리가 필요할 수 있습니다."
            )
        
        if len(recommendations) == 0:
            recommendations.append(
                "현재 k-익명성 수준이 적절합니다. "
                "데이터 유용성과의 균형을 고려하여 최적화할 수 있습니다."
            )
        
        return recommendations
    
    def _generate_html_report(self, report_data: Dict) -> str:
        """HTML 형식의 보고서 생성 (추후 구현)"""
        # 추후 구현
        return "<html><body>Report will be generated here</body></html>"