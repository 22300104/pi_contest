import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt

# k-익명성 계산을 위한 새로운 모듈도 필요합니다
try:
    from modules.privacy_metrics.k_anonymity import KAnonymityAnalyzer
except ImportError:
    # 모듈이 아직 없으면 임시로 처리
    pass


def render_privacy_evaluation_tab():
    """프라이버시 평가 탭 렌더링"""
    st.header("📋 프라이버시 평가")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("먼저 데이터를 업로드해주세요.")
        return
    
    # 처리된 데이터가 있으면 그것을 사용, 없으면 원본 사용
    df = st.session_state.get("df_processed", st.session_state.df)
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📊 k-익명성 분석", "📈 유용성 평가", "🔍 종합 평가"])
    
    with tab1:
        render_k_anonymity_section(df)
    
    with tab2:
        render_utility_evaluation_section(df)
    
    with tab3:
        render_comprehensive_evaluation_section(df)


def render_k_anonymity_section(df: pd.DataFrame):
    """k-익명성 분석 섹션"""
    st.subheader("📊 k-익명성 분석")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 준식별자 선택
        st.markdown("### 준식별자 선택")
        st.info("준식별자(Quasi-Identifier)는 단독으로는 개인을 식별할 수 없지만, 조합하면 개인을 식별할 가능성이 있는 속성들입니다.")
        
        # 컬럼 타입별로 분류
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 준식별자 선택 UI
        selected_qi = []
        
        if categorical_cols:
            st.markdown("**범주형 속성**")
            cat_selected = st.multiselect(
                "범주형 준식별자 선택",
                categorical_cols,
                help="예: 성별, 지역, 직업 등",
                key="cat_qi_select"
            )
            selected_qi.extend(cat_selected)
        
        if numeric_cols:
            st.markdown("**수치형 속성**")
            num_selected = st.multiselect(
                "수치형 준식별자 선택",
                numeric_cols,
                help="예: 나이, 우편번호 등",
                key="num_qi_select"
            )
            selected_qi.extend(num_selected)
    
    with col2:
        # 분석 옵션
        st.markdown("### 분석 옵션")
        
        # 샘플링 옵션 (대용량 데이터 대응)
        data_size = len(df)
        if data_size > 100000:
            use_sampling = st.checkbox(
                "샘플링 사용",
                value=True,
                help=f"전체 {data_size:,}행 중 일부만 분석하여 속도 향상"
            )
            if use_sampling:
                sample_size = st.slider(
                    "샘플 크기",
                    min_value=10000,
                    max_value=min(100000, data_size),
                    value=50000,
                    step=10000,
                    format="%d행"
                )
        else:
            use_sampling = False
            sample_size = data_size
        
        # k값 임계값 설정
        k_threshold = st.number_input(
            "k값 임계값",
            min_value=2,
            max_value=100,
            value=5,
            help="이 값 미만의 k를 가진 레코드는 위험으로 표시됩니다"
        )
    
    # 분석 실행 버튼
    st.markdown("---")
    
    if st.button("🔍 k-익명성 분석 실행", type="primary", disabled=len(selected_qi) == 0):
        if len(selected_qi) == 0:
            st.error("최소 하나 이상의 준식별자를 선택해주세요.")
            return
        
        with st.spinner("k-익명성 분석 중..."):
            # 샘플링 적용
            analysis_df = df.sample(n=sample_size) if use_sampling else df
            
            # k-익명성 계산
            try:
                k_value, k_stats = calculate_k_anonymity(analysis_df, selected_qi)
                
                # 결과 표시
                st.markdown("### 📊 분석 결과")
                
                # 주요 지표 표시
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric(
                        "최소 k값",
                        f"{k_stats['min_k']}",
                        delta=f"{k_stats['min_k'] - k_threshold}" if k_stats['min_k'] < k_threshold else None,
                        delta_color="inverse"
                    )
                
                with col_b:
                    st.metric(
                        "평균 k값",
                        f"{k_stats['avg_k']:.1f}"
                    )
                
                with col_c:
                    st.metric(
                        "중앙값",
                        f"{k_stats['median_k']}"
                    )
                
                with col_d:
                    risk_ratio = k_stats['risk_records'] / len(analysis_df) * 100
                    st.metric(
                        "위험 레코드",
                        f"{k_stats['risk_records']:,}개",
                        delta=f"{risk_ratio:.1f}%",
                        delta_color="inverse"
                    )
                
                # k값 분포 시각화
                st.markdown("### 📈 k값 분포")
                
                create_k_distribution_chart(k_stats['k_distribution'], k_threshold)
                
                # 위험 레코드 상세
                if k_stats['risk_records'] > 0:
                    with st.expander(f"⚠️ 위험 레코드 상세 (k < {k_threshold})", expanded=False):
                        risk_df = k_stats['risk_records_detail']
                        st.dataframe(
                            risk_df.head(100),  # 최대 100개만 표시
                            use_container_width=True
                        )
                        
                        if len(risk_df) > 100:
                            st.info(f"전체 {len(risk_df)}개 중 상위 100개만 표시됩니다.")
                
                # 분석 정보 저장 (다른 탭에서 사용)
                if 'privacy_analysis' not in st.session_state:
                    st.session_state.privacy_analysis = {}
                
                st.session_state.privacy_analysis['k_anonymity'] = {
                    'quasi_identifiers': selected_qi,
                    'k_value': k_value,
                    'k_stats': k_stats,
                    'threshold': k_threshold,
                    'sampled': use_sampling
                }
                
                st.success("✅ k-익명성 분석이 완료되었습니다!")
                
            except Exception as e:
                st.error(f"분석 중 오류 발생: {str(e)}")
    
    elif len(selected_qi) == 0:
        st.info("👆 준식별자를 선택하고 분석을 실행하세요.")


def calculate_k_anonymity(df: pd.DataFrame, quasi_identifiers: List[str]) -> Tuple[int, Dict]:
    """k-익명성 계산 (임시 구현)"""
    # 실제 구현은 modules/privacy_metrics/k_anonymity.py에 있어야 함
    # 여기서는 임시로 간단한 계산만 수행
    
    # 준식별자 조합별 그룹 크기 계산
    group_sizes = df.groupby(quasi_identifiers).size().reset_index(name='count')
    
    # k값은 가장 작은 그룹의 크기
    k_value = group_sizes['count'].min()
    
    # 통계 계산
    k_stats = {
        'min_k': int(group_sizes['count'].min()),
        'max_k': int(group_sizes['count'].max()),
        'avg_k': float(group_sizes['count'].mean()),
        'median_k': int(group_sizes['count'].median()),
        'risk_records': int((group_sizes['count'] < 5).sum()),  # k<5인 그룹 수
        'k_distribution': group_sizes['count'].value_counts().sort_index().to_dict(),
        'risk_records_detail': df[df.index.isin(
            group_sizes[group_sizes['count'] < 5].index
        )] if (group_sizes['count'] < 5).any() else pd.DataFrame()
    }
    
    return k_value, k_stats


def create_k_distribution_chart(k_distribution: Dict[int, int], threshold: int):
    """k값 분포 차트 생성 (Streamlit 내장 차트 사용)"""
    # 데이터 준비
    k_values = list(k_distribution.keys())
    counts = list(k_distribution.values())
    
    # DataFrame으로 변환
    chart_data = pd.DataFrame({
        'k값': k_values,
        '그룹 수': counts,
        '상태': ['위험' if k < threshold else '안전' for k in k_values]
    })
    
    # Streamlit 차트
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 막대 차트
        st.bar_chart(
            data=chart_data.set_index('k값')['그룹 수'],
            height=400
        )
    
    with col2:
        # 통계 정보
        st.metric("임계값", f"k = {threshold}")
        st.metric("위험 그룹", f"{len([k for k in k_values if k < threshold])}개")
        st.metric("안전 그룹", f"{len([k for k in k_values if k >= threshold])}개")
    
    # 추가 정보
    if any(k < threshold for k in k_values):
        st.warning(f"⚠️ k < {threshold}인 그룹이 존재합니다.")
    else:
        st.success(f"✅ 모든 그룹이 k ≥ {threshold}를 만족합니다.")


def render_utility_evaluation_section(df: pd.DataFrame):
    """유용성 평가 섹션"""
    st.subheader("📈 유용성 평가")
    
    # 원본 데이터와 비교
    if 'df' in st.session_state and 'df_processed' in st.session_state:
        original_df = st.session_state.df
        processed_df = st.session_state.df_processed
        
        st.info("원본 데이터와 처리된 데이터를 비교하여 유용성을 평가합니다.")
        
        # 유용성 지표 계산
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 데이터 변경 통계")
            
            # 전체 변경률
            total_cells = original_df.size
            changed_cells = (original_df != processed_df).sum().sum()
            change_rate = (changed_cells / total_cells) * 100
            
            st.metric("전체 변경률", f"{change_rate:.2f}%")
            
            # 컬럼별 변경률
            st.markdown("**컬럼별 변경률**")
            column_changes = {}
            for col in original_df.columns:
                if col in processed_df.columns:
                    changes = (original_df[col] != processed_df[col]).sum()
                    rate = (changes / len(original_df)) * 100
                    column_changes[col] = rate
            
            # 변경률이 높은 순으로 정렬
            sorted_changes = sorted(column_changes.items(), key=lambda x: x[1], reverse=True)
            
            # 상위 10개만 표시
            change_df = pd.DataFrame(sorted_changes[:10], columns=['컬럼', '변경률(%)'])
            st.dataframe(change_df, use_container_width=True)
        
        with col2:
            st.markdown("### 📉 정보 손실 평가")
            
            # 수치형 컬럼에 대한 통계적 유사성
            numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_cols) > 0:
                st.markdown("**수치형 컬럼 통계 변화**")
                
                stats_comparison = []
                for col in numeric_cols:
                    if col in processed_df.columns:
                        orig_mean = original_df[col].mean()
                        proc_mean = processed_df[col].mean()
                        mean_diff = abs(orig_mean - proc_mean) / orig_mean * 100 if orig_mean != 0 else 0
                        
                        orig_std = original_df[col].std()
                        proc_std = processed_df[col].std()
                        std_diff = abs(orig_std - proc_std) / orig_std * 100 if orig_std != 0 else 0
                        
                        stats_comparison.append({
                            '컬럼': col,
                            '평균 변화율(%)': f"{mean_diff:.2f}",
                            '표준편차 변화율(%)': f"{std_diff:.2f}"
                        })
                
                stats_df = pd.DataFrame(stats_comparison[:5])  # 상위 5개만
                st.dataframe(stats_df, use_container_width=True)
            
            # 범주형 컬럼의 다양성 손실
            categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                st.markdown("**범주형 컬럼 다양성 변화**")
                
                diversity_loss = []
                for col in categorical_cols[:5]:  # 상위 5개만
                    if col in processed_df.columns:
                        orig_unique = original_df[col].nunique()
                        proc_unique = processed_df[col].nunique()
                        loss = (orig_unique - proc_unique) / orig_unique * 100 if orig_unique > 0 else 0
                        
                        diversity_loss.append({
                            '컬럼': col,
                            '원본 고유값': orig_unique,
                            '처리후 고유값': proc_unique,
                            '감소율(%)': f"{loss:.1f}"
                        })
                
                diversity_df = pd.DataFrame(diversity_loss)
                st.dataframe(diversity_df, use_container_width=True)
    
    else:
        st.warning("처리된 데이터가 없습니다. 비식별화를 먼저 수행해주세요.")


def render_comprehensive_evaluation_section(df: pd.DataFrame):
    """종합 평가 섹션"""
    st.subheader("🔍 종합 평가")
    
    # k-익명성 결과가 있는지 확인
    has_k_analysis = 'privacy_analysis' in st.session_state and 'k_anonymity' in st.session_state.privacy_analysis
    has_processed_data = 'df_processed' in st.session_state
    
    if has_k_analysis and has_processed_data:
        st.success("✅ 프라이버시와 유용성 평가가 완료되었습니다.")
        
        # 종합 점수 계산 (간단한 예시)
        k_stats = st.session_state.privacy_analysis['k_anonymity']['k_stats']
        
        # 프라이버시 점수 (0-100)
        # 최소 k값이 높을수록 좋음
        privacy_score = min(100, k_stats['min_k'] * 10)
        
        # 유용성 점수 (0-100)
        # 변경률이 낮을수록 좋음 (임시 계산)
        utility_score = 85  # 실제로는 유용성 평가 결과를 바탕으로 계산
        
        # 점수 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("프라이버시 점수", f"{privacy_score}/100")
            st.progress(privacy_score / 100)
        
        with col2:
            st.metric("유용성 점수", f"{utility_score}/100")
            st.progress(utility_score / 100)
        
        with col3:
            total_score = (privacy_score + utility_score) / 2
            st.metric("종합 점수", f"{total_score:.1f}/100")
            st.progress(total_score / 100)
        
        # 권장사항
        st.markdown("### 💡 권장사항")
        
        recommendations = []
        
        if k_stats['min_k'] < 5:
            recommendations.append("⚠️ 최소 k값이 5 미만입니다. 추가적인 비식별화 처리를 권장합니다.")
        
        if k_stats['risk_records'] > len(df) * 0.1:
            recommendations.append("⚠️ 위험 레코드가 전체의 10% 이상입니다. 준식별자를 재검토하거나 더 강한 비식별화를 적용하세요.")
        
        if privacy_score < 70:
            recommendations.append("📌 프라이버시 보호 수준을 높이기 위해 범주화나 일반화를 추가로 적용하는 것을 고려하세요.")
        
        if utility_score < 70:
            recommendations.append("📌 데이터 유용성이 낮습니다. 비식별화 강도를 조절하거나 다른 기법을 시도해보세요.")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.success("✨ 프라이버시와 유용성의 균형이 잘 맞춰져 있습니다!")
        
        # 보고서 다운로드 (추후 구현)
        st.markdown("---")
        st.button("📄 평가 보고서 다운로드", disabled=True, help="준비 중입니다")
    
    else:
        st.info("종합 평가를 위해서는 다음 단계를 완료해주세요:")
        
        if not has_processed_data:
            st.write("1️⃣ 데이터 비식별화 수행")
        else:
            st.write("✅ 데이터 비식별화 완료")
        
        if not has_k_analysis:
            st.write("2️⃣ k-익명성 분석 실행")
        else:
            st.write("✅ k-익명성 분석 완료")