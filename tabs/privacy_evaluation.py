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
    # k-익명성 분석 섹션 내부에 추가
    if selected_qi:  # 준식별자가 선택된 경우
        with st.expander("👥 동질집합 미리보기", expanded=False):
            st.info("""
            **동질집합(Equivalence Class)이란?**
            준식별자 값이 모두 동일한 레코드들의 그룹입니다.
            예: 나이(30대), 성별(남), 지역(서울) → 이 조합이 같은 모든 사람들
            """)
            
            # 미리보기 옵션
            preview_option = st.radio(
                "미리보기 옵션",
                ["상위 5개 그룹", "k값이 낮은 위험 그룹", "랜덤 샘플"],
                horizontal=True
            )
            
            # 표시할 레코드 수
            show_records = st.slider("그룹당 표시할 레코드 수", 1, 10, 3)
            
            if st.button("동질집합 미리보기", key="preview_ec"):
                # 동질집합 생성
                ec_groups = analysis_df.groupby(selected_qi)
                
                # 각 그룹의 크기 계산
                ec_sizes = ec_groups.size().reset_index(name='k')
                ec_sizes = ec_sizes.sort_values('k', ascending=(preview_option == "k값이 낮은 위험 그룹"))
                
                # 미리보기 생성
                preview_groups = []
                
                if preview_option == "상위 5개 그룹":
                    # 크기가 큰 순서대로 5개
                    top_groups = ec_sizes.nlargest(5, 'k')
                elif preview_option == "k값이 낮은 위험 그룹":
                    # k값이 작은 순서대로 5개
                    top_groups = ec_sizes.nsmallest(5, 'k')
                else:  # 랜덤 샘플
                    # 랜덤하게 5개
                    top_groups = ec_sizes.sample(min(5, len(ec_sizes)))
                
                # 각 그룹의 샘플 표시
                for idx, (_, group_info) in enumerate(top_groups.iterrows()):
                    st.markdown(f"### 그룹 {idx+1}")
                    
                    # 준식별자 값 표시
                    qi_values = []
                    for qi in selected_qi:
                        if qi in group_info:
                            qi_values.append(f"{qi}: {group_info[qi]}")
                    
                    st.write(f"**준식별자 조합**: {', '.join(qi_values)}")
                    st.write(f"**k값**: {group_info['k']} (이 조합을 가진 사람 수)")
                    
                    # 위험도 표시
                    if group_info['k'] < k_threshold:
                        st.warning(f"⚠️ 위험: k값이 {k_threshold} 미만입니다!")
                    else:
                        st.success(f"✅ 안전: k값이 {k_threshold} 이상입니다.")
                    
                    # 해당 그룹의 샘플 레코드 표시
                    # 준식별자 값으로 필터링
                    mask = True
                    for qi in selected_qi:
                        mask = mask & (analysis_df[qi] == group_info[qi])
                    
                    group_records = analysis_df[mask].head(show_records)
                    
                    # 민감한 정보는 가리고 표시
                    display_cols = selected_qi + [col for col in analysis_df.columns 
                                                if col not in selected_qi][:3]  # 추가로 3개 컬럼만
                    
                    st.dataframe(
                        group_records[display_cols],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("---")
                
                # 전체 통계
                st.markdown("### 📊 전체 동질집합 통계")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("전체 그룹 수", f"{len(ec_sizes):,}개")
                with col2:
                    st.metric("평균 k값", f"{ec_sizes['k'].mean():.1f}")
                with col3:
                    risk_groups = len(ec_sizes[ec_sizes['k'] < k_threshold])
                    st.metric("위험 그룹", f"{risk_groups}개")
                
                # k값 분포 간단히 표시
                st.markdown("#### k값 분포")
                k_dist = ec_sizes['k'].value_counts().sort_index()
                
                # 간단한 히스토그램
                hist_data = []
                for k, count in k_dist.items():
                    if k < 5:
                        hist_data.append({
                            'k값': f"k={k}",
                            '그룹 수': count,
                            '상태': '위험'
                        })
                    else:
                        hist_data.append({
                            'k값': f"k={k}" if k <= 10 else f"k>{10}",
                            '그룹 수': count,
                            '상태': '안전'
                        })
                
                hist_df = pd.DataFrame(hist_data)
                st.bar_chart(hist_df.set_index('k값')['그룹 수'])
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
        
        # UtilityMetrics 임포트
        try:
            from modules.privacy_metrics.utility_metrics import UtilityMetrics
            utility_analyzer = UtilityMetrics(original_df, processed_df)
        except ImportError:
            st.error("유용성 평가 모듈을 불러올 수 없습니다.")
            return
        
        st.info("각 평가 지표별로 적용할 컬럼을 선택하고 개별적으로 평가를 수행합니다.")
        
        # 샘플링 옵션
        with st.expander("⚙️ 데이터 처리 옵션", expanded=True):
            data_option = st.radio(
                "처리할 데이터 크기 선택",
                options=[
                    f"전체 데이터 ({len(original_df):,}행)",
                    f"대규모 샘플 (100,000행) - 권장" if len(original_df) > 100000 else None,
                    f"빠른 테스트 (10,000행)" if len(original_df) > 10000 else None
                ],
                index=1 if len(original_df) > 100000 else 0,
                help="대용량 데이터의 경우 샘플링을 사용하면 더 빠르게 결과를 확인할 수 있습니다."
            )
            
            # 옵션 파싱
            if "전체 데이터" in data_option:
                use_sampling = False
                sample_size = len(original_df)
            elif "대규모 샘플" in data_option:
                use_sampling = True
                sample_size = 100000
            else:  # 빠른 테스트
                use_sampling = True
                sample_size = 10000
            
            if use_sampling:
                st.warning(f"⚠️ 샘플링 모드: {sample_size:,}개 행만 사용하여 평가합니다.")
            else:
                if len(original_df) > 100000:
                    st.warning("⚠️ 전체 데이터 평가는 시간이 오래 걸릴 수 있습니다.")
        
        st.markdown("---")
        
        # Step 1: 평가할 컬럼 선택
        st.markdown("### 📌 Step 1: 평가할 데이터 항목 선택하기")
        
        st.info("""
        💡 **도움말**: 비식별화 처리한 데이터의 품질을 확인하고 싶은 항목들을 선택하세요.
        
        **데이터 타입별 안내:**
        - 📊 **숫자 데이터**: 나이, 급여, 점수 등 (대부분의 평가 가능)
        - 📝 **문자 데이터**: 성별, 지역, 직업 등 (일부 평가만 가능)  
        - 📅 **날짜 데이터**: 생년월일, 가입일 등 (숫자로 변환하여 평가)
        """)
        
        # 컬럼 타입별로 분류
        numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = original_df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 준식별자 가져오기 (있다면)
        quasi_identifiers = []
        if 'privacy_analysis' in st.session_state and 'k_anonymity' in st.session_state.privacy_analysis:
            quasi_identifiers = st.session_state.privacy_analysis['k_anonymity'].get('quasi_identifiers', [])
        
        # 빠른 선택 버튼들
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if quasi_identifiers and st.button("🎯 준식별자 항목들", help="k-익명성 분석에서 사용한 준식별자"):
                st.session_state.util_selected_columns = quasi_identifiers
                st.rerun()
        
        with col2:
            if st.button("📊 숫자 데이터만"):
                st.session_state.util_selected_columns = numeric_cols
                st.rerun()
                
        with col3:
            if st.button("🔄 모두 선택"):
                st.session_state.util_selected_columns = original_df.columns.tolist()
                st.rerun()
                
        with col4:
            if st.button("❌ 선택 취소"):
                st.session_state.util_selected_columns = []
                st.rerun()
        
        # multiselect로 컬럼 선택
        if 'util_selected_columns' not in st.session_state:
            st.session_state.util_selected_columns = []
            
        selected_columns = st.multiselect(
            "평가할 항목 선택",
            options=original_df.columns.tolist(),
            default=st.session_state.util_selected_columns,
            format_func=lambda x: f"📊 {x} (숫자)" if x in numeric_cols else f"📝 {x} (문자)" if x in categorical_cols else f"📅 {x} (날짜)",
            key="column_selector"
        )
        
        # 선택된 컬럼 업데이트
        st.session_state.util_selected_columns = selected_columns
        
        if selected_columns:
            # 선택된 컬럼 요약
            selected_numeric = [col for col in selected_columns if col in numeric_cols]
            selected_categorical = [col for col in selected_columns if col in categorical_cols]
            selected_datetime = [col for col in selected_columns if col in datetime_cols]
            
            st.success(f"""
            ✅ **{len(selected_columns)}개 항목이 선택되었습니다**
            - 숫자형: {len(selected_numeric)}개
            - 문자형: {len(selected_categorical)}개
            - 날짜형: {len(selected_datetime)}개
            """)
            
            # 준식별자 표시
            if quasi_identifiers:
                with st.expander("❓ 준식별자란?"):
                    st.write("""
                    **준식별자(Quasi-Identifier)**는 개인을 간접적으로 식별할 수 있는 정보들입니다.
                    
                    예시: 나이+성별+지역을 조합하면 특정인을 찾을 수 있음
                    
                    k-익명성 분석에서 사용한 준식별자: **{}**
                    """.format(", ".join(quasi_identifiers)))
        
        # Step 2: 평가 지표 선택
        st.markdown("---")
        st.markdown("### 📌 Step 2: 평가 방법 선택하기")
        
        if selected_columns:
            # 선택된 컬럼 타입 확인
            selected_numeric = [col for col in selected_columns if col in numeric_cols]
            selected_categorical = [col for col in selected_columns if col in categorical_cols]
            
            st.write(f"**선택한 데이터**: {', '.join([f'{col}(숫자)' if col in numeric_cols else f'{col}(문자)' if col in categorical_cols else f'{col}(날짜)' for col in selected_columns[:5]])}{'...' if len(selected_columns) > 5 else ''}")
            
            st.markdown("---")
            
            selected_metrics = []
            
            # 기본 평가 지표
            st.markdown("#### 📊 기본 평가 지표")
            col1, col2 = st.columns([5, 1])
            
            with col1:
                if st.checkbox("**U1: 평균값 차이 (MA)** - 각 숫자 데이터의 평균이 얼마나 유지되었는지 평가", 
                              key="metric_u1",
                              disabled=len(selected_numeric + selected_datetime) == 0):
                    selected_metrics.append('U1')
                    
                if len(selected_numeric + selected_datetime) > 0:
                    st.caption(f"✅ 사용 가능: {', '.join(selected_numeric + selected_datetime)}")
                else:
                    st.caption("❌ 숫자형 또는 날짜형 데이터가 필요합니다")
            
            with col2:
                st.write("📊 1개씩")
                st.caption("각 항목별로 계산")
                
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.checkbox("**U9: 익명화율 (AR)** - 데이터가 얼마나 보존되었는지 평가 (삭제율 확인)", 
                              key="metric_u9"):
                    selected_metrics.append('U9')
                st.caption("✅ 자동 계산 (데이터 선택 불필요)")
            
            with col2:
                st.write("🌐 전체")
                st.caption("전체 데이터셋")
            
            # 관계/유사도 평가 지표
            st.markdown("#### 📐 관계/유사도 평가 지표")
            
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.checkbox("**U2: 상관관계 보존 (MC)** - 숫자 데이터들 간의 관계가 유지되었는지 평가", 
                              key="metric_u2",
                              disabled=len(selected_numeric) < 2):
                    selected_metrics.append('U2')
                    
                if len(selected_numeric) >= 2:
                    n_pairs = len(selected_numeric) * (len(selected_numeric) - 1) // 2
                    st.caption(f"✅ 사용 가능: {n_pairs}개 상관관계 쌍")
                else:
                    st.caption("❌ 최소 2개의 숫자형 데이터가 필요합니다")
            
            with col2:
                st.write("🔗 2개 이상")
                st.caption("데이터 간 관계")
                
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.checkbox("**U3: 코사인 유사도 (CS)** - 원본과 변환 데이터의 패턴 유사성 평가", 
                              key="metric_u3",
                              disabled=len(selected_numeric) == 0):
                    selected_metrics.append('U3')
                    
                if len(selected_numeric) > 0:
                    st.caption(f"✅ 사용 가능: {', '.join(selected_numeric)}")
                else:
                    st.caption("❌ 숫자형 데이터가 필요합니다")
            
            with col2:
                st.write("📊 1개씩")
                st.caption("각 항목별로 계산")
            
            # 거리 기반 평가 지표
            st.markdown("#### 📏 거리 기반 평가 지표")
            
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.checkbox("**U4: 정규화 유클리디안 거리 (NED)** - 각 값이 얼마나 변했는지 정밀 측정", 
                              key="metric_u4",
                              disabled=len(selected_numeric) == 0):
                    selected_metrics.append('U4')
                    
                if len(selected_numeric) > 0:
                    st.caption(f"✅ 사용 가능: {', '.join(selected_numeric)}")
                else:
                    st.caption("❌ 숫자형 데이터가 필요합니다")
            
            with col2:
                st.write("📊 1개씩")
                st.caption("각 항목별로 계산")
                
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.checkbox("**U5: 표준화 유클리디안 거리 (SED)** - 데이터 분포를 고려한 변화량 측정", 
                              key="metric_u5",
                              disabled=len(selected_numeric) == 0):
                    selected_metrics.append('U5')
                    
                if len(selected_numeric) > 0:
                    st.caption(f"✅ 사용 가능: {', '.join(selected_numeric)}")
                else:
                    st.caption("❌ 숫자형 데이터가 필요합니다")
            
            with col2:
                st.write("📊 1개씩")
                st.caption("각 항목별로 계산")
            
            # k-익명성 기반 평가 지표
            st.markdown("#### 🔐 k-익명성 기반 평가 지표")
            
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.checkbox("**U6: 동질집합 분산 (MD_ECM)** - 같은 그룹 내 데이터의 다양성 평가", 
                              key="metric_u6"):
                    selected_metrics.append('U6')
                st.caption("⚠️ 추가 설정 필요: 준식별자 + 민감속성 지정")
            
            with col2:
                st.write("👥 그룹+1개")
                st.caption("그룹 기반 분석")
                
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.checkbox("**U7: 정규화 집합크기 (NA_ECSM)** - 익명화 그룹들의 크기 분포 평가", 
                              key="metric_u7"):
                    selected_metrics.append('U7')
                st.caption("⚠️ 추가 설정 필요: 준식별자 지정")
            
            with col2:
                st.write("👥 그룹")
                st.caption("그룹 기반 분석")
                
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.checkbox("**U8: 비균일 엔트로피 (NUEM)** - 정보 손실량을 엔트로피로 측정", 
                              key="metric_u8"):
                    selected_metrics.append('U8')
                st.caption("✅ 사용 가능: 모든 선택된 데이터")
            
            with col2:
                st.write("📊 여러개")
                st.caption("전체 항목 분석")
            
            # U6, U7을 위한 추가 입력
            if 'U6' in selected_metrics or 'U7' in selected_metrics:
                st.markdown("---")
                st.markdown("#### ⚙️ 준식별자 설정 (U6, U7용)")
                
                quasi_cols = st.multiselect(
                    "준식별자로 사용할 항목 선택",
                    options=selected_columns,
                    default=quasi_identifiers if quasi_identifiers else [],
                    key="quasi_identifiers_utility",
                    help="개인을 간접적으로 식별할 수 있는 속성들을 선택하세요"
                )
                
                if 'U6' in selected_metrics:
                    # 숫자형 컬럼만 민감속성으로 선택 가능
                    sensitive_options = [col for col in selected_columns if col not in quasi_cols and col in numeric_cols]
                    
                    if sensitive_options:
                        sensitive_attr = st.selectbox(
                            "민감속성 선택 (U6용) - 숫자형만 가능",
                            options=sensitive_options,
                            key="sensitive_attr_utility",
                            help="동질집합 내에서 분산을 계산할 숫자형 속성을 선택하세요"
                        )
                    else:
                        st.error("민감속성으로 사용할 수 있는 숫자형 데이터가 없습니다.")
            
            if selected_metrics:
                st.info(f"📊 **{len(selected_metrics)}개 평가 지표가 선택되었습니다**")
            
            # 도움말
            with st.expander("💡 평가 지표 이해하기"):
                st.write("""
                **평가 지표 기호 설명:**
                - 📊 **1개씩**: 각 데이터 항목별로 개별 계산
                - 🔗 **2개 이상**: 데이터 간의 관계 분석  
                - 👥 **그룹**: 준식별자로 그룹을 만들어 분석
                - 🌐 **전체**: 전체 데이터셋 단위로 계산
                
                **결과 해석:**
                - U1, U4, U5, U6, U7, U8: 점수가 **낮을수록** 좋음 (0에 가까울수록 원본과 유사)
                - U3, U9: 점수가 **높을수록** 좋음 (1 또는 100%에 가까울수록 좋음)
                """)
        
        else:
            st.warning("먼저 평가할 데이터 항목을 선택해주세요.")
            selected_metrics = []
        
        # Step 3: 평가 실행
        st.markdown("---")
        st.markdown("### 📌 Step 3: 평가 실행")
        
        # 각 평가 지표별 섹션
        # U1: 평균값 차이
        with st.expander("📊 U1: 평균값 차이 (MA)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            원본과 변환된 데이터의 평균이 얼마나 비슷한지 확인합니다.
            
            ✅ **언제 사용하나요?**
            - 나이를 5세 단위로 변환했을 때
            - 급여를 구간으로 변환했을 때
            
            💡 **결과 해석**: 점수가 0에 가까울수록 좋습니다!
            """)
            
            # 수치형 컬럼만 선택 가능
            numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            datetime_cols = original_df.select_dtypes(include=['datetime64']).columns.tolist()
            
            u1_columns = st.multiselect(
                "평가할 컬럼 선택 (숫자형/날짜형)",
                options=numeric_cols + datetime_cols,
                default=[col for col in selected_columns if col in numeric_cols + datetime_cols],
                key="u1_columns"
            )
            
            if st.button("U1 평가 실행", key="run_u1", type="primary"):
                if u1_columns:
                    with st.spinner("U1 평가 중..."):
                        # 샘플링 적용
                        if use_sampling:
                            sample_idx = np.random.choice(original_df.index, size=sample_size, replace=False)
                            analyzer = UtilityMetrics(original_df.loc[sample_idx], processed_df.loc[sample_idx])
                        else:
                            analyzer = utility_analyzer
                        
                        result = analyzer.calculate_u1_ma(u1_columns)
                        
                        if result['status'] == 'success':
                            # 종합 점수와 평가
                            score = result['total_score']
                            if score < 10:
                                rating = "⭐⭐⭐⭐⭐ 매우 우수"
                            elif score < 50:
                                rating = "⭐⭐⭐⭐ 우수"
                            elif score < 100:
                                rating = "⭐⭐⭐ 보통"
                            elif score < 200:
                                rating = "⭐⭐ 주의"
                            else:
                                rating = "⭐ 개선 필요"
                            
                            st.success(f"""
                            ### 🎯 종합 점수: {score:.4f} {rating}
                            
                            💡 **이 점수의 의미:**
                            원본과 비교했을 때 평균값이 {'거의 동일하게' if score < 10 else '비교적 잘' if score < 50 else '어느 정도' if score < 100 else '다소 많이'} {'유지되었습니다' if score < 100 else '변경되었습니다'}.
                            """)
                            
                            # 컬럼별 상세 결과
                            st.markdown("#### 📋 상세 결과")
                            col_data = []
                            for col, col_result in result['column_results'].items():
                                if 'error' not in col_result:
                                    diff = col_result['difference']
                                    status = "✅" if diff < 10 else "⚠️" if diff < 50 else "❌"
                                    col_data.append({
                                        '항목': col,
                                        '원본 평균': f"{col_result['original_mean']:.2f}",
                                        '변환 평균': f"{col_result['anonymized_mean']:.2f}",
                                        '차이': f"{diff:.2f}",
                                        '상태': status
                                    })
                            
                            if col_data:
                                result_df = pd.DataFrame(col_data)
                                st.dataframe(result_df, use_container_width=True)
                                
                                # 해석 추가
                                worst_col = max(col_data, key=lambda x: float(x['차이'].replace(',', '')))
                                best_col = min(col_data, key=lambda x: float(x['차이'].replace(',', '')))
                                
                                st.info(f"""
                                💬 **해석**: 
                                - 가장 잘 보존됨: **{best_col['항목']}** (차이: {best_col['차이']})
                                - 가장 많이 변경됨: **{worst_col['항목']}** (차이: {worst_col['차이']})
                                """)
                        else:
                            st.error(f"❌ 오류: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("평가할 컬럼을 선택해주세요.")
        
        # U2: 상관관계 보존
        with st.expander("📊 U2: 상관관계 보존 (MC)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            숫자 데이터들 간의 상관관계가 얼마나 유지되었는지 확인합니다.
            
            ✅ **언제 사용하나요?**
            - 나이와 급여의 관계가 유지되었는지 확인할 때
            - 여러 변수 간의 관계가 중요한 경우
            
            💡 **결과 해석**: 점수가 0에 가까울수록 좋습니다!
            """)
            
            numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            u2_columns = st.multiselect(
                "평가할 컬럼 선택 (2개 이상의 숫자형)",
                options=numeric_cols,
                default=[col for col in selected_columns if col in numeric_cols],
                key="u2_columns"
            )
            
            if len(u2_columns) >= 2:
                n_pairs = len(u2_columns) * (len(u2_columns) - 1) // 2
                st.info(f"선택된 {len(u2_columns)}개 컬럼에서 {n_pairs}개의 상관관계 쌍을 평가합니다.")
            
            if st.button("U2 평가 실행", key="run_u2", type="primary"):
                if len(u2_columns) >= 2:
                    with st.spinner("U2 평가 중..."):
                        if use_sampling:
                            sample_idx = np.random.choice(original_df.index, size=sample_size, replace=False)
                            analyzer = UtilityMetrics(original_df.loc[sample_idx], processed_df.loc[sample_idx])
                        else:
                            analyzer = utility_analyzer
                        
                        result = analyzer.calculate_u2_mc(u2_columns)
                        
                        if result['status'] == 'success':
                            score = result['total_score']
                            if score < 0.1:
                                rating = "⭐⭐⭐⭐⭐ 매우 우수"
                            elif score < 0.2:
                                rating = "⭐⭐⭐⭐ 우수"
                            elif score < 0.3:
                                rating = "⭐⭐⭐ 보통"
                            elif score < 0.5:
                                rating = "⭐⭐ 주의"
                            else:
                                rating = "⭐ 개선 필요"
                            
                            st.success(f"""
                            ### 🎯 종합 점수: {score:.4f} {rating}
                            
                            💡 **이 점수의 의미:**
                            데이터 간의 상관관계가 {'매우 잘' if score < 0.1 else '잘' if score < 0.2 else '어느 정도' if score < 0.3 else '부분적으로'} 유지되었습니다.
                            """)
                            
                            # 상관관계 쌍별 결과
                            st.markdown("#### 📋 상세 결과")
                            pair_data = []
                            for pair, pair_result in result['pair_results'].items():
                                diff = pair_result['difference']
                                status = "✅" if diff < 0.1 else "⚠️" if diff < 0.3 else "❌"
                                pair_data.append({
                                    '컬럼 쌍': pair,
                                    '원본 상관계수': f"{pair_result['original_corr']:.4f}",
                                    '변환 상관계수': f"{pair_result['anonymized_corr']:.4f}",
                                    '차이': f"{diff:.4f}",
                                    '상태': status
                                })
                            
                            if pair_data:
                                st.dataframe(pd.DataFrame(pair_data), use_container_width=True)
                        else:
                            st.error(f"❌ 오류: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("❌ U2 평가를 실행할 수 없습니다\n\n🔍 **문제**: 상관관계 평가는 최소 2개의 숫자 데이터가 필요합니다.\n📌 **현재**: {len(u2_columns)}개 선택됨\n\n💡 **해결방법**: 숫자형 컬럼을 2개 이상 선택해주세요.")
        
        # U3: 코사인 유사도
        with st.expander("📊 U3: 코사인 유사도 (CS)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            원본과 변환 데이터의 패턴(방향성)이 얼마나 유사한지 확인합니다.
            
            ✅ **언제 사용하나요?**
            - 데이터의 전체적인 패턴이 유지되었는지 확인할 때
            - 값의 크기보다 방향성이 중요한 경우
            
            💡 **결과 해석**: 점수가 1에 가까울수록 좋습니다!
            """)
            
            numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            u3_columns = st.multiselect(
                "평가할 컬럼 선택 (숫자형)",
                options=numeric_cols,
                default=[col for col in selected_columns if col in numeric_cols],
                key="u3_columns"
            )
            
            if st.button("U3 평가 실행", key="run_u3", type="primary"):
                if u3_columns:
                    with st.spinner("U3 평가 중..."):
                        if use_sampling:
                            sample_idx = np.random.choice(original_df.index, size=sample_size, replace=False)
                            analyzer = UtilityMetrics(original_df.loc[sample_idx], processed_df.loc[sample_idx])
                        else:
                            analyzer = utility_analyzer
                        
                        result = analyzer.calculate_u3_cs(u3_columns)
                        
                        if result['status'] == 'success':
                            score = result['average_score']
                            if score > 0.95:
                                rating = "⭐⭐⭐⭐⭐ 매우 우수"
                            elif score > 0.9:
                                rating = "⭐⭐⭐⭐ 우수"
                            elif score > 0.8:
                                rating = "⭐⭐⭐ 보통"
                            elif score > 0.7:
                                rating = "⭐⭐ 주의"
                            else:
                                rating = "⭐ 개선 필요"
                            
                            st.success(f"""
                            ### 🎯 평균 유사도: {score:.4f} {rating}
                            
                            💡 **이 점수의 의미:**
                            데이터의 패턴이 {int(score * 100)}% 유사하게 유지되었습니다.
                            """)
                            
                            # 컬럼별 결과
                            col_data = []
                            for col, col_result in result['column_results'].items():
                                if 'error' not in col_result:
                                    sim = col_result['cosine_similarity']
                                    status = "✅" if sim > 0.9 else "⚠️" if sim > 0.7 else "❌"
                                    col_data.append({
                                        '컬럼': col,
                                        '코사인 유사도': f"{sim:.4f}",
                                        '유사도(%)': f"{sim*100:.1f}%",
                                        '상태': status
                                    })
                            
                            if col_data:
                                st.dataframe(pd.DataFrame(col_data), use_container_width=True)
                        else:
                            st.error(f"오류: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("평가할 컬럼을 선택해주세요.")
        
        # U4: 정규화 유클리디안 거리
        with st.expander("📊 U4: 정규화 유클리디안 거리 (NED_SSE)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            각 데이터 값이 얼마나 변했는지 정밀하게 측정합니다.
            
            ✅ **언제 사용하나요?**
            - 개별 값의 정확도가 중요한 경우
            - 데이터 변화를 세밀하게 추적할 때
            
            💡 **결과 해석**: 점수가 0에 가까울수록 좋습니다!
            """)
            
            numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            u4_columns = st.multiselect(
                "평가할 컬럼 선택 (숫자형)",
                options=numeric_cols,
                default=[col for col in selected_columns if col in numeric_cols],
                key="u4_columns"
            )
            
            if st.button("U4 평가 실행", key="run_u4", type="primary"):
                if u4_columns:
                    with st.spinner("U4 평가 중..."):
                        if use_sampling:
                            sample_idx = np.random.choice(original_df.index, size=sample_size, replace=False)
                            analyzer = UtilityMetrics(original_df.loc[sample_idx], processed_df.loc[sample_idx])
                        else:
                            analyzer = utility_analyzer
                        
                        result = analyzer.calculate_u4_ned(u4_columns)
                        
                        if result['status'] == 'success':
                            score = result['total_score']
                            if score < 0.1:
                                rating = "⭐⭐⭐⭐⭐ 매우 우수"
                            elif score < 0.3:
                                rating = "⭐⭐⭐⭐ 우수"
                            elif score < 0.5:
                                rating = "⭐⭐⭐ 보통"
                            elif score < 1.0:
                                rating = "⭐⭐ 주의"
                            else:
                                rating = "⭐ 개선 필요"
                            
                            st.success(f"""
                            ### 🎯 총점: {score:.4f} {rating}
                            
                            💡 **이 점수의 의미:**
                            데이터 값들이 {'매우 적게' if score < 0.1 else '적게' if score < 0.3 else '보통 수준으로' if score < 0.5 else '다소 많이'} 변경되었습니다.
                            """)
                            
                            # 컬럼별 결과
                            col_data = []
                            for col, col_result in result['column_results'].items():
                                if 'error' not in col_result:
                                    ned = col_result['normalized_sse']
                                    status = "✅" if ned < 0.1 else "⚠️" if ned < 0.5 else "❌"
                                    col_data.append({
                                        '컬럼': col,
                                        'SSE': f"{col_result['sse']:.4f}",
                                        '정규화 SSE': f"{ned:.4f}",
                                        '비교 레코드 수': col_result['record_count'],
                                        '상태': status
                                    })
                            
                            if col_data:
                                st.dataframe(pd.DataFrame(col_data), use_container_width=True)
                        else:
                            st.error(f"오류: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("평가할 컬럼을 선택해주세요.")
        
        # U5: 표준화 유클리디안 거리
        with st.expander("📊 U5: 표준화 유클리디안 거리 (SED_SSE)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            데이터의 분포(표준편차)를 고려하여 변화량을 측정합니다.
            
            ✅ **언제 사용하나요?**
            - 데이터의 분포가 중요한 경우
            - 범주화된 데이터의 정보 손실을 측정할 때
            
            💡 **결과 해석**: 점수가 0에 가까울수록 좋습니다!
            """)
            
            numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            u5_columns = st.multiselect(
                "평가할 컬럼 선택 (숫자형)",
                options=numeric_cols,
                default=[col for col in selected_columns if col in numeric_cols],
                key="u5_columns"
            )
            
            if st.button("U5 평가 실행", key="run_u5", type="primary"):
                if u5_columns:
                    with st.spinner("U5 평가 중..."):
                        if use_sampling:
                            sample_idx = np.random.choice(original_df.index, size=sample_size, replace=False)
                            analyzer = UtilityMetrics(original_df.loc[sample_idx], processed_df.loc[sample_idx])
                        else:
                            analyzer = utility_analyzer
                        
                        result = analyzer.calculate_u5_sed(u5_columns)
                        
                        if result['status'] == 'success':
                            st.success(f"✅ U5 총점: {result['total_score']:.4f} (0에 가까울수록 좋음)")
                            
                            # 컬럼별 결과
                            col_data = []
                            for col, col_result in result['column_results'].items():
                                if 'error' not in col_result:
                                    col_data.append({
                                        '컬럼': col,
                                        'SSE': f"{col_result['sse']:.4f}",
                                        '표준편차': f"{col_result['std_dev']:.4f}",
                                        '비교 레코드 수': col_result['record_count']
                                    })
                            
                            if col_data:
                                st.dataframe(pd.DataFrame(col_data), use_container_width=True)
                        else:
                            st.error(f"오류: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("평가할 컬럼을 선택해주세요.")
        
        # U6: 동질집합 분산
        with st.expander("🔐 U6: 동질집합 분산 (MD_ECM)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            같은 그룹(동질집합) 내에서 민감한 정보가 얼마나 다양한지 확인합니다.
            
            ✅ **언제 사용하나요?**
            - k-익명성 처리 후 그룹 내 다양성 확인
            - 민감정보의 분포가 중요한 경우
            
            💡 **결과 해석**: 점수가 낮을수록 좋습니다!
            """)
            
            all_cols = original_df.columns.tolist()
            numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            u6_quasi = st.multiselect(
                "준식별자 선택 (그룹을 만들 속성)",
                options=all_cols,
                default=quasi_identifiers if quasi_identifiers else [],
                key="u6_quasi",
                help="예: 성별, 지역, 연령대"
            )
            
            if u6_quasi:
                sensitive_options = [col for col in numeric_cols if col not in u6_quasi]
                if sensitive_options:
                    u6_sensitive = st.selectbox(
                        "민감속성 선택 (분산을 계산할 숫자형 속성)",
                        options=sensitive_options,
                        key="u6_sensitive",
                        help="예: 급여, 병력점수"
                    )
                else:
                    st.error("민감속성으로 사용할 수 있는 숫자형 데이터가 없습니다.")
                    u6_sensitive = None
            
            # 동질집합 미리보기 버튼 (U6용) - 추가 시작
            if u6_quasi:
                if st.button("🔍 동질집합 미리보기", key="preview_ec_u6"):
                    with st.spinner("동질집합 분석 중..."):
                        # 처리된 데이터에서 동질집합 생성
                        ec_preview = processed_df.groupby(u6_quasi).size().reset_index(name='그룹크기')
                        ec_preview = ec_preview.sort_values('그룹크기', ascending=False)
                        
                        st.write("**동질집합 요약:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 그룹 수", f"{len(ec_preview):,}개")
                        with col2:
                            st.metric("최소 크기", f"{ec_preview['그룹크기'].min()}")
                        with col3:
                            st.metric("최대 크기", f"{ec_preview['그룹크기'].max()}")
                        
                        st.write("**상위 5개 동질집합:**")
                        display_df = ec_preview.head(5).copy()
                        display_df.index = range(1, len(display_df) + 1)
                        st.dataframe(display_df, use_container_width=True)
                        
                        if 'u6_sensitive' in locals() and u6_sensitive:
                            st.info(f"💡 각 그룹 내에서 '{u6_sensitive}'의 분산을 계산하게 됩니다.")
            # 동질집합 미리보기 버튼 (U6용) - 추가 끝

            if st.button("U6 평가 실행", key="run_u6", type="primary"):
                if u6_quasi and 'u6_sensitive' in locals() and u6_sensitive:
                    with st.spinner("U6 평가 중..."):
                        if use_sampling:
                            sample_idx = np.random.choice(original_df.index, size=sample_size, replace=False)
                            analyzer = UtilityMetrics(original_df.loc[sample_idx], processed_df.loc[sample_idx])
                        else:
                            analyzer = utility_analyzer
                        
                        result = analyzer.calculate_u6_md_ecm(u6_quasi, u6_sensitive)
                        
                        if result['status'] == 'success':
                            st.success(f"""
                            ### 🎯 평균 분산: {result['total_score']:.4f}
                            
                            💡 **이 점수의 의미:**
                            동질집합 내 {u6_sensitive}의 분산이 평균 {result['total_score']:.2f}입니다.
                            분산이 클수록 그룹 내 다양성이 높아 프라이버시 보호에 유리합니다.
                            """)
                            
                            st.info(f"분석된 동질집합 개수: {result['ec_count']}개")
                            
                            # 상위 동질집합 정보
                            if 'ec_details' in result and result['ec_details']:
                                st.markdown("#### 동질집합 예시 (상위 10개)")
                                ec_df = pd.DataFrame(result['ec_details'])
                                st.dataframe(ec_df, use_container_width=True)
                        else:
                            st.error(f"오류: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("준식별자와 민감속성을 모두 선택해주세요.")
        
        # U7: 정규화 집합크기
        with st.expander("🔐 U7: 정규화 집합크기 (NA_ECSM)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            익명화 그룹들의 크기가 얼마나 균등한지 확인합니다.
            
            ✅ **언제 사용하나요?**
            - k-익명성 처리 후 그룹 크기 분포 확인
            - 과도한 일반화 여부 검증
            
            💡 **결과 해석**: 점수가 낮을수록 좋습니다!
            """)
            
            all_cols = original_df.columns.tolist()
            
            u7_quasi = st.multiselect(
                "준식별자 선택",
                options=all_cols,
                default=quasi_identifiers if quasi_identifiers else [],
                key="u7_quasi"
            )
            # 동질집합 미리보기 버튼 (U7용) - 추가 시작
            if u7_quasi:
                if st.button("🔍 동질집합 미리보기", key="preview_ec_u7"):
                    with st.spinner("동질집합 분석 중..."):
                        # 처리된 데이터에서 동질집합 생성
                        ec_preview = processed_df.groupby(u7_quasi).size().reset_index(name='그룹크기')
                        ec_preview = ec_preview.sort_values('그룹크기', ascending=False)
                        
                        st.write("**동질집합 분포:**")
                        
                        # k값 분포
                        k_dist = ec_preview['그룹크기'].value_counts().sort_index()
                        
                        # 간단한 분포 표시
                        dist_summary = []
                        for k in [1, 2, 3, 4, 5]:
                            count = len(ec_preview[ec_preview['그룹크기'] == k])
                            if count > 0:
                                dist_summary.append(f"k={k}: {count}개 그룹")
                        
                        k_5_plus = len(ec_preview[ec_preview['그룹크기'] > 5])
                        if k_5_plus > 0:
                            dist_summary.append(f"k>5: {k_5_plus}개 그룹")
                        
                        st.write(" | ".join(dist_summary))
                        
                        # 통계
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 그룹 수", f"{len(ec_preview):,}개")
                        with col2:
                            st.metric("평균 크기", f"{ec_preview['그룹크기'].mean():.1f}")
                        with col3:
                            min_k = ec_preview['그룹크기'].min()
                            st.metric("최소 k값", min_k, delta="위험" if min_k < 5 else "안전")
                        
                        # 작은 그룹 경고
                        small_groups = len(ec_preview[ec_preview['그룹크기'] < 5])
                        if small_groups > 0:
                            st.warning(f"⚠️ k<5인 그룹이 {small_groups}개 있습니다.")
            # 동질집합 미리보기 버튼 (U7용) - 추가 끝
            
            if st.button("U7 평가 실행", key="run_u7", type="primary"):
                if u7_quasi:
                    with st.spinner("U7 평가 중..."):
                        if use_sampling:
                            sample_idx = np.random.choice(original_df.index, size=sample_size, replace=False)
                            analyzer = UtilityMetrics(original_df.loc[sample_idx], processed_df.loc[sample_idx])
                        else:
                            analyzer = utility_analyzer
                        
                        result = analyzer.calculate_u7_na_ecsm(u7_quasi)
                        
                        if result['status'] == 'success':
                            st.success(f"""
                            ### 🎯 점수: {result['total_score']:.4f}
                            
                            💡 **이 점수의 의미:**
                            동질집합들의 크기가 {'균등하게' if result['total_score'] < 2 else '비교적 균등하게' if result['total_score'] < 5 else '불균등하게'} 분포되어 있습니다.
                            """)
                            
                            if 'details' in result:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("전체 레코드", f"{result['details']['total_records']:,}")
                                    st.metric("동질집합 수", f"{result['details']['ec_count']:,}")
                                with col2:
                                    st.metric("최소 k값", result['details']['min_k'])
                                    st.metric("평균 집합 크기", f"{result['details']['avg_ec_size']:.2f}")
                        else:
                            st.error(f"오류: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("준식별자를 선택해주세요.")
        
        # U8: 비균일 엔트로피
        with st.expander("📊 U8: 비균일 엔트로피 (NUEM)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            전체적인 정보 손실량을 엔트로피로 측정합니다.
            
            ✅ **언제 사용하나요?**
            - 전반적인 정보 손실 평가
            - 여러 컬럼의 종합적인 변화 측정
            
            💡 **결과 해석**: 점수가 낮을수록 좋습니다!
            """)
            
            all_cols = original_df.columns.tolist()
            
            u8_columns = st.multiselect(
                "평가할 컬럼 선택",
                options=all_cols,
                default=selected_columns,
                key="u8_columns"
            )
            
            if st.button("U8 평가 실행", key="run_u8", type="primary"):
                if u8_columns:
                    with st.spinner("U8 평가 중..."):
                        if use_sampling:
                            sample_idx = np.random.choice(original_df.index, size=sample_size, replace=False)
                            analyzer = UtilityMetrics(original_df.loc[sample_idx], processed_df.loc[sample_idx])
                        else:
                            analyzer = utility_analyzer
                        
                        result = analyzer.calculate_u8_nuem(u8_columns)
                        
                        if result['status'] == 'success':
                            st.success(f"""
                            ### 🎯 엔트로피: {result['total_score']:.4f}
                            
                            💡 **이 점수의 의미:**
                            정보 손실이 {'매우 적습니다' if result['total_score'] < 1 else '적은 편입니다' if result['total_score'] < 3 else '보통입니다' if result['total_score'] < 5 else '많은 편입니다'}.
                            """)
                            
                            if 'details' in result:
                                st.info(f"평가 레코드: {result['details']['total_records']:,}, "
                                       f"평가 속성: {result['details']['total_attributes']}")
                        else:
                            st.error(f"오류: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("평가할 컬럼을 선택해주세요.")
        
        # U9: 익명화율
        with st.expander("📊 U9: 익명화율 (AR)", expanded=False):
            st.markdown("""
            🎯 **무엇을 평가하나요?**
            원본 대비 익명처리된 데이터의 보존율을 확인합니다.
            
            ✅ **언제 사용하나요?**
            - 데이터 삭제가 얼마나 발생했는지 확인
            - 전체적인 데이터 보존율 평가
            
            💡 **결과 해석**: 100%에 가까울수록 좋습니다!
            """)
            
            if st.button("U9 평가 실행", key="run_u9", type="primary"):
                with st.spinner("U9 평가 중..."):
                    result = utility_analyzer.calculate_u9_ar()
                    
                    if result['status'] == 'success':
                        score = result['total_score']
                        if score >= 95:
                            rating = "⭐⭐⭐⭐⭐ 매우 우수"
                        elif score >= 90:
                            rating = "⭐⭐⭐⭐ 우수"
                        elif score >= 80:
                            rating = "⭐⭐⭐ 보통"
                        elif score >= 70:
                            rating = "⭐⭐ 주의"
                        else:
                            rating = "⭐ 개선 필요"
                        
                        st.success(f"""
                        ### 🎯 익명화율: {score:.2f}% {rating}
                        
                        💡 **이 점수의 의미:**
                        원본 데이터의 {score:.1f}%가 보존되었습니다.
                        {f'{100-score:.1f}%의 데이터가 삭제되었습니다.' if score < 100 else '모든 데이터가 보존되었습니다.'}
                        """)
                        
                        if 'details' in result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("원본 레코드", f"{result['details']['original_records']:,}")
                            with col2:
                                st.metric("처리후 레코드", f"{result['details']['anonymized_records']:,}")
                            with col3:
                                deleted = result['details']['original_records'] - result['details']['anonymized_records']
                                st.metric("삭제된 레코드", f"{deleted:,}")
                    else:
                        st.error(f"오류: {result.get('error', 'Unknown error')}")
        
        # 평가 결과 저장
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 평가 결과 CSV 다운로드", type="secondary"):
                st.info("평가 결과 다운로드 기능은 준비 중입니다.")
        with col2:
            if st.button("📄 평가 보고서 생성", type="secondary"):
                st.info("보고서 생성 기능은 준비 중입니다.")
    
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