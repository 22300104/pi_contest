# privacy_evaluation.py (개선 버전)

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import json, uuid, time

# k-익명성 계산을 위한 새로운 모듈도 필요합니다
try:
    from modules.privacy_metrics.k_anonymity import KAnonymityAnalyzer
except ImportError:
    # 모듈이 아직 없으면 임시로 처리
    pass

def get_column_types():
    """전역 설정된 컬럼 타입 반환"""
    return {
        'numeric': st.session_state.get('global_numeric_cols', []),
        'categorical': st.session_state.get('global_categorical_cols', []),
        'datetime': st.session_state.get('global_datetime_cols', [])
    }

def render_privacy_evaluation_tab():
    """프라이버시 평가 탭 렌더링"""
    st.header("📋 프라이버시 평가")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("먼저 데이터를 업로드해주세요.")
        return
    
    # 처리된 데이터가 있으면 그것을 사용, 없으면 원본 사용
    df = st.session_state.get("df_processed", st.session_state.df)
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📊 k-익명성 분석", "📈 유용성 평가"])
    
    with tab1:
        render_k_anonymity_section(df)
    
    with tab2:
        render_utility_evaluation_section(df)

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
        
        # 🔴 표본률 입력 추가
        st.markdown("#### 📊 표본률 설정")
        sample_rate = st.number_input(
            "표본률 (f)",
            min_value=0.001,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.3f",
            help="전체 모집단 대비 현재 데이터의 비율 (1.0 = 전체 데이터)"
        )
        
        # 표본률에 따른 설명
        if sample_rate < 1.0:
            st.info(f"📌 현재 데이터는 전체 모집단의 {sample_rate*100:.1f}%입니다")
        else:
            st.info("📌 전체 모집단 데이터로 분석합니다")
        
            # 🔴 표본률 입력 추가
    st.markdown("#### 📊 표본률 설정")
    sample_rate = st.number_input(
        "표본률 (f)",
        min_value=0.001,
        max_value=1.0,
        value=1.0,
        step=0.01,
        format="%.3f",
        help="전체 모집단 대비 현재 데이터의 비율 (1.0 = 전체 데이터)"
    )
    
    # 표본률에 따른 설명
    if sample_rate < 1.0:
        st.info(f"📌 현재 데이터는 전체 모집단의 {sample_rate*100:.1f}%입니다")
    else:
        st.info("📌 전체 모집단 데이터로 분석합니다")
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
    
    # 동질집합 미리보기 섹션
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
            
            if st.button("동질집합 확인하기", key="preview_ec_k"):
                with st.spinner("동질집합 분석 중..."):
                    # 샘플링 제거 - df를 직접 사용
                    preview_df = df
                    
                    # 동질집합 생성
                    ec_groups = preview_df.groupby(selected_qi)
                    
                    # 각 그룹의 크기 계산
                    ec_sizes = ec_groups.size().reset_index(name='k')
                    
                    if len(ec_sizes) == 0:
                        st.warning("동질집합을 생성할 수 없습니다.")
                    else:
                        # 정렬 옵션에 따라 정렬
                        if preview_option == "k값이 낮은 위험 그룹":
                            ec_sizes = ec_sizes.sort_values('k', ascending=True)
                            top_groups = ec_sizes.head(5)
                        elif preview_option == "상위 5개 그룹":
                            ec_sizes = ec_sizes.sort_values('k', ascending=False)
                            top_groups = ec_sizes.head(5)
                        else:  # 랜덤
                            top_groups = ec_sizes.sample(min(5, len(ec_sizes)))
                        
                        # k값 순서 정렬 옵션 추가
                        sort_by_k = st.checkbox("k값 순서로 정렬", value=True, key="sort_by_k")
                        if sort_by_k and preview_option != "랜덤 샘플":
                            ascending = preview_option == "k값이 낮은 위험 그룹"
                            top_groups = top_groups.sort_values('k', ascending=ascending)
                
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
                        mask = mask & (preview_df[qi] == group_info[qi])
                    
                    group_records = preview_df[mask].head(show_records)
                    
                    # 민감한 정보는 가리고 표시
                    display_cols = selected_qi + [col for col in preview_df.columns 
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
                    if k < k_threshold:
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
    


    # privacy_evaluation.py의 render_k_anonymity_section 함수 내
    # "동질집합 미리보기" 섹션 다음에 추가

    with st.expander("📊 EC별 통계 분석", expanded=False):
        st.info("""
        **EC별 통계 분석이란?**
        각 동질집합(EC)별로 선택한 속성들의 통계를 계산합니다.
        - 수치형: 평균, 표준편차, 최소/최대값
        - 범주형: 고유값 수, 엔트로피, 최빈값
        """)
        
        # 통계 대상 컬럼 선택
        all_cols = df.columns.tolist()
        available_cols = [col for col in all_cols if col not in selected_qi]
        
        target_cols = st.multiselect(
            "통계를 계산할 속성 선택",
            available_cols,
            help="준식별자를 제외한 속성들을 선택하세요",
            key="ec_stat_target_cols"
        )
        
        if target_cols:
            # EC 필터 옵션
            use_filter = st.checkbox("특정 EC만 조회", key="ec_filter_check")
            
            ec_selection = None
            if use_filter:
                st.markdown("**조회할 EC 조건 입력**")
                ec_filters = []
                
                # 각 준식별자별 필터 입력
                filter_cols = st.columns(len(selected_qi))
                for i, qi in enumerate(selected_qi):
                    with filter_cols[i]:
                        # 해당 컬럼의 고유값 가져오기
                        unique_vals = df[qi].dropna().unique()
                        
                        if len(unique_vals) <= 20:
                            # 값이 적으면 선택박스
                            selected_val = st.selectbox(
                                f"{qi}",
                                ["전체"] + list(unique_vals),
                                key=f"ec_filter_{qi}"
                            )
                            if selected_val != "전체":
                                ec_filters.append({qi: selected_val})
                        else:
                            # 값이 많으면 텍스트 입력
                            input_val = st.text_input(
                                f"{qi}",
                                placeholder="값 입력",
                                key=f"ec_filter_input_{qi}"
                            )
                            if input_val:
                                # 숫자 변환 시도
                                try:
                                    if df[qi].dtype in ['int64', 'float64']:
                                        ec_filters.append({qi: float(input_val)})
                                    else:
                                        ec_filters.append({qi: input_val})
                                except:
                                    ec_filters.append({qi: input_val})
                
                # 필터 조합
                if ec_filters:
                    ec_selection = [dict(pair for d in ec_filters for pair in d.items())]
            
            # 통계 계산 버튼
            if st.button("📊 EC별 통계 계산", key="calc_ec_stats"):
                with st.spinner("EC별 통계 계산 중..."):
                    try:
                        # 통계 계산
                        ec_stats_df = calculate_ec_statistics(
                            df=df,
                            ec_cols=selected_qi,
                            target_cols=target_cols,
                            ec_selection=ec_selection
                        )
                        
                        if len(ec_stats_df) == 0:
                            st.warning("조건에 맞는 EC가 없습니다.")
                        else:
                            # 결과 표시
                            st.success(f"✅ {len(ec_stats_df)}개 EC에 대한 통계 계산 완료!")
                            
                            # 요약 정보
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("전체 EC 수", f"{len(ec_stats_df):,}개")
                            with col2:
                                avg_size = ec_stats_df['EC_SIZE'].mean()
                                st.metric("평균 EC 크기", f"{avg_size:.1f}")
                            with col3:
                                total_records = ec_stats_df['EC_SIZE'].sum()
                                st.metric("총 레코드 수", f"{total_records:,}")
                            
                            # 통계 테이블 표시
                            st.markdown("### 📊 EC별 통계 결과")
                            
                            # 표시할 컬럼 선택
                            display_cols = selected_qi + ['EC_SIZE']
                            for target_col in target_cols:
                                # 해당 target_col 관련 통계 컬럼들 추가
                                stat_cols = [col for col in ec_stats_df.columns if col.startswith(f'{target_col}_')]
                                display_cols.extend(stat_cols)
                            
                            # 데이터프레임 표시
                            st.dataframe(
                                ec_stats_df[display_cols],
                                use_container_width=True,
                                height=400
                            )
                            
                            # 다운로드 버튼
                            csv = ec_stats_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 CSV 다운로드",
                                data=csv.encode('utf-8-sig'),
                                file_name=f"ec_statistics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # 세션 상태에 저장
                            if 'ec_statistics' not in st.session_state:
                                st.session_state.ec_statistics = {}
                            st.session_state.ec_statistics['latest'] = {
                                'df': ec_stats_df,
                                'ec_cols': selected_qi,
                                'target_cols': target_cols,
                                'timestamp': pd.Timestamp.now()
                            }
                            
                                                        # EC별 엔트로피 분포 시각화
                            if any(col.endswith('_entropy') for col in ec_stats_df.columns):
                                with st.expander("📈 엔트로피 분포 시각화", expanded=False):
                                    entropy_cols = [col for col in ec_stats_df.columns if col.endswith('_entropy')]
                                    
                                    for ent_col in entropy_cols:
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        
                                        # 히스토그램
                                        ec_stats_df[ent_col].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
                                        ax.set_xlabel('엔트로피')
                                        ax.set_ylabel('EC 수')
                                        ax.set_title(f'{ent_col.replace("_entropy", "")} 엔트로피 분포')
                                        
                                        # 평균선 추가
                                        mean_entropy = ec_stats_df[ent_col].mean()
                                        ax.axvline(mean_entropy, color='red', linestyle='--', 
                                                label=f'평균: {mean_entropy:.3f}')
                                        ax.legend()
                                        
                                        st.pyplot(fig)
                                        plt.close()
                    
                    except Exception as e:
                        st.error(f"통계 계산 중 오류 발생: {str(e)}")
                        st.exception(e)


    # k-익명성 분석 실행 버튼
    if st.button("🔍 k-익명성 분석 실행", type="primary", disabled=len(selected_qi) == 0):
        if len(selected_qi) == 0:
            st.error("최소 하나 이상의 준식별자를 선택해주세요.")
            return
        
        with st.spinner("k-익명성 분석 중..."):
            # 샘플링 적용
            if use_sampling and sample_size < len(df): 
                analysis_df = df.sample(n=sample_size)
            else:
                analysis_df = df
            
            # k-익명성 계산
            try:
                k_value, k_stats = calculate_k_anonymity(
                    analysis_df,
                    selected_qi,
                    k_threshold,
                    sample_rate  # 🔴 추가
                )
                
                # 결과 표시
                st.markdown("### 📊 분석 결과")
                

                # 주요 지표 표시 (🔴 EMP 추가로 5개 컬럼으로 변경)
                col_a, col_b, col_c, col_d, col_e = st.columns(5)

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

                # 🔴 EMP 메트릭 추가
                with col_e:
                    emp_value = k_stats['emp']
                    emp_percent = emp_value * 100
                    st.metric(
                        "EMP",
                        f"{emp_percent:.2f}%",
                        delta=k_stats['emp_risk_level'],
                        delta_color="inverse" if emp_value > 0.05 else "normal"
                    )

                # 🔴 EMP 상세 정보 추가
                st.markdown("### 🎯 EMP (Expected Match Probability) 분석")
                emp_col1, emp_col2, emp_col3 = st.columns(3)

                with emp_col1:
                    st.info(f"""
                    **EMP 값**: {k_stats['emp']:.6f} ({k_stats['emp']*100:.3f}%)
                    **위험 수준**: {k_stats['emp_risk_level']}
                    """)

                with emp_col2:
                    st.info(f"""
                    **표본률**: {k_stats['sample_rate']}
                    **평균 개인 위험도**: {k_stats['avg_individual_risk']:.4f}
                    """)

                with emp_col3:
                    st.info(f"""
                    **고위험 레코드**: {k_stats['high_risk_records']:,}개
                    **전체 위험도 합**: {k_stats['total_risk_sum']:.2f}
                    """)

                # EMP 해석 가이드
                with st.expander("💡 EMP 해석 가이드", expanded=False):
                    st.markdown("""
                    **EMP(Expected Match Probability)**는 데이터셋에서 개인이 재식별될 기대 확률입니다.
                    
                    - **< 1%**: 매우 안전 (재식별 위험 매우 낮음)
                    - **1-5%**: 안전 (재식별 위험 낮음)
                    - **5-10%**: 주의 필요 (중간 수준 위험)
                    - **10-20%**: 위험 (재식별 위험 높음)
                    - **> 20%**: 매우 위험 (즉시 조치 필요)
                    
                    **계산 방법**: Skinner-Elliot 모델 기반
                    - 개별 위험도 = 1 / (표본률 × EC크기)
                    - EMP = (표본률 / 전체레코드수) × Σ개별위험도
                    """)
                
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
                    'sampled': use_sampling,
                    'sample_rate': sample_rate,  # 🔴 추가
                    'emp': k_stats['emp'],  # 🔴 추가
                    'emp_risk_level': k_stats['emp_risk_level']  # 🔴 추가
}
                
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

def calculate_k_anonymity(
        df: pd.DataFrame,
        quasi_identifiers: List[str],
        k_threshold: int = 5,
        sample_rate: float = 1.0  # 🔴 새 파라미터 추가
) -> Tuple[int, Dict]:
    """
    선택한 준식별자에 대해 k-익명성 통계 및 EMP 계산
    
    Args:
        df: 데이터프레임
        quasi_identifiers: 준식별자 리스트
        k_threshold: k값 임계값
        sample_rate: 표본률 (f값, 0 < f <= 1)
    
    Returns:
        k_value : 전체 데이터의 최소 k
        k_stats : 상세 통계 딕셔너리 (EMP 포함)
    """
    # 1) 동질집합 크기 계산
    group_sizes = (
        df.groupby(quasi_identifiers)
          .size()
          .reset_index(name='count')
    )

    k_value = int(group_sizes['count'].min())

    # 2) 위험 레코드( k < k_threshold ) 집합 추출
    risk_ec = group_sizes[group_sizes['count'] < k_threshold][quasi_identifiers]
    risk_records_detail = df.merge(
        risk_ec,
        on=quasi_identifiers,
        how='inner'
    )
    
    # 🔴 3) EMP 계산 추가
    # EMP 계산을 위한 개별 위험도 계산
    n = len(df)  # 전체 레코드 수
    total_risk = 0.0
    
    # 각 레코드의 EC 크기를 찾아서 위험도 계산
    for _, row in df.iterrows():
        # 해당 레코드가 속한 EC 찾기
        ec_condition = True
        for qi in quasi_identifiers:
            ec_condition &= (group_sizes[qi] == row[qi])
        
        # EC 크기 찾기 (더 효율적인 방법)
        ec_match = group_sizes
        for qi in quasi_identifiers:
            ec_match = ec_match[ec_match[qi] == row[qi]]
        
        if len(ec_match) > 0:
            ec_size = ec_match.iloc[0]['count']
            # Skinner-Elliot 공식: risk_i = 1 / (f * |EC_i|)
            risk_i = 1 / (sample_rate * ec_size)
            if risk_i > 1:
                risk_i = 1  # 위험도는 최대 1
            total_risk += risk_i
    
    # 더 효율적인 방법: merge를 사용
    # 각 레코드에 EC 크기 정보 추가
    df_with_ec_size = df.merge(
        group_sizes.rename(columns={'count': 'ec_size'}),
        on=quasi_identifiers,
        how='left'
    )
    
    # 각 레코드의 위험도 계산
    df_with_ec_size['risk_i'] = 1 / (sample_rate * df_with_ec_size['ec_size'])
    df_with_ec_size['risk_i'] = df_with_ec_size['risk_i'].clip(upper=1)  # 최대값 1로 제한
    
    total_risk = df_with_ec_size['risk_i'].sum()
    
    # EMP = (f / N) * Σ risk_i
    emp = (sample_rate / n) * total_risk

    # 4) 통계 딕셔너리 작성
    k_stats = {
        'min_k': k_value,
        'max_k': int(group_sizes['count'].max()),
        'avg_k': float(group_sizes['count'].mean()),
        'median_k': int(group_sizes['count'].median()),
        'k_distribution': group_sizes['count']
                          .value_counts()
                          .sort_index()
                          .to_dict(),
        'risk_records': len(risk_records_detail),
        'risk_records_detail': risk_records_detail,
        # 🔴 EMP 관련 통계 추가
        'emp': emp,
        'sample_rate': sample_rate,
        'total_risk_sum': total_risk,
        'avg_individual_risk': total_risk / n,
        'high_risk_records': len(df_with_ec_size[df_with_ec_size['risk_i'] >= 0.5]),  # 위험도 50% 이상
        'emp_risk_level': get_emp_risk_level(emp)  # 위험 수준 평가
    }

    return k_value, k_stats

# 🔴 EMP 위험 수준 평가 함수 추가
def get_emp_risk_level(emp: float) -> str:
    """EMP 값에 따른 위험 수준 평가"""
    if emp < 0.01:
        return "매우 낮음"
    elif emp < 0.05:
        return "낮음"
    elif emp < 0.1:
        return "중간"
    elif emp < 0.2:
        return "높음"
    else:
        return "매우 높음"

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

def render_utility_evaluation_section(_: pd.DataFrame):
    """유용성 평가 탭 – 개선된 버전"""
    st.subheader("📈 유용성 평가")
    
    # 지표 메타 정보 (개선된 임계값)
    METRIC_INFO = {
        'U1': {
            'name': '평균값 차이',
            'desc': '두 데이터셋 평균값이 얼마나 다른지 측정 (0에 가까울수록 좋음)',
            'thresholds': {'excellent': 0.05, 'good': 0.1, 'fair': 0.5},
            'lower_better': True
        },
        'U2': {
            'name': '상관계수 보존',
            'desc': '원본·비식별 상관계수 차이 평균 (0에 가까울수록 좋음)',
            'thresholds': {'excellent': 0.01, 'good': 0.05, 'fair': 0.1},
            'lower_better': True
        },
        'U3': {
            'name': '코사인 유사도',
            'desc': '벡터 유사도 평균 (1에 가까울수록 좋음)',
            'thresholds': {'excellent': 0.99, 'good': 0.95, 'fair': 0.9},
            'lower_better': False
        },
        'U4': {
            'name': '정규화 거리',
            'desc': '정규화 SSE 합 (0에 가까울수록 좋음)',
            'thresholds': {'excellent': 0.01, 'good': 0.05, 'fair': 0.1},
            'lower_better': True
        },
        'U5': {
            'name': '표준화 거리',
            'desc': '표준화 SSE 합 (0에 가까울수록 좋음)',
            'thresholds': {'excellent': 0.1, 'good': 0.5, 'fair': 1.0},
            'lower_better': True
        },
        'U6': {
            'name': '동질집합 분산',
            'desc': '동질집합 내 민감값 분산 평균 (낮을수록 정보 유지)',
            'thresholds': {'excellent': 0.5, 'good': 1.0, 'fair': 2.0},
            'lower_better': True
        },
        'U7': {
            'name': '정규화 집합크기',
            'desc': '(N/N_EC)/k : 동질집합 크기 지표 (낮을수록 안전)',
            'thresholds': {'excellent': 1.0, 'good': 2.0, 'fair': 5.0},
            'lower_better': True
        },
        'U8': {
            'name': '비균일 엔트로피',
            'desc': '변경 레코드 엔트로피 (0에 가까울수록 원본과 유사)',
            'thresholds': {'excellent': 0.1, 'good': 0.3, 'fair': 0.5},
            'lower_better': True
        },
        'U9': {
            'name': '익명화율',
            'desc': '비식별 데이터가 얼마나 남았는지 (%) (높을수록 활용 ↑)',
            'thresholds': {'excellent': 95, 'good': 90, 'fair': 80},
            'lower_better': False
        },
    }
    
    # 도움말 토글
    show_help = st.toggle("👶 처음이라면 도움말 보기", value=False)
    if show_help:
        st.info("**유용성(U) 지표란?**\n\n" + 
                "\n".join([f"• **{k} ({v['name']})** : {v['desc']}" 
                          for k, v in METRIC_INFO.items()]))

    # 1. 데이터 존재 확인
    if 'df' not in st.session_state or 'df_processed' not in st.session_state:
        st.warning("먼저 데이터를 업로드하고 비식별화를 완료해주세요.")
        return

    orig_df = st.session_state.df
    proc_df = st.session_state.df_processed

    # 2. 무조건 변환된 데이터 타입 사용
    st.info("📌 변환된 데이터의 타입을 기준으로 평가합니다.")
    
    # 변환된 데이터 기준으로 컬럼 분류
    numeric_cols = proc_df.select_dtypes(include='number').columns.tolist()
    all_cols = proc_df.columns.tolist()

    # 3. 평가 대상 컬럼
    if 'util_cols' not in st.session_state:
        st.session_state.util_cols = numeric_cols

    st.markdown("### ① 평가 대상 컬럼")
    left, right = st.columns([3, 1])
    with right:
        if st.button("숫자형만"):
            st.session_state.util_cols = numeric_cols
            st.rerun()

        if st.button("전체"):
            st.session_state.util_cols = all_cols
            st.rerun()

        if st.button("초기화"):
            st.session_state.util_cols = []
            st.rerun()

    with left:
        sel_cols = st.multiselect(
            "컬럼 선택", all_cols,
            default=st.session_state.util_cols, key="util_cols"
        )
    
    if not sel_cols:
        st.info("컬럼을 한 개 이상 선택하세요.")
        return
    sel_num = [c for c in sel_cols if c in numeric_cols]

    # UtilityMetrics 준비
    from modules.privacy_metrics.utility_metrics import UtilityMetrics
    utility_analyzer = UtilityMetrics(orig_df, proc_df) 

    # 4. 지표 선택 & QI 옵션
    st.markdown("### ② 지표 선택")
    
    metrics = st.multiselect(
        "실행할 지표", list(METRIC_INFO.keys()),
        default=['U1', 'U2', 'U9'],
        format_func=lambda m: f"{m} – {METRIC_INFO[m]['name']}"
    )

    # 선택한 지표 설명 패널
    if metrics:
        with st.container():
            st.markdown("#### 선택 지표 설명")
            for m in metrics:
                st.markdown(f"**{m} – {METRIC_INFO[m]['name']}**  \n"
                            f"{METRIC_INFO[m]['desc']}")
    
    qi_cols, sens_attr = [], None
    if any(m in metrics for m in ('U6', 'U7')):
        with st.expander("🔐 QI·민감속성", expanded=True):
            qi_cols = st.multiselect("준식별자(QI)", options=sel_cols)
            if qi_cols:
                cand = [c for c in sel_num if c not in qi_cols]
                if cand:
                    sens_attr = st.selectbox("민감속성", cand)

    # 5. 샘플링
    st.markdown("### ③ 샘플링")
    use_samp = st.toggle("샘플링 사용", value=True)
    samp_rows = st.slider(
        "샘플 행 수", 10_000, min(1_000_000, len(orig_df)),
        100_000, step=10_000, disabled=not use_samp, format="%d 행"
    )
    analysis_df = orig_df.sample(samp_rows, random_state=42) if use_samp and samp_rows < len(orig_df) else orig_df

    # 6. 실행
    st.markdown("### ④ 평가 실행")
    if st.button("🚀 선택한 지표 실행", type="primary"):
        run_id = uuid.uuid4().hex[:8]
        summary, detail_results = [], {}
        prog = st.progress(0.0)
        total = len(metrics)

        # 결과 리스트에 행 추가하는 헬퍼 (개선)
        def push(metric: str, res: dict, used_cols: list):
            """summary·detail 두 곳에 결과를 저장"""
            metric_info = METRIC_INFO[metric]
            
            # U2 상관계수는 특별 처리
            if metric == 'U2' and res.get('status') == 'success':
                # 전체 점수
                score = res.get('total_score', 0)
                summary.append({
                    '지표': metric,
                    '지표명': metric_info['name'],
                    '컬럼': "전체 상관계수",
                    '점수': round(score, 4) if isinstance(score, (int, float)) else score,
                    '평가': get_score_badge(metric, score, METRIC_INFO)
                })
                
                # 개별 쌍 결과
                if 'pair_results' in res:
                    for pair, pair_res in res['pair_results'].items():
                        summary.append({
                            '지표': metric,
                            '지표명': metric_info['name'],
                            '컬럼': pair,
                            '점수': round(pair_res['difference'], 4),
                            '평가': get_score_badge(metric, pair_res['difference'], METRIC_INFO)
                        })
            
            # 컬럼별 점수를 분해해서 보여줘야 하는 지표
            elif metric in ('U1', 'U3', 'U4', 'U5') and res.get('status') == 'success':
                for col, det in res['column_results'].items():
                    if 'error' in det:
                        continue
                    val = det.get('difference') or det.get('cosine_similarity') \
                          or det.get('normalized_sse') or det.get('sse')
                    summary.append({
                        '지표': metric,
                        '지표명': metric_info['name'],
                        '컬럼': col,
                        '점수': round(val, 4) if isinstance(val, (int, float)) else val,
                        '평가': get_score_badge(metric, val, METRIC_INFO)
                    })
            else:
                score = res.get('total_score') or res.get('average_score')
                summary.append({
                    '지표': metric,
                    '지표명': metric_info['name'],
                    '컬럼': ", ".join(used_cols) if used_cols else '-',
                    '점수': round(score, 4) if isinstance(score, (int, float)) else score,
                    '평가': get_score_badge(metric, score, METRIC_INFO)
                })
            detail_results[metric] = res

        # 선택한 지표 순차 실행
        for i, m in enumerate(metrics, 1):
            prog.progress(i / total, text=f"{m} 계산 중…")

            if m == 'U1':
                push(m, utility_analyzer.calculate_u1_ma(sel_num), sel_num)
            elif m == 'U2':
                push(m, utility_analyzer.calculate_u2_mc(sel_num), sel_num)
            elif m == 'U3':
                push(m, utility_analyzer.calculate_u3_cs(sel_num), sel_num)
            elif m == 'U4':
                push(m, utility_analyzer.calculate_u4_ned(sel_num), sel_num)
            elif m == 'U5':
                push(m, utility_analyzer.calculate_u5_sed(sel_num), sel_num)
            elif m == 'U6' and qi_cols and sens_attr:
                push(m, utility_analyzer.calculate_u6_md_ecm(qi_cols, sens_attr), [sens_attr])
            elif m == 'U7' and qi_cols:
                push(m, utility_analyzer.calculate_u7_na_ecsm(qi_cols), qi_cols)
            elif m == 'U8':
                push(m, utility_analyzer.calculate_u8_nuem(sel_num), sel_num)
            elif m == 'U9':
                push(m, utility_analyzer.calculate_u9_ar(), [])

        prog.empty()

        # 히스토리 세션 스토리지
        st.session_state.setdefault('util_history', []).append({
            'id':     run_id,
            'time':   time.strftime("%H:%M:%S"),
            'rows':   len(analysis_df),
            'summary': summary,
            'detail':  detail_results,
        })

    # 7. 결과 표시 (개선)
    if st.session_state.get("util_history"):
        latest = st.session_state.util_history[-1]
        st.markdown(f"### ⑤ 결과 요약 ({latest['time']})")

        # A. 주요 메트릭 카드
        card_metrics = ["U1", "U2", "U3", "U9"]
        cols = st.columns(len(card_metrics))

        for col, m in zip(cols, card_metrics):
            # 해당 지표의 첫 번째 결과 찾기
            row = next((r for r in latest["summary"] if r["지표"] == m), None)
            if not row:
                col.empty()
                continue

            # 색상과 아이콘 결정
            badge = row['평가']
            emoji = badge.split()[0] if badge else "⚪"
            
            col.metric(
                label=f"{m} - {METRIC_INFO[m]['name']}",
                value=f"{row['점수']} {emoji}",
                help=badge
            )

        # B. 개선된 요약표
        df_sum = pd.DataFrame(latest["summary"])
        
        # 컬럼 순서 조정
        display_cols = ['지표', '지표명', '컬럼', '점수', '평가']
        df_sum = df_sum[display_cols]
        
        # 스타일 적용
        styled_df = df_sum.style.apply(lambda x: [
            'background-color: #e8f5e9' if '🟢' in str(v) else
            'background-color: #fff3e0' if '🟡' in str(v) else
            'background-color: #ffebee' if '🔴' in str(v) else ''
            for v in x
        ], axis=1)
        
        st.dataframe(styled_df, hide_index=True, use_container_width=True)

        # C. 상세 결과
        with st.expander("🔍 상세 결과 보기"):
            for metric in latest["detail"]:
                st.markdown(f"#### {metric} - {METRIC_INFO[metric]['name']}")
                st.json(latest["detail"][metric])

        # D. 다운로드
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ 요약 CSV",
                df_sum.to_csv(index=False, encoding="utf-8-sig").encode(),
                "utility_summary.csv",
            )
        with col2:
            st.download_button(
                "⬇️ 상세 JSON",
                json.dumps(latest["detail"], ensure_ascii=False, indent=2).encode("utf-8"),
                "utility_detail.json",
                mime="application/json",
            )

        # E. 실행 히스토리
        with st.expander("🕑 실행 히스토리"):
            for h in reversed(st.session_state.util_history):
                if st.button(f"{h['time']} ({h['rows']} rows)", key=h["id"]):
                    # 선택한 히스토리를 맨 뒤로 보내고 다시 렌더
                    st.session_state.util_history.append(
                        st.session_state.util_history.pop(
                            st.session_state.util_history.index(h)
                        )
                    )
                    st.rerun()

def calculate_ec_statistics(df: pd.DataFrame, ec_cols: List[str], target_cols: List[str], 
                          ec_selection: List[Dict] = None) -> pd.DataFrame:
    """
    EC별 통계 계산
    
    Args:
        df: 전체 데이터프레임
        ec_cols: EC(동질집합) 기준 컬럼 리스트
        target_cols: 통계 산출 대상 컬럼 리스트
        ec_selection: 선택한 EC 조합 (옵션)
    
    Returns:
        EC별 통계가 포함된 DataFrame
    """
    import scipy.stats as stats
    
    # 1. 결측값 처리 옵션
    df_clean = df.copy()
    
    # 2. EC별 그룹화
    ec_groups = df_clean.groupby(ec_cols)
    
    # 3. 결과 저장을 위한 리스트
    results = []
    
    # 4. 각 EC에 대해 통계 계산
    for ec_values, group in ec_groups:
        row_data = {}
        
        # EC 식별자 추가
        if isinstance(ec_values, tuple):
            for i, col in enumerate(ec_cols):
                row_data[col] = ec_values[i]
        else:
            row_data[ec_cols[0]] = ec_values
        
        # EC 크기
        row_data['EC_SIZE'] = len(group)
        
        # 각 target column에 대한 통계
        for target_col in target_cols:
            if target_col not in group.columns:
                continue
                
            col_data = group[target_col].dropna()
            
            # 데이터 타입 확인
            if pd.api.types.is_numeric_dtype(col_data):
                # 수치형 통계
                row_data[f'{target_col}_mean'] = col_data.mean() if len(col_data) > 0 else None
                row_data[f'{target_col}_std'] = col_data.std() if len(col_data) > 1 else None
                row_data[f'{target_col}_min'] = col_data.min() if len(col_data) > 0 else None
                row_data[f'{target_col}_max'] = col_data.max() if len(col_data) > 0 else None
                row_data[f'{target_col}_count'] = len(col_data)
            else:
                # 범주형 통계
                row_data[f'{target_col}_nunique'] = col_data.nunique()
                
                # 최빈값
                if len(col_data) > 0:
                    mode_result = col_data.mode()
                    row_data[f'{target_col}_mode'] = mode_result[0] if len(mode_result) > 0 else None
                    row_data[f'{target_col}_mode_ratio'] = (col_data == row_data[f'{target_col}_mode']).sum() / len(col_data)
                
                # 엔트로피 계산
                value_counts = col_data.value_counts()
                if len(value_counts) > 1:
                    probabilities = value_counts / len(col_data)
                    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                    row_data[f'{target_col}_entropy'] = entropy
                else:
                    row_data[f'{target_col}_entropy'] = 0.0
        
        results.append(row_data)
    
    # DataFrame으로 변환
    result_df = pd.DataFrame(results)
    
    # EC_SIZE로 정렬
    result_df = result_df.sort_values('EC_SIZE', ascending=True)
    
    # EC 선택 필터링 (옵션)
    if ec_selection:
        # 선택한 EC만 필터링
        mask = pd.Series([False] * len(result_df))
        for selection in ec_selection:
            condition = pd.Series([True] * len(result_df))
            for col, val in selection.items():
                if col in result_df.columns:
                    condition &= (result_df[col] == val)
            mask |= condition
        result_df = result_df[mask]
    
    return result_df


def get_score_badge(metric: str, value: Any, metric_info: Dict) -> str:
    """점수를 평가하여 배지 반환"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "⚪ 평가 불가"
    
    if not isinstance(value, (int, float)):
        return "⚪ 비수치"
    
    info = metric_info[metric]
    thresholds = info['thresholds']
    lower_better = info['lower_better']
    
    if lower_better:
        if value <= thresholds['excellent']:
            return "🟢 우수"
        elif value <= thresholds['good']:
            return "🟡 양호"
        elif value <= thresholds['fair']:
            return "🟠 보통"
        else:
            return "🔴 주의"
    else:
        if value >= thresholds['excellent']:
            return "🟢 우수"
        elif value >= thresholds['good']:
            return "🟡 양호"
        elif value >= thresholds['fair']:
            return "🟠 보통"
        else:
            return "🔴 주의"