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

# 모든 분석 결과를 추적하는 헬퍼 함수
def get_analysis_summary():
    """모든 분석 결과 요약"""
    summary = {
        'data_info': {
            'original_rows': len(st.session_state.df) if 'df' in st.session_state else 0,
            'processed_rows': len(st.session_state.df_processed) if 'df_processed' in st.session_state else 0,
        },
        'analyses': {}
    }
    
    # k-익명성 결과
    if 'privacy_analysis' in st.session_state:
        if 'k_anonymity' in st.session_state.privacy_analysis:
            k_anon = st.session_state.privacy_analysis['k_anonymity']
            summary['analyses']['k_anonymity'] = {
                'quasi_identifiers': k_anon.get('quasi_identifiers', []),
                'k_value': k_anon.get('k_value'),
                'emp': k_anon.get('emp'),
                'sampled': k_anon.get('sampled', False)
            }
    
    # EC 통계 결과
    if 'ec_statistics' in st.session_state:
        if 'latest' in st.session_state.ec_statistics:
            ec_stat = st.session_state.ec_statistics['latest']
            summary['analyses']['ec_statistics'] = {
                'ec_cols': ec_stat.get('ec_cols', []),
                'target_cols': ec_stat.get('target_cols', []),
                'ec_count': len(ec_stat.get('df', [])),
                'sampled': ec_stat.get('sampled', False)
            }
    
    return summary

# 사이드바에 분석 상태 표시 (선택사항)
def show_analysis_status():
    """사이드바에 분석 상태 표시"""
    with st.sidebar:
        st.markdown("### 📊 분석 상태")
        
        summary = get_analysis_summary()
        
        # 데이터 정보
        st.markdown("**데이터**")
        st.text(f"원본: {summary['data_info']['original_rows']:,}행")
        st.text(f"처리: {summary['data_info']['processed_rows']:,}행")
        
        # 분석 정보
        if summary['analyses']:
            st.markdown("**완료된 분석**")
            
            if 'k_anonymity' in summary['analyses']:
                k_info = summary['analyses']['k_anonymity']
                st.text(f"✅ k-익명성 (k={k_info['k_value']})")
                if k_info['emp']:
                    st.text(f"   EMP: {k_info['emp']:.3%}")
            
            if 'ec_statistics' in summary['analyses']:
                ec_info = summary['analyses']['ec_statistics']
                st.text(f"✅ EC 통계 ({ec_info['ec_count']}개)")
                
# privacy_evaluation.py 상단에 추가
def sync_quasi_identifiers():
    """탭 간 준식별자 동기화를 위한 헬퍼 함수"""
    if 'k_anonymity' in st.session_state.get('privacy_analysis', {}):
        return st.session_state.privacy_analysis['k_anonymity'].get('quasi_identifiers', [])
    return []

def get_analysis_dataframe():
    """모든 분석에서 동일한 데이터프레임 사용"""
    # 처리된 데이터가 있으면 그것을 사용, 없으면 원본 사용
    return st.session_state.get("df_processed", st.session_state.df)


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
    
    # 공통 데이터프레임 사용
    df = get_analysis_dataframe()
    
    # 3개 탭으로 분리
    tab1, tab2, tab3 = st.tabs([
        "📊 k-익명성 분석", 
        "📈 EC별 통계 분석",
        "📋 유용성 평가"
    ])
    
    with tab1:
        render_k_anonymity_section(df)
    
    with tab2:
        render_ec_statistics_section(df)
    
    with tab3:
        render_utility_evaluation_section(df)

def render_ec_statistics_section(df: pd.DataFrame):
    """EC별 통계 분석 섹션"""
    st.subheader("📈 EC별 통계 분석")
    
    # 일관성 옵션
    st.info("💡 Tip: k-익명성 분석과 동일한 준식별자를 사용하려면 아래 옵션을 활용하세요")
    
    # k-익명성에서 사용한 준식별자 가져오기
    k_anon_qi = sync_quasi_identifiers()
    
    use_same_qi = False
    if k_anon_qi:
        use_same_qi = st.checkbox(
            f"k-익명성 분석과 동일한 준식별자 사용 ({', '.join(k_anon_qi)})",
            value=True,
            key="use_same_qi_ec"
        )
    
    # 준식별자 선택
    st.markdown("### 1️⃣ 준식별자 선택")
    
    if use_same_qi and k_anon_qi:
        selected_qi = k_anon_qi
        st.success(f"✅ k-익명성과 동일한 준식별자 사용: {', '.join(selected_qi)}")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            all_cols = df.columns.tolist()
            selected_qi = st.multiselect(
                "EC 생성을 위한 준식별자 선택",
                all_cols,
                default=k_anon_qi if not use_same_qi else [],
                help="동질집합을 만들 기준 컬럼들을 선택하세요",
                key="ec_stat_qi_selection"
            )
        
        with col2:
            if selected_qi:
                # EC 수 미리 계산
                ec_count = df.groupby(selected_qi).ngroups
                st.metric("예상 EC 수", f"{ec_count:,}개")
                st.metric("평균 EC 크기", f"{len(df) / ec_count:.1f}")
    
    if not selected_qi:
        st.warning("준식별자를 선택해주세요")
        return
    
    # 통계 대상 선택
    st.markdown("### 2️⃣ 통계 대상 선택")
    
    available_cols = [col for col in df.columns if col not in selected_qi]
    
    # 컬럼 타입별 분류
    numeric_cols = df[available_cols].select_dtypes(include='number').columns.tolist()
    categorical_cols = df[available_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if numeric_cols:
            st.markdown("**수치형 속성**")
            selected_numeric = st.multiselect(
                "수치형",
                numeric_cols,
                help="평균, 표준편차, 최소/최대값이 계산됩니다"
            )
    
    with col2:
        if categorical_cols:
            st.markdown("**범주형 속성**")
            selected_categorical = st.multiselect(
                "범주형",
                categorical_cols,
                help="고유값 수, 엔트로피, 최빈값이 계산됩니다"
            )
    
    target_cols = (selected_numeric if 'selected_numeric' in locals() else []) + \
                  (selected_categorical if 'selected_categorical' in locals() else [])
    
    if not target_cols:
        st.warning("통계를 계산할 속성을 선택해주세요")
        return
    
    # 실행 옵션
    st.markdown("### 3️⃣ 실행 옵션")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 데이터 일관성 체크
        data_check = st.checkbox(
            "데이터 검증",
            value=True,
            help="실행 전 데이터 일관성을 확인합니다"
        )
    
    with col2:
        # 샘플링 옵션
        if len(df) > 50000:
            use_sampling = st.checkbox(
                "샘플링 사용",
                value=True,
                help=f"전체 {len(df):,}행 중 일부만 분석"
            )
        else:
            use_sampling = False
    
    with col3:
        if use_sampling:
            sample_size = st.number_input(
                "샘플 크기",
                min_value=10000,
                max_value=len(df),
                value=min(50000, len(df)),
                step=10000,
                key="ec_sample_size"
            )
    
    # 실행 버튼
    if st.button("📊 EC별 통계 계산 실행", type="primary", use_container_width=True):
        
        # 데이터 검증
        if data_check:
            with st.spinner("데이터 일관성 확인 중..."):
                # 현재 df와 세션의 df가 같은지 확인
                if not df.equals(get_analysis_dataframe()):
                    st.error("⚠️ 데이터가 변경되었습니다. 페이지를 새로고침해주세요.")
                    return
                
                # 선택한 컬럼이 모두 존재하는지 확인
                missing_cols = set(selected_qi + target_cols) - set(df.columns)
                if missing_cols:
                    st.error(f"⚠️ 다음 컬럼이 없습니다: {missing_cols}")
                    return
        
        # 통계 계산
        with st.spinner("EC별 통계 계산 중..."):
            try:
                # 샘플링 적용
                analysis_df = df.sample(sample_size) if use_sampling else df
                
                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("동질집합 생성 중...")
                progress_bar.progress(0.3)
                
                # 통계 계산
                ec_stats_df = calculate_ec_statistics(
                    df=analysis_df,
                    ec_cols=selected_qi,
                    target_cols=target_cols
                )
                
                progress_bar.progress(0.7)
                status_text.text("결과 정리 중...")
                
                # 결과 저장 (일관성을 위해)
                if 'ec_statistics' not in st.session_state:
                    st.session_state.ec_statistics = {}
                
                st.session_state.ec_statistics['latest'] = {
                    'df': ec_stats_df,
                    'ec_cols': selected_qi,
                    'target_cols': target_cols,
                    'timestamp': pd.Timestamp.now(),
                    'sampled': use_sampling,
                    'sample_size': sample_size if use_sampling else len(df)
                }
                
                # 결과 표시
                display_ec_statistics_results(ec_stats_df, selected_qi, target_cols)
                
                progress_bar.progress(1.0)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                progress_bar.empty()
                status_text.empty()

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
        
        # 표본률 입력
        st.markdown("#### 📊 표본률 설정")
        sample_rate = st.number_input(
            "표본률 (f)",
            min_value=0.001,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.3f",
            help="전체 모집단 대비 현재 데이터의 비율 (1.0 = 전체 데이터)",
            key="k_anonymity_sample_rate"
        )
        
        # 표본률에 따른 설명
        if sample_rate < 1.0:
            st.info(f"📌 현재 데이터는 전체 모집단의 {sample_rate*100:.1f}%입니다")
        else:
            st.info("📌 전체 모집단 데이터로 분석합니다")
        
        # 샘플링 옵션 (대용량 데이터 대응)
        data_size = len(df)
        use_sampling = False  # 🔴 기본값 먼저 설정
        sample_size = data_size  # 🔴 기본값 설정

        if data_size > 500000:
            st.error(f"""
            ⚠️ **매우 큰 데이터셋** ({data_size:,}행)
            
            권장사항:
            1. 준식별자를 5개 이하로 선택
            2. 먼저 샘플로 테스트 후 전체 실행
            3. 카테고리가 많은 컬럼은 제외
            """)
            
            # 대용량 데이터일 때 샘플링 옵션 제공
            use_sampling = st.checkbox(
                "샘플링 사용",
                value=True,
                help=f"전체 {data_size:,}행 중 일부만 분석하여 속도 향상"
            )
            
        elif data_size > 100000:
            st.warning(f"""
            ⚠️ **대용량 데이터** ({data_size:,}행)
            
            예상 소요 시간: {data_size // 50000}~{data_size // 25000}분
            """)
            
            # 중간 크기 데이터일 때도 샘플링 옵션 제공
            use_sampling = st.checkbox(
                "샘플링 사용",
                value=True,
                help=f"전체 {data_size:,}행 중 일부만 분석하여 속도 향상"
            )

        # 샘플링을 사용하는 경우에만 샘플 크기 설정
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
        
        # EC 통계 안내 메시지
        st.info(
            "💡 동질집합별 상세 통계를 보려면 **'EC별 통계 분석'** 탭을 이용하세요. "
            "현재 선택한 준식별자가 자동으로 연동됩니다."
        )
    
    st.markdown("---")
    
    # k-익명성 분석 실행 버튼
    # k-익명성 분석 실행 버튼 부분 개선
    if st.button("🔍 k-익명성 분석 실행", type="primary", disabled=len(selected_qi) == 0):
        
        # 예상 시간 계산
        estimated_time = estimate_analysis_time(len(df), len(selected_qi))
        
        if estimated_time > 30:  # 30초 이상
            st.warning(f"""
            ⏱️ 예상 소요 시간: **{estimated_time//60}분 {estimated_time%60}초**
            
            💡 시간 단축 방법:
            - 준식별자 수 줄이기 (현재: {len(selected_qi)}개)
            - 샘플링 사용하기
            - 카테고리가 적은 컬럼 선택
            """)
            
            if not st.checkbox("계속 진행하시겠습니까?"):
                st.stop()
        
        # 실행
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
                    sample_rate
                )
                
                # 결과 표시
                st.markdown("### 📊 분석 결과")
                
                # 주요 지표 표시 (EMP 포함 5개 컬럼)
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
                
                # EMP 메트릭 추가
                with col_e:
                    emp_value = k_stats['emp']
                    emp_percent = emp_value * 100
                    st.metric(
                        "EMP",
                        f"{emp_percent:.2f}%",
                        delta=k_stats['emp_risk_level'],
                        delta_color="inverse" if emp_value > 0.05 else "normal"
                    )
                
                # EMP 상세 정보 추가
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
                    'sample_rate': sample_rate,
                    'emp': k_stats['emp'],
                    'emp_risk_level': k_stats['emp_risk_level']
                }
                
                st.success("✅ k-익명성 분석이 완료되었습니다!")
                
            except Exception as e:
                st.error(f"분석 중 오류 발생: {str(e)}")
    
    elif len(selected_qi) == 0:
        st.info("👆 준식별자를 선택하고 분석을 실행하세요.")


import time

def calculate_k_anonymity_with_timing(df, quasi_identifiers, k_threshold, sample_rate):
    """시간 측정과 함께 k-익명성 계산"""
    
    start_time = time.time()
    step_times = {}
    
    # 각 단계별 시간 측정
    step_start = time.time()
    group_sizes = df.groupby(quasi_identifiers).size().reset_index(name='count')
    step_times['groupby'] = time.time() - step_start
    
    # ... 나머지 계산 ...
    
    total_time = time.time() - start_time
    
    # 성능 정보 표시
    with st.expander("⏱️ 성능 분석", expanded=False):
        st.write(f"총 소요 시간: {total_time:.2f}초")
        for step, duration in step_times.items():
            st.write(f"- {step}: {duration:.2f}초 ({duration/total_time*100:.1f}%)")
    
    return k_value, k_stats


def calculate_k_anonymity(
        df: pd.DataFrame,
        quasi_identifiers: List[str],
        k_threshold: int = 5,
        sample_rate: float = 1.0
) -> Tuple[int, Dict]:
    """
    최적화된 k-익명성 및 EMP 계산
    """
    # 진행 상황 표시를 위한 placeholder
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # 1) 동질집합 크기 계산 (10%)
        progress_placeholder.text("동질집합 생성 중...")
        progress_bar.progress(0.1)
        
        group_sizes = (
            df.groupby(quasi_identifiers)
              .size()
              .reset_index(name='count')
        )
        
        k_value = int(group_sizes['count'].min())
        
        # 2) 위험 레코드 추출 (20%)
        progress_placeholder.text("위험 레코드 분석 중...")
        progress_bar.progress(0.2)
        
        risk_ec = group_sizes[group_sizes['count'] < k_threshold][quasi_identifiers]
        
        if len(risk_ec) > 0:
            risk_records_detail = df.merge(
                risk_ec,
                on=quasi_identifiers,
                how='inner'
            )
        else:
            risk_records_detail = pd.DataFrame()
        
        # 3) EMP 계산 (효율적인 방법만 사용) (50%)
        progress_placeholder.text("EMP 위험도 계산 중...")
        progress_bar.progress(0.5)
        
        n = len(df)
        
        # 🔴 최적화: merge 한 번만 수행
        df_with_ec_size = df.merge(
            group_sizes.rename(columns={'count': 'ec_size'}),
            on=quasi_identifiers,
            how='left'
        )
        
        # 벡터화된 계산
        df_with_ec_size['risk_i'] = 1 / (sample_rate * df_with_ec_size['ec_size'])
        df_with_ec_size['risk_i'] = df_with_ec_size['risk_i'].clip(upper=1)
        
        total_risk = df_with_ec_size['risk_i'].sum()
        emp = (sample_rate / n) * total_risk
        
        # 4) 통계 계산 (80%)
        progress_placeholder.text("통계 정리 중...")
        progress_bar.progress(0.8)
        
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
            'emp': emp,
            'sample_rate': sample_rate,
            'total_risk_sum': total_risk,
            'avg_individual_risk': total_risk / n,
            'high_risk_records': len(df_with_ec_size[df_with_ec_size['risk_i'] >= 0.5]),
            'emp_risk_level': get_emp_risk_level(emp)
        }
        
        # 완료 (100%)
        progress_bar.progress(1.0)
        progress_placeholder.text("분석 완료!")
        
        # 잠시 후 progress bar 제거
        time.sleep(0.5)
        progress_bar.empty()
        progress_placeholder.empty()
        
        return k_value, k_stats
        
    except Exception as e:
        progress_bar.empty()
        progress_placeholder.empty()
        raise e


def preview_equivalence_classes(df: pd.DataFrame, selected_qi: List[str], 
                               preview_option: str, k_threshold: int):
    """최적화된 동질집합 미리보기"""
    
    with st.spinner("동질집합 분석 중..."):
        # Progress bar 추가
        progress = st.progress(0)
        
        # 1. EC 계산 (캐싱 활용)
        @st.cache_data
        def compute_ec_sizes(_df, qi_list):
            return _df.groupby(qi_list).size().reset_index(name='k')
        
        progress.progress(0.3)
        ec_sizes = compute_ec_sizes(df, selected_qi)
        
        progress.progress(0.6)
        
        # 2. 정렬 (벡터화)
        if preview_option == "k값이 낮은 위험 그룹":
            top_groups = ec_sizes.nsmallest(5, 'k')
        elif preview_option == "상위 5개 그룹":
            top_groups = ec_sizes.nlargest(5, 'k')
        else:
            top_groups = ec_sizes.sample(min(5, len(ec_sizes)))
        
        progress.progress(1.0)
        progress.empty()
        
        return ec_sizes, top_groups
    
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

def display_ec_statistics_results(ec_stats_df: pd.DataFrame, selected_qi: List[str], target_cols: List[str]):
    """EC 통계 결과 표시 (최적화)"""
    
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
    
    # 표시 옵션
    st.markdown("### 📊 결과 표시 옵션")
    col1, col2 = st.columns(2)
    
    with col1:
        # 표시할 행 수 제한
        max_rows = st.slider(
            "표시할 최대 EC 수",
            min_value=10,
            max_value=min(1000, len(ec_stats_df)),
            value=min(100, len(ec_stats_df)),
            step=10
        )
    
    with col2:
        # 정렬 옵션
        sort_by = st.selectbox(
            "정렬 기준",
            ['EC_SIZE'] + [col for col in ec_stats_df.columns if col.endswith('_entropy')],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        sort_order = st.radio("정렬 순서", ["오름차순", "내림차순"], horizontal=True)
    
    # 테이블 표시
    st.markdown("### 📋 EC별 통계 테이블")
    
    # 정렬 적용
    display_df = ec_stats_df.sort_values(
        sort_by, 
        ascending=(sort_order == "오름차순")
    ).head(max_rows)
    
    # 컬럼 선택
    display_cols = selected_qi + ['EC_SIZE']
    for target_col in target_cols:
        stat_cols = [col for col in display_df.columns if col.startswith(f'{target_col}_')]
        display_cols.extend(stat_cols)
    
    # 데이터프레임 표시
    st.dataframe(
        display_df[display_cols],
        use_container_width=True,
        height=400
    )
    
    # 다운로드
    csv = ec_stats_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 전체 결과 CSV 다운로드",
        data=csv.encode('utf-8-sig'),
        file_name=f"ec_statistics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # 시각화는 선택적으로
    if st.checkbox("📈 엔트로피 시각화 보기", value=False, key="show_entropy_viz_main"):
        display_entropy_visualization(display_df)


# 사용자에게 선택권을 주는 방식
def suggest_dtype_optimization(df: pd.DataFrame):
    """데이터 타입 최적화 제안 (실행하지 않음)"""
    
    suggestions = []
    potential_memory_save = 0
    
    for col in df.columns:
        if df[col].dtype == 'int64':
            if df[col].min() >= 0 and df[col].max() <= 255:
                suggestions.append(f"• {col}: int64 → uint8 (메모리 87.5% 절약)")
                potential_memory_save += df[col].memory_usage() * 0.875
    
    if suggestions:
        with st.expander("💡 성능 최적화 제안", expanded=False):
            st.info(f"""
            데이터 타입을 최적화하면 분석 속도를 높일 수 있습니다.
            
            **제안사항:**
            {chr(10).join(suggestions)}
            
            **예상 메모리 절약**: {potential_memory_save / 1024 / 1024:.1f} MB
            """)
            
            if st.button("데이터 타입 최적화 적용"):
                # 사용자가 명시적으로 동의한 경우만 적용
                optimized_df = optimize_dtypes(df.copy())
                st.session_state.df_optimized = optimized_df
                st.success("✅ 최적화 완료!")
                
def estimate_analysis_time(n_rows: int, n_qi: int) -> int:
    """분석 시간 예측 (초)"""
    # 경험적 공식 (조정 필요)
    base_time = 0.00001 * n_rows  # 행당 기본 시간
    qi_factor = 1.5 ** n_qi  # 준식별자 수에 따른 지수적 증가
    return int(base_time * qi_factor)


def display_entropy_visualization(ec_stats_df: pd.DataFrame):
    """엔트로피 시각화 (최적화)"""
    entropy_cols = [col for col in ec_stats_df.columns if col.endswith('_entropy')]
    
    if not entropy_cols:
        st.info("엔트로피 데이터가 없습니다.")
        return
    
    # 시각화할 컬럼 선택
    selected_entropy = st.selectbox(
        "시각화할 엔트로피 컬럼",
        entropy_cols,
        format_func=lambda x: x.replace('_entropy', '').replace('_', ' ').title()
    )
    
    # 단일 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 히스토그램
    data = ec_stats_df[selected_entropy].dropna()
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('엔트로피')
    ax.set_ylabel('EC 수')
    ax.set_title(f'{selected_entropy.replace("_entropy", "")} 엔트로피 분포')
    
    # 통계 정보 추가
    mean_val = data.mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'평균: {mean_val:.3f}')
    ax.legend()
    
    # 표시
    st.pyplot(fig)
    
    # 메모리 해제 중요!
    plt.close(fig)
    
    # 요약 통계
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("최소", f"{data.min():.3f}")
    with col2:
        st.metric("평균", f"{data.mean():.3f}")
    with col3:
        st.metric("최대", f"{data.max():.3f}")
    with col4:
        st.metric("표준편차", f"{data.std():.3f}")


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