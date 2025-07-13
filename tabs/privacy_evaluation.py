# privacy_evaluation.py

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
                    k_threshold
                )
                
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

def calculate_k_anonymity(
        df: pd.DataFrame,
        quasi_identifiers: List[str],
        k_threshold: int = 5
) -> Tuple[int, Dict]:
    """
    선택한 준식별자에 대해 k-익명성 통계 계산
    Returns
        k_value : 전체 데이터의 최소 k
        k_stats : 상세 통계 딕셔너리
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

    # 3) 통계 딕셔너리 작성
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
        'risk_records_detail': risk_records_detail
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

def render_utility_evaluation_section(_: pd.DataFrame):
    """유용성 평가 탭 – 리뉴얼 + 버그 수정 버전"""
    st.subheader("📈 유용성 평가")
    
    # 지표 메타 정보
    METRIC_INFO = {
        'U1': ('평균값 차이',
            '두 데이터셋 평균값이 얼마나 다른지 측정 (0에 가까울수록 좋음)'),
        'U2': ('상관계수 보존',
            '원본·비식별 상관계수 차이 평균 (0에 가까울수록 좋음)'),
        'U3': ('코사인 유사도',
            '벡터 유사도 평균 (1에 가까울수록 좋음)'),
        'U4': ('정규화 거리',
            '정규화 SSE 합 (0에 가까울수록 좋음)'),
        'U5': ('표준화 거리',
            '표준화 SSE 합 (0에 가까울수록 좋음)'),
        'U6': ('동질집합 분산',
            '동질집합 내 민감값 분산 평균 (낮을수록 정보 유지)'),
        'U7': ('정규화 집합크기',
            '(N/N_EC)/k : 동질집합 크기 지표 (낮을수록 안전)'),
        'U8': ('비균일 엔트로피',
            '변경 레코드 엔트로피 (0에 가까울수록 원본과 유사)'),
        'U9': ('익명화율',
            '비식별 데이터가 얼마나 남았는지 (%) (높을수록 활용 ↑)'),
    }
    
    # 도움말 토글
    show_help = st.toggle("👶 처음이라면 도움말 보기", value=False)
    if show_help:
        md = "**유용성(U) 지표란?**  \n"
        for k, (name, desc) in METRIC_INFO.items():
            md += f"• **{k} {name}** : {desc}  \n"
        st.info(md)

    # 1. 데이터 존재 확인
    if 'df' not in st.session_state or 'df_processed' not in st.session_state:
        st.warning("먼저 데이터를 업로드·비식별화 해 주세요.")
        return

    orig_df = st.session_state.df
    proc_df = st.session_state.df_processed

    # 3. 타입 비교 기준 & 컬럼 목록
    type_ref = st.radio(
        "타입 비교 기준", ["원본 데이터", "변환 후 데이터"],
        index=0, horizontal=True
    )
    base_df = orig_df if type_ref == "원본 데이터" else proc_df
    numeric_cols = base_df.select_dtypes(include='number').columns.tolist()
    all_cols = base_df.columns.tolist()

    # 4. 평가 대상 컬럼
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

    # 원본 vs 변환후 기준에 맞춰 숫자 변환 & UtilityMetrics 준비
    from modules.preprocessor import DataPreprocessor
    pre = DataPreprocessor()

    base_orig = orig_df.copy()
    if type_ref == "변환 후 데이터":
        for col in sel_cols:
            if base_orig[col].dtype == "object":
                converted, _ = pre.safe_type_conversion(base_orig[col], "numeric")
                base_orig[col] = converted

    from modules.privacy_metrics.utility_metrics import UtilityMetrics
    utility_analyzer = UtilityMetrics(orig_df, proc_df) 

    # 5. 지표 선택 & QI 옵션
    st.markdown("### ② 지표 선택")
    
    metrics = st.multiselect(
        "실행할 지표", list(METRIC_INFO.keys()),
        default=['U1', 'U2', 'U9'],
        format_func=lambda m: f"{m} – {METRIC_INFO[m][0]}"
    )

    # 선택한 지표 설명 패널
    if metrics:
        with st.container():
            st.markdown("#### 선택 지표 설명")
            for m in metrics:
                st.markdown(f"**{m} – {METRIC_INFO[m][0]}**  \n"
                            f"{METRIC_INFO[m][1]}")
    
    qi_cols, sens_attr = [], None
    if any(m in metrics for m in ('U6', 'U7')):
        with st.expander("🔐 QI·민감속성", expanded=True):
            qi_cols = st.multiselect("준식별자(QI)", options=sel_cols)
            if qi_cols:
                cand = [c for c in sel_num if c not in qi_cols]
                if cand:
                    sens_attr = st.selectbox("민감속성", cand)

    # 6. 샘플링
    st.markdown("### ③ 샘플링")
    use_samp = st.toggle("샘플링 사용", value=True)
    samp_rows = st.slider(
        "샘플 행 수", 10_000, min(1_000_000, len(orig_df)),
        100_000, step=10_000, disabled=not use_samp, format="%d 행"
    )
    analysis_df = orig_df.sample(samp_rows, random_state=42) if use_samp and samp_rows < len(orig_df) else orig_df

    # 7. 실행
    st.markdown("### ④ 평가 실행")
    if st.button("🚀 Run selected metrics", type="primary"):
        run_id = uuid.uuid4().hex[:8]
        summary, detail_results = [], {}
        prog = st.progress(0.0)
        total = len(metrics)

        # 결과 리스트에 행 추가하는 헬퍼
        def push(metric: str, res: dict, used_cols: list):
            """summary·detail 두 곳에 결과를 저장"""
            # 컬럼별 점수를 분해해서 보여줘야 하는 지표
            if metric in ('U1', 'U3', 'U4', 'U5') and res.get('status') == 'success':
                for col, det in res['column_results'].items():
                    if 'error' in det:
                        continue
                    val = det.get('difference') or det.get('cosine_similarity') \
                          or det.get('normalized_sse') or det.get('sse')
                    summary.append({
                        '지표': metric, '컬럼': col,
                        '점수': round(val, 4) if isinstance(val, (int, float)) else val
                    })
            else:
                score = res.get('total_score') or res.get('average_score')
                summary.append({
                    '지표': metric,
                    '컬럼': ", ".join(used_cols) if used_cols else '-',
                    '점수': round(score, 4) if isinstance(score, (int, float)) else score
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

    # 8. 결과 표시
    # ---------- 카드용 배지 & 해석 함수 ---------- #
    def badge(metric: str, val: float) -> tuple[str, str]:
        """점수 → (이모지 배지, 해석 문자열)"""
        if metric == "U1":                       # 평균값 차이 (작을수록 좋음)
            if val < 0.1:  return "🟢", "거의 차이 없음"
            if val < 1.0:  return "🟡", "차이 있지만 양호"
            return "🔴", "평균값 차이 큼"

        if metric == "U2":                       # 상관계수 차이 (작을수록 좋음)
            if val < 0.02: return "🟢", "상관관계 잘 보존"
            if val < 0.10: return "🟡", "다소 손상"
            return "🔴", "상관관계 크게 손상"

        if metric == "U3":                       # 코사인 유사도 (클수록 좋음)
            if val > 0.98: return "🟢", "거의 동일"
            if val > 0.90: return "🟡", "대체로 유사"
            return "🔴", "유사도 낮음"

        if metric == "U9":                       # 익명화율 (클수록 좋음)
            if val > 90:  return "🟢", "데이터 대부분 보존"
            if val > 70:  return "🟡", "적당히 보존"
            return "🔴", "많이 손실"

        return "⚪", "참고값"                    # 나머지 지표

    if st.session_state.get("util_history"):
        latest = st.session_state.util_history[-1]
        st.markdown(f"### ⑤ 결과 요약 ({latest['time']})")

        # A. 카드용 해석 함수
        def verdict(metric: str, value) -> str:
            """점수를 초보자용 배지 텍스트로 변환"""
            if not isinstance(value, (int, float)):
                return "⚪ 참고값"

            good = "🟢 매우 유사"
            ok   = "🟢 양호"

            if metric == "U1" and value < 0.1:
                return good
            elif metric == "U2" and value < 0.05:
                return good
            elif metric == "U3" and value > 0.95:
                return good
            elif metric in ["U4", "U5"] and value < 0.05:
                return ok
            elif metric == "U9" and value > 90:
                return "🟢 활용도 ↑"
            
            return "⚪ 참고값"

        # B. 주요 메트릭 카드
        # ---------- 주요 메트릭 카드 ---------- #
        card_metrics = ["U1", "U2", "U3", "U9"]
        cols = st.columns(len(card_metrics))

        for col, m in zip(cols, card_metrics):
            row = next((r for r in latest["summary"] if r["지표"] == m), None)
            if not row:
                col.empty()
                continue

            emoji, expl = badge(m, row["점수"])          # ← 새 함수 호출
            col.metric(
                label=f"{m} {emoji}",
                value=row["점수"],
                help=expl                                # 마우스 오버 시 해석
            )

        # C. 요약표
        df_sum = (
            pd.DataFrame(latest["summary"])[["지표", "컬럼", "점수"]]
              .sort_values(["지표", "컬럼"])
        )
        st.dataframe(df_sum, hide_index=True, use_container_width=True)

        # D. 상세 결과
        for r in latest["summary"]:
            with st.expander(f"🔍 {r['지표']} – {r['컬럼']}"):
                st.json(latest["detail"][r["지표"]])

        # E. 다운로드
        st.download_button(
            "⬇️ 요약 CSV",
            df_sum.to_csv(index=False, encoding="utf-8-sig").encode(),
            "utility_summary.csv",
        )
        st.download_button(
            "⬇️ 상세 JSON",
            json.dumps(latest["detail"], ensure_ascii=False, indent=2).encode("utf-8"),
            "utility_detail.json",
            mime="application/json",
        )

        # F. 실행 히스토리
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