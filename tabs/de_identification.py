import streamlit as st
import pandas as pd
from typing import Optional, Tuple

# 내부 모듈 -------------------------------------------------------------
from modules.de_identification.rounding import RoundingProcessor
from modules.de_identification.masking import MaskingProcessor
from modules.de_identification.deletion import DeletionProcessor
from modules.de_identification.substitution import SubstitutionProcessor

# ---------------------------------------------------------------------
# 공통 유틸 ------------------------------------------------------------
# ---------------------------------------------------------------------

def _fmt_num(x: float, decimals: int | None = None) -> str:
    if pd.isna(x):
        return ""
    if decimals is None:                       # ← 추가
        decimals = 0 if float(x).is_integer() else 2
    return f"{x:,.{decimals}f}"



def _update_session_df(new_df: pd.DataFrame):
    if "df_processed" not in st.session_state:
        st.session_state.df_processed = st.session_state.df.copy()
    st.session_state.df_processed = new_df


def _log_history(hist_type: str, column: str, details: str):
    if "processing_history" not in st.session_state:
        st.session_state.processing_history = []
    st.session_state.processing_history.append({
        "type": hist_type,
        "column": column,
        "details": details,
    })

# ---------------------------------------------------------------------
# 메인 탭 --------------------------------------------------------------
# ---------------------------------------------------------------------

def render_de_identification_tab():
    st.header("🔐 데이터 비식별화")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("먼저 데이터를 업로드해주세요.")
        return

    df = st.session_state.get("df_processed", st.session_state.df)

    technique = st.selectbox(
        "비식별화 기법 선택",
        [
            "라운딩 (Rounding)",
            "마스킹 (Masking)",
            "부분 삭제 (Deletion)",
            "치환 (Substitution)",
        ],
    )

    if technique.startswith("라운딩"):
        render_rounding_section(df)
    elif technique.startswith("마스킹"):
        render_masking_section(df)
    elif technique.startswith("부분 삭제"):
        render_deletion_section(df)
    else:
        render_substitution_section(df)

# ---------------------------------------------------------------------
# 1️⃣  라운딩 + 이상치(클리핑) -----------------------------------------
# ---------------------------------------------------------------------

def render_rounding_section(df: pd.DataFrame):
    st.subheader("📊 라운딩 + 이상치(클리핑)")

    # ── 대상 컬럼 & 라운딩 방식 선택 ───────────────────────────
    left, right = st.columns(2)

    with left:
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not num_cols:
            st.error("숫자형 컬럼이 없습니다. 먼저 문자→숫자 변환을 진행하세요.")
            return
        col = st.selectbox("처리할 컬럼", num_cols)

        r_map   = {"내림 ↓": "floor", "반올림 ↔": "round", "올림 ↑": "ceil"}
        r_disp  = st.radio("처리 방식", list(r_map.keys()), horizontal=True)
        r_type  = r_map[r_disp]

    with right:
        place_kind = st.radio("자리수 선택", ["소수점 자리", "정수 자리"])
        if place_kind == "소수점 자리":
            dec        = st.number_input("소수점 몇째 자리?", 0, 10, 2, 1)
            int_place  = None
            place_desc = f"소수점 {dec}째 자리에서 {r_disp}"
        else:
            int_place  = st.selectbox(
                "어느 자리?", [10, 100, 1000, 10000, 100000],
                format_func=lambda x: f"{x:,}의 자리"
            )
            dec        = None
            place_desc = f"{int_place:,}의 자리에서 {r_disp}"

    # ── “전부 양수 데이터” 여부 체크 ──────────────────────────
    positive_only = st.checkbox("양수 데이터 전용 (0 미만 무시)", value=False)

    # ── 클리핑(이상치) 옵션 ───────────────────────────────────
    st.markdown("### 🚧 이상치(클리핑) 옵션")
    mode   = st.radio("임계값 선택", ["사용 안 함", "시그마(σ) 기반", "IQR 기반", "수동 입력"],
                      horizontal=True)
    lower: float | None = None
    upper: float | None = None

    if mode == "시그마(σ) 기반":
        k     = st.number_input("k 값 (mean ± k·σ)", 1.0, 5.0, 3.0, 0.5)
        stats = RoundingProcessor.get_statistics(df[col], sigma_k=k,
                                                 positive_only=positive_only)   # ← 옵션 전달
        lower, upper = stats["sigma_range"]

    elif mode == "IQR 기반":
        k     = st.number_input("k 값 (Q1 ± k·IQR)", 0.5, 3.0, 1.5, 0.5)
        stats = RoundingProcessor.get_statistics(df[col], iqr_k=k,
                                                 positive_only=positive_only)   # ← 옵션 전달
        lower, upper = stats["iqr_range"]

    elif mode == "수동 입력":
        l_c, u_c = st.columns(2)
        with l_c:
            lower = st.number_input("하한(≤)", value=float(df[col].min()))
        with u_c:
            upper = st.number_input("상한(≥)", value=float(df[col].max()))
        if lower >= upper:
            st.error("하한은 상한보다 작아야 합니다.")
            lower = upper = None

    bounds = None if mode == "사용 안 함" else (lower, upper)
    if bounds:
        st.caption(f"범위: {_fmt_num(lower)} ~ {_fmt_num(upper)}")

    # ── 미리보기 ───────────────────────────────────────────────
    st.markdown("### 미리보기")
    st.info(f"✨ {place_desc}{' (클리핑 적용)' if bounds else ''}")

    try:
        preview = RoundingProcessor.get_preview(
            df, col, r_type,
            dec, int_place,
            outlier_bounds=bounds,
            sample_size=5,
        )
        if not preview.empty:
            st.dataframe(preview.applymap(_fmt_num), use_container_width=True)
        else:
            st.warning("미리보기할 데이터가 없습니다.")
    except Exception as err:
        st.error(f"미리보기 오류: {err}")

    # ── 실제 적용 ─────────────────────────────────────────────
    apply_btn, undo_btn = st.columns(2)

    with apply_btn:
        if st.button("✅ 적용", type="primary"):
            try:
                new_series = RoundingProcessor.round_column(
                    df, col, r_type,
                    dec, int_place,
                    outlier_bounds=bounds,
                )
                affected = (df[col] != new_series).sum()
                _update_session_df(
                    st.session_state.get("df_processed", df).assign(**{col: new_series})
                )

                detail = f"{affected:,}행 변환 · {place_desc}"
                if bounds:
                    detail = f"클리핑({_fmt_num(lower)}~{_fmt_num(upper)}) → " + detail
                _log_history("라운딩", col, detail)

                st.success(f"✅ 라운딩 완료 — 변경 {affected:,}행")
                st.rerun()
            except Exception as err:
                st.error(f"처리 오류: {err}")

    with undo_btn:
        if st.button("↩️ 되돌리기"):
            st.info("되돌리기 기능은 준비 중입니다.")

    # ── 처리 이력 ─────────────────────────────────────────────
    if st.session_state.get("processing_history"):
        st.markdown("### 📝 처리 이력")
        for i, h in enumerate(st.session_state.processing_history, 1):
            st.text(f"{i}. {h['type']}: {h['column']} - {h['details']}")
# ---------------------------------------------------------------------
# 2️⃣  마스킹 / 삭제 / 치환 (원본 로직 사용) -----------------------------
# ---------------------------------------------------------------------







def render_masking_section(df: pd.DataFrame):
    """마스킹 섹션 렌더링 (삭제와 동일한 구조)"""
    st.subheader("🎭 마스킹 처리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 모든 컬럼 선택 가능
        selected_column = st.selectbox(
            "처리할 컬럼 선택",
            df.columns.tolist(),
            key="mask_column_select"
        )
        
        # 마스킹 방식 선택 (삭제와 동일한 4가지)
        masking_mode = st.radio(
            "마스킹 방식",
            ["구분자 기반", "위치/범위 기반", "조건 기반", "스마트 마스킹"],
            help="데이터에 맞는 마스킹 방식을 선택하세요",
            key="mask_mode_radio"
        )
        
        # 마스킹 문자
        mask_char = st.text_input("마스킹 문자", value="*", max_chars=1, key="mask_char_input")
    
    with col2:
        # 마스킹 타입별 옵션
        params = {'mask_char': mask_char}
        
        if masking_mode == "구분자 기반":
            st.markdown("### 구분자 기반 마스킹")
            params['delimiter'] = st.text_input("구분자", value="-", help="데이터를 나누는 구분자", key="mask_delimiter")
            
            params['keep_position'] = st.radio(
                "유지할 위치",
                ["왼쪽", "오른쪽", "가운데"],
                horizontal=True,
                key="mask_keep_pos"
            )
            params['keep_position'] = {'왼쪽': 'left', '오른쪽': 'right', '가운데': 'middle'}[params['keep_position']]
            
            params['occurrence'] = st.radio(
                "구분자 처리",
                ["첫 번째만", "마지막만", "모든 구분자"],
                horizontal=True,
                key="mask_occurrence"
            )
            params['occurrence'] = {'첫 번째만': 'first', '마지막만': 'last', '모든 구분자': 'all'}[params['occurrence']]
            
            if params['occurrence'] == 'all':
                params['keep_count'] = st.number_input("유지할 부분 개수", min_value=1, value=1, key="mask_keep_count")
            
            masking_type = "delimiter"
            
        elif masking_mode == "위치/범위 기반":
            st.markdown("### 위치/범위 기반 마스킹")
            
            # 단위 선택
            params['unit'] = st.radio("처리 단위", ["글자", "단어"], horizontal=True, key="mask_unit")
            params['unit'] = 'character' if params['unit'] == "글자" else 'word'
            
            # 모드 선택
            mode_display = st.selectbox(
                "처리 방식",
                ["단순 마스킹", "특정 위치", "범위 지정", "간격 마스킹", "중요 부분만 유지"],
                key="mask_pos_mode"
            )
            
            # 단어 단위일 때 마스킹 스타일 추가
            if params['unit'] == 'word':
                mask_style_display = st.selectbox(
                    "단어 마스킹 스타일",
                    ["전체 마스킹", "첫 글자만 보존", "마지막 글자만 보존", "양끝 글자 보존", "앞부분 마스킹", "뒷부분 마스킹"],
                    help="각 단어를 어떻게 마스킹할지 선택",
                    key="mask_word_style"
                )
                style_map = {
                    "전체 마스킹": "full",
                    "첫 글자만 보존": "keep_first",
                    "마지막 글자만 보존": "keep_last",
                    "양끝 글자 보존": "keep_edges",
                    "앞부분 마스킹": "partial_front",
                    "뒷부분 마스킹": "partial_back"
                }
                params['mask_style'] = style_map[mask_style_display]
            
            if mode_display == "단순 마스킹":
                params['mode'] = 'simple'
                params['position'] = st.radio(
                    "마스킹 위치",
                    ["앞부분", "뒷부분", "양쪽"],
                    horizontal=True,
                    key="mask_simple_pos"
                )
                params['position'] = {'앞부분': 'front', '뒷부분': 'back', '양쪽': 'both'}[params['position']]
                
                if params['position'] == 'both' and params['unit'] == 'word':
                    col_a, col_b = st.columns(2)
                    with col_a:
                        params['front_count'] = st.number_input("앞에서 마스킹", min_value=0, value=1, key="mask_front_cnt")
                    with col_b:
                        params['back_count'] = st.number_input("뒤에서 마스킹", min_value=0, value=1, key="mask_back_cnt")
                else:
                    params['count'] = st.number_input(
                        f"마스킹할 {params['unit'] == 'character' and '글자' or '단어'} 수", 
                        min_value=1, value=3, key="mask_count"
                    )
                
                params['preserve_minimum'] = st.number_input("최소 보존 개수", min_value=0, value=1, key="mask_preserve")
                
            elif mode_display == "특정 위치":
                params['mode'] = 'specific'
                params['operation'] = st.radio("작업", ["마스킹", "보존"], horizontal=True, key="mask_spec_op")
                params['operation'] = 'mask' if params['operation'] == "마스킹" else 'preserve'
                
                positions_str = st.text_input(
                    f"{params['operation'] == 'mask' and '마스킹' or '보존'}할 위치 (쉼표 구분)", 
                    value="2,4", 
                    help="예: 1,3,5",
                    key="mask_positions"
                )
                try:
                    params['positions'] = [int(p.strip()) for p in positions_str.split(',') if p.strip()]
                except:
                    params['positions'] = []
                
                if params['operation'] == 'mask':
                    st.info(f"지정한 위치의 {params['unit'] == 'word' and '단어' or '글자'}를 마스킹합니다")
                else:
                    st.info(f"지정한 위치의 {params['unit'] == 'word' and '단어' or '글자'}만 보존합니다")
                
            elif mode_display == "범위 지정":
                params['mode'] = 'range'
                params['operation'] = st.radio("작업", ["마스킹", "보존"], horizontal=True, key="mask_range_op")
                params['operation'] = 'mask' if params['operation'] == "마스킹" else 'preserve'
                
                range_str = st.text_input(
                    f"{params['operation'] == 'mask' and '마스킹' or '보존'}할 범위", 
                    value="2-4", 
                    help="예: 2-4, 6-8 (쉼표로 여러 범위 가능)",
                    key="mask_ranges"
                )
                try:
                    ranges = []
                    for r in range_str.split(','):
                        if '-' in r:
                            start, end = r.strip().split('-')
                            ranges.append((int(start), int(end)))
                    params['ranges'] = ranges
                except:
                    params['ranges'] = []
                
            elif mode_display == "간격 마스킹":
                params['mode'] = 'interval'
                interval_type = st.radio("간격 유형", ["N번째마다", "홀짝"], horizontal=True, key="mask_interval")
                
                if interval_type == "N번째마다":
                    params['interval_type'] = 'every_n'
                    params['n'] = st.number_input("간격 (N)", min_value=2, value=2, key="mask_n")
                    params['offset'] = st.number_input("시작 오프셋", min_value=0, value=0, key="mask_offset")
                    st.info(f"{params['offset']}부터 시작하여 {params['n']}번째마다 마스킹")
                else:
                    params['interval_type'] = 'odd_even'
                    params['keep_odd'] = st.radio("유지할 위치", ["홀수", "짝수"], horizontal=True, key="mask_odd") == "홀수"
                    st.info(f"{'짝수' if params['keep_odd'] else '홀수'} 번째를 마스킹")
                    
            else:  # 중요 부분만 유지 (단어 단위만)
                if params['unit'] == 'word':
                    params['mode'] = 'important'
                    params['keep_count'] = st.number_input("유지할 단어 수", min_value=1, value=2, key="mask_imp_cnt")
                    params['criteria'] = st.radio(
                        "선택 기준",
                        ["처음", "마지막", "가장 긴"],
                        horizontal=True,
                        key="mask_criteria"
                    )
                    params['criteria'] = {'처음': 'first', '마지막': 'last', '가장 긴': 'longest'}[params['criteria']]
                else:
                    st.warning("이 옵션은 단어 단위에서만 사용 가능합니다")
                    params['mode'] = 'simple'
            
            masking_type = "position"
            
        elif masking_mode == "조건 기반":
            st.markdown("### 조건 기반 마스킹")
            
            condition_type_display = st.selectbox(
                "조건 유형",
                ["길이 조건", "패턴 매칭", "문자 타입별", "사전 기반"],
                key="mask_cond_type"
            )
            
            if condition_type_display == "길이 조건":
                params['condition_type'] = 'length'
                params['unit'] = st.radio("단위", ["단어", "전체"], horizontal=True, key="mask_len_unit")
                params['unit'] = 'word' if params['unit'] == "단어" else 'character'
                
                col_a, col_b = st.columns(2)
                with col_a:
                    params['min_length'] = st.number_input("최소 길이", min_value=0, value=1, key="mask_min_len")
                with col_b:
                    max_val = st.number_input("최대 길이 (0=무제한)", min_value=0, value=0, key="mask_max_len")
                    params['max_length'] = float('inf') if max_val == 0 else max_val
                
                if params['unit'] == 'word':
                    st.info(f"{params['min_length']}~{params['max_length']}글자 단어를 마스킹")
                else:
                    st.info(f"전체 길이가 {params['min_length']}~{params['max_length']}인 경우 전체 마스킹")
                    
            elif condition_type_display == "패턴 매칭":
                params['condition_type'] = 'pattern'
                
                pattern_preset = st.selectbox(
                    "패턴 프리셋",
                    ["사용자 정의", "숫자", "괄호 내용", "특수문자", "공백", "URL", "이메일", "전화번호"],
                    key="mask_pattern_preset"
                )
                
                preset_patterns = {
                    "숫자": r'\d+',
                    "괄호 내용": r'\([^)]*\)|\[[^\]]*\]',
                    "특수문자": r'[^\w\s가-힣]',
                    "공백": r'\s+',
                    "URL": r'https?://[^\s]+',
                    "이메일": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                    "전화번호": r'\d{2,4}[\-\s]?\d{3,4}[\-\s]?\d{4}'
                }
                
                if pattern_preset == "사용자 정의":
                    params['pattern'] = st.text_input("정규식 패턴", value=r'\d+', key="mask_pattern")
                else:
                    params['pattern'] = preset_patterns[pattern_preset]
                    st.code(f"패턴: {params['pattern']}")
                
                params['mask_matched'] = st.checkbox("매칭된 부분 마스킹", value=True, 
                                                    help="체크 해제 시 매칭되지 않은 부분 마스킹",
                                                    key="mask_matched")
                params['unit'] = st.radio("처리 단위", ["매칭 부분", "단어 전체"], horizontal=True, key="mask_pat_unit")
                params['unit'] = 'match' if params['unit'] == "매칭 부분" else 'word'
                
                if params['unit'] == 'match':
                    params['max_masks'] = st.number_input("최대 마스킹 횟수 (-1: 무제한)", min_value=-1, value=-1, key="mask_max")
                    
            elif condition_type_display == "문자 타입별":
                params['condition_type'] = 'char_type'
                
                char_type_map = {
                    "숫자": "digits",
                    "문자 (한글/영문)": "letters",
                    "한글만": "korean",
                    "영문만": "english",
                    "특수문자": "special",
                    "공백": "whitespace",
                    "사용자 정의": "custom",
                    "지정 문자만 유지": "except"
                }
                
                char_type_display = st.selectbox("마스킹할 문자 타입", list(char_type_map.keys()), key="mask_char_type")
                params['char_type'] = char_type_map[char_type_display]
                
                if params['char_type'] == "custom":
                    params['characters'] = st.text_input("마스킹할 문자들", value="!@#$%", key="mask_chars")
                elif params['char_type'] == "except":
                    params['keep_pattern'] = st.text_input("유지할 문자 패턴", value=r'[가-힣a-zA-Z0-9\s]', key="mask_keep_pat")
                    
            else:  # 사전 기반
                params['condition_type'] = 'dictionary'
                params['unit'] = st.radio("매칭 단위", ["단어", "전체 일치"], horizontal=True, key="mask_dict_unit")
                params['unit'] = 'word' if params['unit'] == "단어" else 'exact'
                params['case_sensitive'] = st.checkbox("대소문자 구분", value=False, key="mask_case")
                
                dict_text = st.text_area(
                    "마스킹할 목록 (한 줄에 하나)", 
                    value="홍길동\n김철수\n이영희",
                    height=150,
                    key="mask_dict"
                )
                params['dictionary'] = [w.strip() for w in dict_text.split('\n') if w.strip()]
                
                st.info(f"{len(params['dictionary'])}개 항목이 등록되었습니다")
            
            masking_type = "condition"
            
        else:  # 스마트 마스킹
            st.markdown("### 스마트 마스킹")
            
            smart_type_display = st.selectbox(
                "스마트 마스킹 유형",
                ["자동 감지", "개인정보 수준별", "중복/불필요 정보"],
                key="mask_smart_type"
            )
            
            if smart_type_display == "자동 감지":
                params['smart_type'] = 'auto_detect'
                st.info("""
                자동으로 데이터 형태를 감지하여 마스킹:
                - 이메일: 로컬 파트 마스킹
                - 전화번호: 중간/뒷자리 마스킹  
                - 주민번호: 뒷자리 마스킹
                - URL: 경로 마스킹
                - 날짜: 월일 마스킹
                - 주소: 상세주소 마스킹
                """)
                
            elif smart_type_display == "개인정보 수준별":
                params['smart_type'] = 'personal_info'
                params['level'] = st.radio(
                    "마스킹 수준",
                    ["낮음", "중간", "높음"],
                    horizontal=True,
                    help="높을수록 더 많은 정보를 마스킹",
                    key="mask_level"
                )
                params['level'] = {'낮음': 'low', '중간': 'medium', '높음': 'high'}[params['level']]
                
                level_info = {
                    'low': "최소 마스킹: 민감한 부분만 부분적으로",
                    'medium': "중간 마스킹: 식별 가능한 정보 대부분",
                    'high': "최대 마스킹: 거의 모든 정보"
                }
                st.info(level_info[params['level']])
                
            else:  # 중복/불필요 정보
                params['smart_type'] = 'redundant'
                params['mask_parentheses'] = st.checkbox("괄호 내용 마스킹", value=True, key="mask_paren")
                params['mask_duplicates'] = st.checkbox("연속 중복 문자 마스킹", value=True, key="mask_dup")
                params['mask_special'] = st.checkbox("앞뒤 특수문자 마스킹", value=True, key="mask_trim")
                st.info("중복 문자, 괄호 내용, 불필요한 앞뒤 문자를 마스킹합니다")
            
            masking_type = "smart"
    
    # 미리보기
    st.markdown("### 미리보기")
    
    try:
        preview_df = MaskingProcessor.get_preview(
            df,
            selected_column,
            masking_type,
            sample_size=5,
            **params
        )
        
        if not preview_df.empty:
            st.dataframe(preview_df, use_container_width=True)
        else:
            st.warning("미리보기할 데이터가 없습니다.")
    
    except Exception as e:
        st.error(f"미리보기 생성 중 오류: {str(e)}")
    
    # 적용 버튼
    if st.button("✅ 적용", type="primary", key="apply_masking"):
        try:
            # 마스킹 적용
            processed_column = MaskingProcessor.mask_column(
                df,
                selected_column,
                masking_type,
                **params
            )
            
            # 데이터프레임 업데이트
            if 'df_processed' not in st.session_state:
                st.session_state.df_processed = st.session_state.df.copy()
            
            st.session_state.df_processed[selected_column] = processed_column
            
            # 처리 기록 저장
            if 'processing_history' not in st.session_state:
                st.session_state.processing_history = []
            
            # 상세 설명 생성
            details = f"{masking_mode}"
            if masking_mode == "위치/범위 기반":
                details += f" ({params.get('unit', 'character')} 단위)"
            elif masking_mode == "조건 기반":
                details += f" ({params.get('condition_type', '')})"
            elif masking_mode == "스마트 마스킹":
                details += f" ({params.get('smart_type', '')})"
            
            st.session_state.processing_history.append({
                'type': '마스킹',
                'column': selected_column,
                'details': details
            })
            
            st.success(f"✅ '{selected_column}' 컬럼에 마스킹이 적용되었습니다!")
            st.rerun()
            
        except Exception as e:
            st.error(f"처리 중 오류 발생: {str(e)}")


def render_deletion_section(df: pd.DataFrame):
    """부분 삭제 섹션 렌더링 (리팩토링 버전)"""
    st.subheader("✂️ 부분 삭제 처리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 모든 컬럼 선택 가능
        selected_column = st.selectbox(
            "처리할 컬럼 선택",
            df.columns.tolist(),
            key="del_column_select"
        )
        
        # 삭제 방식 선택 (정리된 4가지)
        deletion_mode = st.radio(
            "삭제 방식",
            ["구분자 기반", "위치/범위 기반", "조건 기반", "스마트 삭제"],
            help="데이터에 맞는 삭제 방식을 선택하세요",
            key="del_mode_radio"
        )
    
    with col2:
        # 삭제 타입별 옵션
        params = {}
        
        if deletion_mode == "구분자 기반":
            st.markdown("### 구분자 기반 삭제")
            params['delimiter'] = st.text_input("구분자", value="-", help="데이터를 나누는 구분자", key="del_delimiter")
            
            params['keep_position'] = st.radio(
                "유지할 위치",
                ["왼쪽", "오른쪽", "가운데"],
                horizontal=True,
                key="del_keep_pos"
            )
            params['keep_position'] = {'왼쪽': 'left', '오른쪽': 'right', '가운데': 'middle'}[params['keep_position']]
            
            params['occurrence'] = st.radio(
                "구분자 처리",
                ["첫 번째만", "마지막만", "모든 구분자"],
                horizontal=True,
                key="del_occurrence"
            )
            params['occurrence'] = {'첫 번째만': 'first', '마지막만': 'last', '모든 구분자': 'all'}[params['occurrence']]
            
            if params['occurrence'] == 'all':
                params['keep_count'] = st.number_input("유지할 부분 개수", min_value=1, value=1, key="del_keep_count")
            
            deletion_type = "delimiter"
            
        elif deletion_mode == "위치/범위 기반":
            st.markdown("### 위치/범위 기반 삭제")
            
            # 단위 선택
            params['unit'] = st.radio("처리 단위", ["글자", "단어"], horizontal=True, key="del_unit")
            params['unit'] = 'character' if params['unit'] == "글자" else 'word'
            
            # 모드 선택
            mode_display = st.selectbox(
                "처리 방식",
                ["단순 삭제", "특정 위치", "범위 지정", "간격 삭제", "중요 부분만 유지"],
                key="del_pos_mode"
            )
            
            if mode_display == "단순 삭제":
                params['mode'] = 'simple'
                params['position'] = st.radio(
                    "삭제 위치",
                    ["앞부분", "뒷부분", "양쪽"],
                    horizontal=True,
                    key="del_simple_pos"
                )
                params['position'] = {'앞부분': 'front', '뒷부분': 'back', '양쪽': 'both'}[params['position']]
                
                if params['position'] == 'both' and params['unit'] == 'word':
                    col_a, col_b = st.columns(2)
                    with col_a:
                        params['front_count'] = st.number_input("앞에서 삭제", min_value=0, value=1, key="del_front_cnt")
                    with col_b:
                        params['back_count'] = st.number_input("뒤에서 삭제", min_value=0, value=1, key="del_back_cnt")
                else:
                    params['count'] = st.number_input(
                        f"삭제할 {params['unit'] == 'character' and '글자' or '단어'} 수", 
                        min_value=1, value=3, key="del_count"
                    )
                
                params['preserve_minimum'] = st.number_input("최소 보존 개수", min_value=0, value=1, key="del_preserve")
                
            elif mode_display == "특정 위치":
                params['mode'] = 'specific'
                params['operation'] = st.radio("작업", ["삭제", "유지"], horizontal=True, key="del_spec_op")
                params['operation'] = 'delete' if params['operation'] == "삭제" else 'keep'
                
                positions_str = st.text_input(
                    f"{params['operation'] == 'delete' and '삭제' or '유지'}할 위치 (쉼표 구분)", 
                    value="2,4", 
                    help="예: 1,3,5",
                    key="del_positions"
                )
                try:
                    params['positions'] = [int(p.strip()) for p in positions_str.split(',') if p.strip()]
                except:
                    params['positions'] = []
                
                if params['operation'] == 'delete':
                    st.info(f"지정한 위치의 {params['unit'] == 'word' and '단어' or '글자'}를 삭제합니다")
                else:
                    st.info(f"지정한 위치의 {params['unit'] == 'word' and '단어' or '글자'}만 유지합니다")
                
            elif mode_display == "범위 지정":
                params['mode'] = 'range'
                params['operation'] = st.radio("작업", ["삭제", "유지"], horizontal=True, key="del_range_op")
                params['operation'] = 'delete' if params['operation'] == "삭제" else 'keep'
                
                range_str = st.text_input(
                    f"{params['operation'] == 'delete' and '삭제' or '유지'}할 범위", 
                    value="2-4", 
                    help="예: 2-4, 6-8 (쉼표로 여러 범위 가능)",
                    key="del_ranges"
                )
                try:
                    ranges = []
                    for r in range_str.split(','):
                        if '-' in r:
                            start, end = r.strip().split('-')
                            ranges.append((int(start), int(end)))
                    params['ranges'] = ranges
                except:
                    params['ranges'] = []
                
            elif mode_display == "간격 삭제":
                params['mode'] = 'interval'
                interval_type = st.radio("간격 유형", ["N번째마다", "홀짝"], horizontal=True, key="del_interval")
                
                if interval_type == "N번째마다":
                    params['interval_type'] = 'every_n'
                    params['n'] = st.number_input("간격 (N)", min_value=2, value=2, key="del_n")
                    params['offset'] = st.number_input("시작 오프셋", min_value=0, value=0, key="del_offset")
                    st.info(f"{params['offset']}부터 시작하여 {params['n']}번째마다 삭제")
                else:
                    params['interval_type'] = 'odd_even'
                    params['keep_odd'] = st.radio("유지할 위치", ["홀수", "짝수"], horizontal=True, key="del_odd") == "홀수"
                    st.info(f"{'홀수' if params['keep_odd'] else '짝수'} 번째만 유지")
                    
            else:  # 중요 부분만 유지 (단어 단위만)
                if params['unit'] == 'word':
                    params['mode'] = 'important'
                    params['keep_count'] = st.number_input("유지할 단어 수", min_value=1, value=2, key="del_imp_cnt")
                    params['criteria'] = st.radio(
                        "선택 기준",
                        ["처음", "마지막", "가장 긴"],
                        horizontal=True,
                        key="del_criteria"
                    )
                    params['criteria'] = {'처음': 'first', '마지막': 'last', '가장 긴': 'longest'}[params['criteria']]
                else:
                    st.warning("이 옵션은 단어 단위에서만 사용 가능합니다")
                    params['mode'] = 'simple'
            
            deletion_type = "position"
            
        elif deletion_mode == "조건 기반":
            st.markdown("### 조건 기반 삭제")
            
            condition_type_display = st.selectbox(
                "조건 유형",
                ["길이 조건", "패턴 매칭", "문자 타입별", "사전 기반"],
                key="del_cond_type"
            )
            
            if condition_type_display == "길이 조건":
                params['condition_type'] = 'length'
                params['unit'] = st.radio("단위", ["단어", "전체"], horizontal=True, key="del_len_unit")
                params['unit'] = 'word' if params['unit'] == "단어" else 'character'
                
                col_a, col_b = st.columns(2)
                with col_a:
                    params['min_length'] = st.number_input("최소 길이", min_value=0, value=1, key="del_min_len")
                with col_b:
                    max_val = st.number_input("최대 길이 (0=무제한)", min_value=0, value=0, key="del_max_len")
                    params['max_length'] = float('inf') if max_val == 0 else max_val
                
                if params['unit'] == 'word':
                    st.info(f"{params['min_length']}~{params['max_length']}글자 단어를 삭제")
                else:
                    st.info(f"전체 길이가 {params['min_length']}~{params['max_length']}인 경우 전체 삭제")
                    
            elif condition_type_display == "패턴 매칭":
                params['condition_type'] = 'pattern'
                
                pattern_preset = st.selectbox(
                    "패턴 프리셋",
                    ["사용자 정의", "숫자", "괄호 내용", "특수문자", "공백", "URL", "이메일 도메인", "전화번호"],
                    key="del_pattern_preset"
                )
                
                preset_patterns = {
                    "숫자": r'\d+',
                    "괄호 내용": r'\([^)]*\)|\[[^\]]*\]',
                    "특수문자": r'[^\w\s가-힣]',
                    "공백": r'\s+',
                    "URL": r'https?://[^\s]+',
                    "이메일 도메인": r'@[^\s]+',
                    "전화번호": r'\d{2,4}[\-\s]?\d{3,4}[\-\s]?\d{4}'
                }
                
                if pattern_preset == "사용자 정의":
                    params['pattern'] = st.text_input("정규식 패턴", value=r'\d+', key="del_pattern")
                else:
                    params['pattern'] = preset_patterns[pattern_preset]
                    st.code(f"패턴: {params['pattern']}")
                
                params['delete_matched'] = st.checkbox("매칭된 부분 삭제", value=True, 
                                                      help="체크 해제 시 매칭되지 않은 부분 삭제",
                                                      key="del_matched")
                params['unit'] = st.radio("처리 단위", ["매칭 부분", "단어 전체"], horizontal=True, key="del_pat_unit")
                params['unit'] = 'match' if params['unit'] == "매칭 부분" else 'word'
                
                if params['unit'] == 'match':
                    params['max_deletions'] = st.number_input("최대 삭제 횟수 (-1: 무제한)", min_value=-1, value=-1, key="del_max")
                    
            elif condition_type_display == "문자 타입별":
                params['condition_type'] = 'char_type'
                
                char_type_map = {
                    "숫자": "digits",
                    "문자 (한글/영문)": "letters",
                    "한글만": "korean",
                    "영문만": "english",
                    "특수문자": "special",
                    "공백": "whitespace",
                    "사용자 정의": "custom",
                    "지정 문자만 유지": "except"
                }
                
                char_type_display = st.selectbox("삭제할 문자 타입", list(char_type_map.keys()), key="del_char_type")
                params['char_type'] = char_type_map[char_type_display]
                
                if params['char_type'] == "custom":
                    params['characters'] = st.text_input("삭제할 문자들", value="!@#$%", key="del_chars")
                elif params['char_type'] == "except":
                    params['keep_pattern'] = st.text_input("유지할 문자 패턴", value=r'[가-힣a-zA-Z0-9\s]', key="del_keep_pat")
                    
            else:  # 사전 기반
                params['condition_type'] = 'dictionary'
                params['unit'] = st.radio("매칭 단위", ["단어", "전체 일치"], horizontal=True, key="del_dict_unit")
                params['unit'] = 'word' if params['unit'] == "단어" else 'exact'
                params['case_sensitive'] = st.checkbox("대소문자 구분", value=False, key="del_case")
                
                dict_text = st.text_area(
                    "삭제할 목록 (한 줄에 하나)", 
                    value="은\n는\n이\n가\n을\n를",
                    height=150,
                    key="del_dict"
                )
                params['dictionary'] = [w.strip() for w in dict_text.split('\n') if w.strip()]
                
                st.info(f"{len(params['dictionary'])}개 항목이 등록되었습니다")
            
            deletion_type = "condition"
            
        else:  # 스마트 삭제
            st.markdown("### 스마트 삭제")
            
            smart_type_display = st.selectbox(
                "스마트 삭제 유형",
                ["자동 감지", "개인정보 수준별", "중복/불필요 정보"],
                key="del_smart_type"
            )
            
            if smart_type_display == "자동 감지":
                params['smart_type'] = 'auto_detect'
                st.info("""
                자동으로 데이터 형태를 감지하여 삭제:
                - 이메일: 도메인 삭제
                - 전화번호: 뒷자리 삭제  
                - 주민번호: 뒷자리 삭제
                - URL: 경로 삭제
                - 날짜: 월일 삭제
                - 주소: 상세주소 삭제
                """)
                
            elif smart_type_display == "개인정보 수준별":
                params['smart_type'] = 'personal_info'
                params['level'] = st.radio(
                    "삭제 수준",
                    ["낮음", "중간", "높음"],
                    horizontal=True,
                    help="높을수록 더 많은 정보를 삭제",
                    key="del_level"
                )
                params['level'] = {'낮음': 'low', '중간': 'medium', '높음': 'high'}[params['level']]
                
                level_info = {
                    'low': "최소 삭제: 민감한 부분만 부분적으로",
                    'medium': "중간 삭제: 식별 가능한 정보 대부분",
                    'high': "최대 삭제: 거의 모든 정보"
                }
                st.info(level_info[params['level']])
                
            else:  # 중복/불필요 정보
                params['smart_type'] = 'redundant'
                params['remove_parentheses'] = st.checkbox("괄호 내용 제거", value=True, key="del_paren")
                params['remove_duplicates'] = st.checkbox("연속 중복 문자 제거", value=True, key="del_dup")
                params['trim_special'] = st.checkbox("앞뒤 특수문자 제거", value=True, key="del_trim")
                st.info("중복 문자, 괄호 내용, 불필요한 앞뒤 문자를 정리합니다")
            
            deletion_type = "smart"
    
    # 미리보기
    st.markdown("### 미리보기")
    
    try:
        preview_df = DeletionProcessor.get_preview(
            df,
            selected_column,
            deletion_type,
            sample_size=5,
            **params
        )
        
        if not preview_df.empty:
            st.dataframe(preview_df, use_container_width=True)
        else:
            st.warning("미리보기할 데이터가 없습니다.")
    
    except Exception as e:
        st.error(f"미리보기 생성 중 오류: {str(e)}")
    
    # 적용 버튼
    if st.button("✅ 적용", type="primary", key="apply_deletion"):
        try:
            # 삭제 적용
            processed_column = DeletionProcessor.delete_column(
                df,
                selected_column,
                deletion_type,
                **params
            )
            
            # 데이터프레임 업데이트
            if 'df_processed' not in st.session_state:
                st.session_state.df_processed = st.session_state.df.copy()
            
            st.session_state.df_processed[selected_column] = processed_column
            
            # 처리 기록 저장
            if 'processing_history' not in st.session_state:
                st.session_state.processing_history = []
            
            # 상세 설명 생성
            details = f"{deletion_mode}"
            if deletion_mode == "위치/범위 기반":
                details += f" ({params.get('unit', 'character')} 단위)"
            elif deletion_mode == "조건 기반":
                details += f" ({params.get('condition_type', '')})"
            elif deletion_mode == "스마트 삭제":
                details += f" ({params.get('smart_type', '')})"
            
            st.session_state.processing_history.append({
                'type': '부분 삭제',
                'column': selected_column,
                'details': details
            })
            
            st.success(f"✅ '{selected_column}' 컬럼에 부분 삭제가 적용되었습니다!")
            st.rerun()
            
        except Exception as e:
            st.error(f"처리 중 오류 발생: {str(e)}")

def render_substitution_section(df: pd.DataFrame):
    """치환 섹션 렌더링"""
    st.subheader("🔄 치환 처리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 컬럼 선택
        selected_column = st.selectbox(
            "처리할 컬럼 선택",
            df.columns.tolist(),
            key="sub_column"
        )
        
        # 데이터 타입 확인
        column_dtype = df[selected_column].dtype
        is_numeric = pd.api.types.is_numeric_dtype(column_dtype)
        
        # 치환 유형 선택
        if is_numeric:
            substitution_type = st.radio(
                "치환 유형",
                ["숫자형 구간 치환", "개별 값 치환"],
                help="숫자형 데이터의 치환 방식을 선택하세요",
                key="sub_type"
            )
        else:
            substitution_type = "개별 값 치환"
            st.info("문자형 데이터는 개별 값 치환을 사용합니다")
    
    with col2:
        params = {}
        
        if substitution_type == "숫자형 구간 치환":
            st.markdown("### 구간 설정")
            
            # 구간 설정 방식
            interval_method = st.selectbox(
                "구간 설정 방식",
                ["수동 설정", "등간격", "분위수", "표준편차"],
                key="interval_method"
            )
            
            if interval_method == "수동 설정":
                st.info("각 구간의 경계값과 치환값을 입력하세요")
                
                # 구간 개수
                num_intervals = st.number_input("구간 개수", min_value=2, max_value=10, value=3, key="num_intervals")
                
                # 구간 설정
                intervals = []
                for i in range(num_intervals):
                    with st.expander(f"구간 {i+1}", expanded=True):
                        col_a, col_b, col_c = st.columns([2, 2, 3])
                        with col_a:
                            if i == 0:
                                min_val = st.number_input(f"최소값", value=0.0, key=f"min_{i}")
                            else:
                                min_val = intervals[-1]['max']
                                st.text(f"최소값: {min_val}")
                        with col_b:
                            if i == num_intervals - 1:
                                # 마지막 구간은 데이터의 최대값 참고
                                try:
                                    data_max = pd.to_numeric(df[selected_column], errors='coerce').max()
                                    default_max = float(data_max) if not pd.isna(data_max) else 100.0
                                except:
                                    default_max = 100.0
                                max_val = st.number_input(f"최대값", value=default_max, key=f"max_{i}")
                            else:
                                max_val = st.number_input(f"최대값", value=float((i+1)*30.0), key=f"max_{i}")
                        with col_c:
                            replace_val = st.text_input(f"치환값", value=f"구간{i+1}", key=f"replace_{i}")
                        
                        col_d, col_e = st.columns(2)
                        with col_d:
                            include_min = st.checkbox(f"최소값 포함 (≥)", value=True, key=f"include_min_{i}")
                        with col_e:
                            include_max = st.checkbox(f"최대값 포함 (≤)", value=(i == num_intervals-1), key=f"include_max_{i}")
                        
                        intervals.append({
                            'min': min_val,
                            'max': max_val,
                            'value': replace_val,
                            'include_min': include_min,
                            'include_max': include_max
                        })
                
                params['method'] = 'manual'
                params['intervals'] = intervals
                
            elif interval_method == "등간격":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_intervals = st.number_input("구간 개수", min_value=2, max_value=20, value=5, key="n_equal_intervals")
                with col_b:
                    label_type = st.radio("라벨 유형", ["자동", "사용자 정의"], key="label_type_equal")
                
                if label_type == "사용자 정의":
                    labels = []
                    for i in range(n_intervals):
                        label = st.text_input(f"구간 {i+1} 라벨", value=f"구간{i+1}", key=f"label_equal_{i}")
                        labels.append(label)
                    params['labels'] = labels
                
                params['method'] = 'equal'
                params['n_intervals'] = n_intervals
                
            elif interval_method == "분위수":
                col_a, col_b = st.columns(2)
                with col_a:
                    quantile_type = st.selectbox(
                        "분위수 유형",
                        ["4분위수", "5분위수", "10분위수", "사용자 정의"],
                        key="quantile_type"
                    )
                    
                    if quantile_type == "4분위수":
                        n_quantiles = 4
                        default_labels = ["Q1", "Q2", "Q3", "Q4"]
                    elif quantile_type == "5분위수":
                        n_quantiles = 5
                        default_labels = ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]
                    elif quantile_type == "10분위수":
                        n_quantiles = 10
                        default_labels = [f"D{i+1}" for i in range(10)]
                    else:
                        n_quantiles = st.number_input("분위수", min_value=2, max_value=20, value=4, key="n_custom_quantiles")
                        default_labels = [f"분위{i+1}" for i in range(n_quantiles)]
                
                with col_b:
                    use_default_labels = st.checkbox("기본 라벨 사용", value=True, key="use_default_quantile_labels")
                
                if not use_default_labels:
                    labels = []
                    for i in range(n_quantiles):
                        label = st.text_input(f"분위 {i+1} 라벨", value=default_labels[i], key=f"label_quantile_{i}")
                        labels.append(label)
                    params['labels'] = labels
                else:
                    params['labels'] = default_labels
                
                params['method'] = 'quantile'
                params['n_quantiles'] = n_quantiles
                
            else:  # 표준편차
                col_a, col_b = st.columns(2)
                with col_a:
                    n_std = st.number_input("표준편차 배수", min_value=0.5, max_value=3.0, value=1.0, step=0.5, key="n_std")
                with col_b:
                    use_default_labels = st.checkbox("기본 라벨 사용", value=True, key="use_default_std_labels")
                
                if not use_default_labels:
                    labels = []
                    label_names = ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]
                    for i, default_label in enumerate(label_names):
                        label = st.text_input(f"구간 {i+1} 라벨", value=default_label, key=f"label_std_{i}")
                        labels.append(label)
                    params['labels'] = labels
                
                params['method'] = 'std'
                params['n_std'] = n_std
            
            substitution_type = "numeric"
            
        else:  # 개별 값 치환
            st.markdown("### 개별 값 매핑")
            
            # 매핑 방식 선택
            mapping_method = st.radio(
                "매핑 방식",
                ["개별 선택", "패턴 매칭", "전체 목록"],
                horizontal=True,
                key="mapping_method"
            )
            
            if mapping_method == "개별 선택":
                # 고유값 확인
                unique_values = df[selected_column].dropna().unique()
                st.info(f"고유값 개수: {len(unique_values)}개")
                
                if len(unique_values) > 50:
                    st.warning("고유값이 많습니다. 패턴 매칭 사용을 권장합니다.")
                
                # 다중 선택
                selected_values = st.multiselect(
                    "치환할 값 선택",
                    options=sorted(unique_values),
                    default=[],
                    key="selected_values"
                )
                
                if selected_values:
                    replace_value = st.text_input("선택한 값들을 다음으로 치환:", value="기타", key="multi_replace_value")
                    mappings = {val: replace_value for val in selected_values}
                    
                    # 나머지 값 처리
                    handle_rest = st.checkbox("선택하지 않은 값도 처리", key="handle_rest")
                    if handle_rest:
                        default_value = st.text_input("나머지 값들을 다음으로 치환:", value="기타", key="default_value")
                        params['default_value'] = default_value
                else:
                    mappings = {}
                
                params['mappings'] = mappings
                substitution_type = "categorical"
                
            elif mapping_method == "패턴 매칭":
                st.info("정규식 패턴을 사용하여 값을 치환합니다")
                
                # 패턴 개수
                num_patterns = st.number_input("패턴 개수", min_value=1, max_value=10, value=2, key="num_patterns")
                
                patterns = []
                for i in range(num_patterns):
                    with st.expander(f"패턴 {i+1}", expanded=True):
                        col_a, col_b = st.columns([3, 2])
                        with col_a:
                            pattern = st.text_input(
                                "패턴 (정규식)",
                                value="서울|경기|인천" if i == 0 else "",
                                key=f"pattern_{i}",
                                help="예: 서울|경기|인천, ^A.*"
                            )
                        with col_b:
                            value = st.text_input(
                                "치환값",
                                value="수도권" if i == 0 else "",
                                key=f"pattern_value_{i}"
                            )
                        
                        if pattern and value:
                            patterns.append({'pattern': pattern, 'value': value})
                
                params['patterns'] = patterns
                substitution_type = "pattern"
                
            else:  # 전체 목록
                st.info("모든 고유값에 대한 매핑을 지정합니다")
                
                unique_values = df[selected_column].dropna().unique()
                
                # 스크롤 가능한 영역으로 변경
                if len(unique_values) > 10:
                    st.warning(f"고유값이 {len(unique_values)}개로 많습니다.")
                    
                    # Expander 내부에 스크롤 적용
                    with st.expander(f"📝 전체 값 매핑 ({len(unique_values)}개 항목)", expanded=True):
                        # 이 expander 내부에만 스크롤 적용
                        st.markdown("""
                            <style>
                            div[data-testid="stExpander"] div[data-testid="stVerticalBlock"]:has(input) {
                                max-height: 400px;
                                overflow-y: auto;
                                padding-right: 10px;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        mappings = {}
                        # 모든 값 표시 (20개 제한 제거)
                        for val in sorted(unique_values):
                            new_val = st.text_input(
                                f"{val} →", 
                                value=str(val), 
                                key=f"map_{val}"
                            )
                            if new_val != str(val):
                                mappings[val] = new_val
                else:
                    # 10개 이하는 그냥 표시
                    mappings = {}
                    for val in sorted(unique_values):
                        new_val = st.text_input(f"{val} →", value=str(val), key=f"map_{val}")
                        if new_val != str(val):
                            mappings[val] = new_val
    
    # 미리보기
    st.markdown("### 미리보기")
    
    try:
        # SubstitutionProcessor import 필요
        from modules.de_identification.substitution import SubstitutionProcessor
        
        preview_df = SubstitutionProcessor.get_preview(
            df,
            selected_column,
            substitution_type,
            sample_size=10,
            **params
        )
        
        if not preview_df.empty:
            st.dataframe(preview_df, use_container_width=True)
            
            # 통계 정보 표시
            with st.expander("치환 통계", expanded=False):
                stats = SubstitutionProcessor.get_statistics(
                    df,
                    selected_column,
                    substitution_type,
                    **params
                )
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("원본 고유값", stats['original']['unique_count'])
                with col_stat2:
                    st.metric("치환 후 고유값", stats['substituted']['unique_count'])
        else:
            st.warning("미리보기할 데이터가 없습니다.")
    
    except Exception as e:
        st.error(f"미리보기 생성 중 오류: {str(e)}")
    
    # 적용 버튼
    if st.button("✅ 적용", type="primary", key="apply_substitution"):
        try:
            # 치환 적용
            from modules.de_identification.substitution import SubstitutionProcessor
            
            processed_column = SubstitutionProcessor.substitute_column(
                df,
                selected_column,
                substitution_type,
                **params
            )
            
            # 데이터프레임 업데이트
            if 'df_processed' not in st.session_state:
                st.session_state.df_processed = st.session_state.df.copy()
            
            st.session_state.df_processed[selected_column] = processed_column
            
            # 처리 기록 저장
            if 'processing_history' not in st.session_state:
                st.session_state.processing_history = []
            
            # 상세 설명 생성
            if substitution_type == "numeric":
                method_name = params.get('method', 'manual')
                method_display = {
                    'manual': '수동 구간',
                    'equal': '등간격',
                    'quantile': '분위수',
                    'std': '표준편차'
                }
                details = f"숫자형 치환 ({method_display.get(method_name, method_name)})"
            elif substitution_type == "pattern":
                details = f"패턴 치환 ({len(params.get('patterns', []))}개 패턴)"
            else:
                details = f"범주형 치환 ({len(params.get('mappings', {}))}개 매핑)"
            
            st.session_state.processing_history.append({
                'type': '치환',
                'column': selected_column,
                'details': details
            })
            
            st.success(f"✅ '{selected_column}' 컬럼에 치환이 적용되었습니다!")
            st.rerun()
            
        except Exception as e:
            st.error(f"처리 중 오류 발생: {str(e)}")

# 공통 UI 컴포넌트 함수들
def render_processing_history():
    """처리 이력 표시"""
    if 'processing_history' in st.session_state and st.session_state.processing_history:
        with st.expander("📝 전체 처리 이력", expanded=False):
            for i, history in enumerate(st.session_state.processing_history, 1):
                st.text(f"{i}. {history['type']}: {history['column']} - {history['details']}")
            
            if st.button("🗑️ 이력 초기화", key="clear_history"):
                st.session_state.processing_history = []
                st.rerun()


def render_data_export():
    """처리된 데이터 내보내기"""
    if 'df_processed' in st.session_state:
        st.markdown("---")
        st.subheader("💾 처리된 데이터 내보내기")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV 다운로드
            csv = st.session_state.df_processed.to_csv(index=False)
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel 다운로드 (필요시 구현)
            st.button("📥 Excel 다운로드", disabled=True, help="준비 중")
        
        with col3:
            # 원본과 비교
            if st.button("🔍 원본과 비교"):
                st.info("비교 기능은 준비 중입니다")