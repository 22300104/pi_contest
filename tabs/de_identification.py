import streamlit as st
import pandas as pd
from modules.de_identification.rounding import RoundingProcessor
from modules.de_identification.masking import MaskingProcessor


def render_de_identification_tab():
    """비식별화 처리 탭 렌더링"""
    st.header("🔐 데이터 비식별화")
    
    # 세션에 데이터가 있는지 확인
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("먼저 데이터를 업로드해주세요.")
        return
    
    # 전처리된 데이터가 있으면 사용, 없으면 원본 사용
    df = st.session_state.get('df_processed', st.session_state.df)
    
    # 비식별화 기법 선택
    technique = st.selectbox(
        "비식별화 기법 선택",
        ["라운딩 (Rounding)", "마스킹 (Masking)"]
    )
    
    if technique == "라운딩 (Rounding)":
        render_rounding_section(df)
    elif technique == "마스킹 (Masking)":
        render_masking_section(df)


def render_rounding_section(df: pd.DataFrame):
    """라운딩 섹션 렌더링"""
    st.subheader("📊 라운딩 처리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 숫자형 컬럼만 필터링 (int64, float64)
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numeric_columns:
            st.error("숫자형 컬럼이 없습니다. 먼저 '데이터 전처리' 탭에서 문자형 데이터를 숫자형으로 변환해주세요.")
            return
        
        # 컬럼 선택
        selected_column = st.selectbox(
            "처리할 컬럼 선택",
            numeric_columns
        )
        
        # 처리 방식 선택
        rounding_type_map = {
            "내림 ↓": "floor",
            "반올림 ↔": "round",
            "올림 ↑": "ceil"
        }
        
        rounding_type_display = st.radio(
            "처리 방식",
            list(rounding_type_map.keys()),
            horizontal=True
        )
        rounding_type = rounding_type_map[rounding_type_display]
    
    with col2:
        # 자리수 타입 선택
        place_type = st.radio(
            "자리수 선택",
            ["소수점 자리", "정수 자리"]
        )
        
        if place_type == "소수점 자리":
            decimal_places = st.number_input(
                "소수점 몇째 자리에서 처리할까요?",
                min_value=0,
                max_value=10,
                value=2,
                step=1
            )
            integer_place = None
            place_description = f"소수점 {decimal_places}째 자리에서 {rounding_type_display}"
        else:
            integer_place = st.selectbox(
                "어느 자리에서 처리할까요?",
                [10, 100, 1000, 10000, 100000],
                format_func=lambda x: f"{x:,}의 자리"
            )
            decimal_places = None
            place_description = f"{integer_place:,}의 자리에서 {rounding_type_display}"
    
    # 미리보기
    st.markdown("### 미리보기")
    st.info(f"✨ {place_description}")
    
    try:
        preview_df = RoundingProcessor.get_preview(
            df,
            selected_column,
            rounding_type,
            decimal_places,
            integer_place,
            sample_size=5
        )
        
        if not preview_df.empty:
            st.dataframe(preview_df, use_container_width=True)
        else:
            st.warning("미리보기할 데이터가 없습니다.")
    
    except Exception as e:
        st.error(f"미리보기 생성 중 오류: {str(e)}")
    
    # 적용 버튼
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("✅ 적용", type="primary"):
            try:
                # 라운딩 적용
                processed_column = RoundingProcessor.round_column(
                    df,
                    selected_column,
                    rounding_type,
                    decimal_places,
                    integer_place
                )
                
                # 데이터프레임 업데이트
                # df_processed가 없으면 생성
                if 'df_processed' not in st.session_state:
                    st.session_state.df_processed = st.session_state.df.copy()
                
                # 처리된 데이터 업데이트
                st.session_state.df_processed[selected_column] = processed_column
                
                # 처리 기록 저장
                if 'processing_history' not in st.session_state:
                    st.session_state.processing_history = []
                
                st.session_state.processing_history.append({
                    'type': '라운딩',
                    'column': selected_column,
                    'details': place_description
                })
                
                st.success(f"✅ '{selected_column}' 컬럼에 라운딩이 적용되었습니다!")
                st.rerun()
                
            except Exception as e:
                st.error(f"처리 중 오류 발생: {str(e)}")
    
    with col2:
        if st.button("↩️ 되돌리기"):
            # 원본 데이터로 되돌리기 기능 (추후 구현)
            st.info("되돌리기 기능은 준비 중입니다.")
    
    # 처리 이력 표시
    if 'processing_history' in st.session_state and st.session_state.processing_history:
        st.markdown("### 📝 처리 이력")
        for i, history in enumerate(st.session_state.processing_history, 1):
            st.text(f"{i}. {history['type']}: {history['column']} - {history['details']}")

def render_masking_section(df: pd.DataFrame):
    """마스킹 섹션 렌더링"""
    st.subheader("🎭 마스킹 처리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 모든 컬럼 선택 가능 (문자열로 변환하여 처리)
        selected_column = st.selectbox(
            "처리할 컬럼 선택",
            df.columns.tolist()
        )
        
        # 마스킹 타입 선택
        masking_mode = st.radio(
            "마스킹 방식",
            ["기본 마스킹", "위치 기반", "패턴 기반", "조건부", "특수 마스킹"],
            help="상황에 맞는 마스킹 방식을 선택하세요"
        )
        
        # 마스킹 문자
        mask_char = st.text_input("마스킹 문자", value="*", max_chars=1)
    
    with col2:
        # 마스킹 타입별 옵션
        params = {}
        
        if masking_mode == "기본 마스킹":
            st.markdown("### 기본 마스킹 옵션")
            direction = st.radio("방향", ["앞에서", "뒤에서"], horizontal=True)
            params['direction'] = 'front' if direction == "앞에서" else 'back'
            params['length'] = st.number_input("마스킹할 글자 수", min_value=1, value=3)
            params['min_preserve'] = st.number_input("최소 보존 글자 수", min_value=0, value=1)
            masking_type = "basic"
            
        elif masking_mode == "위치 기반":
            st.markdown("### 위치 기반 마스킹")
            position_type = st.selectbox(
                "마스킹 유형",
                ["처음 N글자", "마지막 N글자", "범위 지정", "중간 마스킹"]
            )
            
            if position_type == "처음 N글자":
                params['mask_type'] = 'first_n'
                params['n'] = st.number_input("마스킹할 글자 수", min_value=1, value=3)
            elif position_type == "마지막 N글자":
                params['mask_type'] = 'last_n'
                params['n'] = st.number_input("마스킹할 글자 수", min_value=1, value=4)
            elif position_type == "범위 지정":
                params['mask_type'] = 'range'
                col_a, col_b = st.columns(2)
                with col_a:
                    params['start'] = st.number_input("시작 위치", min_value=1, value=2)
                with col_b:
                    params['end'] = st.number_input("끝 위치", min_value=1, value=5)
            else:  # 중간 마스킹
                params['mask_type'] = 'middle'
                params['preserve'] = st.number_input("양끝 보존 글자 수", min_value=1, value=2)
            
            masking_type = "position"
            
        elif masking_mode == "패턴 기반":
            st.markdown("### 패턴 기반 마스킹")
            pattern_type = st.selectbox(
                "패턴 유형",
                ["구분자 이후 마스킹", "특정 문자 이전 마스킹"]
            )
            
            if pattern_type == "구분자 이후 마스킹":
                params['pattern_type'] = 'after_delimiter'
                params['delimiter'] = st.text_input("구분자", value="-")
                params['position'] = st.radio("위치", ["첫 번째 이후", "마지막만"], horizontal=True)
                params['position'] = 'first' if params['position'] == "첫 번째 이후" else 'last'
            else:
                params['pattern_type'] = 'before_char'
                params['char'] = st.text_input("기준 문자", value="@")
                params['preserve'] = st.number_input("앞부분 보존 글자 수", min_value=0, value=1)
            
            masking_type = "pattern"
            
        elif masking_mode == "조건부":
            st.markdown("### 조건부 마스킹")
            condition_type = st.selectbox(
                "조건 유형",
                ["길이별 다른 처리", "비율 마스킹"]
            )
            
            if condition_type == "길이별 다른 처리":
                params['condition_type'] = 'by_length'
                st.info("""
                자동 규칙:
                - 3글자 이하: 마지막 1글자 마스킹
                - 4-6글자: 마지막 2글자 마스킹
                - 7글자 이상: 양끝 2글자 제외 가운데 마스킹
                """)
            else:
                params['condition_type'] = 'percentage'
                params['percent'] = st.slider("마스킹 비율(%)", 10, 90, 50)
                params['position'] = st.radio("위치", ["앞부분", "뒷부분", "고르게"], horizontal=True)
                params['position'] = {'앞부분': 'front', '뒷부분': 'back', '고르게': 'distributed'}[params['position']]
            
            masking_type = "condition"
            
        else:  # 특수 마스킹
            st.markdown("### 특수 마스킹")
            special_type = st.selectbox(
                "특수 유형",
                ["숫자만 마스킹", "문자만 마스킹", "형식 유지 마스킹"]
            )
            
            if special_type == "숫자만 마스킹":
                params['special_type'] = 'numbers_only'
                st.info("예: 홍길동123 → 홍길동***")
            elif special_type == "문자만 마스킹":
                params['special_type'] = 'letters_only'
                st.info("예: ABC123 → ***123")
            else:
                params['special_type'] = 'keep_format'
                st.info("예: 1234-5678 → ****-****")
            
            masking_type = "special"
    
    # 미리보기
    st.markdown("### 미리보기")
    
    try:
        params['mask_char'] = mask_char
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
    if st.button("✅ 적용", type="primary"):
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
            
            st.session_state.processing_history.append({
                'type': '마스킹',
                'column': selected_column,
                'details': f"{masking_mode} - {mask_char}"
            })
            
            st.success(f"✅ '{selected_column}' 컬럼에 마스킹이 적용되었습니다!")
            st.rerun()
            
        except Exception as e:
            st.error(f"처리 중 오류 발생: {str(e)}")