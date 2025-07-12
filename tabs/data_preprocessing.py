import streamlit as st
import pandas as pd
import re

def render_data_preprocessing_tab():
    """데이터 타입 변환 탭 렌더링"""
    st.header("📊 데이터 타입 변환")
    
    # 세션에 데이터가 있는지 확인
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("먼저 데이터를 업로드해주세요.")
        return
    
    df = st.session_state.df
    
    # 전처리된 데이터프레임이 없으면 생성
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = df.copy()
    
    # 타입 변환 섹션만 렌더링
    render_type_conversion_section(df)

def render_type_conversion_section(df):
    """데이터 타입 변환 섹션"""
    # 서브헤더 제거 (이미 메인 헤더가 있음)
    
    # 전처리된 데이터가 있으면 그것을 사용
    display_df = st.session_state.get('df_processed', df)
    
    # 현재 데이터 타입 표시
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("### 현재 데이터 타입")
        type_df = pd.DataFrame({
            '컬럼명': display_df.columns,
            '현재 타입': display_df.dtypes.astype(str),
            '샘플 데이터': [display_df[col].dropna().head(1).values[0] if len(display_df[col].dropna()) > 0 else 'N/A' for col in display_df.columns]
        })
    
    # 동적 높이 설정 (10개 이하는 전체 표시, 그 이상은 스크롤)
    if len(type_df) > 10:
        st.dataframe(type_df, height=400, use_container_width=True)
    else:
        st.dataframe(type_df, use_container_width=True)
    
    with col2:
        st.markdown("### 숫자형 변환 가능 컬럼")
        
        # object 타입 컬럼 분석
        object_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not object_columns:
            st.success("✅ 모든 컬럼이 이미 적절한 타입입니다.")
            return
        
        # 모든 object 컬럼에 대해 변환 가능성 분석
        conversion_info = []
        
        # null로 간주할 값들 정의
        null_values = ['na', 'n/a', 'n.a', 'n.a.', 'NA', 'N/A', 'N.A', 'N.A.',
                      'null', 'NULL', 'Null', 'none', 'None', 'NONE',
                      '-', '--', '---', '.', '..', '...', 
                      '?', '??', '???', 'missing', 'Missing', 'MISSING',
                      '없음', '해당없음', '미상', '알수없음', '모름',
                      '', ' ', '  ']
        null_values_lower = [str(v).lower() for v in null_values]
        
        for col in object_columns:
            # 적응형 샘플링: null이 많으면 샘플 크기를 늘림
            sample_size = 100  # 100만 행에는 100개면 충분
            max_attempts = 3  # 최대 3번까지만 (100 -> 1,000 -> 10,000)
            sample = pd.Series()
            
            for attempt in range(max_attempts):
                current_size = min(sample_size * (10 ** attempt), len(df[col]))
                temp_sample = df[col].head(current_size)
                
                # null 값 필터링
                temp_sample_clean = []
                for val in temp_sample:
                    if pd.isna(val):
                        continue
                    if str(val).strip().lower() in null_values_lower:
                        continue
                    temp_sample_clean.append(val)
                
                if len(temp_sample_clean) > 0 or current_size >= len(df[col]):
                    sample = pd.Series(temp_sample_clean)
                    break
            
            # 변환 가능성 분석
            numeric_count = 0
            patterns = set()
            
            if len(sample) > 0:
                for val in sample:
                    try:
                        # 다양한 패턴 체크
                        str_val = str(val).strip()
                        
                        # null 값인지 다시 확인
                        if str_val.lower() in null_values_lower:
                            continue
                        
                        # 패턴 감지
                        if ',' in str_val:
                            patterns.add('쉼표(,)')
                        if '/' in str_val:
                            patterns.add('슬래시(/)')
                        if ' ' in str_val:
                            patterns.add('공백')
                        if '.' in str_val and str_val.count('.') == 1:
                            patterns.add('소수점(.)')
                        if any(c in str_val for c in ['원', '$', '￦', '₩', '%']):
                            patterns.add('단위/기호')
                        
                        # 변환 시도
                        cleaned = str_val.replace(',', '').replace(' ', '').strip()
                        # 단위 제거
                        for unit in ['원', '$', '￦', '₩', '%']:
                            cleaned = cleaned.replace(unit, '')
                        
                        float(cleaned)
                        numeric_count += 1
                    except Exception:
                        pass
                
                conversion_rate = (numeric_count / len(sample)) * 100
            else:
                conversion_rate = 0
            
            # 실제 null 개수 계산
            actual_null_count = 0
            for val in df[col]:
                if pd.isna(val) or str(val).strip().lower() in null_values_lower:
                    actual_null_count += 1
            
            total_count = len(df[col])
            null_ratio = (actual_null_count / total_count) * 100
            
            conversion_info.append({
                'column': col,
                'rate': conversion_rate,
                'patterns': patterns,
                'samples': sample.head(3).tolist() if len(sample) > 0 else [],
                'null_count': actual_null_count,
                'total_count': total_count,
                'null_ratio': null_ratio,
                'sample_size_checked': len(sample)
            })
        
        if conversion_info:
            st.markdown("**변환할 컬럼을 선택하세요:**")
            
            # 전체 선택/해제
            col_all, col_info = st.columns([1, 3])
            with col_all:
                select_all = st.checkbox("전체 선택", key="select_all_columns")
            
            # 개별 컬럼 선택
            selected_columns = []
            for info in conversion_info:
                col = info['column']
                default = select_all or st.session_state.get(f'convert_{col}', False)
                
                with st.container():
                    col_check, col_info = st.columns([1, 4])
                    with col_check:
                        # 이미 변환된 컬럼은 표시만 하고 선택 불가
                        if st.session_state.get(f'converted_{col}', False):
                            st.checkbox(col, value=True, disabled=True, key=f"select_{col}")
                            st.caption("✅ 변환됨")
                        else:
                            selected = st.checkbox(
                                col,
                                value=default,
                                key=f"select_{col}"
                            )
                            if selected:
                                selected_columns.append(col)
                    
                    with col_info:
                        # null이 아닌 값 기준으로 상태 표시
                        if info['null_count'] == info['total_count']:
                            st.error("❌ 모든 값이 null")
                        elif info['rate'] == 0:
                            st.error("❌ 숫자로 변환 불가")
                            st.caption(f"null 비율: {info['null_ratio']:.1f}%")
                        elif info['rate'] >= 90:
                            st.success(f"✅ 안전: {info['rate']:.1f}% 변환 가능")
                            st.caption(f"null 비율: {info['null_ratio']:.1f}%")
                        elif info['rate'] >= 50:
                            st.warning(f"⚠️ 주의: {info['rate']:.1f}% 변환 가능")
                            st.caption(f"null 비율: {info['null_ratio']:.1f}%")
                        else:
                            st.error(f"⛔ 위험: {info['rate']:.1f}% 변환 가능")
                            st.caption(f"null 비율: {info['null_ratio']:.1f}%")
                        
                        # 추가 정보
                        if info['patterns']:
                            st.caption(f"발견된 패턴: {', '.join(info['patterns'])}")
                        if info['samples']:
                            st.caption(f"예시: {', '.join(map(str, info['samples'][:2]))}")
                        st.caption(f"검사한 샘플: {info['sample_size_checked']:,}개")
            
            # 변환 옵션
            st.markdown("---")
            st.markdown("### 🔧 변환 옵션")
            
            st.info("""
            💡 **변환 옵션 안내**
            - 아래 옵션들은 숫자로 변환하기 전에 텍스트에서 제거할 문자들입니다
            - 예시: "1,234" → "1234", "100 원" → "100"
            """)
            
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                remove_comma = st.checkbox(
                    "쉼표(,) 제거", 
                    value=True,
                    help="천단위 구분 쉼표를 제거합니다. 예: 1,234 → 1234"
                )
                remove_space = st.checkbox(
                    "공백 제거", 
                    value=True,
                    help="숫자 사이나 앞뒤의 공백을 제거합니다. 예: '1 234' → '1234'"
                )
            
            with col_opt2:
                remove_slash = st.checkbox(
                    "슬래시(/) 제거", 
                    value=False,
                    help="날짜나 분수 표현의 슬래시를 제거합니다. 예: '10/20' → '1020'"
                )
                
            with st.expander("🔍 고급 옵션", expanded=False):
                st.info("""
                💡 **사용법 안내**
                - 제거하고 싶은 문자나 기호를 쉼표(,)로 구분해서 입력하세요
                - 예시: 
                  - 괄호 제거: `(, )`
                  - 특수문자 제거: `*, /, ^, #`
                  - 텍스트 제거: `년, 월, 일`
                  - 복잡한 패턴: `/*/*, (주), [참고]`
                """)
                
                custom_pattern = st.text_input(
                    "추가로 제거할 문자", 
                    placeholder="예: (, ), *, /, ^",
                    help="여러 문자를 제거하려면 쉼표로 구분하세요"
                )
                
                # 입력 예시 표시
                if custom_pattern:
                    st.caption(f"✅ 제거될 문자: {custom_pattern}")
                
                # 사용자가 입력한 문자를 정규식으로 변환
                custom_regex = None
                if custom_pattern:
                    # 쉼표로 구분된 문자들을 처리
                    chars_to_remove = [c.strip() for c in custom_pattern.split(',') if c.strip()]
                    if chars_to_remove:
                        st.caption(f"📝 {len(chars_to_remove)}개 패턴을 제거합니다")
                        # 특수문자 이스케이프 처리
                        escaped_chars = [re.escape(c) for c in chars_to_remove]
                        custom_regex = '|'.join(escaped_chars)
            
            # 변환 실행
            col_preview, col_execute = st.columns([2, 1])
            
            with col_preview:
                if st.button("🔍 변환 미리보기", disabled=len(selected_columns) == 0):
                    if selected_columns:
                        preview_df = pd.DataFrame()
                        for col in selected_columns[:3]:  # 최대 3개 컬럼만 미리보기
                            original = df[col].head(5)
                            converted = convert_column(
                                df[col].head(5), 
                                remove_comma, 
                                remove_space, 
                                remove_slash, 
                                custom_regex
                            )
                            preview_df[f'{col} (원본)'] = original
                            preview_df[f'{col} (변환)'] = converted
                        
                        st.dataframe(preview_df)
            
            # 🔽 기존 코드의 "변환 실행" 위치를 찾아 그대로 교체하세요
            with col_execute:
                if st.button("✅ 변환 실행", type="primary", disabled=len(selected_columns) == 0):
                    if selected_columns:
                        progress_bar = st.progress(0)
                        status_text  = st.empty()

                        success_count        = 0          # 변환 완료된 컬럼 수
                        total_rows_converted = 0          # 변환된 행(값) 총합
                        per_col_rows         = {}         # 👉 컬럼별 변환 행 수 기록용

                        for i, col in enumerate(selected_columns):
                            status_text.text(f"변환 중: {col}")
                            progress_bar.progress((i + 1) / len(selected_columns))

                            try:
                                # ① 원본 백업 (옵션) ― 이미 df_processed에 사본을 쓰고 있다면 생략 가능
                                original_series = df[col].copy()

                                # ② 변환
                                converted = convert_column(
                                    df[col],
                                    remove_comma,
                                    remove_space,
                                    remove_slash,
                                    custom_regex
                                )

                                # ③ 세션 상태 갱신
                                st.session_state.df_processed[col] = converted
                                st.session_state[f'converted_{col}'] = True
                                success_count += 1

                                # ④ 변환된 행 수 계산
                                rows_converted = converted.notna().sum()
                                per_col_rows[col] = rows_converted
                                total_rows_converted += rows_converted

                            except Exception as e:
                                st.error(f"{col} 변환 실패: {str(e)}")

                        # 진행 표시 없애기
                        progress_bar.empty()
                        status_text.empty()

                        # 원본 데이터에도 반영
                        st.session_state.df = st.session_state.df_processed.copy()

                        # ⑤ 결과 메시지 (컬럼 수 + 총 행 수)
                        st.success(
                            f"✅ {success_count}/{len(selected_columns)}개 컬럼 변환 완료! "
                            f"총 {total_rows_converted:,}행 변환되었습니다."
                        )

                        # ⑥ 상세 내역 토글 — 원한다면 추가
                        with st.expander("📄 컬럼별 변환 행 수", expanded=False):
                            for c, n_rows in per_col_rows.items():
                                st.write(f"• **{c}** : {n_rows:,} 행")

                        st.balloons()
                        st.rerun()

        else:
            st.info("ℹ️ 문자형(object) 컬럼이 없습니다.")
        
        # 원본으로 되돌리기
        if any(st.session_state.get(f'converted_{col}', False) for col in df.columns):
            st.markdown("---")
            if st.button("↩️ 모든 변환 취소 (원본으로 되돌리기)"):
                # 원본 데이터 복원
                if 'df_original' in st.session_state:
                    st.session_state.df = st.session_state.df_original.copy()
                    st.session_state.df_processed = st.session_state.df_original.copy()
                
                # 변환 플래그 초기화
                for col in df.columns:
                    if f'converted_{col}' in st.session_state:
                        st.session_state[f'converted_{col}'] = False
                
                st.success("✅ 원본 데이터로 복원되었습니다.")
                st.rerun()

def convert_column(series, remove_comma=True, remove_space=True, remove_slash=False, custom_pattern=None):
    """컬럼 데이터를 숫자형으로 변환"""
    # null로 간주할 값들
    null_values = ['na', 'n/a', 'n.a', 'n.a.', 'NA', 'N/A', 'N.A', 'N.A.',
                  'null', 'NULL', 'Null', 'none', 'None', 'NONE',
                  '-', '--', '---', '.', '..', '...', 
                  '?', '??', '???', 'missing', 'Missing', 'MISSING',
                  '없음', '해당없음', '미상', '알수없음', '모름',
                  '', ' ', '  ']
    null_values_lower = [str(v).lower() for v in null_values]
    
    # 문자열로 변환
    str_series = series.astype(str)
    
    # null 값들을 NaN으로 변환
    def clean_value(x):
        if str(x).strip().lower() in null_values_lower:
            return None
        return x
    
    str_series = str_series.apply(clean_value)
    
    # 패턴 제거
    if remove_comma:
        str_series = str_series.str.replace(',', '', regex=False)
    if remove_space:
        str_series = str_series.str.strip()
        str_series = str_series.str.replace(' ', '', regex=False)
    if remove_slash:
        str_series = str_series.str.replace('/', '', regex=False)
    
    # 일반적인 단위 제거
    common_units = ['원', '$', '￦', '₩', '%', '개', '명', '건', '회']
    for unit in common_units:
        str_series = str_series.str.replace(unit, '', regex=False)
    
    if custom_pattern:
        try:
            str_series = str_series.str.replace(custom_pattern, '', regex=True)
        except Exception:
            st.warning(f"잘못된 정규식: {custom_pattern}")
    
    # 숫자로 변환
    return pd.to_numeric(str_series, errors='coerce')