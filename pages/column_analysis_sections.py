import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def get_analysis_df(df: pd.DataFrame, selected_column: str) -> pd.DataFrame:
    """처리된 데이터가 있으면 반환, 없으면 원본 반환"""
    if 'df_processed' in st.session_state and selected_column in st.session_state.df_processed.columns:
        if st.session_state.get(f'processed_{selected_column}', False):
            return st.session_state.df_processed
    return df

def render_data_preprocessing(df: pd.DataFrame, selected_column: str):
    """데이터 전처리 섹션"""
    st.subheader("🔧 데이터 전처리")
    
    # 현재 분석에 사용 중인 데이터 가져오기
    analysis_df = get_analysis_df(df, selected_column)
    col_data = analysis_df[selected_column]
    
    # 처리 상태 표시
    if st.session_state.get(f'processed_{selected_column}', False):
        st.success("✅ 전처리된 데이터 사용 중")
    
    # Null 값 처리
    if col_data.dtype == 'object':
        with st.expander("Null 값 처리"):
            # 빈도 높은 짧은 값들 감지
            try:
                value_counts = col_data.value_counts()
                potential_nulls = []
                
                for value, count in value_counts.items():
                    if isinstance(value, str) and len(value.strip()) <= 3:
                        if count > len(col_data) * 0.01:
                            potential_nulls.append(value)
                
                if potential_nulls:
                    st.info(f"잠재적 null: {', '.join([f'"{v}"' for v in potential_nulls[:10]])}")
            except:
                pass
            
            custom_nulls = st.text_area(
                "추가 null 표현 (한 줄에 하나씩):",
                help="예: -\nN/A\n없음\n해당없음",
                key=f"custom_nulls_{selected_column}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Null 처리 적용", key=f"null_process_{selected_column}"):
                    custom_null_list = [x.strip() for x in custom_nulls.split('\n') if x.strip()]
                    
                    # 원본 데이터를 복사하여 처리
                    if 'df_processed' not in st.session_state:
                        st.session_state.df_processed = df.copy()
                    
                    processed_data = st.session_state.preprocessor.detect_and_convert_nulls(
                        df[selected_column].copy(), custom_null_list
                    )
                    
                    st.session_state.df_processed[selected_column] = processed_data
                    st.session_state[f'processed_{selected_column}'] = True
                    st.success("✅ Null 처리 완료! 아래에서 분석 결과를 확인하세요.")
            
            with col2:
                if st.session_state.get(f'processed_{selected_column}', False):
                    if st.button("원본으로 복원", key=f"restore_{selected_column}"):
                        st.session_state.df_processed[selected_column] = df[selected_column].copy()
                        st.session_state[f'processed_{selected_column}'] = False
                        st.info("원본 데이터로 복원되었습니다.")
    
    # 타입 변환
    with st.expander("타입 변환"):
        col1, col2 = st.columns(2)
        
        with col1:
            if col_data.dtype == 'object':
                if st.button("숫자형으로 변환 시도", key=f"to_numeric_{selected_column}"):
                    # 현재 데이터 (처리된 것이 있으면 그것을) 변환
                    current_data = st.session_state.df_processed[selected_column] if 'df_processed' in st.session_state else df[selected_column]
                    
                    converted, stats = st.session_state.preprocessor.safe_type_conversion(
                        current_data, 'numeric'
                    )
                    if stats['success']:
                        if 'df_processed' not in st.session_state:
                            st.session_state.df_processed = df.copy()
                        
                        st.session_state.df_processed[selected_column] = converted
                        st.session_state[f'processed_{selected_column}'] = True
                        st.success("✅ 변환 성공! 아래에서 분석 결과를 확인하세요.")
                        
                        # 변환 통계 표시
                        with st.container():
                            st.json(stats['conversion_stats'])
                    else:
                        st.error("변환 실패")
                        if stats.get('errors'):
                            st.error(f"오류: {stats['errors'][0]}")
        
        with col2:
            if col_data.dtype == 'object':
                if st.button("날짜형으로 변환 시도", key=f"to_datetime_{selected_column}"):
                    current_data = st.session_state.df_processed[selected_column] if 'df_processed' in st.session_state else df[selected_column]
                    
                    converted, stats = st.session_state.preprocessor.safe_type_conversion(
                        current_data, 'datetime'
                    )
                    if stats['success']:
                        if 'df_processed' not in st.session_state:
                            st.session_state.df_processed = df.copy()
                        
                        st.session_state.df_processed[selected_column] = converted
                        st.session_state[f'processed_{selected_column}'] = True
                        st.success("✅ 변환 성공! 아래에서 분석 결과를 확인하세요.")
                    else:
                        st.error("변환 실패")

def render_data_type_selection(col_data: pd.Series) -> Tuple[str, str]:
    """데이터 타입 선택 섹션"""
    st.subheader("🏷️ 데이터 타입 선택")
    
    status_col, main_col, sub_col = st.columns([1, 1, 1])
    
    with status_col:
        st.markdown("**현재 데이터 상태**")
        if pd.api.types.is_numeric_dtype(col_data):
            st.success("✅ 숫자형")
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            st.success("✅ 날짜/시간형")
        else:
            st.info("📝 문자형")
        
        st.metric("고유값 비율", f"{col_data.nunique()/len(col_data)*100:.1f}%")
    
    with main_col:
        main_type = st.radio(
            "주 데이터 타입:",
            ["numeric", "categorical", "text", "datetime"],
            format_func=lambda x: {
                "numeric": "🔢 숫자형",
                "categorical": "📝 범주형",
                "text": "💬 텍스트",
                "datetime": "📅 날짜/시간"
            }[x]
        )
    
    with sub_col:
        if main_type == "numeric":
            sub_type = st.radio(
                "세부 타입:",
                ["continuous", "discrete", "binary"],
                format_func=lambda x: {
                    "continuous": "〰️ 연속형",
                    "discrete": "🔢 이산형",
                    "binary": "☯️ 이진형"
                }[x]
            )
        elif main_type == "categorical":
            sub_type = st.radio(
                "세부 타입:",
                ["nominal", "ordinal", "binary"],
                format_func=lambda x: {
                    "nominal": "🏷️ 명목형",
                    "ordinal": "📊 순서형",
                    "binary": "☯️ 이진형"
                }[x]
            )
        elif main_type == "text":
            sub_type = st.radio(
                "세부 타입:",
                ["short", "long"],
                format_func=lambda x: {
                    "short": "📝 짧은 텍스트",
                    "long": "📄 긴 텍스트"
                }[x]
            )
        else:
            sub_type = "datetime"
            st.info("📅 날짜/시간형")
    
    return main_type, sub_type

def render_statistical_analysis(df: pd.DataFrame, selected_column: str, main_type: str, sub_type: str):
    """통계 분석 섹션"""
    st.markdown("### 📊 통계 분석")
    
    appropriate_stats = st.session_state.detector.get_appropriate_stats(main_type, sub_type)
    
    if appropriate_stats:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_stat = st.radio(
                "통계 방법 선택:",
                appropriate_stats,
                key="stat_radio"
            )
        
        with col2:
            if selected_stat:
                try:
                    result = st.session_state.detector.calculate_statistics(
                        df, selected_column, selected_stat
                    )
                    
                    # 결과를 보기 좋게 표시
                    if isinstance(result, dict):
                        if len(result) > 4:
                            subcol1, subcol2 = st.columns(2)
                            items = list(result.items())
                            mid = len(items) // 2
                            
                            with subcol1:
                                for key, value in items[:mid]:
                                    st.metric(key, value)
                            with subcol2:
                                for key, value in items[mid:]:
                                    st.metric(key, value)
                        else:
                            cols = st.columns(len(result))
                            for i, (key, value) in enumerate(result.items()):
                                with cols[i]:
                                    st.metric(key, value)
                    else:
                        st.write(result)
                except Exception as e:
                    st.error(f"계산 오류: {str(e)}")

def render_visualization_section(df: pd.DataFrame, selected_column: str, main_type: str, sub_type: str):
    """시각화 섹션"""
    st.markdown("### 📉 시각화")
    
    appropriate_viz = st.session_state.detector.get_appropriate_visualizations(main_type, sub_type)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_viz = st.radio(
            "시각화 방법 선택:",
            appropriate_viz,
            key="viz_radio"
        )
        
        # 시각화 파라미터
        if selected_viz:
            params = {}
            
            if "상위" in selected_viz or "파레토" in selected_viz:
                params['top_n'] = st.slider("표시할 항목 수:", 5, 50, 20, key="top_n_slider")
            elif selected_viz == "히스토그램":
                params['bins'] = st.slider("구간 수:", 10, 100, 30, key="bins_slider")
            elif selected_viz == "산점도":
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                if selected_column in numeric_cols:
                    numeric_cols.remove(selected_column)
                
                if numeric_cols:
                    params['other_column'] = st.selectbox(
                        "Y축 변수 선택:", 
                        numeric_cols,
                        key="scatter_y_axis"
                    )
                    params['show_regression'] = st.checkbox(
                        "회귀선 표시", 
                        value=False,
                        key="show_regression"
                    )
                else:
                    st.warning("다른 수치형 열이 없어 인덱스를 X축으로 사용합니다.")
    
    with col2:
        if selected_viz:
            with st.container():
                try:
                    fig = st.session_state.detector.create_visualization(
                        df, selected_column, selected_viz, params if 'params' in locals() else {}
                    )
                    st.pyplot(fig, use_container_width=False)
                    plt.close()
                except Exception as e:
                    st.error(f"시각화 오류: {str(e)}")
                    st.info("데이터를 확인하고 다른 방법을 시도해보세요.")

def render_summary_section(df: pd.DataFrame, selected_column: str, main_type: str, sub_type: str):
    """전체 요약 섹션"""
    st.divider()
    st.markdown("### 📑 전체 요약")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("📊 모든 통계 한번에 보기", key="show_all_stats"):
            render_all_statistics(df, selected_column, main_type, sub_type)
    
    with col2:
        if st.checkbox("🎨 모든 시각화 한번에 보기", key="show_all_viz"):
            render_visualization_gallery(df, selected_column, main_type, sub_type)

def render_all_statistics(df: pd.DataFrame, selected_column: str, main_type: str, sub_type: str):
    """모든 통계 표시"""
    with st.container():
        st.markdown("#### 전체 통계 요약")
        
        appropriate_stats = st.session_state.detector.get_appropriate_stats(main_type, sub_type)
        
        for stat_name in appropriate_stats:
            try:
                st.markdown(f"**{stat_name}**")
                result = st.session_state.detector.calculate_statistics(
                    df, selected_column, stat_name
                )
                
                if isinstance(result, dict):
                    result_str = " | ".join([f"{k}: {v}" for k, v in result.items()])
                    st.text(result_str)
                else:
                    st.write(result)
                
                st.divider()
            except Exception as e:
                st.error(f"{stat_name} 오류: {str(e)[:50]}...")

def render_visualization_gallery(df: pd.DataFrame, selected_column: str, main_type: str, sub_type: str):
    """시각화 갤러리"""
    with st.container():
        st.markdown("#### 시각화 갤러리")
        
        appropriate_viz = st.session_state.detector.get_appropriate_visualizations(main_type, sub_type)
        viz_to_show = appropriate_viz[:4]
        
        for i in range(0, len(viz_to_show), 2):
            subcol1, subcol2 = st.columns(2)
            
            for j, col in enumerate([subcol1, subcol2]):
                if i + j < len(viz_to_show):
                    with col:
                        viz_type = viz_to_show[i + j]
                        try:
                            st.markdown(f"**{viz_type}**")
                            params = {}
                            if "상위" in viz_type:
                                params['top_n'] = 10
                            elif viz_type == "히스토그램":
                                params['bins'] = 20
                            elif viz_type == "산점도":
                                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                                if selected_column in numeric_cols:
                                    numeric_cols.remove(selected_column)
                                if numeric_cols:
                                    params['other_column'] = numeric_cols[0]
                            
                            fig = st.session_state.detector.create_visualization(
                                df, selected_column, viz_type, params
                            )
                            fig.set_size_inches(4, 3)
                            st.pyplot(fig, use_container_width=False)
                            plt.close()
                        except Exception as e:
                            st.error(f"생성 실패")