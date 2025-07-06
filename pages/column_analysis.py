import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.ui_components import data_type_guide, data_quality_check
from pages.column_analysis_sections import (
    get_analysis_df,  # 이 줄 추가!
    render_data_preprocessing,
    render_data_type_selection,
    render_statistical_analysis,
    render_visualization_section,
    render_summary_section
)

# ... 나머지 코드는 동일

def render_column_analysis(df: pd.DataFrame):
    """속성별 분석 탭 렌더링"""
    st.header("속성별 분석")
    
    # 데이터 타입별 가이드
    data_type_guide()
    
    # 속성 선택
    selected_column = st.selectbox("분석할 속성 선택:", df.columns.tolist())
    
    if selected_column:
        # 처리된 데이터가 있으면 사용
        analysis_df = get_analysis_df(df, selected_column)
        col_data = analysis_df[selected_column]
        
        # 데이터 품질 검사
        data_quality_check(col_data)
        
        # 샘플 데이터 표시 - analysis_df 사용
        render_data_sample(analysis_df, selected_column)
        
        # 데이터 전처리
        render_data_preprocessing(df, selected_column)  # 원본 df 전달
        
        # 전처리 후 다시 데이터 가져오기 (처리된 것이 있으면 그것을)
        analysis_df = get_analysis_df(df, selected_column)
        col_data = analysis_df[selected_column]
        
        # 데이터 타입 선택
        main_type, sub_type = render_data_type_selection(col_data)
        
        # 분석 섹션
        st.divider()
        st.subheader("📈 데이터 분석")
        
        # 기본 정보 - analysis_df 사용
        render_basic_info(analysis_df, selected_column)
        
        # 통계 분석 - analysis_df 사용
        render_statistical_analysis(analysis_df, selected_column, main_type, sub_type)
        
        # 시각화 - analysis_df 사용
        render_visualization_section(analysis_df, selected_column, main_type, sub_type)
        
        # 전체 요약 - analysis_df 사용
        render_summary_section(analysis_df, selected_column, main_type, sub_type)
        
def render_data_sample(df: pd.DataFrame, selected_column: str):
    """데이터 샘플 표시"""
    st.subheader("📋 데이터 샘플")
    
    col_data = df[selected_column]
    
    # 기본 정보 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("전체 행", f"{len(col_data):,}")
    with col2:
        st.metric("고유값", f"{col_data.nunique():,}")
    with col3:
        st.metric("결측값", f"{col_data.isnull().sum():,}")
    with col4:
        st.metric("타입", str(col_data.dtype))
    
    # 샘플 표시
    st.text("데이터 샘플 (최대 10개):")
    try:
        sample_data = col_data.head(10)
        sample_display = []
        for i, value in enumerate(sample_data):
            if pd.isna(value):
                sample_display.append(f"{i}: <NA>")
            else:
                str_value = str(value)
                if len(str_value) > 50:
                    str_value = str_value[:50] + "..."
                sample_display.append(f"{i}: {str_value}")
        
        st.code('\n'.join(sample_display))
    except Exception as e:
        st.error(f"샘플 표시 오류: {str(e)}")

def render_basic_info(df: pd.DataFrame, selected_column: str):
    """기본 정보 표시"""
    with st.expander("📋 기본 정보", expanded=True):
        basic_info = st.session_state.detector.calculate_statistics(df, selected_column, "기본정보")
        col1, col2 = st.columns(2)
        items = list(basic_info.items())
        mid = len(items) // 2
        
        with col1:
            for key, value in items[:mid]:
                st.text(f"{key}: {value}")
        with col2:
            for key, value in items[mid:]:
                st.text(f"{key}: {value}")