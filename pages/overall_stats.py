import streamlit as st
import pandas as pd
from utils.helpers import format_number, get_memory_usage

def render_overall_stats(df: pd.DataFrame):
    """전체 통계 탭 렌더링"""
    st.header("전체 통계 정보")
    
    # 기본 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 행 수", format_number(len(df)))
    with col2:
        st.metric("총 열 수", len(df.columns))
    with col3:
        total_cells = len(df) * len(df.columns) if len(df.columns) > 0 else 1
        missing_ratio = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0
        st.metric("결측값 비율", f"{missing_ratio:.2f}%")
    with col4:
        st.metric("메모리 사용량", get_memory_usage(df))
    
    # 데이터 타입 분포
    st.subheader("데이터 타입 분포")
    try:
        dtype_chart = st.session_state.detector.create_dtype_distribution_chart(df)
        st.pyplot(dtype_chart)
    except Exception as e:
        st.error(f"차트 생성 오류: {str(e)}")
    
    # 열별 정보
    with st.expander("📋 열별 상세 정보"):
        col_info = []
        for col in df.columns:
            col_info.append({
                '열 이름': col,
                '데이터 타입': str(df[col].dtype),
                '결측값': df[col].isnull().sum(),
                '고유값': df[col].nunique(),
                '결측률': f"{df[col].isnull().sum() / len(df) * 100:.1f}%"
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df)