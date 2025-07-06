import streamlit as st
import pandas as pd

def render_data_preview(df: pd.DataFrame):
    """데이터 미리보기 탭 렌더링"""
    st.header("데이터 미리보기")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_rows = st.slider("표시할 행 수:", 5, 100, 20)
    
    with col2:
        show_random = st.checkbox("랜덤 샘플링", value=False)
    
    with col3:
        selected_columns = st.multiselect(
            "표시할 열 선택 (비어있으면 전체):",
            df.columns.tolist()
        )
    
    try:
        # 표시할 데이터 준비
        if show_random:
            display_df = df.sample(n=min(n_rows, len(df)))
        else:
            display_df = df.head(n_rows)
        
        # 선택된 열만 표시
        if selected_columns:
            display_df = display_df[selected_columns]
        
        # 데이터 표시
        st.dataframe(display_df)
        
        # 데이터 정보
        with st.expander("📊 데이터 정보"):
            st.text(f"표시된 행: {len(display_df)}")
            st.text(f"표시된 열: {len(display_df.columns)}")
            
            # 메모리 사용량
            memory_usage = display_df.memory_usage(deep=True).sum()
            st.text(f"메모리 사용량: {memory_usage / 1024 / 1024:.2f} MB")
            
    except Exception as e:
        st.error(f"데이터 표시 오류: {str(e)}")