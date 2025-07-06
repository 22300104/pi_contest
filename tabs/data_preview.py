import streamlit as st

def render_data_preview_tab():
    """데이터 미리보기 탭"""
    df = st.session_state.df
    
    st.header("데이터 미리보기")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_rows = st.slider("표시할 행 수:", 5, 100, 20)
    
    with col2:
        show_random = st.checkbox("랜덤 샘플링", value=False)
    
    with col3:
        selected_columns = st.multiselect(
            "표시할 열 선택:",
            df.columns.tolist(),
            default=[]
        )
    
    # 데이터 표시
    display_df = df
    
    if selected_columns:
        display_df = display_df[selected_columns]
    
    if show_random:
        display_df = display_df.sample(n=min(n_rows, len(display_df)))
    else:
        display_df = display_df.head(n_rows)
    
    st.dataframe(display_df)
    
    # 간단한 정보
    with st.expander("📊 표시된 데이터 정보"):
        st.text(f"행: {len(display_df)}")
        st.text(f"열: {len(display_df.columns)}")
        