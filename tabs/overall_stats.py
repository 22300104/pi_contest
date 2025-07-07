import streamlit as st
import matplotlib.pyplot as plt
from utils.helpers import format_number, get_memory_usage

def render_overall_stats_tab():
    """전체 통계 탭"""
    # 전처리된 데이터가 있으면 그것을 사용, 없으면 원본 사용
    df = st.session_state.get('df_processed', st.session_state.df)
    
    st.header("전체 통계 정보")
    
    # 전처리 상태 알림
    if 'df_processed' in st.session_state:
        converted_cols = [col for col in df.columns if st.session_state.get(f'converted_{col}', False)]
        if converted_cols:
            st.info(f"📊 전처리된 데이터 기준입니다. 변환된 컬럼: {', '.join(converted_cols)}")
    
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
    
    # 데이터 타입 분포 - 버튼으로 실행
    st.divider()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("📊 데이터 타입 분포 보기", key="show_dtype"):
            st.session_state.show_dtype_chart = True
    
    with col2:
        if st.session_state.get('show_dtype_chart', False):
            with st.spinner("차트 생성 중..."):
                try:
                    fig = st.session_state.detector.create_dtype_distribution_chart(df)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"차트 생성 오류: {str(e)}")
    
    # 열별 정보
    with st.expander("📋 열별 상세 정보"):
        col_info = []
        for col in df.columns:
            col_data = {
                '열 이름': col,
                '타입': str(df[col].dtype),
                '결측값': df[col].isnull().sum(),
                '고유값': df[col].nunique(),
                '결측률': f"{df[col].isnull().sum() / len(df) * 100:.1f}%"
            }
            
            # 변환된 컬럼 표시
            if st.session_state.get(f'converted_{col}', False):
                col_data['상태'] = '✅ 변환됨'
            else:
                col_data['상태'] = ''
            
            col_info.append(col_data)
        
        st.dataframe(col_info)