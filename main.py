import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from modules.data_detector import DataDetector
from modules.preprocessor import DataPreprocessor
from modules.ui_components import file_upload_section, initialize_session_state
from tabs.overall_stats import render_overall_stats_tab
from tabs.column_analysis import render_column_analysis_tab
from tabs.data_preview import render_data_preview_tab

# 페이지 설정
st.set_page_config(page_title="Excel/CSV 통계 분석", layout="wide")
st.title("📊 Excel/CSV 파일 통계 분석 도구")

# 세션 상태 초기화
initialize_session_state()

# 파일 업로드
file_upload_section()

# 데이터가 로드된 경우 탭 표시
if st.session_state.df is not None:
    tab1, tab2, tab3 = st.tabs(["📈 전체 통계", "🔍 속성별 분석", "📊 데이터 미리보기"])
    
    with tab1:
        render_overall_stats_tab()
    
    with tab2:
        render_column_analysis_tab()
    
    with tab3:
        render_data_preview_tab()
else:
    st.info("👆 파일을 업로드하여 시작하세요.")