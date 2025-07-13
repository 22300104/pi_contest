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
from tabs.de_identification import render_de_identification_tab
from tabs.data_preprocessing import render_data_preprocessing_tab
from tabs.privacy_evaluation import render_privacy_evaluation_tab  # 새로운 import

# 페이지 설정
st.set_page_config(page_title="Excel/CSV 통계 분석", layout="wide")
st.title("📊 Excel/CSV 파일 통계 분석 도구")

# 세션 상태 초기화
initialize_session_state()

# 파일 업로드
file_upload_section()

# 데이터가 로드된 경우 메뉴 표시
if st.session_state.df is not None:
    # 구분선
    st.markdown("---")
   
    # 라디오 버튼으로 메뉴 선택 (순서 변경 및 새 탭 추가)
    menu = st.radio(
        "메뉴 선택",
        [
            "📈 전체 통계", 
            "📊 데이터 타입 변환", 
            "🔍 속성별 분석", 
            "🔐 비식별화", 
            "📋 프라이버시 평가",  # 새로운 탭
            "📥 미리보기 및 다운로드"  # 마지막으로 이동
        ],
        horizontal=True,
        key="main_menu_selection"
    )
    st.markdown("---")
   
    # 선택된 메뉴에 따라 콘텐츠 표시
    if menu == "📈 전체 통계":
        render_overall_stats_tab()
    elif menu == "📊 데이터 타입 변환":
        render_data_preprocessing_tab()
    elif menu == "🔍 속성별 분석":
        render_column_analysis_tab()
    elif menu == "🔐 비식별화":
        render_de_identification_tab()
    elif menu == "📋 프라이버시 평가":  # 새로운 조건
        render_privacy_evaluation_tab()
    elif menu == "📥 미리보기 및 다운로드":
        render_data_preview_tab()
else:
    st.info("👆 파일을 업로드하여 시작하세요.")