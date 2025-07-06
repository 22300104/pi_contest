import streamlit as st
import pandas as pd
from modules.data_detector import DataDetector
from modules.preprocessor import DataPreprocessor
from utils.helpers import format_number

def initialize_session_state():
    """세션 상태 초기화"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'detector' not in st.session_state:
        st.session_state.detector = DataDetector()
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    # 통계/시각화 표시 상태
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = {}
    if 'show_viz' not in st.session_state:
        st.session_state.show_viz = {}

def file_upload_section():
    """파일 업로드 섹션"""
    uploaded_file = st.file_uploader("Excel 또는 CSV 파일을 업로드하세요", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            with st.spinner('데이터를 로드하는 중...'):
                try:
                    file_info = st.session_state.detector.detect_file_info(uploaded_file)
                    df, warnings = st.session_state.preprocessor.safe_load_data(uploaded_file, file_info)
                    
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.last_uploaded_file = uploaded_file.name
                        # 상태 초기화
                        st.session_state.show_stats = {}
                        st.session_state.show_viz = {}
                        
                        st.success(f"✅ 파일 로드 완료! (행: {format_number(len(df))}, 열: {len(df.columns)})")
                        
                        if warnings:
                            with st.expander("⚠️ 로드 경고", expanded=True):
                                for warning in warnings:
                                    st.warning(warning)
                    else:
                        st.error("파일 로드에 실패했습니다.")
                        
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

def data_type_guide():
    """데이터 타입별 가이드"""
    with st.expander("📚 데이터 타입별 분석 가이드", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔢 숫자형 (Numeric)**
            - **연속형**: 히스토그램, 박스플롯, 밀도플롯, 기술통계, 분위수, 이상치
            - **이산형**: 막대그래프, 빈도분석, 기술통계
            - **이진형**: 파이차트, 비율분석
            
            **📝 범주형 (Categorical)**
            - **명목형**: 막대그래프, 파이차트, 빈도분석
            - **순서형**: 막대그래프, 순서통계
            - **이진형**: 파이차트, 비율분석
            """)
        
        with col2:
            st.markdown("""
            **💬 텍스트 (Text)**
            - **짧은 텍스트**: 단어빈도, 패턴분석
            - **긴 텍스트**: 길이분포, 텍스트통계
            
            **📅 날짜/시간 (Datetime)**
            - 시계열그래프, 월별/요일별 분포
            - 시간범위, 주기분석
            """)

def data_quality_check(col_data):
    """데이터 품질 검사"""
    with st.expander("🔍 데이터 품질 검사", expanded=False):
        try:
            issues = st.session_state.preprocessor.detect_data_issues(col_data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 데이터 품질**")
                for key, value in issues['data_quality'].items():
                    st.text(f"{key}: {value}")
            
            with col2:
                if issues['warnings']:
                    st.markdown("**⚠️ 문제점**")
                    for warning in issues['warnings']:
                        st.text(f"• {warning}")
                else:
                    st.success("✅ 양호")
            
            with col3:
                if issues['suggestions']:
                    st.markdown("**💡 제안**")
                    for suggestion in issues['suggestions']:
                        st.text(f"• {suggestion}")
        except Exception as e:
            st.error(f"품질 검사 오류: {str(e)}")