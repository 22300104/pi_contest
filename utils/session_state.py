import streamlit as st
from modules.data_detector import DataDetector
from modules.preprocessor import DataPreprocessor
from modules.anonymizer import DataAnonymizer

def initialize_session_state():
    """세션 상태 초기화"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'detector' not in st.session_state:
        st.session_state.detector = DataDetector()
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    if 'anonymizer' not in st.session_state:
        st.session_state.anonymizer = DataAnonymizer()
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None