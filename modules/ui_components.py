import streamlit as st
import pandas as pd
from modules.data_detector import DataDetector
from modules.preprocessor import DataPreprocessor
import io
try:
    import msoffcrypto
    MSOFFCRYPTO_AVAILABLE = True
except ImportError:
    MSOFFCRYPTO_AVAILABLE = False

def initialize_session_state():
    """세션 상태 초기화"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'detector' not in st.session_state:
        st.session_state.detector = DataDetector()
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = {}
    if 'show_viz' not in st.session_state:
        st.session_state.show_viz = {}

def file_upload_section():
    """파일 업로드 섹션"""
    uploaded_file = st.file_uploader(
        "Excel 또는 CSV 파일을 업로드하세요",
        type=['xlsx', 'xls', 'csv'],
        help="최대 200MB까지 업로드 가능합니다"
    )
    
    if uploaded_file is not None:
        try:
            # CSV 파일 처리
            if uploaded_file.name.endswith('.csv'):
                # 기존 CSV 처리 코드 그대로 유지
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    encodings = ['cp949', 'euc-kr', 'latin1', 'iso-8859-1']
                    
                    for encoding in encodings:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            st.info(f"📝 파일 인코딩: {encoding}")
                            break
                        except:
                            continue
                    else:
                        raise Exception("지원하지 않는 파일 인코딩입니다.")
                
                st.session_state.df = df
                st.success(f"✅ 파일 업로드 성공! (크기: {len(df):,} 행 × {len(df.columns)} 열)")
            
            # Excel 파일 처리
            else:
                handle_excel_file(uploaded_file)
            
            # 원본 데이터 백업
            if 'df' in st.session_state and st.session_state.df is not None:
                if 'df_original' not in st.session_state:
                    st.session_state.df_original = st.session_state.df.copy()
                    
        except Exception as e:
            st.error(f"❌ 파일 읽기 오류: {str(e)}")
            if uploaded_file.name.endswith('.csv'):
                st.info("💡 CSV 파일의 경우 인코딩 문제일 수 있습니다. Excel 파일로 변환하여 시도해보세요.")

def handle_excel_file(uploaded_file):
    """Excel 파일 처리 (디버깅 정보 포함)"""
    
    # 파일 정보 출력
    st.info(f"""
    📄 파일 정보:
    - 이름: {uploaded_file.name}
    - 크기: {uploaded_file.size / 1024 / 1024:.2f} MB
    - 타입: {uploaded_file.type}
    """)
    
    # 먼저 비밀번호 보호 확인
    if MSOFFCRYPTO_AVAILABLE:
        try:
            uploaded_file.seek(0)
            # msoffcrypto로 먼저 확인
            office_file = msoffcrypto.OfficeFile(uploaded_file)
            
            if office_file.is_encrypted():
                st.warning("🔒 이 파일은 암호화되어 있습니다.")
                handle_encrypted_excel(uploaded_file)
                return
        except:
            # 암호화 확인 실패시 일반 방법으로 계속 진행
            pass
    
    # 파일 내용 확인 (처음 몇 바이트)
    uploaded_file.seek(0)
    header = uploaded_file.read(8)
    uploaded_file.seek(0)
    
    # ZIP 파일 시그니처 확인 (xlsx는 실제로 zip 형식)
    if header.startswith(b'PK'):
        st.success("✅ 올바른 Excel 2007+ 형식 (ZIP 기반)")
    else:
        st.warning("⚠️ 파일 형식이 예상과 다릅니다.")
        st.text(f"파일 헤더: {header.hex()}")
    
    # 각 엔진으로 시도하면서 구체적인 오류 메시지 표시
    engines = ['openpyxl', 'xlrd', None]
    
    for engine in engines:
        try:
            st.info(f"🔄 시도 중: {engine or 'default'} 엔진")
            uploaded_file.seek(0)
            
            if engine:
                df = pd.read_excel(uploaded_file, engine=engine)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success(f"✅ 성공! 엔진: {engine or 'default'}")
            st.success(f"✅ 파일 로드 완료! (크기: {len(df):,} 행 × {len(df.columns)} 열)")
            return
            
        except Exception as e:
            st.error(f"❌ {engine or 'default'} 엔진 실패: {type(e).__name__}: {str(e)}")
            
            # 특정 오류에 대한 추가 정보
            if "OLE2" in str(e):
                st.info("💡 이 파일은 구형 Excel 형식이거나 손상되었을 수 있습니다.")
            elif "encrypted" in str(e).lower():
                st.info("💡 암호화된 파일일 수 있습니다.")
    
    # 모든 방법 실패
    st.error("❌ 파일을 읽을 수 없습니다.")
    
    # 대안 제시
    with st.expander("🛠️ 해결 방법"):
        st.markdown("""
        **1. Excel에서 다시 저장하기:**
        - Excel에서 파일 열기
        - '파일' → '다른 이름으로 저장'
        - 파일 형식: 'Excel 통합 문서 (*.xlsx)' 선택
        - 새 이름으로 저장
        
        **2. CSV로 변환하기:**
        - Excel에서 파일 열기
        - '파일' → '다른 이름으로 저장'
        - 파일 형식: 'CSV UTF-8 (쉼표로 분리)' 선택
        
        **3. 온라인 변환 도구 사용:**
        - Google Sheets에 업로드 후 다시 다운로드
        - 온라인 Excel to CSV 변환기 사용
        """)


def handle_encrypted_excel(uploaded_file):
    """비밀번호로 보호된 Excel 파일 처리"""
    st.warning("🔒 이 파일은 비밀번호로 보호되어 있습니다.")
    password = st.text_input("비밀번호를 입력하세요:", type="password", key="excel_pwd")
    
    if password:
        try:
            # 파일 복호화
            uploaded_file.seek(0)
            decrypted = io.BytesIO()
            
            office_file = msoffcrypto.OfficeFile(uploaded_file)
            office_file.load_key(password=password)
            office_file.decrypt(decrypted)
            
            # 복호화된 파일 처리
            decrypted.seek(0)
            excel_file = pd.ExcelFile(decrypted)
            
            if len(excel_file.sheet_names) > 1:
                st.success("✅ 비밀번호가 맞습니다!")
                selected_sheet = st.selectbox(
                    f"분석할 시트를 선택하세요 ({len(excel_file.sheet_names)}개):",
                    excel_file.sheet_names,
                    key="encrypted_sheet_selector"
                )
                
                if st.button("선택한 시트 불러오기", key="load_encrypted_sheet"):
                    decrypted.seek(0)
                    df = pd.read_excel(decrypted, sheet_name=selected_sheet)
                    st.session_state.df = df
                    st.success(f"✅ '{selected_sheet}' 시트 로드 완료! (크기: {len(df):,} 행 × {len(df.columns)} 열)")
            else:
                df = pd.read_excel(decrypted)
                st.session_state.df = df
                st.success(f"✅ 파일 로드 완료! (크기: {len(df):,} 행 × {len(df.columns)} 열)")
                
        except Exception as e:
            if "password" in str(e).lower():
                st.error("❌ 비밀번호가 틀렸습니다.")
            else:
                st.error(f"❌ 오류: {str(e)}")
def data_type_guide():
    """데이터 타입 가이드"""
    with st.expander("📚 데이터 타입 선택 가이드", expanded=False):
        st.markdown("""
        ### 주 데이터 타입
        - **숫자형**: 계산 가능한 수치 데이터
        - **범주형**: 그룹이나 카테고리를 나타내는 데이터
        - **텍스트**: 자유로운 문자 데이터
        - **날짜/시간**: 시간 정보를 포함한 데이터
        
        ### 세부 타입
        - **연속형**: 실수 값 (예: 키, 몸무게)
        - **이산형**: 정수 값 (예: 개수, 횟수)
        - **명목형**: 순서가 없는 범주 (예: 성별, 지역)
        - **순서형**: 순서가 있는 범주 (예: 등급, 만족도)
        """)

def data_quality_check(col_data):
    """데이터 품질 체크"""
    issues = []
    
    # 결측값 체크
    missing_rate = (col_data.isnull().sum() / len(col_data)) * 100
    if missing_rate > 20:
        issues.append(f"⚠️ 결측값이 {missing_rate:.1f}%로 많습니다")
    
    # 고유값 체크
    unique_rate = (col_data.nunique() / len(col_data)) * 100
    if unique_rate < 1:
        issues.append(f"⚠️ 고유값이 {unique_rate:.1f}%로 매우 적습니다")
    elif unique_rate > 95:
        issues.append(f"ℹ️ 고유값이 {unique_rate:.1f}%로 매우 많습니다")
    
    # 이슈 표시
    if issues:
        for issue in issues:
            st.warning(issue)