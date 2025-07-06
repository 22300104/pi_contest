import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from modules.data_detector import DataDetector
from modules.preprocessor import DataPreprocessor
from modules.anonymizer import DataAnonymizer
from utils.helpers import format_number, get_memory_usage

st.set_page_config(page_title="Excel/CSV 통계 분석", layout="wide")
st.title("📊 Excel/CSV 파일 통계 분석 도구")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'detector' not in st.session_state:
    st.session_state.detector = DataDetector()
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()

uploaded_file = st.file_uploader("Excel 또는 CSV 파일을 업로드하세요", type=['xlsx', 'xls', 'csv'])

if uploaded_file is not None:
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        with st.spinner('데이터를 로드하는 중...'):
            try:
                file_info = st.session_state.detector.detect_file_info(uploaded_file)
                df = st.session_state.preprocessor.load_data(uploaded_file, file_info)
                
                if df is not None:
                    st.session_state.df = df
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.success(f"✅ 파일 로드 완료! (행: {format_number(len(df))}, 열: {len(df.columns)})")
                    
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

if st.session_state.df is not None:
    df = st.session_state.df
    
    tab1, tab2, tab3 = st.tabs(["📈 전체 통계", "🔍 속성별 분석", "📊 데이터 미리보기"])
    
    with tab1:
        st.header("전체 통계 정보")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 행 수", format_number(len(df)))
        with col2:
            st.metric("총 열 수", len(df.columns))
        with col3:
            st.metric("결측값 비율", f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%")
        with col4:
            st.metric("메모리 사용량", get_memory_usage(df))
    
    with tab2:
            st.header("속성별 분석")
            
            # 속성 선택
            selected_column = st.selectbox("분석할 속성 선택:", df.columns.tolist())
            
            if selected_column:
                # 샘플 데이터 표시
                st.subheader("📋 데이터 샘플")
                col_data = df[selected_column]
                
                # 기본 정보 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("고유값 수", f"{col_data.nunique():,}")
                with col2:
                    st.metric("결측값", f"{col_data.isnull().sum():,}")
                with col3:
                    st.metric("데이터 타입", str(col_data.dtype))
                
                # 샘플 표시
                st.text("처음 10개 값:")
                sample_values = col_data.dropna().head(10).tolist()
                st.code(', '.join(map(str, sample_values)))
                
                # 숫자 변환 가능성 체크
                if col_data.dtype == 'object':
                    try:
                        # 숫자 변환 시도
                        numeric_test = pd.to_numeric(col_data.dropna().head(100), errors='coerce')
                        conversion_rate = numeric_test.notna().sum() / len(numeric_test)
                        
                        if conversion_rate > 0.8:  # 80% 이상 변환 가능하면
                            st.warning(f"💡 이 열의 {conversion_rate*100:.1f}%가 숫자로 변환 가능합니다.")
                            
                            if st.checkbox("숫자형으로 변환하여 분석하기"):
                                # 숫자로 변환
                                col_data = pd.to_numeric(col_data, errors='coerce')
                                df[selected_column] = col_data  # 원본 데이터프레임도 업데이트
                                st.success("✅ 숫자형으로 변환되었습니다.")
                                
                                # 변환 결과 표시
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("변환 성공", f"{col_data.notna().sum():,}개")
                                with col2:
                                    st.metric("변환 실패 (NaN)", f"{col_data.isna().sum() - df[selected_column].isna().sum():,}개")
                    except:
                        pass
                
                # 데이터 타입 선택
                st.subheader("🏷️ 데이터 타입 선택")
                
                # 현재 데이터 타입에 따른 추천
                if pd.api.types.is_numeric_dtype(col_data):
                    st.info("💡 추천: numeric (숫자형)")
                elif col_data.dtype == 'object':
                    unique_ratio = col_data.nunique() / len(col_data)
                    if unique_ratio < 0.05:
                        st.info("💡 추천: categorical (범주형)")
                    else:
                        st.info("💡 추천: text (텍스트)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    main_type = st.radio(
                        "주 데이터 타입을 선택하세요:",
                        ["numeric (숫자형)", "categorical (범주형)", "text (텍스트)", "datetime (날짜/시간)"],
                        help="데이터의 주요 특성을 선택해주세요"
                    )
                    main_type = main_type.split()[0]  # 괄호 앞부분만 추출
                
                with col2:
                    if main_type == "numeric":
                        sub_type = st.radio(
                            "세부 타입을 선택하세요:",
                            ["continuous (연속형) - 실수, 측정값", 
                            "discrete (이산형) - 정수, 개수",
                            "binary (이진형) - 0/1, Yes/No"],
                            help="continuous: 키, 몸무게, 온도 등\ndiscrete: 나이, 개수, 순위 등\nbinary: 성별, 합격여부 등"
                        )
                    elif main_type == "categorical":
                        sub_type = st.radio(
                            "세부 타입을 선택하세요:",
                            ["nominal (명목형) - 순서 없음",
                            "ordinal (순서형) - 순서 있음",
                            "binary (이진형) - 두 가지 값"],
                            help="nominal: 지역, 혈액형 등\nordinal: 학력, 등급 등\nbinary: Yes/No, True/False 등"
                        )
                    elif main_type == "text":
                        sub_type = st.radio(
                            "세부 타입을 선택하세요:",
                            ["short (짧은 텍스트) - 이름, 단어",
                            "long (긴 텍스트) - 문장, 설명"],
                            help="short: 이름, 제목 등\nlong: 리뷰, 설명 등"
                        )
                    else:  # datetime
                        sub_type = "datetime"
                        # 날짜 형식 변환 옵션
                        if col_data.dtype == 'object':
                            st.warning("💡 날짜/시간 형식으로 변환이 필요할 수 있습니다.")
                            if st.checkbox("날짜/시간 형식으로 변환하기"):
                                try:
                                    col_data = pd.to_datetime(col_data, errors='coerce')
                                    df[selected_column] = col_data
                                    st.success("✅ 날짜/시간 형식으로 변환되었습니다.")
                                except Exception as e:
                                    st.error(f"변환 실패: {str(e)}")
                    
                    sub_type = sub_type.split()[0]  # 괄호 앞부분만 추출
                
                # 선택된 타입과 실제 타입 불일치 경고
                if main_type == "numeric" and not pd.api.types.is_numeric_dtype(col_data):
                    st.warning("⚠️ 선택한 타입(숫자형)과 실제 데이터 타입이 다릅니다. 위에서 변환을 시도해보세요.")
                
                # 선택된 타입에 따른 통계 및 시각화
                st.divider()
                
                # 통계 분석
                st.subheader("📊 통계 분석")
                
                # 기본 정보는 항상 표시
                with st.expander("기본 정보", expanded=True):
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
                
                # 데이터 타입에 맞는 통계 옵션
                appropriate_stats = st.session_state.detector.get_appropriate_stats(main_type, sub_type)
                
                if appropriate_stats:
                    selected_stats = st.multiselect(
                        "추가 통계를 선택하세요:",
                        appropriate_stats,
                        help="데이터 타입에 적합한 통계 방법들입니다"
                    )
                    
                    if selected_stats and st.button("통계 계산", key="calc_stats"):
                        cols = st.columns(min(len(selected_stats), 3))
                        for idx, stat in enumerate(selected_stats):
                            with cols[idx % 3]:
                                with st.container():
                                    st.markdown(f"**{stat}**")
                                    try:
                                        result = st.session_state.detector.calculate_statistics(df, selected_column, stat)
                                        if isinstance(result, dict):
                                            for key, value in result.items():
                                                st.text(f"{key}: {value}")
                                        else:
                                            st.write(result)
                                    except Exception as e:
                                        st.error(f"⚠️ 오류: {str(e)}")
                
                # 시각화
                st.subheader("📉 시각화")
                
                # 데이터 타입에 맞는 시각화 옵션 (고유값 무관)
                appropriate_viz = st.session_state.detector.get_appropriate_visualizations(
                    main_type, sub_type
                )
                
                selected_viz = st.selectbox(
                    "시각화 방법을 선택하세요:",
                    ["선택하세요"] + appropriate_viz,
                    help="선택한 데이터 타입에 적합한 시각화 방법들입니다"
                )
                
                if selected_viz != "선택하세요":
                    # 시각화별 파라미터
                    params = {}
                    
                    if selected_viz in ["상위N막대그래프", "상위 N개 막대그래프"]:
                        params['top_n'] = st.slider("표시할 항목 수:", 5, 50, 20)
                    elif selected_viz == "히스토그램":
                        params['bins'] = st.slider("구간(bin) 수:", 10, 100, 30)
                    elif selected_viz == "파레토차트":
                        params['top_n'] = st.slider("표시할 항목 수:", 10, 30, 20)
                    
                    if st.button("시각화 생성", key="create_viz"):
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            with st.spinner("시각화 생성 중..."):
                                try:
                                    fig = st.session_state.detector.create_visualization(
                                        df, selected_column, selected_viz, params
                                    )
                                    st.pyplot(fig)
                                    plt.close()
                                except Exception as e:
                                    st.error(f"⚠️ 시각화 오류: {str(e)}")
                                    st.info("💡 다른 시각화 방법을 시도해보세요.")
    
    with tab3:
        st.header("데이터 미리보기")
        
        col1, col2 = st.columns(2)
        with col1:
            n_rows = st.slider("표시할 행 수:", 5, 100, 20)
        with col2:
            show_random = st.checkbox("랜덤 샘플링", value=False)
        
        if show_random:
            st.dataframe(df.sample(n=min(n_rows, len(df))))
        else:
            st.dataframe(df.head(n_rows))

else:
    st.info("👆 파일을 업로드하여 시작하세요.")