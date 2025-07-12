import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.ui_components import data_type_guide, data_quality_check

def render_column_analysis_tab():
    """속성별 분석 탭"""
    df = st.session_state.df
    
    st.header("속성별 분석")
    
    # 데이터 타입 가이드
    data_type_guide()
    
    # 선택된 컬럼을 세션 상태에 저장
    if 'selected_column_for_analysis' not in st.session_state:
        st.session_state.selected_column_for_analysis = df.columns.tolist()[0]
    
    # 속성 선택 - 세션 상태 사용
    selected_column = st.selectbox(
        "분석할 속성 선택:", 
        df.columns.tolist(),
        index=df.columns.tolist().index(st.session_state.selected_column_for_analysis),
        key="column_analysis_selectbox",
        on_change=lambda: setattr(st.session_state, 'selected_column_for_analysis', st.session_state.column_analysis_selectbox)
    )
    
    # 선택된 컬럼 업데이트
    st.session_state.selected_column_for_analysis = selected_column
    
    if selected_column:
        if 'df_processed' in st.session_state:
    # df_processed에 해당 컬럼이 있고, 원본과 다르면 처리된 것으로 간주
            if selected_column in st.session_state.df_processed.columns:
                # 원본과 비교해서 하나라도 다르면 처리된 것
                if not st.session_state.df[selected_column].equals(st.session_state.df_processed[selected_column]):
                    df_analysis = st.session_state.df_processed
                    col_data = df_analysis[selected_column]
                    st.info(f"📊 처리된 데이터를 사용 중입니다. (타입: {col_data.dtype})")
                else:
                    df_analysis = df
                    col_data = df[selected_column]
            else:
                df_analysis = df
                col_data = df[selected_column]
        else:
            df_analysis = df
            col_data = df[selected_column]
        

        # 데이터 품질 검사
        data_quality_check(col_data)
        
        # 기본 정보
        render_basic_info(col_data)
        
        # 데이터 타입 선택
        main_type, sub_type = render_data_type_selection(col_data, selected_column)
        
        # 통계 분석 섹션
        render_statistics_section(df_analysis, selected_column, main_type, sub_type)
        
        # 시각화 섹션
        render_visualization_section(df_analysis, selected_column, main_type, sub_type)
        
        # 다중 변수 분석 섹션
        render_multivariate_analysis_section(df_analysis)

def render_basic_info(col_data):
    """기본 정보 표시"""
    st.subheader("📋 기본 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("전체 행", f"{len(col_data):,}")
    with col2:
        st.metric("고유값", f"{col_data.nunique():,}")
    with col3:
        st.metric("결측값", f"{col_data.isnull().sum():,}")
    with col4:
        st.metric("타입", str(col_data.dtype))

def render_data_type_selection(col_data, selected_column):
    """데이터 타입 선택"""
    st.subheader("🏷️ 데이터 타입 선택")
    
    # 각 컬럼별로 선택된 타입을 세션 상태에 저장
    main_type_key = f"main_type_{selected_column}"
    sub_type_key = f"sub_type_{selected_column}"
    
    # 기본값 설정
    if main_type_key not in st.session_state:
        st.session_state[main_type_key] = "numeric"
    if sub_type_key not in st.session_state:
        st.session_state[sub_type_key] = "continuous"
    
    col1, col2 = st.columns(2)
    
    with col1:
        main_type = st.radio(
            "주 데이터 타입:",
            ["numeric", "categorical", "text", "datetime"],
            format_func=lambda x: {
                "numeric": "🔢 숫자형",
                "categorical": "📝 범주형",
                "text": "💬 텍스트",
                "datetime": "📅 날짜/시간"
            }[x],
            key=f"main_data_type_radio_{selected_column}",
            index=["numeric", "categorical", "text", "datetime"].index(st.session_state[main_type_key])
        )
        st.session_state[main_type_key] = main_type
    
    with col2:
        if main_type == "numeric":
            sub_type = st.radio(
                "세부 타입:",
                ["continuous", "discrete", "binary"],
                format_func=lambda x: {
                    "continuous": "〰️ 연속형",
                    "discrete": "🔢 이산형",
                    "binary": "☯️ 이진형"
                }[x],
                key=f"numeric_subtype_radio_{selected_column}"
            )
        elif main_type == "categorical":
            sub_type = st.radio(
                "세부 타입:",
                ["nominal", "ordinal", "binary"],
                format_func=lambda x: {
                    "nominal": "🏷️ 명목형",
                    "ordinal": "📊 순서형",
                    "binary": "☯️ 이진형"
                }[x],
                key=f"categorical_subtype_radio_{selected_column}"
            )
        elif main_type == "text":
            sub_type = st.radio(
                "세부 타입:",
                ["short", "long"],
                format_func=lambda x: {
                    "short": "📝 짧은 텍스트",
                    "long": "📄 긴 텍스트"
                }[x],
                key=f"text_subtype_radio_{selected_column}"
            )
        else:
            sub_type = "datetime"
        
        st.session_state[sub_type_key] = sub_type
    
    return main_type, sub_type

def render_statistics_section(df, selected_column, main_type, sub_type):
    """통계 분석 섹션"""
    st.divider()
    st.subheader("📊 통계 분석")
    
    appropriate_stats = st.session_state.detector.get_appropriate_stats(main_type, sub_type)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_stats = st.multiselect(
            "통계 방법 선택:",
            appropriate_stats,
            default=[],
            key=f"stats_multiselect_{selected_column}"
        )
        
        if st.button("📊 통계 실행", key=f"run_stats_{selected_column}"):
            st.session_state.show_stats[selected_column] = True
    
    with col2:
        if st.session_state.show_stats.get(selected_column, False) and selected_stats:
            for stat_type in selected_stats:
                with st.expander(f"📈 {stat_type}", expanded=True):
                    try:
                        result = st.session_state.detector.calculate_statistics(
                            df, selected_column, stat_type
                        )
                        
                        if isinstance(result, dict):
                            for key, value in result.items():
                                st.text(f"{key}: {value}")
                        else:
                            st.write(result)
                    except Exception as e:
                        st.error(f"계산 오류: {str(e)}")

def render_visualization_section(df, selected_column, main_type, sub_type):
    """시각화 섹션"""
    st.divider()
    st.subheader("📉 시각화")
    
    appropriate_viz = st.session_state.detector.get_appropriate_visualizations(main_type, sub_type)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_viz = st.selectbox(
            "시각화 방법:",
            appropriate_viz,
            key=f"viz_select_{selected_column}"
        )
        
        # 시각화 파라미터
        params = {}
        
        if "상위" in selected_viz or "파레토" in selected_viz:
            params['top_n'] = st.slider("표시할 항목 수:", 5, 50, 20, key=f"viz_top_n_slider_{selected_column}")
        elif selected_viz == "히스토그램":
            params['bins'] = st.slider("구간 수:", 10, 100, 30, key=f"viz_bins_slider_{selected_column}")
        elif selected_viz == "산점도":
            # 숫자형 열 찾기 (실제 숫자형 + 변환된 숫자형)
            numeric_cols = []
            
            # 원본 데이터프레임의 숫자형 열
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                elif 'df_processed' in st.session_state and col in st.session_state.df_processed.columns:
                    if pd.api.types.is_numeric_dtype(st.session_state.df_processed[col]):
                        numeric_cols.append(col)
            
            # 중복 제거 및 현재 열 제외
            numeric_cols = list(set(numeric_cols))
            if selected_column in numeric_cols:
                numeric_cols.remove(selected_column)
            
            if numeric_cols:
                params['other_column'] = st.selectbox(
                    "Y축 변수:", 
                    numeric_cols,
                    help="숫자형 또는 숫자로 변환된 열만 표시됩니다",
                    key=f"scatter_y_axis_select_{selected_column}"
                )
                params['show_regression'] = st.checkbox("회귀선 표시", key=f"show_regression_checkbox_{selected_column}")
            else:
                st.warning("다른 숫자형 열이 없습니다. 먼저 데이터를 숫자형으로 변환해주세요.")
        
        if st.button("📉 시각화 실행", key=f"run_viz_{selected_column}"):
            st.session_state.show_viz[selected_column] = True
            st.session_state.viz_params = params
    
    with col2:
        if st.session_state.show_viz.get(selected_column, False):
            with st.spinner("시각화 생성 중..."):
                try:
                    params = st.session_state.get('viz_params', {})
                    fig = st.session_state.detector.create_visualization(
                        df, selected_column, selected_viz, params
                    )
                    st.pyplot(fig, use_container_width=False)
                    plt.close()
                except Exception as e:
                    st.error(f"시각화 오류: {str(e)}")

def render_multivariate_analysis_section(df):
    """다중 변수 분석 섹션"""
    st.divider()
    st.subheader("🔍 다중 변수 분석")
    
    with st.expander("📊 여러 변수 분포 비교", expanded=False):
        # 숫자형 열 찾기
        numeric_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect(
                "비교할 변수들을 선택하세요 (2개 이상):",
                numeric_cols,
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
                key="multivar_column_select"
            )
            
            if len(selected_cols) >= 2:
                viz_type = st.radio(
                    "시각화 방법:",
                    ["박스플롯 비교", "바이올린플롯 비교", "상관관계 히트맵", "산점도 매트릭스"],
                    horizontal=True,
                    key="multivar_viz_type_radio"
                )
                
                if st.button("📊 다중 변수 시각화 실행", key="multivar_viz"):
                    with st.spinner("시각화 생성 중..."):
                        try:
                            fig = create_multivariate_visualization(df, selected_cols, viz_type)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.error(f"시각화 오류: {str(e)}")
        else:
            st.warning("숫자형 변수가 2개 이상 필요합니다. 데이터 전처리를 통해 숫자형으로 변환해주세요.")

def create_multivariate_visualization(df, columns, viz_type):
    """다중 변수 시각화 생성"""
    
    if viz_type == "박스플롯 비교":
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 데이터 정규화 (스케일이 다른 경우를 위해)
        normalized_data = []
        for col in columns:
            data = df[col].dropna()
            if len(data) > 0:
                normalized = (data - data.mean()) / data.std() if data.std() > 0 else data
                normalized_data.append(normalized)
        
        if normalized_data:
            bp = ax.boxplot(normalized_data, labels=columns, patch_artist=True)
            
            # 박스 색상 설정
            colors = plt.cm.Set3(range(len(normalized_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title("변수별 분포 비교 (정규화됨)")
            ax.set_ylabel("정규화된 값")
            plt.xticks(rotation=45, ha='right')
        
    elif viz_type == "바이올린플롯 비교":
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 데이터를 long format으로 변환
        plot_data = []
        for col in columns:
            data = df[col].dropna()
            if len(data) > 0:
                normalized = (data - data.mean()) / data.std() if data.std() > 0 else data
                for val in normalized:
                    plot_data.append({'Variable': col, 'Value': val})
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.violinplot(data=plot_df, x='Variable', y='Value', ax=ax)
            ax.set_title("변수별 분포 비교 (바이올린플롯)")
            plt.xticks(rotation=45, ha='right')
        
    elif viz_type == "상관관계 히트맵":
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 상관관계 계산
        correlation_matrix = df[columns].corr()
        
        # 히트맵 그리기
        mask = None
        if len(columns) > 10:  # 변수가 많으면 상삼각 행렬만 표시
            mask = [[False if i >= j else True for i in range(len(columns))] for j in range(len(columns))]
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, ax=ax, fmt='.3f', mask=mask,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("변수 간 상관관계 히트맵")
        
    elif viz_type == "산점도 매트릭스":
        # 변수가 많으면 경고
        if len(columns) > 5:
            st.warning("변수가 많아 처음 5개만 표시합니다.")
            columns = columns[:5]
        
        # seaborn의 pairplot 사용
        g = sns.pairplot(df[columns].dropna(), 
                        diag_kind='hist',
                        plot_kws={'alpha': 0.6, 's': 30},
                        diag_kws={'bins': 20})
        g.fig.suptitle("산점도 매트릭스", y=1.02)
        fig = g.fig
    
    plt.tight_layout()
    return fig if 'fig' in locals() else plt.gcf()