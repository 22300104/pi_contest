import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

def render_data_preview_tab():
    """데이터 미리보기 탭"""
    # 전처리된 데이터가 있으면 그것을 사용, 없으면 원본 사용
    df = st.session_state.get('df_processed', st.session_state.df)
    
    st.header("📥 미리보기 및 다운로드")
    
    # 전처리/비식별화 상태 표시
    if 'df_processed' in st.session_state:
        processing_info = []
        
        # 타입 변환된 컬럼
        converted_cols = [col for col in df.columns if st.session_state.get(f'converted_{col}', False)]
        if converted_cols:
            processing_info.append(f"타입 변환: {len(converted_cols)}개 컬럼")
        
        # 라운딩 처리 이력
        if 'processing_history' in st.session_state and st.session_state.processing_history:
            rounding_count = sum(1 for h in st.session_state.processing_history if h['type'] == '라운딩')
            if rounding_count > 0:
                processing_info.append(f"라운딩: {rounding_count}개 컬럼")
        
        if processing_info:
            st.success(f"✅ 처리된 데이터입니다. ({', '.join(processing_info)})")
    
    # 미리보기 옵션과 다운로드 버튼을 같은 행에 배치
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        n_rows = st.slider("표시할 행 수:", 5, 100, 20)
    
    with col2:
        selected_columns = st.multiselect(
            "표시할 열 선택:",
            df.columns.tolist(),
            default=[]
        )
    
    with col3:
        st.markdown("### 📥 다운로드")
        
        # 탭으로 구분
        csv_tab, excel_tab = st.tabs(["📄 CSV", "📊 Excel"])
        
        with csv_tab:
            st.caption("• 빠른 생성\n• 텍스트 형식\n• ⚠️ 데이터만 포함")
            
            # CSV는 데이터만 저장
            csv_data = df.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="CSV 다운로드",
                data=csv_data,
                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
            
            st.info("ℹ️ CSV는 데이터만 포함합니다. 처리 이력이 필요하면 Excel을 선택하세요.")
        
        with excel_tab:
            st.caption("• 여러 시트\n• 서식 지원\n• ✅ 처리 이력 포함")
            
            # 데이터 크기 확인
            if len(df) > 50000:
                st.warning(f"⚠️ {len(df):,}행은 시간이 걸립니다")
            
            if st.button("Excel 생성", use_container_width=True):
                with st.spinner("생성 중..."):
                    excel_data = create_excel_with_history(df)
                    
                    st.download_button(
                        label="Excel 다운로드",
                        data=excel_data,
                        file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="excel_dl"
                    )
            
            st.success("✅ Excel은 처리 이력과 타입 변환 정보를 별도 시트에 포함합니다.")
    
    # 데이터 표시
    display_df = df
    
    if selected_columns:
        display_df = display_df[selected_columns]
    
    # 순차적으로 표시 (head만 사용)
    display_df = display_df.head(n_rows)
    
    st.dataframe(display_df, use_container_width=True)
    
    # 데이터 정보
    with st.expander("📊 표시된 데이터 정보"):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("표시된 행 수", f"{len(display_df):,}")
            st.metric("표시된 열 수", f"{len(display_df.columns)}")
        with col_info2:
            st.metric("전체 행 수", f"{len(df):,}")
            st.metric("전체 열 수", f"{len(df.columns)}")
    
    # 원본 복구 섹션
    if 'df_original' in st.session_state:
        st.markdown("---")
        st.subheader("🔄 원본 데이터 복구")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 처리된 컬럼 목록 생성
            processed_columns = []
            
            # 타입 변환된 컬럼
            for col in df.columns:
                if st.session_state.get(f'converted_{col}', False):
                    processed_columns.append({
                        'name': col,
                        'type': '타입변환',
                        'current': str(df[col].dtype)
                    })
            
            # 라운딩 처리된 컬럼
            if 'processing_history' in st.session_state:
                for history in st.session_state.processing_history:
                    if history['type'] == '라운딩':
                        col_name = history['column']
                        # 중복 체크
                        existing = next((item for item in processed_columns if item['name'] == col_name), None)
                        if existing:
                            existing['type'] += ', 라운딩'
                        else:
                            processed_columns.append({
                                'name': col_name,
                                'type': '라운딩',
                                'current': history['details']
                            })
            
            if processed_columns:
                st.write("**처리된 컬럼 목록:**")
                for col_info in processed_columns:
                    col_display, btn_col = st.columns([4, 1])
                    with col_display:
                        st.text(f"• {col_info['name']}: {col_info['type']} ({col_info['current']})")
                    with btn_col:
                        if st.button("원본으로", key=f"restore_{col_info['name']}"):
                            # 경고 메시지 추가
                            st.warning(f"⚠️ '{col_info['name']}' 컬럼의 모든 변경사항(타입 변환, 비식별화 등)이 취소됩니다.")
                            
                            # 원본 데이터에서 해당 컬럼만 복구
                            if 'df_processed' in st.session_state:
                                st.session_state.df_processed[col_info['name']] = st.session_state.df_original[col_info['name']].copy()
                            else:
                                st.session_state.df[col_info['name']] = st.session_state.df_original[col_info['name']].copy()
                            
                            # 플래그 초기화
                            if f'converted_{col_info["name"]}' in st.session_state:
                                st.session_state[f'converted_{col_info["name"]}'] = False
                            
                            # 처리 이력에서 제거
                            if 'processing_history' in st.session_state:
                                st.session_state.processing_history = [
                                    h for h in st.session_state.processing_history 
                                    if h['column'] != col_info['name']
                                ]
                            
                            st.success(f"✅ '{col_info['name']}' 컬럼이 원본으로 복구되었습니다.")
                            st.rerun()
            else:
                st.info("처리된 컬럼이 없습니다.")
        
        with col2:
            st.markdown("###")  # 여백
            if st.button("🔄 전체 초기화", type="secondary", use_container_width=True):
                if st.session_state.get('confirm_reset', False):
                    # 실제 초기화 실행
                    st.session_state.df = st.session_state.df_original.copy()
                    if 'df_processed' in st.session_state:
                        st.session_state.df_processed = st.session_state.df_original.copy()
                    
                    # 모든 플래그 초기화
                    for key in list(st.session_state.keys()):
                        if key.startswith('converted_'):
                            del st.session_state[key]
                    
                    # 처리 이력 초기화
                    if 'processing_history' in st.session_state:
                        st.session_state.processing_history = []
                    
                    st.session_state.confirm_reset = False
                    st.success("✅ 모든 데이터가 원본으로 초기화되었습니다.")
                    st.rerun()
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ 모든 데이터 타입 변환과 비식별화 처리가 취소됩니다. 정말로 초기화하시려면 다시 한번 클릭하세요.")


def create_csv_with_history(df):
    """처리 이력이 포함된 CSV 파일 생성"""
    output = BytesIO()
    
    # 1. 메인 데이터만 먼저 쓰기 (헤더 없이)
    df.to_csv(output, index=False, encoding='utf-8-sig')
    
    # 2. 구분선과 처리 정보 추가
    output.write("\n\n".encode('utf-8-sig'))
    output.write("="*50 + "\n".encode('utf-8-sig'))
    output.write("처리 정보\n".encode('utf-8-sig'))
    output.write("="*50 + "\n".encode('utf-8-sig'))
    
    # 처리 일시
    output.write(f"\n처리 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n".encode('utf-8-sig'))
    
    # 처리 이력
    if 'processing_history' in st.session_state and st.session_state.processing_history:
        output.write("\n[처리 이력]\n".encode('utf-8-sig'))
        output.write("순번,처리 유형,대상 컬럼,처리 내용\n".encode('utf-8-sig'))
        
        for i, history in enumerate(st.session_state.processing_history, 1):
            line = f"{i},{history['type']},{history['column']},{history['details']}\n"
            output.write(line.encode('utf-8-sig'))
    
    # 타입 변환 정보
    converted_cols = [col for col in df.columns if st.session_state.get(f'converted_{col}', False)]
    if converted_cols:
        output.write("\n[타입 변환 정보]\n".encode('utf-8-sig'))
        output.write("컬럼명,원본 타입,변환 타입\n".encode('utf-8-sig'))
        
        for col in converted_cols:
            line = f"{col},object,{str(df[col].dtype)}\n"
            output.write(line.encode('utf-8-sig'))
    
    output.seek(0)
    return output.getvalue()


def create_excel_with_history(df):
    """처리 이력과 함께 엑셀 파일 생성"""
    output = BytesIO()
    
    # 대용량 데이터 처리를 위한 최적화
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 메인 데이터 시트
        df.to_excel(writer, sheet_name='데이터', index=False)
        
        # 처리 이력 시트 생성
        if 'processing_history' in st.session_state and st.session_state.processing_history:
            history_data = []
            for i, history in enumerate(st.session_state.processing_history, 1):
                history_data.append({
                    '순번': i,
                    '처리 유형': history['type'],
                    '대상 컬럼': history['column'],
                    '처리 내용': history['details']
                })
            
            history_df = pd.DataFrame(history_data)
            history_df.to_excel(writer, sheet_name='처리이력', index=False)
        
        # 타입 변환 정보 시트
        converted_cols = [col for col in df.columns if st.session_state.get(f'converted_{col}', False)]
        if converted_cols:
            conversion_data = []
            for col in converted_cols:
                conversion_data.append({
                    '컬럼명': col,
                    '원본 타입': 'object',
                    '변환 타입': str(df[col].dtype),
                    '상태': '✅ 완료'
                })
            
            conversion_df = pd.DataFrame(conversion_data)
            conversion_df.to_excel(writer, sheet_name='타입변환', index=False)
    
    output.seek(0)
    return output.getvalue()