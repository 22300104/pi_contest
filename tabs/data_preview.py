import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

__all__ = [
    "render_data_preview_tab",
]

# -----------------------------------------------------------------------------
# 📦  Helper functions  ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _ensure_original_copy(df: pd.DataFrame):
    """Ensure that the original dataframe is stored exactly once."""
    if "df_original" not in st.session_state:
        st.session_state.df_original = df.copy()


def _collect_processed_columns(df: pd.DataFrame):
    """Gather every column that has been processed (타입변환 / 비식별화 등)."""
    processed = []

    # 1️⃣  타입 변환 플래그 -----------------------------------------------------
    for col in df.columns:
        if st.session_state.get(f"converted_{col}"):
            processed.append({
                "name": col,
                "type": "타입변환",
                "current": str(df[col].dtype),
            })

    # 2️⃣  처리 이력 -----------------------------------------------------------
    for h in st.session_state.get("processing_history", []):
        col_name = h["column"]
        col_type = h["type"]  # 라운딩 · 마스킹 · 부분 삭제 · 치환 …
        details  = h["details"]

        exist = next((p for p in processed if p["name"] == col_name), None)
        if exist:
            if col_type not in exist["type"]:
                exist["type"] += f", {col_type}"
        else:
            processed.append({
                "name": col_name,
                "type": col_type,
                "current": details,
            })

    return processed


# -----------------------------------------------------------------------------
# 🚀  High‑speed Excel export with progress bar  --------------------------------
# -----------------------------------------------------------------------------

def _create_excel_with_history(df: pd.DataFrame):
    """Create an XLSX file with separate sheets for data, history and conversion."""

    # 진행 단계 계산
    steps = 1  # 데이터 시트
    if st.session_state.get("processing_history"):
        steps += 1
    if any(st.session_state.get(f"converted_{c}") for c in df.columns):
        steps += 1

    progress = st.progress(0.0)
    done = 0

    out = BytesIO()
    # pandas<2.1 호환: options→engine_kwargs
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        # 1️⃣  데이터 ----------------------------------------------------------------
        df.to_excel(writer, sheet_name="데이터", index=False)
        done += 1
        progress.progress(done / steps)

        # 2️⃣  처리 이력 ------------------------------------------------------------
        if st.session_state.get("processing_history"):
            hist_df = pd.DataFrame([
                {
                    "순번": i + 1,
                    "처리 유형": h["type"],
                    "대상 컬럼": h["column"],
                    "처리 내용": h["details"],
                }
                for i, h in enumerate(st.session_state.processing_history)
            ])
            hist_df.to_excel(writer, sheet_name="처리이력", index=False)
            done += 1
            progress.progress(done / steps)

        # 3️⃣  타입 변환 정보 ------------------------------------------------------
        conv = [c for c in df.columns if st.session_state.get(f"converted_{c}")]
        if conv:
            conv_df = pd.DataFrame([
                {
                    "컬럼명": c,
                    "원본 타입": "object",
                    "변환 타입": str(df[c].dtype),
                    "상태": "✅ 완료",
                }
                for c in conv
            ])
            conv_df.to_excel(writer, sheet_name="타입변환", index=False)
            done += 1
            progress.progress(done / steps)

    progress.empty()
    out.seek(0)
    return out.getvalue()


# -----------------------------------------------------------------------------
# 🖥️  Main render function  ----------------------------------------------------
# -----------------------------------------------------------------------------

def render_data_preview_tab():
    """Render the Preview / Export tab."""

    if "df" not in st.session_state:
        st.warning("먼저 데이터를 업로드해주세요.")
        return

    # Ensure original copy is stored
    _ensure_original_copy(st.session_state.df)

    # Use processed dataframe if available
    df = st.session_state.get("df_processed", st.session_state.df)

    st.header("📥 미리보기 및 다운로드")

    # --- 상태 배너 -----------------------------------------------------------
    banners = []
    if any(st.session_state.get(f"converted_{c}") for c in df.columns):
        banners.append("타입 변환 완료")
    if st.session_state.get("processing_history"):
        banners.append(f"비식별/라운딩 등 {len(st.session_state.processing_history)}건")
    if banners:
        st.success("✅ " + " · ".join(banners))

    # --- 미리보기 옵션 -------------------------------------------------------
    col_rows, col_cols, col_dl = st.columns([1, 2, 2])

    with col_rows:
        n_rows = st.slider("표시할 행 수", 5, 100, 20)

    with col_cols:
        selected_cols = st.multiselect("표시할 열", df.columns.tolist(), default=[])

    with col_dl:
        st.markdown("### 📥 다운로드")
        tab_csv, tab_xlsx = st.tabs(["📄 CSV", "📊 Excel"])

        with tab_csv:
            st.caption("• 빠름 • 텍스트 형식 • 데이터만 포함")
            data_csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSV 다운로드",
                data_csv,
                file_name=f"data_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

        with tab_xlsx:
            st.caption("• 여러 시트 • 처리 이력 포함")
            if len(df) > 50_000:
                st.warning(f"⚠️ {len(df):,}행은 생성에 시간이 걸립니다")
            if st.button("Excel 생성", use_container_width=True):
                xlsx_bytes = _create_excel_with_history(df)
                st.download_button(
                    "Excel 다운로드",
                    xlsx_bytes,
                    file_name=f"data_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="excel_dl",
                )

    # --- 데이터 표시 ---------------------------------------------------------
    disp = df[selected_cols] if selected_cols else df
    st.dataframe(disp.head(n_rows), use_container_width=True)

    with st.expander("📊 표시된 데이터 정보"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("표시된 행 수", f"{len(disp):,}")
            st.metric("표시된 열 수", f"{len(disp.columns):,}")
        with col_b:
            st.metric("전체 행 수", f"{len(df):,}")
            st.metric("전체 열 수", f"{len(df.columns):,}")

    # --- 원본 복구 -----------------------------------------------------------
    if "df_original" in st.session_state:
        st.markdown("---")
        st.subheader("🔄 원본 데이터 복구")

        processed_cols = _collect_processed_columns(df)
        col_list, col_reset = st.columns([3, 1])

        # 개별 컬럼 복구 -------------------------------------------------------
        with col_list:
            if processed_cols:
                st.write("**처리된 컬럼 목록:**")
                for info in processed_cols:
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.text(f"• {info['name']}: {info['type']} ({info['current']})")
                    with c2:
                        if st.button("원본으로", key=f"restore_{info['name']}"):
                            # 컬럼 복원
                            st.session_state.df_processed[info['name']] = st.session_state.df_original[info['name']].copy()
                            st.session_state.pop(f"converted_{info['name']}", None)
                            # 관련 처리 이력 제거
                            st.session_state.processing_history = [h for h in st.session_state.get("processing_history", []) if h["column"] != info['name']]
                            st.success(f"✅ '{info['name']}' 컬럼이 복구되었습니다.")
                            st.rerun()
            else:
                st.info("처리된 컬럼이 없습니다.")

        # 전체 초기화 ----------------------------------------------------------
        with col_reset:
            st.markdown("###  ")  # vertical align
            if st.button("🔄 전체 초기화", type="secondary", use_container_width=True):
                if st.session_state.get("confirm_reset"):
                    # 데이터프레임 복원
                    st.session_state.df_processed = st.session_state.df_original.copy()
                    # 모든 변환 플래그 제거
                    for k in list(st.session_state.keys()):
                        if k.startswith("converted_"):
                            del st.session_state[k]
                    # 이력 초기화
                    st.session_state.processing_history = []
                    st.session_state.confirm_reset = False
                    st.success("✅ 모든 데이터가 원본으로 초기화되었습니다.")
                    st.rerun()
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ 정말 초기화할까요? 다시 클릭하면 실행됩니다.")
