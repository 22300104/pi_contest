import pandas as pd
import numpy as np
from typing import Literal, Tuple, Dict, Optional


class RoundingProcessor:
    """
    숫자 컬럼 라운딩 + 이상치(클리핑) 처리 전담 클래스
    ──────────────────────────────────────────────
    1. 시그마(평균±kσ) / IQR(Q1±k·IQR) 임계값 산출
    2. (선택) 클리핑 → 라운딩 처리
    3. 미리보기 / 통계 계산 헬퍼
    """

    # ───────────────────────────────────
    # A. 통계 + 임계값 계산
    # ───────────────────────────────────
    @staticmethod
    def get_statistics(
        series: pd.Series,
        sigma_k: float = 3.0,
        iqr_k: float = 1.5,
        *,                           # ← 키워드 전용 인자 구역
        positive_only: bool = False  # ← ① 새 옵션
    ) -> Dict[str, Tuple[float, float]]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            raise ValueError("유효한 숫자 데이터가 없습니다.")

        mean, std = clean.mean(), clean.std(ddof=0)
        q1, q3   = clean.quantile(0.25), clean.quantile(0.75)
        iqr      = q3 - q1

        sigma_low, sigma_high = mean - sigma_k*std, mean + sigma_k*std
        iqr_low,   iqr_high   = q1   - iqr_k*iqr,  q3   + iqr_k*iqr

        # ② 전부 양수 데이터일 때 음수 하한 제거
        if positive_only:
            sigma_low = max(0, sigma_low)
            iqr_low   = max(0, iqr_low)

        return {
            "sigma_range": (sigma_low, sigma_high),
            "iqr_range":   (iqr_low,   iqr_high),
            "mean": mean,
            "std": std,
            "q1": q1,
            "q3": q3,
        }


    # ───────────────────────────────────
    # B. 실제 라운딩 + 이상치 컷
    # ───────────────────────────────────
    @staticmethod
    def round_column(
        df: pd.DataFrame,
        column_name: str,
        rounding_type: Literal["floor", "ceil", "round"],
        decimal_places: Optional[int] = None,
        integer_place: Optional[int] = None,
        *,
        outlier_bounds: Optional[Tuple[float, float]] = None,
    ) -> pd.Series:
        if column_name not in df.columns:
            raise ValueError(f"컬럼 '{column_name}'을 찾을 수 없습니다.")

        col = pd.to_numeric(df[column_name], errors="coerce")

        # ① 클리핑
        if outlier_bounds is not None:
            low, high = outlier_bounds
            col = col.clip(lower=low, upper=high)

        # ② 라운딩
        if decimal_places is not None:
            scale = 10 ** decimal_places
            if rounding_type == "floor":
                col = np.floor(col * scale) / scale
            elif rounding_type == "ceil":
                col = np.ceil(col * scale) / scale
            else:
                col = np.round(col, decimal_places)

        elif integer_place is not None:
            if rounding_type == "floor":
                col = np.floor(col / integer_place) * integer_place
            elif rounding_type == "ceil":
                col = np.ceil(col / integer_place) * integer_place
            else:
                col = np.round(col / integer_place) * integer_place
        else:
            raise ValueError("decimal_places 또는 integer_place 중 하나는 지정해야 합니다.")

        return col

    # ───────────────────────────────────
    # C. 미리보기
    # ───────────────────────────────────
    @staticmethod
    def get_preview(
        df: pd.DataFrame,
        column_name: str,
        rounding_type: Literal["floor", "ceil", "round"],
        decimal_places: Optional[int] = None,
        integer_place: Optional[int] = None,
        *,
        outlier_bounds: Optional[Tuple[float, float]] = None,
        sample_size: int = 5,
    ) -> pd.DataFrame:
        valid = df[df[column_name].notna()][column_name]
        if valid.empty:
            return pd.DataFrame({"원본값": [], "결과값": []})

        idx = valid.sample(min(sample_size, len(valid))).index
        original = df.loc[idx, column_name]

        rounded = RoundingProcessor.round_column(
            df.loc[idx],
            column_name,
            rounding_type,
            decimal_places,
            integer_place,
            outlier_bounds=outlier_bounds,
        )

        return pd.DataFrame({"원본값": original.values, "결과값": rounded.values})
