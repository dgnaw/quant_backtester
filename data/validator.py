from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .schema import REQUIRED_OHLC_COLUMNS
from .validation_report import ValidationReport


@dataclass
class DataQualityPolicy:
    allow_zero_volume: bool = True
    allow_missing_volume: bool = True
    require_positive_ohlc: bool = True
    reject_duplicates: bool = True
    reject_unsorted: bool = True


class DataValidator:
    """
    Kiểm tra tính toàn vẹn của dữ liệu OHLCV theo DataQualityPolicy.
    Tuyệt đối KHÔNG sửa đổi (mutate) DataFrame đầu vào.
    """
    def __init__(self, policy: DataQualityPolicy | None = None):
        self.policy = policy if policy is not None else DataQualityPolicy()

    def validate(self, data: pd.DataFrame) -> ValidationReport:
        report = ValidationReport(is_valid=True, total_rows=len(data), invalid_rows=0)
        invalid_row_set: set[int] = set()

        # A. Schema Validation
        col_map = {str(c).strip().lower(): c for c in data.columns}
        missing = [c for c in REQUIRED_OHLC_COLUMNS if c not in col_map]
        if missing:
            report.add_error("MISSING_COLUMNS", f"Thiếu các cột bắt buộc: {missing}")
            return report

        if len(data) == 0:
            report.add_error("EMPTY_DATA", "Dữ liệu rỗng (0 dòng).")
            return report

        # Làm việc trên bản sao chỉ để ép kiểu kiểm tra, không chạm vào data gốc
        time_col = data[col_map["time"]]
        parsed_time = pd.to_datetime(time_col, errors="coerce")

        # D. Timestamp Validation
        bad_time_mask = parsed_time.isna()
        if bad_time_mask.any():
            idxs = data.index[bad_time_mask].tolist()
            invalid_row_set.update(idxs)
            report.add_error("INVALID_TIMESTAMP", f"Có {len(idxs)} timestamp không hợp lệ.", row_indices=idxs)

        valid_times = parsed_time.dropna()
        if not valid_times.is_monotonic_increasing:
            if self.policy.reject_unsorted:
                report.add_error("UNSORTED_TIMESTAMP", "Dữ liệu chưa được sắp xếp tăng dần theo thời gian.")
            else:
                report.add_warning("UNSORTED_TIMESTAMP", "Dữ liệu chưa được sắp xếp tăng dần theo thời gian.")

        dup_mask = parsed_time.duplicated(keep=False) & ~bad_time_mask
        if dup_mask.any():
            idxs = data.index[dup_mask].tolist()
            invalid_row_set.update(idxs)
            if self.policy.reject_duplicates:
                report.add_error("DUPLICATE_TIMESTAMP", f"Phát hiện {len(idxs)} dòng có timestamp bị trùng.", row_indices=idxs)
            else:
                report.add_warning("DUPLICATE_TIMESTAMP", f"Phát hiện {len(idxs)} dòng có timestamp bị trùng.", row_indices=idxs)

        # B. Numeric Validation cho OHLC
        ohlc_cols = ["open", "high", "low", "close"]
        numeric_df = pd.DataFrame(index=data.index)
        for c in ohlc_cols:
            numeric_df[c] = pd.to_numeric(data[col_map[c]], errors="coerce")

        nan_mask = numeric_df.isna().any(axis=1)
        if nan_mask.any():
            idxs = data.index[nan_mask].tolist()
            invalid_row_set.update(idxs)
            report.add_error("NAN_OHLC", f"Có {len(idxs)} dòng chứa NaN hoặc kiểu dữ liệu không phải số trong OHLC.", row_indices=idxs)

        inf_mask = np.isinf(numeric_df.to_numpy(dtype=float, na_value=0.0)).any(axis=1)
        if inf_mask.any():
            idxs = data.index[inf_mask].tolist()
            invalid_row_set.update(idxs)
            report.add_error("INF_OHLC", f"Có {len(idxs)} dòng chứa giá trị vô cực (Inf/-Inf) trong OHLC.", row_indices=idxs)

        if self.policy.require_positive_ohlc:
            non_pos_mask = (numeric_df <= 0).any(axis=1) & ~nan_mask
            if non_pos_mask.any():
                idxs = data.index[non_pos_mask].tolist()
                invalid_row_set.update(idxs)
                report.add_error("NON_POSITIVE_OHLC", f"Có {len(idxs)} dòng có giá OHLC <= 0.", row_indices=idxs)

        # C. OHLC Consistency Validation: High >= max(Open, Close), Low <= min(Open, Close), High >= Low
        clean_mask = ~(nan_mask | inf_mask)
        if clean_mask.any():
            sub = numeric_df.loc[clean_mask]
            inconsistent = (
                (sub["high"] < sub[["open", "close", "low"]].max(axis=1))
                | (sub["low"] > sub[["open", "close", "high"]].min(axis=1))
                | (sub["high"] < sub["low"])
            )
            if inconsistent.any():
                idxs = sub.index[inconsistent].tolist()
                invalid_row_set.update(idxs)
                report.add_error(
                    "INCONSISTENT_OHLC",
                    f"Có {len(idxs)} nến vi phạm bất đẳng thức High >= max(Open, Close) hoặc Low <= min(Open, Close).",
                    row_indices=idxs,
                )

        # Volume Validation
        vol_col_name = next((col_map[v] for v in ("volume", "real_volume", "tick_volume") if v in col_map), None)
        if vol_col_name is None:
            if self.policy.allow_missing_volume:
                report.add_info("MISSING_VOLUME_COLUMN", "Không có cột volume; sử dụng mặc định khi chuẩn hóa.")
            else:
                report.add_error("MISSING_VOLUME_COLUMN", "Thiếu cột volume bắt buộc theo chính sách.")
        else:
            vol_series = pd.to_numeric(data[vol_col_name], errors="coerce")
            neg_vol = vol_series < 0
            if neg_vol.any():
                idxs = data.index[neg_vol].tolist()
                invalid_row_set.update(idxs)
                report.add_error("NEGATIVE_VOLUME", f"Có {len(idxs)} dòng có volume âm.", row_indices=idxs)

            zero_vol = vol_series == 0
            if zero_vol.any():
                idxs = data.index[zero_vol].tolist()
                if self.policy.allow_zero_volume:
                    report.add_warning("ZERO_VOLUME", f"Có {len(idxs)} dòng có volume = 0.", row_indices=idxs)
                else:
                    invalid_row_set.update(idxs)
                    report.add_error("ZERO_VOLUME", f"Có {len(idxs)} dòng có volume = 0 (bị từ chối theo policy).", row_indices=idxs)

        report.invalid_rows = len(invalid_row_set)
        report.add_info("VALIDATION_SUMMARY", f"Đã kiểm tra {len(data)} nến, phát hiện {report.invalid_rows} nến lỗi.")
        return report


def validate_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    """Hàm tương thích ngược (backward-compatible) với phiên bản cũ."""
    missing = set(REQUIRED_OHLC_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Thiếu các cột: {sorted(missing)}")
    df = data.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Có timestamp không hợp lệ.")
    df = df.sort_values("time").reset_index(drop=True)
    if df["time"].duplicated().any():
        raise ValueError("Timestamp bị trùng.")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("OHLC chứa giá trị thiếu hoặc không phải số.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Giá OHLC phải lớn hơn 0.")
    bad = (df.high < df[["open", "low", "close"]].max(axis=1)) | (df.low > df[["open", "high", "close"]].min(axis=1))
    if bad.any():
        raise ValueError(f"Có {int(bad.sum())} nến OHLC không hợp lệ.")
    return df

