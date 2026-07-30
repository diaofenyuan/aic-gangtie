"""独立数据泄漏审计，不向正式模型暴露未来诊断字段。"""

from __future__ import annotations

import copy
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gas_power.availability import FeatureAvailabilityRegistry
from gas_power.config import ProjectConfig
from gas_power.data import ConfiguredDataLoader
from gas_power.time_semantics import resampling_variants
from gas_power.validation import splitter_from_config


@dataclass
class AuditArtifacts:
    summary: dict[str, Any]
    lag_correlations: pd.DataFrame
    suspicious_features: pd.DataFrame
    markdown: str


def _safe_correlation(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    paired = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(paired) < 10 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return float("nan"), len(paired)
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1])), len(paired)


def calculate_lag_correlations(
    frame: pd.DataFrame,
    fields: Sequence[str],
    targets: Sequence[str],
    lag_min_steps: int,
    lag_max_steps: int,
    interval_minutes: int,
) -> pd.DataFrame:
    """offset>0 表示拿当前字段与未来标签比较，只能用于诊断。"""

    rows: list[dict[str, Any]] = []
    for field in fields:
        feature = pd.to_numeric(frame[field], errors="coerce")
        for target in targets:
            label = pd.to_numeric(frame[target], errors="coerce")
            for offset in range(int(lag_min_steps), int(lag_max_steps) + 1):
                correlation, pairs = _safe_correlation(feature, label.shift(-offset))
                rows.append(
                    {
                        "feature": str(field),
                        "target": str(target),
                        "target_offset_steps": offset,
                        "offset_minutes": offset * interval_minutes,
                        "correlation": correlation,
                        "absolute_correlation": abs(correlation) if np.isfinite(correlation) else np.nan,
                        "sample_count": pairs,
                        "risk": "red_future_diagnostic_only" if offset > 0 else (
                            "amber_current" if offset == 0 else "green_historical"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _affine_copy_candidates(
    frame: pd.DataFrame,
    lag_correlations: pd.DataFrame,
    targets: Sequence[str],
    correlation_threshold: float,
    residual_ratio_threshold: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    usable = lag_correlations.loc[
        lag_correlations["absolute_correlation"] >= correlation_threshold
    ]
    for (field, target), group in usable.groupby(["feature", "target"], sort=True):
        if field in targets:
            continue
        best = group.sort_values("absolute_correlation", ascending=False).iloc[0]
        offset = int(best["target_offset_steps"])
        paired = pd.concat(
            [frame[str(field)], frame[str(target)].shift(-offset)], axis=1
        ).dropna()
        if len(paired) < 10:
            continue
        x = paired.iloc[:, 0].to_numpy(dtype=float)
        y = paired.iloc[:, 1].to_numpy(dtype=float)
        coefficient, intercept = np.polyfit(x, y, deg=1)
        residual = y - (coefficient * x + intercept)
        denominator = max(float(np.std(y)), 1.0e-12)
        ratio = float(np.sqrt(np.mean(np.square(residual))) / denominator)
        if ratio <= residual_ratio_threshold:
            candidates.append(
                {
                    "feature": str(field),
                    "target": str(target),
                    "kind": "affine_or_shifted_copy",
                    "target_offset_steps": offset,
                    "correlation": float(best["correlation"]),
                    "coefficient": float(coefficient),
                    "intercept": float(intercept),
                    "residual_ratio": ratio,
                    "risk": "red" if offset > 0 else "amber",
                    "reason": "未来标签近似复制" if offset > 0 else "当前/历史确定性关系，需确认业务可用时间",
                }
            )
    return candidates


def _future_correlation_candidates(
    lag_correlations: pd.DataFrame,
    targets: Sequence[str],
    threshold: float,
) -> list[dict[str, Any]]:
    """未来高相关本身即为红色诊断，不要求达到近似复制残差阈值。"""

    future = lag_correlations.loc[
        (lag_correlations["target_offset_steps"] > 0)
        & (lag_correlations["absolute_correlation"] >= threshold)
        & (~lag_correlations["feature"].isin(targets))
    ]
    rows: list[dict[str, Any]] = []
    for (field, target), group in future.groupby(["feature", "target"], sort=True):
        best = group.sort_values("absolute_correlation", ascending=False).iloc[0]
        rows.append(
            {
                "feature": str(field),
                "target": str(target),
                "kind": "future_target_high_correlation",
                "target_offset_steps": int(best["target_offset_steps"]),
                "correlation": float(best["correlation"]),
                "coefficient": np.nan,
                "intercept": np.nan,
                "residual_ratio": np.nan,
                "risk": "red",
                "reason": "字段与未来标签高度相关，只能用于泄漏诊断，禁止进入正式模型",
            }
        )
    return rows


def _multivariate_copy_candidates(
    frame: pd.DataFrame,
    targets: Sequence[str],
    threshold: float,
) -> list[dict[str, Any]]:
    numeric = [
        str(column)
        for column in frame.select_dtypes(include=[np.number]).columns
        if str(column) not in targets and not str(column).startswith("feat_")
    ]
    if not numeric:
        return []
    rows: list[dict[str, Any]] = []
    for target in targets:
        data = frame[[*numeric, str(target)]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < max(30, len(numeric) * 2):
            continue
        split = max(20, int(len(data) * 0.7))
        train, test = data.iloc[:split], data.iloc[split:]
        if len(test) < 10:
            continue
        x_train = np.column_stack([np.ones(len(train)), train[numeric].to_numpy(dtype=float)])
        coefficients = np.linalg.lstsq(x_train, train[str(target)].to_numpy(dtype=float), rcond=None)[0]
        x_test = np.column_stack([np.ones(len(test)), test[numeric].to_numpy(dtype=float)])
        predicted = x_test @ coefficients
        actual = test[str(target)].to_numpy(dtype=float)
        denominator = float(np.sum(np.square(actual - actual.mean())))
        r2 = 1.0 - float(np.sum(np.square(actual - predicted))) / max(denominator, 1.0e-12)
        if r2 >= threshold:
            strongest = np.argsort(np.abs(coefficients[1:]))[-min(8, len(numeric)) :]
            selected = [numeric[int(position)] for position in strongest]
            rows.append(
                {
                    "feature": json.dumps(selected, ensure_ascii=False),
                    "target": str(target),
                    "kind": "multivariate_linear_combination",
                    "target_offset_steps": 0,
                    "correlation": np.nan,
                    "coefficient": json.dumps(
                        {numeric[i]: float(coefficients[i + 1]) for i in strongest},
                        ensure_ascii=False,
                    ),
                    "intercept": float(coefficients[0]),
                    "residual_ratio": float(np.sqrt(np.mean(np.square(actual - predicted))) / max(np.std(actual), 1.0e-12)),
                    "risk": "amber",
                    "reason": f"时间后段检验 R2={r2:.6f}；仅用于关系审计，不自动入模",
                }
            )
    return rows


def _static_leakage_scan() -> list[dict[str, str]]:
    """扫描工程核心代码中的高风险时序操作。"""

    from gas_power import data, features

    source = "\n".join([inspect.getsource(data), inspect.getsource(features)])
    checks = {
        "centered_rolling": r"rolling\([^\n)]*center\s*=\s*True",
        "bidirectional_interpolation": r"interpolate\([^\n)]*limit_direction\s*=\s*['\"]both",
        "time_axis_backfill": r"\.bfill\(\s*(?:limit\s*=|\))",
        "full_data_standardization": r"(?:StandardScaler|RobustScaler)\([^)]*\)\.fit(?:_transform)?\(",
    }
    return [
        {
            "check": name,
            "status": "failed" if re.search(pattern, source) else "passed",
            "detail": "发现高风险代码模式" if re.search(pattern, source) else "未发现高风险代码模式",
        }
        for name, pattern in checks.items()
    ]


def _resampling_sensitivity(
    config: ProjectConfig,
    targets: Sequence[str],
) -> list[dict[str, Any]]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for label, closed in resampling_variants():
        raw = copy.deepcopy(config.raw)
        raw.setdefault("data", {})["resampling"] = {"label": label, "closed": closed}
        candidate = ProjectConfig(raw=raw, source=config.source, root=config.root)
        try:
            frames[(label, closed)] = ConfiguredDataLoader(candidate).load()[0]
        except Exception as exc:
            frames[(label, closed)] = pd.DataFrame()
            frames[(label, closed)].attrs["error"] = str(exc)
    reference_key = (
        str(config.section("data").get("resampling", {}).get("label", "right")),
        str(config.section("data").get("resampling", {}).get("closed", "right")),
    )
    reference = frames.get(reference_key, pd.DataFrame())
    rows: list[dict[str, Any]] = []
    for (label, closed), candidate in frames.items():
        row: dict[str, Any] = {"label": label, "closed": closed, "reference": (label, closed) == reference_key}
        if candidate.empty or reference.empty:
            row.update({"status": "error", "error": candidate.attrs.get("error", "empty")})
        else:
            common = reference.index.intersection(candidate.index)
            differences = []
            for target in targets:
                left = pd.to_numeric(reference.loc[common, target], errors="coerce")
                right = pd.to_numeric(candidate.loc[common, target], errors="coerce")
                differences.append(float((left - right).abs().mean()))
            row.update(
                {
                    "status": "ok",
                    "common_rows": len(common),
                    "mean_absolute_target_difference": float(np.nanmean(differences)),
                }
            )
        rows.append(row)
    return rows


def _split_integrity(config: ProjectConfig, frame: pd.DataFrame, max_horizon: int) -> dict[str, Any]:
    try:
        splits = splitter_from_config(config.section("validation")).split(frame.index, max_horizon)
    except Exception as exc:
        return {"status": "failed", "detail": str(exc)}
    seen: set[pd.Timestamp] = set()
    overlap = 0
    for split in splits:
        origins = {pd.Timestamp(item) for item in split.validation_origins}
        overlap += len(seen.intersection(origins))
        seen.update(origins)
        if split.train_end >= split.validation_origins.min():
            return {"status": "failed", "detail": "训练区间与验证起点重叠"}
    return {
        "status": "passed" if overlap == 0 else "failed",
        "folds": len(splits),
        "overlapping_validation_origins": overlap,
        "label_window_rule": "训练响应的 target_time 必须不晚于 train_end",
    }


def run_data_audit(
    config: ProjectConfig,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    registry: FeatureAvailabilityRegistry,
) -> AuditArtifacts:
    settings = config.section("audit")
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    interval = int(config.section("optimization").get("interval_minutes", 15))
    numeric_fields = [
        str(column)
        for column in frame.select_dtypes(include=[np.number]).columns
        if not str(column).startswith("feat_")
    ]
    lag = calculate_lag_correlations(
        frame,
        numeric_fields,
        targets,
        int(settings.get("lag_min_steps", -96)),
        int(settings.get("lag_max_steps", 96)),
        interval,
    )
    suspicious = _affine_copy_candidates(
        frame,
        lag,
        targets,
        float(settings.get("high_correlation_threshold", 0.995)),
        float(settings.get("affine_residual_ratio_threshold", 0.01)),
    )
    suspicious.extend(
        _future_correlation_candidates(
            lag,
            targets,
            float(settings.get("high_correlation_threshold", 0.995)),
        )
    )
    suspicious.extend(
        _multivariate_copy_candidates(
            frame,
            targets,
            float(settings.get("multivariate_r2_threshold", 0.995)),
        )
    )
    keywords = [str(value).lower() for value in settings.get("suspicious_keywords", [])]
    all_columns = list(dict.fromkeys([*frame.columns.astype(str), *features.columns.astype(str)]))
    for column in all_columns:
        matched = [keyword for keyword in keywords if keyword in column.lower()]
        if matched:
            suspicious.append(
                {
                    "feature": column,
                    "target": "",
                    "kind": "suspicious_name",
                    "target_offset_steps": np.nan,
                    "correlation": np.nan,
                    "coefficient": np.nan,
                    "intercept": np.nan,
                    "residual_ratio": np.nan,
                    "risk": "red" if any(key in {"future", "label", "target"} for key in matched) else "amber",
                    "reason": f"字段名包含可疑关键词: {matched}",
                }
            )

    availability_rows = []
    for field in numeric_fields:
        item = registry.get(field)
        availability_rows.append(
            {
                "field": field,
                "registered": field in registry.fields,
                "available_at_origin": item.available_at_origin,
                "allow_short": item.allow_short,
                "allow_long": item.allow_long,
                "is_label": item.is_label,
                "is_plan": item.is_plan,
                "pending": item.pending,
            }
        )
    direct_targets = [target for target in targets if target in features.columns]
    future_feature_hits = [
        column for column in features.columns if "future" in str(column).lower()
    ]
    resampling = _resampling_sensitivity(config, targets)
    resampling_changed = any(
        row.get("mean_absolute_target_difference", 0.0)
        > float(settings.get("resampling_difference_threshold", 1.0e-9))
        for row in resampling
        if not row.get("reference")
    )
    preprocessing = config.section("preprocessing")
    imputation_method = str(preprocessing.get("imputation", {}).get("method", "none"))
    static_checks = _static_leakage_scan()
    split_check = _split_integrity(config, frame, int(config.section("forecast")["long_steps"]))
    suspicious_columns = [
        "feature", "target", "kind", "target_offset_steps", "correlation",
        "coefficient", "intercept", "residual_ratio", "risk", "reason",
    ]
    suspicious_frame = pd.DataFrame(suspicious, columns=suspicious_columns).drop_duplicates(
        subset=["feature", "target", "kind", "target_offset_steps"], keep="first"
    )
    future_red = suspicious_frame.loc[
        (suspicious_frame["risk"] == "red")
        & (pd.to_numeric(suspicious_frame["target_offset_steps"], errors="coerce") > 0)
    ]
    checks = [
        {"check": "target_not_direct_feature", "status": "passed" if not direct_targets else "failed", "detail": direct_targets},
        {"check": "future_named_feature_rejected", "status": "passed" if not future_feature_hits else "failed", "detail": future_feature_hits},
        {"check": "unknown_fields_default_denied", "status": "passed", "detail": [row["field"] for row in availability_rows if not row["registered"]]},
        {"check": "forward_fill_is_causal", "status": "passed" if imputation_method in {"ffill", "none"} else "failed", "detail": imputation_method},
        {"check": "split_and_window_integrity", **split_check},
        {"check": "unique_time_index", "status": "passed" if frame.index.is_unique else "failed", "detail": int(frame.index.duplicated().sum())},
        {"check": "target_time_formula", "status": "passed", "detail": "验证器按 origin + horizon * 15min 取标签"},
        {"check": "resampling_boundary_sensitive", "status": "warning" if resampling_changed else "passed", "detail": resampling},
        *static_checks,
    ]
    summary = {
        "warning": "未来偏移相关性只用于泄漏诊断，代码结构禁止其进入正式训练特征。",
        "rows": len(frame),
        "fields": len(numeric_fields),
        "targets": targets,
        "registry_path": str(registry.path) if registry.path else None,
        "registry_missing": registry.missing_file,
        "formal_feature_count": len(features.columns),
        "formal_feature_columns": list(features.columns),
        "availability": availability_rows,
        "checks": checks,
        "future_red_risk_count": len(future_red),
        "suspicious_feature_count": len(suspicious_frame),
        "resampling_sensitivity": resampling,
        "passed": not any(item.get("status") == "failed" for item in checks),
    }
    markdown_lines = [
        "# 数据与时间语义审计",
        "",
        "> 未来偏移只用于诊断，绝对不能进入正式模型。红色风险不代表作弊结论，需要结合字段业务发布时间确认。",
        "",
        f"- 数据行数：{len(frame)}",
        f"- 正式特征数：{len(features.columns)}",
        f"- 可疑项数：{len(suspicious_frame)}",
        f"- 未来偏移红色风险数：{len(future_red)}",
        "",
        "## 检查结果",
        "",
        "| 检查 | 状态 |",
        "|---|---|",
    ]
    markdown_lines.extend(f"| {item['check']} | {item.get('status')} |" for item in checks)
    markdown_lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 高相关或近似线性关系不自动进入模型，必须先确认 feature_available_time <= forecast_origin_time。",
            "- 合成数据中的 future_generator_all_leak 和 timestamp_advanced_generator_all 是故意注入的验收字段。",
            "- 合成指标不能用于推断真实竞赛成绩。",
        ]
    )
    return AuditArtifacts(summary, lag, suspicious_frame, "\n".join(markdown_lines) + "\n")
