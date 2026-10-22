"""Evaluation Statistics and Uncertainty Analysis Module.

Provides reproducible calculation of:
1. Deterministic bootstrap confidence intervals (percentile method) for test
   metrics (e.g. Recall and ROC-AUC) on held-out test predictions.
2. Expected-cost analysis across decision thresholds based on explicit cost
   assumptions (Loss = FN * Cost_Miss + FP * Cost_Alarm).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
THRESHOLD_EVAL_PATH = RESULTS_DIR / "threshold_evaluation.json"
UNCERTAINTY_OUTPUT_PATH = RESULTS_DIR / "evaluation_uncertainty.json"


def compute_bootstrap_ci(
    y_true: Union[np.ndarray, List[int]],
    y_score: Union[np.ndarray, List[float]],
    metric: str = "recall",
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Computes a deterministic percentile bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Binary predicted labels (for recall) or predicted risk probabilities (for ROC-AUC).
    metric : str, default='recall'
        Metric to compute: 'recall' or 'roc_auc'.
    n_bootstraps : int, default=2000
        Number of bootstrap resamples with replacement.
    ci_level : float, default=0.95
        Two-sided confidence level (e.g. 0.95 for 95% CI).
    random_state : int, default=42
        Seed for reproducible NumPy RandomState.

    Returns
    -------
    dict
        Dictionary containing point estimate, CI bounds, replicate count,
        confidence level, random seed, and method description.
    """
    y_t = np.asarray(y_true).astype(int)
    y_s = np.asarray(y_score).astype(float)

    if len(y_t) != len(y_s):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_score)={len(y_s)}")

    if metric not in ("recall", "roc_auc"):
        raise ValueError(f"Unsupported metric: '{metric}'. Supported: 'recall', 'roc_auc'")

    # Compute point estimate on original sample
    if metric == "recall":
        point_est = float(recall_score(y_t, y_s.astype(int)))
    else:
        point_est = float(roc_auc_score(y_t, y_s))

    rng = np.random.RandomState(random_state)
    n = len(y_t)
    replicate_scores: List[float] = []
    skipped_resamples = 0
    attempts = 0
    max_attempts = n_bootstraps * 10

    while len(replicate_scores) < n_bootstraps and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(n, size=n, replace=True)
        sample_yt = y_t[idx]
        sample_ys = y_s[idx]

        unique_classes = np.unique(sample_yt)
        # ROC-AUC requires both classes (0 and 1) to be present
        if len(unique_classes) < 2:
            skipped_resamples += 1
            continue

        if metric == "recall":
            # If no positive instances in resample, skip to preserve valid calculation
            if sample_yt.sum() == 0:
                skipped_resamples += 1
                continue
            score = float(recall_score(sample_yt, sample_ys.astype(int)))
        else:
            score = float(roc_auc_score(sample_yt, sample_ys))

        replicate_scores.append(score)

    if len(replicate_scores) < n_bootstraps:
        raise RuntimeError(
            f"Could not obtain {n_bootstraps} valid bootstrap resamples after {attempts} attempts "
            f"({skipped_resamples} skipped). Insufficient class diversity in bootstrap samples."
        )

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    ci_lower = float(np.percentile(replicate_scores, lower_pct))
    ci_upper = float(np.percentile(replicate_scores, upper_pct))

    return {
        "metric": metric,
        "point_estimate": point_est,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "n_bootstraps_requested": n_bootstraps,
        "valid_replicates": len(replicate_scores),
        "skipped_resamples": skipped_resamples,
        "method": "empirical_percentile_bootstrap",
        "random_state": random_state,
        "ci_bounds_formatted": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        "ci_percentage_formatted": f"[{ci_lower * 100:.1f}%, {ci_upper * 100:.1f}%]",
    }


def compute_expected_cost(
    confusion_matrix: Union[np.ndarray, List[List[int]]],
    cost_ratio: float = 8.0,
    cost_alarm: float = 1.0,
) -> Dict[str, Any]:
    """Computes expected operational loss from a 2x2 confusion matrix under cost assumptions.

    Parameters
    ----------
    confusion_matrix : 2x2 array-like
        Format: [[TN, FP], [FN, TP]]
    cost_ratio : float, default=8.0
        Assumed ratio of missed-delay cost to false-alarm review cost (C_delay / C_alarm).
    cost_alarm : float, default=1.0
        Baseline cost unit for false alarms.

    Returns
    -------
    dict
        Dictionary with FN, FP, unit costs, and total expected cost.
    """
    cm = np.asarray(confusion_matrix)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}")

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    cost_missed_delay = float(cost_ratio * cost_alarm)
    expected_loss = float((fn * cost_missed_delay) + (fp * cost_alarm))

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cost_ratio": cost_ratio,
        "cost_alarm": cost_alarm,
        "cost_missed_delay": cost_missed_delay,
        "expected_loss": expected_loss,
        "loss_formula": f"FN * {cost_missed_delay:.1f} + FP * {cost_alarm:.1f} = {fn} * {cost_missed_delay:.1f} + {fp} * {cost_alarm:.1f}",
    }


def compute_break_even_ratio(fn_a: int, fp_a: int, fn_b: int, fp_b: int) -> Optional[float]:
    """Calculates the break-even cost ratio R = C_delay / C_alarm where Loss(A) = Loss(B).

    Formula:
        fn_a * R + fp_a = fn_b * R + fp_b
        R * (fn_a - fn_b) = fp_b - fp_a
        R = (fp_a - fp_b) / (fn_b - fn_a)
    """
    delta_fn = fn_b - fn_a
    delta_fp = fp_a - fp_b
    if delta_fn == 0:
        return None
    return float(delta_fp / delta_fn)


def evaluate_cost_model_across_thresholds(
    threshold_eval: Optional[Dict[str, Any]] = None,
    cost_ratio: float = 8.0,
    cost_alarm: float = 1.0,
) -> Dict[str, Any]:
    """Evaluates expected cost across thresholds 0.40, 0.50, and 0.55 on held-out test data.

    Returns full comparative table with break-even ratios and cost differences.
    """
    if threshold_eval is None:
        if not THRESHOLD_EVAL_PATH.exists():
            raise FileNotFoundError(f"Missing artifact: {THRESHOLD_EVAL_PATH}")
        with open(THRESHOLD_EVAL_PATH, "r", encoding="utf-8") as f:
            te_data = json.load(f)
        threshold_eval = te_data["test_evaluation"]

    eval_040 = threshold_eval["operational_threshold_0_40"]
    eval_050 = threshold_eval["default_threshold_0_50"]
    eval_055 = threshold_eval["f1_maximizing_threshold_0_55"]

    cost_040 = compute_expected_cost(eval_040["confusion_matrix"], cost_ratio, cost_alarm)
    cost_050 = compute_expected_cost(eval_050["confusion_matrix"], cost_ratio, cost_alarm)
    cost_055 = compute_expected_cost(eval_055["confusion_matrix"], cost_ratio, cost_alarm)

    break_even_040_vs_050 = compute_break_even_ratio(
        cost_040["fn"], cost_040["fp"], cost_050["fn"], cost_050["fp"]
    )
    break_even_040_vs_055 = compute_break_even_ratio(
        cost_040["fn"], cost_040["fp"], cost_055["fn"], cost_055["fp"]
    )

    base_loss = cost_040["expected_loss"]
    loss_diff_050_pct = ((cost_050["expected_loss"] - base_loss) / base_loss) * 100.0
    loss_diff_055_pct = ((cost_055["expected_loss"] - base_loss) / base_loss) * 100.0

    return {
        "assumptions": {
            "cost_ratio": cost_ratio,
            "cost_alarm_unit": cost_alarm,
            "cost_missed_delay_unit": cost_ratio * cost_alarm,
            "nature_of_cost_values": "Illustrative operational assumptions based on domain judgment, not empirically measured financial losses or proven automated optima.",
        },
        "threshold_0_40": {
            "threshold": 0.40,
            "label": "Operational Policy (Selected on Val)",
            "fn": cost_040["fn"],
            "fp": cost_040["fp"],
            "expected_cost_units": cost_040["expected_loss"],
            "relative_cost_vs_040": "0.0% (Baseline)",
        },
        "threshold_0_50": {
            "threshold": 0.50,
            "label": "Default Model Setting",
            "fn": cost_050["fn"],
            "fp": cost_050["fp"],
            "expected_cost_units": cost_050["expected_loss"],
            "relative_cost_vs_040": f"+{loss_diff_050_pct:.1f}%",
        },
        "threshold_0_55": {
            "threshold": 0.55,
            "label": "Validation F1-Maximizing Setting",
            "fn": cost_055["fn"],
            "fp": cost_055["fp"],
            "expected_cost_units": cost_055["expected_loss"],
            "relative_cost_vs_040": f"+{loss_diff_055_pct:.1f}%",
        },
        "break_even_analysis": {
            "break_even_ratio_040_vs_050": break_even_040_vs_050,
            "break_even_ratio_040_vs_055": break_even_040_vs_055,
            "interpretation": (
                f"Threshold 0.40 incurs strictly lower expected loss than default 0.50 whenever "
                f"missed delays are at least {break_even_040_vs_050:.3f}x costlier than false alarms "
                f"(R >= {break_even_040_vs_050:.3f})."
            ),
        },
    }


def compute_all_reproducible_evaluation_stats(
    seed: int = 42,
    n_bootstraps: int = 2000,
) -> Dict[str, Any]:
    """Generates the full suite of reproducible evaluation stats and confidence intervals."""
    if not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing test predictions artifact: {TEST_PREDICTIONS_PATH}")

    df_test = pd.read_csv(TEST_PREDICTIONS_PATH)
    y_true = df_test["true_target"].values
    y_prob = df_test["predicted_delay_probability"].values
    y_pred_040 = (y_prob >= 0.40).astype(int)

    # 1. Recall bootstrap CI at operational threshold 0.40
    recall_ci = compute_bootstrap_ci(
        y_true=y_true,
        y_score=y_pred_040,
        metric="recall",
        n_bootstraps=n_bootstraps,
        ci_level=0.95,
        random_state=seed,
    )

    # 2. ROC-AUC bootstrap CI
    roc_auc_ci = compute_bootstrap_ci(
        y_true=y_true,
        y_score=y_prob,
        metric="roc_auc",
        n_bootstraps=n_bootstraps,
        ci_level=0.95,
        random_state=seed,
    )

    # 3. Cost model evaluation
    cost_analysis = evaluate_cost_model_across_thresholds()

    return {
        "metadata": {
            "description": "Reproducible uncertainty and cost model evaluation for port delay forecasting assignment",
            "seed": seed,
            "n_bootstraps": n_bootstraps,
            "ci_method": "empirical_percentile_bootstrap",
            "test_sample_size": len(y_true),
            "test_positive_samples": int(y_true.sum()),
            "test_negative_samples": int(len(y_true) - y_true.sum()),
            "test_base_rate": float(y_true.mean()),
            "validation_base_rate": 0.156,
        },
        "bootstrap_confidence_intervals": {
            "recall_threshold_0_40": recall_ci,
            "roc_auc": roc_auc_ci,
        },
        "expected_cost_model": cost_analysis,
    }


def generate_and_save_uncertainty_artifact() -> Path:
    """Executes full evaluation stats suite and writes results/evaluation_uncertainty.json."""
    stats = compute_all_reproducible_evaluation_stats(seed=42, n_bootstraps=2000)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNCERTAINTY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return UNCERTAINTY_OUTPUT_PATH


if __name__ == "__main__":
    out_path = generate_and_save_uncertainty_artifact()
    print(f"Reproducible evaluation uncertainty stats saved to: {out_path}")
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bci = data["bootstrap_confidence_intervals"]
    rec = bci["recall_threshold_0_40"]
    auc = bci["roc_auc"]
    print(f"Recall at 0.40: {rec['point_estimate']*100:.2f}% | 95% CI: {rec['ci_percentage_formatted']}")
    print(f"ROC-AUC:        {auc['point_estimate']:.4f} | 95% CI: [{auc['ci_lower']:.3f}, {auc['ci_upper']:.3f}]")
    ec = data["expected_cost_model"]
    print("\nExpected Cost Summary (Cost Ratio = 8:1):")
    for t_key in ["threshold_0_40", "threshold_0_50", "threshold_0_55"]:
        item = ec[t_key]
        print(f"  {item['label']} (t={item['threshold']}): FN={item['fn']}, FP={item['fp']} -> Cost={item['expected_cost_units']:.0f} ({item['relative_cost_vs_040']})")
    print(f"Break-even ratio (0.40 vs 0.50): {ec['break_even_analysis']['break_even_ratio_040_vs_050']:.3f}")
