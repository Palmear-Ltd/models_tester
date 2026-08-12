#!/usr/bin/env python
"""CLI: evaluate candidate infestation decision rules against a labeled score manifest.

Consumes the manifest + per-window score cache produced by offline_score.py and compares
several session-level decision rules — the current threshold+count-band method (control)
plus several probability-aggregation candidates, including SPRT evidence accumulation —
on: 3-way confusion (HEALTHY/SUSPICIOUS/INFESTED vs. ground truth), ROC/AUC on each rule's
continuous session statistic, and a per-month FP/FN-rate breakdown that directly tests
whether a candidate is more season-robust than the current method.

Any data-driven fitting (ROC-optimal cutoffs, SPRT's fitted-mode Beta likelihoods) is done
on a held-out train split only; ALL reported metrics are computed on a separate test split,
so a candidate can't simply memorize the corpus it's graded on.

ewma_peak is fit at several EWMA spans (EWMA_SPAN_SWEEP), each ranked on the test split
exactly like any other candidate -- the winning span ships alongside its cutoff. Different
models were found (2026-08) to want different spans (an older/larger architecture favored
near-zero smoothing, opposite of the span=5.0 originally validated only against 9_1_2), so
span is no longer assumed fixed across models.

Usage:
  <full-deps-python> evaluate_decision_rules.py --manifest manifest.csv \
      --report-out evaluation_report.csv [--plots-dir plots/]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.decision.baselines import (
    current_method,
    ewma_peak,
    ewma_peak_score,
    mean_score,
    mean_threshold,
    median_score,
    median_threshold,
)
from app.decision.fixed_sample import classify as quantile_classify
from app.decision.fixed_sample import fit_quantile_thresholds
from app.decision.likelihood import FittedLikelihood, fit_beta_params
from app.decision.rms_confidence import RmsConfidenceConfig
from app.decision.sprt import SPRTAccumulator, SPRTConfig
from app.decision.threshold import DEFAULT_SPAN, ThresholdConfig

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:  # pragma: no cover - sklearn is a declared project dependency
    LogisticRegression = None
    roc_auc_score = None
    roc_curve = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# app.decision.baselines.ewma_peak_score and app.decision.threshold.EwmaPeakDecision both
# default to this; kept as the reference/fallback span (DEFAULT_SPAN in threshold.py).
EWMA_SPAN = DEFAULT_SPAN

# Different models were found (2026-08) to have differently-shaped per-window score
# sequences -- an older/larger architecture (9_1_1) turned out to want near-zero
# smoothing, opposite of the span=5.0 fit validated only against 9_1_2. So span is now
# swept and fit per model, same as any other candidate rule: each value here becomes its
# own ewma_peak(span=...) candidate, fit on train and ranked on test like everything
# else. The winner's span ships in that model's decision_threshold.json alongside its
# cutoff -- ThresholdConfig already carries both, so no app code change is needed.
EWMA_SPAN_SWEEP = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)

ALPHA_BETA_SWEEP = (0.01, 0.05, 0.10)
QUANTILE_ALPHA_BETA_SWEEP = (0.01, 0.05, 0.10, 0.15, 0.20, 0.30)
WEIGHT_SWEEP = (0.1, 0.2, 0.4)


@dataclass
class SessionRecord:
    path: str
    label: str  # "T" (infested) or "F" (healthy)
    month_bucket: str
    scores: list = field(default_factory=list)


def load_manifest(manifest_path):
    records = []
    excluded = {"ambiguous": 0, "no_label": 0, "no_cache": 0, "empty_scores": 0}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ambiguous_prefix") in ("True", "1", "true"):
                excluded["ambiguous"] += 1
                continue
            if row.get("label") not in ("T", "F"):
                excluded["no_label"] += 1
                continue
            cache_path = row["cache_path"]
            if not os.path.exists(cache_path):
                excluded["no_cache"] += 1
                continue
            with open(cache_path, "r", encoding="utf-8") as cf:
                scores = json.load(cf)["scores"]
            if not scores:
                excluded["empty_scores"] += 1
                continue
            records.append(
                SessionRecord(
                    path=row["path"],
                    label=row["label"],
                    month_bucket=row.get("month_bucket") or "unknown",
                    scores=scores,
                )
            )
    return records, excluded


def stratified_split(records, test_frac=0.3, seed=42):
    """Deterministic per-label shuffle+split, so fitting and evaluation never share files."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    train, test = [], []
    for group in by_label.values():
        group = list(group)
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_frac)) if len(group) > 1 else 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def roc_optimal_threshold(stats, binary_labels, default=0.5):
    """Youden's J-optimal cutoff on a continuous statistic (binary_labels: 1=infested)."""
    if roc_curve is None or len(set(binary_labels)) < 2:
        return default
    fpr, tpr, thresholds = roc_curve(binary_labels, stats)
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best])


def _sprt_candidate(name, config):
    def predict(scores, cfg=config):
        acc = SPRTAccumulator(cfg)
        for s in scores:
            acc.update(s)
        return acc.result().final_state

    def stat(scores, cfg=config):
        acc = SPRTAccumulator(cfg)
        for s in scores:
            acc.update(s)
        return acc.result().cumulative_llr

    return {"name": name, "predict": predict, "stat": stat, "config": config}


def _quantile_candidate(name, stat_fn, thresholds, span=None):
    def predict(scores, sf=stat_fn, th=thresholds):
        return quantile_classify(sf(scores), th)

    def stat(scores, sf=stat_fn):
        return sf(scores)

    candidate = {"name": name, "predict": predict, "stat": stat, "quantile_thresholds": thresholds}
    if span is not None:
        candidate["span"] = span
    return candidate


def build_candidates(train_records):
    train_scores_by_label = defaultdict(list)
    for r in train_records:
        train_scores_by_label[r.label].extend(r.scores)

    candidates = []

    # Control: main.py's existing threshold+count-band rule, unmodified.
    candidates.append(
        {
            "name": "current_method(thresh=0.5,susp=17,inf=27)",
            "predict": lambda scores: current_method(scores, 0.5, 17, 27),
            "stat": lambda scores: sum(1 for s in scores if s > 0.5),
        }
    )

    train_binary = [1 if r.label == "T" else 0 for r in train_records]

    train_means = [mean_score(r.scores) for r in train_records]
    mean_cutoff = roc_optimal_threshold(train_means, train_binary)
    candidates.append(
        {
            "name": f"mean_threshold(cutoff={mean_cutoff:.3f})",
            "predict": lambda scores, t=mean_cutoff: mean_threshold(scores, t).final_state,
            "stat": lambda scores: mean_score(scores),
        }
    )

    train_medians = [median_score(r.scores) for r in train_records]
    median_cutoff = roc_optimal_threshold(train_medians, train_binary)
    candidates.append(
        {
            "name": f"median_threshold(cutoff={median_cutoff:.3f})",
            "predict": lambda scores, t=median_cutoff: median_threshold(scores, t).final_state,
            "stat": lambda scores: median_score(scores),
        }
    )

    for span in EWMA_SPAN_SWEEP:
        train_ewma = [ewma_peak_score(r.scores, span=span) for r in train_records]
        ewma_cutoff = roc_optimal_threshold(train_ewma, train_binary)
        candidates.append(
            {
                "name": f"ewma_peak(span={span:g},cutoff={ewma_cutoff:.3f})",
                "predict": lambda scores, t=ewma_cutoff, sp=span: ewma_peak(scores, t, span=sp).final_state,
                "stat": lambda scores, sp=span: ewma_peak_score(scores, span=sp),
                # Carried at full precision (the name rounds to 3dp) so --threshold-out can
                # ship this value verbatim as the app's decision_threshold.json.
                "fitted_cutoff": ewma_cutoff,
                "span": span,
            }
        )

    # Fixed-sample-size quantile-band thresholds: recordings here are a fixed ~20s/40
    # windows, not open-ended, so this asks "what statistic value separates the classes
    # at target error rates given exactly this many windows" directly from labeled data,
    # rather than assuming open-ended sequential evidence accumulation (SPRT) applies.
    train_by_label = defaultdict(list)
    for r in train_records:
        train_by_label[r.label].append(r)
    healthy_sessions = train_by_label.get("F", [])
    infested_sessions = train_by_label.get("T", [])
    if len(healthy_sessions) >= 2 and len(infested_sessions) >= 2:
        stat_fns = {"mean": (mean_score, None), "median": (median_score, None)}
        for span in EWMA_SPAN_SWEEP:
            stat_fns[f"ewma_peak_span{span:g}"] = (
                lambda scores, sp=span: ewma_peak_score(scores, span=sp),
                span,
            )
        for stat_name, (stat_fn, stat_span) in stat_fns.items():
            healthy_stats = [stat_fn(r.scores) for r in healthy_sessions]
            infested_stats = [stat_fn(r.scores) for r in infested_sessions]
            for alpha in QUANTILE_ALPHA_BETA_SWEEP:
                beta = alpha
                thresholds = fit_quantile_thresholds(healthy_stats, infested_stats, alpha, beta)
                candidates.append(
                    _quantile_candidate(
                        f"fixed_n_quantile_{stat_name}(alpha={alpha},beta={beta},"
                        f"t_low={thresholds.t_low:.3f},t_high={thresholds.t_high:.3f})",
                        stat_fn,
                        thresholds,
                        span=stat_span,
                    )
                )
    else:
        print("WARNING: not enough train-split sessions per class to fit quantile thresholds; skipping.")

    for alpha in ALPHA_BETA_SWEEP:
        beta = alpha
        for weight in WEIGHT_SWEEP:
            cfg = SPRTConfig(alpha=alpha, beta=beta, weight=weight, likelihood_mode="logit")
            candidates.append(_sprt_candidate(f"sprt_logit(alpha={alpha},beta={beta},weight={weight})", cfg))

    healthy_scores = train_scores_by_label.get("F", [])
    infested_scores = train_scores_by_label.get("T", [])
    if len(healthy_scores) >= 2 and len(infested_scores) >= 2:
        fitted = FittedLikelihood(
            healthy_beta=fit_beta_params(healthy_scores),
            infested_beta=fit_beta_params(infested_scores),
        )
        for alpha in ALPHA_BETA_SWEEP:
            beta = alpha
            for weight in WEIGHT_SWEEP:
                cfg = SPRTConfig(
                    alpha=alpha, beta=beta, weight=weight, likelihood_mode="fitted", fitted_params=fitted
                )
                candidates.append(_sprt_candidate(f"sprt_fitted(alpha={alpha},beta={beta},weight={weight})", cfg))
    else:
        print("WARNING: not enough train-split scores to fit per-class Beta densities; skipping sprt_fitted.")

    return candidates


def confusion_3way(predictions, labels):
    tp = sum(1 for p, l in zip(predictions, labels) if p == "INFESTED" and l == "T")
    tn = sum(1 for p, l in zip(predictions, labels) if p == "HEALTHY" and l == "F")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "INFESTED" and l == "F")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "HEALTHY" and l == "T")
    n = len(labels)
    n_susp = sum(1 for p in predictions if p == "SUSPICIOUS")
    n_decided = n - n_susp

    def safe_div(a, b):
        return a / b if b else float("nan")

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if not math.isnan(precision) and not math.isnan(recall) else float("nan")

    fp_cons, fn_cons = fp, fn
    for p, l in zip(predictions, labels):
        if p == "SUSPICIOUS":
            if l == "T":
                fn_cons += 1
            else:
                fp_cons += 1

    return {
        "n": n,
        "n_suspicious": n_susp,
        "coverage": safe_div(n_decided, n),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_excl_suspicious": safe_div(tp + tn, n_decided),
        "precision_excl_suspicious": precision,
        "recall_excl_suspicious": recall,
        "f1_excl_suspicious": f1,
        "accuracy_conservative": safe_div(tp + tn, n),
        "fp_conservative": fp_cons,
        "fn_conservative": fn_cons,
    }


def fpr_fnr_by_month(predictions, labels, months):
    """Per-month FP rate (among F sessions) and FN rate (among T sessions). A SUSPICIOUS
    verdict counts as an error in the relevant direction (conservative)."""
    buckets = defaultdict(lambda: {"f_total": 0, "f_fp": 0, "t_total": 0, "t_fn": 0})
    for pred, label, month in zip(predictions, labels, months):
        b = buckets[month]
        if label == "F":
            b["f_total"] += 1
            if pred != "HEALTHY":
                b["f_fp"] += 1
        else:
            b["t_total"] += 1
            if pred != "INFESTED":
                b["t_fn"] += 1
    result = {}
    for month, b in sorted(buckets.items()):
        result[month] = {
            "fpr": (b["f_fp"] / b["f_total"]) if b["f_total"] else None,
            "fnr": (b["t_fn"] / b["t_total"]) if b["t_total"] else None,
            "f_total": b["f_total"],
            "t_total": b["t_total"],
        }
    return result


def evaluate_rule(name, predict_fn, stat_fn, records, binary_labels):
    predictions = [predict_fn(r.scores) for r in records]
    stats = [stat_fn(r.scores) for r in records]
    labels = [r.label for r in records]
    months = [r.month_bucket for r in records]

    confusion = confusion_3way(predictions, labels)
    auc = None
    if roc_auc_score is not None and len(set(binary_labels)) > 1:
        try:
            auc = float(roc_auc_score(binary_labels, stats))
        except ValueError:
            auc = None
    month_breakdown = fpr_fnr_by_month(predictions, labels, months)
    fprs = [b["fpr"] for b in month_breakdown.values() if b["fpr"] is not None]
    fnrs = [b["fnr"] for b in month_breakdown.values() if b["fnr"] is not None]

    return {
        "name": name,
        "confusion": confusion,
        "auc": auc,
        "month_breakdown": month_breakdown,
        "fpr_month_std": float(np.std(fprs)) if len(fprs) > 1 else None,
        "fnr_month_std": float(np.std(fnrs)) if len(fnrs) > 1 else None,
    }


def _fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:.3f}"


def print_summary_table(results):
    header = (
        f"{'rule':45s} {'n':>5s} {'cov':>6s} {'acc':>6s} {'prec':>6s} "
        f"{'rec':>6s} {'f1':>6s} {'acc_cons':>9s} {'auc':>6s} {'fpr_std':>8s} {'fnr_std':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        c = r["confusion"]
        print(
            f"{r['name']:45s} {c['n']:5d} {_fmt(c['coverage']):>6s} "
            f"{_fmt(c['accuracy_excl_suspicious']):>6s} {_fmt(c['precision_excl_suspicious']):>6s} "
            f"{_fmt(c['recall_excl_suspicious']):>6s} {_fmt(c['f1_excl_suspicious']):>6s} "
            f"{_fmt(c['accuracy_conservative']):>9s} {_fmt(r['auc']):>6s} "
            f"{_fmt(r['fpr_month_std']):>8s} {_fmt(r['fnr_month_std']):>8s}"
        )


def write_report_csv(results, path):
    fieldnames = [
        "name", "n", "n_suspicious", "coverage", "tp", "tn", "fp", "fn",
        "accuracy_excl_suspicious", "precision_excl_suspicious", "recall_excl_suspicious",
        "f1_excl_suspicious", "accuracy_conservative", "auc", "fpr_month_std", "fnr_month_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"name": r["name"], "auc": r["auc"], "fpr_month_std": r["fpr_month_std"], "fnr_month_std": r["fnr_month_std"]}
            row.update(r["confusion"])
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote report: {path}")


def print_month_breakdown(title, result):
    print(f"\n--- Season/month breakdown: {title} ---")
    print(f"{'month':10s} {'F_total':>8s} {'FP_rate':>8s} {'T_total':>8s} {'FN_rate':>8s}")
    for month, b in result["month_breakdown"].items():
        print(f"{month:10s} {b['f_total']:8d} {_fmt(b['fpr']):>8s} {b['t_total']:8d} {_fmt(b['fnr']):>8s}")


def _beta_pdf(xs, a, b):
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return np.exp(log_norm + (a - 1.0) * np.log(xs) + (b - 1.0) * np.log(1.0 - xs))


def make_plots(plots_dir, train_records, test_records, candidates, results):
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return
    os.makedirs(plots_dir, exist_ok=True)

    healthy_scores = [s for r in train_records if r.label == "F" for s in r.scores]
    infested_scores = [s for r in train_records if r.label == "T" for s in r.scores]
    fitted_cand = next((c for c in candidates if c.get("config") and c["config"].likelihood_mode == "fitted"), None)
    if healthy_scores and infested_scores:
        fig, ax = plt.subplots()
        ax.hist(healthy_scores, bins=40, range=(0, 1), alpha=0.5, density=True, label="Healthy (F) windows")
        ax.hist(infested_scores, bins=40, range=(0, 1), alpha=0.5, density=True, label="Infested (T) windows")
        if fitted_cand is not None:
            xs = np.linspace(0.001, 0.999, 200)
            fitted = fitted_cand["config"].fitted_params
            ax.plot(xs, _beta_pdf(xs, *fitted.healthy_beta), label="Fitted healthy Beta")
            ax.plot(xs, _beta_pdf(xs, *fitted.infested_beta), label="Fitted infested Beta")
        ax.set_xlabel("window score")
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(plots_dir, "score_histograms.png"), dpi=120)
        plt.close(fig)

    if roc_curve is not None:
        binary = [1 if r.label == "T" else 0 for r in test_records]
        if len(set(binary)) > 1:
            fig, ax = plt.subplots()
            for cand in candidates:
                if "current_method" in cand["name"] or "alpha=0.05,beta=0.05,weight=0.2" in cand["name"]:
                    stats = [cand["stat"](r.scores) for r in test_records]
                    fpr, tpr, _ = roc_curve(binary, stats)
                    ax.plot(fpr, tpr, label=cand["name"])
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(fontsize=7)
            fig.savefig(os.path.join(plots_dir, "roc_curves.png"), dpi=120)
            plt.close(fig)

    best_sprt = next((r for r in results if r["name"].startswith("sprt_")), None)
    best_cand = next((c for c in candidates if c["name"] == best_sprt["name"]), None) if best_sprt else None
    if best_cand is not None:
        example_healthy = next((r for r in test_records if r.label == "F"), None)
        example_infested = next((r for r in test_records if r.label == "T"), None)
        fig, ax = plt.subplots()
        a, b = best_cand["config"].boundaries
        for rec, label in ((example_healthy, "example healthy (F)"), (example_infested, "example infested (T)")):
            if rec is None:
                continue
            acc = SPRTAccumulator(best_cand["config"])
            trace = [acc.update(s).cumulative_llr for s in rec.scores]
            ax.plot(trace, label=label)
        ax.axhline(a, color="red", linestyle="--", label="A (INFESTED boundary)")
        ax.axhline(b, color="green", linestyle="--", label="B (HEALTHY boundary)")
        ax.set_xlabel("window index")
        ax.set_ylabel("cumulative LLR (nats)")
        ax.set_title(best_cand["name"])
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(plots_dir, "llr_traces.png"), dpi=120)
        plt.close(fig)

    print(f"Wrote plots to {plots_dir}")


def write_threshold_config(path, cutoff, span):
    """Writes the app's decision_threshold.json at full float precision.

    main.py:load_resources reads this from next to the model, so shipping a refit is
    just dropping this file into models/<name>/ -- no code change."""
    config = ThresholdConfig(cutoff=float(cutoff), span=float(span))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(config.to_json())
    return config


def ewma_quantile_cross_check(candidates, span):
    """The fixed-n quantile band fitted on the ewma_peak statistic at the given span.

    An independent second estimate of where the classes separate: the original fit was
    trusted because this and the ROC-optimal cutoff agreed to within 0.0004. A wide
    disagreement means the refit needs a human look, not a silent ship. `span` must match
    the ewma_peak candidate the cutoff was actually fit at -- comparing against a
    different span's band would be an apples-to-oranges check."""
    bands = [
        c["quantile_thresholds"]
        for c in candidates
        if c["name"].startswith("fixed_n_quantile_ewma_peak_span") and c.get("span") == span
        and "quantile_thresholds" in c
    ]
    if not bands:
        return None
    return sorted(bands, key=lambda b: b.t_high - b.t_low)[len(bands) // 2]


def load_rms_manifest(path):
    """Reads scripts/rms_scan.py's output CSV into `path -> {"mean_rms", "peak_rms"}`
    session aggregates (mean / max of the per-window RMS trace). Sessions with no scanned
    windows (e.g. a file too short to fill the buffer once) are omitted."""
    aggregates = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                values = json.loads(row["rms_json"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if not values:
                continue
            aggregates[row["path"]] = {
                "mean_rms": float(sum(values) / len(values)),
                "peak_rms": float(max(values)),
            }
    return aggregates


def _logistic_prob(a, b, x):
    """sigmoid(a*x + b), clamped against OverflowError on a degenerate/near-separable
    1D fit (LogisticRegression can return very large coefficients when a branch's
    train examples happen to be perfectly separable in log-RMS)."""
    z = a * x + b
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    ez = math.exp(max(z, -700.0))
    return ez / (1.0 + ez)


def fit_logistic_1d(x_values, y_labels):
    """Fits sigmoid(a*x + b) via 1D logistic regression. Returns (a, b), or None if
    sklearn is unavailable or the labels are single-class (nothing to separate)."""
    if LogisticRegression is None or len(x_values) < 2 or len(set(y_labels)) < 2:
        return None
    X = np.asarray(x_values, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y_labels, dtype=np.int64)
    model = LogisticRegression()
    model.fit(X, y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def _rms_pool_examples(records, rms_by_path, cutoff, span, feature):
    """For each record with a matching RMS aggregate, resolves its predicted state under
    the given (already-fitted) ewma_peak cutoff/span and buckets it into the INFESTED
    pool (TP/FP) or HEALTHY pool (TN/FN) per the design doc. Returns
    (infested_x, infested_y, healthy_x, healthy_y): x is log(rms) under `feature`
    ("mean_rms"/"peak_rms"), y is 1 if the verdict was correct, 0 otherwise."""
    infested_x, infested_y, healthy_x, healthy_y = [], [], [], []
    for r in records:
        agg = rms_by_path.get(r.path)
        if agg is None:
            continue
        rms_value = agg[feature]
        if rms_value is None or rms_value <= 0.0 or not math.isfinite(rms_value):
            continue
        predicted = ewma_peak(r.scores, cutoff, span=span).final_state
        x = math.log(rms_value)
        if predicted == "INFESTED":
            infested_x.append(x)
            infested_y.append(1 if r.label == "T" else 0)
        else:
            healthy_x.append(x)
            healthy_y.append(1 if r.label == "F" else 0)
    return infested_x, infested_y, healthy_x, healthy_y


def _branch_auc(a, b, xs, ys):
    if a is None or roc_auc_score is None or len(set(ys)) < 2:
        return None
    probs = [_logistic_prob(a, b, x) for x in xs]
    try:
        return float(roc_auc_score(ys, probs))
    except ValueError:
        return None


# Below this many FN (or FP) train examples in a branch, a sharp tercile split on the
# fitted probability overstates confidence in a curve fit on very few points -- widen
# the trust-tier "medium" band instead (25th/75th percentile) rather than the usual
# 33rd/67th. The design doc's own risk flag: the FN branch was n=97 corpus-wide at
# writing time, and a 70/30 train/test split shrinks that further.
THIN_BRANCH_THRESHOLD = 30


def fit_rms_confidence(records, train, test, cutoff, span, rms_manifest_path):
    """Fits the two sigmoid(a*log(rms)+b) verdict-confidence branches (design: docs/
    superpowers/specs/2026-08-09-rms-verdict-confidence-design.md), trying both mean_rms
    and peak_rms and keeping whichever gets the better held-out TEST-split AUC (averaged
    across both branches). Returns a dict of fitted RmsConfidenceConfig fields, or None
    if there isn't enough labeled+RMS-matched data to fit either feature."""
    rms_by_path = load_rms_manifest(rms_manifest_path)
    matched = sum(1 for r in records if r.path in rms_by_path)
    print("\n=== RMS verdict-confidence fit ===")
    print(f"  RMS manifest: {rms_manifest_path} ({len(rms_by_path)} sessions with RMS)")
    print(f"  {matched}/{len(records)} labeled sessions have a matching RMS aggregate")
    print(f"  Conditioned on ewma_peak cutoff={cutoff!r}, span={span}")

    best = None
    for feature in ("mean_rms", "peak_rms"):
        train_inf_x, train_inf_y, train_heal_x, train_heal_y = _rms_pool_examples(
            train, rms_by_path, cutoff, span, feature
        )
        test_inf_x, test_inf_y, test_heal_x, test_heal_y = _rms_pool_examples(
            test, rms_by_path, cutoff, span, feature
        )
        infested_fit = fit_logistic_1d(train_inf_x, train_inf_y)
        healthy_fit = fit_logistic_1d(train_heal_x, train_heal_y)
        if infested_fit is None or healthy_fit is None:
            print(f"  {feature}: not enough data/labels in one branch to fit; skipping.")
            continue
        a_inf, b_inf = infested_fit
        a_heal, b_heal = healthy_fit
        infested_auc = _branch_auc(a_inf, b_inf, test_inf_x, test_inf_y)
        healthy_auc = _branch_auc(a_heal, b_heal, test_heal_x, test_heal_y)
        aucs = [v for v in (infested_auc, healthy_auc) if v is not None]
        combined_auc = float(np.mean(aucs)) if aucs else None
        n_fn_train = sum(1 for y in train_heal_y if y == 0)
        n_fp_train = sum(1 for y in train_inf_y if y == 0)
        print(
            f"  {feature}: INFESTED n_train={len(train_inf_y)} test_auc={_fmt(infested_auc)} | "
            f"HEALTHY n_train={len(train_heal_y)} (n_fn_train={n_fn_train}, n_fp_train={n_fp_train}) "
            f"test_auc={_fmt(healthy_auc)} | combined={_fmt(combined_auc)}"
        )
        candidate = {
            "feature": feature,
            "a_infested": a_inf, "b_infested": b_inf,
            "a_healthy": a_heal, "b_healthy": b_heal,
            "n_infested_fit": len(train_inf_y), "n_healthy_fit": len(train_heal_y),
            "n_fn_train": n_fn_train, "n_fp_train": n_fp_train,
            "combined_auc": combined_auc,
            "test_inf_x": test_inf_x, "test_heal_x": test_heal_x,
        }
        if best is None or (
            combined_auc is not None and (best["combined_auc"] is None or combined_auc > best["combined_auc"])
        ):
            best = candidate

    if best is None:
        print("  Could not fit either feature (insufficient labeled+RMS-matched data per branch).")
        return None

    print(f"  Winning feature: {best['feature']} (combined held-out test AUC={_fmt(best['combined_auc'])})")

    # Trust-tier cutoffs, chosen on the winning curve's fitted probability across ALL
    # held-out TEST sessions (both pools combined -- probability is a single 0..1
    # "P(verdict correct)" scale regardless of which branch produced it, so pooling is
    # the right unit to set tiers on).
    all_probs = (
        [_logistic_prob(best["a_infested"], best["b_infested"], x) for x in best["test_inf_x"]]
        + [_logistic_prob(best["a_healthy"], best["b_healthy"], x) for x in best["test_heal_x"]]
    )
    thin = best["n_fn_train"] < THIN_BRANCH_THRESHOLD or best["n_fp_train"] < THIN_BRANCH_THRESHOLD
    lo_q, hi_q = (25.0, 75.0) if thin else (33.0, 67.0)
    if all_probs:
        lo_edge = float(np.percentile(all_probs, lo_q))
        hi_edge = float(np.percentile(all_probs, hi_q))
        if hi_edge <= lo_edge:
            hi_edge = lo_edge + 1e-6
    else:
        lo_edge, hi_edge = 0.4, 0.6
    print(
        f"  tier_edges: ({lo_edge:.4f}, {hi_edge:.4f})  "
        f"[{lo_q:g}th/{hi_q:g}th percentile of {len(all_probs)} held-out session probabilities"
        + (", widened: a branch's error class is thin (<%d)]" % THIN_BRANCH_THRESHOLD if thin else "]")
    )

    return {
        "a_infested": best["a_infested"], "b_infested": best["b_infested"],
        "a_healthy": best["a_healthy"], "b_healthy": best["b_healthy"],
        "feature": best["feature"], "tier_edges": (lo_edge, hi_edge),
        "n_infested_fit": best["n_infested_fit"], "n_healthy_fit": best["n_healthy_fit"],
    }


def write_rms_confidence_config(path, fitted):
    """Writes the app's rms_confidence.json. main.py:load_resources reads this from next
    to the model, same as decision_threshold.json -- shipping a refit is just dropping
    this file into models/<name>/, no code change."""
    config = RmsConfidenceConfig(
        a_infested=float(fitted["a_infested"]),
        b_infested=float(fitted["b_infested"]),
        a_healthy=float(fitted["a_healthy"]),
        b_healthy=float(fitted["b_healthy"]),
        feature=fitted["feature"],
        tier_edges=(float(fitted["tier_edges"][0]), float(fitted["tier_edges"][1])),
        n_infested_fit=int(fitted["n_infested_fit"]),
        n_healthy_fit=int(fitted["n_healthy_fit"]),
    )
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(config.to_json())
    return config


def run(args):
    records, excluded = load_manifest(args.manifest)
    n_t = sum(1 for r in records if r.label == "T")
    n_f = sum(1 for r in records if r.label == "F")
    print(f"Loaded {len(records)} labeled sessions ({n_t} T / {n_f} F); excluded: {excluded}")
    if len(records) < 10:
        print("WARNING: very few labeled sessions loaded; results below are not statistically reliable.")

    train, test = stratified_split(records, test_frac=args.test_frac, seed=args.seed)
    print(
        f"Train split: {len(train)} sessions (used only to fit thresholds/likelihoods) | "
        f"Test split: {len(test)} sessions (all metrics below are computed on this split only)"
    )

    candidates = build_candidates(train)
    test_binary = [1 if r.label == "T" else 0 for r in test]

    results = [evaluate_rule(c["name"], c["predict"], c["stat"], test, test_binary) for c in candidates]
    # Sort by conservative accuracy (SUSPICIOUS counted as wrong), not F1-excluding-
    # suspicious — the latter rewards abstaining on hard cases, which can make a rule that
    # only ever decides on a handful of easy sessions look artificially best.
    results.sort(
        key=lambda r: r["confusion"]["accuracy_conservative"] if not math.isnan(r["confusion"]["accuracy_conservative"]) else -1,
        reverse=True,
    )

    print()
    print_summary_table(results)
    write_report_csv(results, args.report_out)

    current = next((r for r in results if r["name"].startswith("current_method")), None)
    best = results[0] if results else None
    if current:
        print_month_breakdown("CURRENT METHOD (control)", current)
    if best and best is not current:
        print_month_breakdown(f"BEST CANDIDATE: {best['name']}", best)

    if args.plots_dir:
        make_plots(args.plots_dir, train, test, candidates, results)

    _report_fitted_cutoff(args, candidates, results)

    if args.rms_manifest:
        picked = _best_ewma_cutoff_and_span(candidates, results)
        if picked is None:
            print("\nNo ewma_peak candidate was fitted; cannot fit RMS verdict confidence.")
        else:
            rms_cutoff, rms_span = picked
            fitted = fit_rms_confidence(records, train, test, rms_cutoff, rms_span, args.rms_manifest)
            if fitted is not None and args.rms_confidence_out:
                write_rms_confidence_config(args.rms_confidence_out, fitted)
                print(f"\nWrote {args.rms_confidence_out}")
                print("  Copy it to models/<model>/rms_confidence.json to activate it.")
            elif fitted is not None:
                print("\n(Pass --rms-confidence-out models/<model>/rms_confidence.json to ship this.)")


def _best_ewma_cutoff_and_span(candidates, results):
    """Picks the (cutoff, span) of whichever ewma_peak(span=...) candidate the test-split
    ranking already preferred -- shared by the cutoff report and the RMS-confidence fit
    below, which conditions its pools on this same already-fitted cutoff."""
    best_ewma_result = next((r for r in results if r["name"].startswith("ewma_peak(")), None)
    if best_ewma_result is None:
        return None
    ewma = next(c for c in candidates if c["name"] == best_ewma_result["name"])
    return ewma["fitted_cutoff"], ewma["span"]


def _report_fitted_cutoff(args, candidates, results):
    """Prints the fitted EWMA-peak cutoff at full precision (the summary table rounds to
    3dp) and, with --threshold-out, ships it as the app's decision_threshold.json.

    Several span variants were fit (EWMA_SPAN_SWEEP) and ranked on the test split
    alongside every other candidate rule; this picks whichever span variant that ranking
    already preferred, rather than assuming a single fixed span."""
    picked = _best_ewma_cutoff_and_span(candidates, results)
    if picked is None:
        print("\nNo ewma_peak candidate was fitted; nothing to emit.")
        return
    cutoff, span = picked
    print("\n=== Fitted EWMA-peak cutoff ===")
    print(f"  Best span (of {EWMA_SPAN_SWEEP}, by test-set accuracy_conservative): {span:g}")
    print(f"  ROC-optimal (Youden J) cutoff: {cutoff!r}  (span={span})")

    band = ewma_quantile_cross_check(candidates, span)
    if band is None:
        print("  Quantile-band cross-check: unavailable (too few sessions per label).")
    else:
        delta = min(abs(cutoff - band.t_low), abs(cutoff - band.t_high))
        print(f"  Quantile-band cross-check:     t_low={band.t_low:.6f} t_high={band.t_high:.6f}")
        print(f"  Closest-edge disagreement:     {delta:.6f}", end="")
        # The 2026-06 fit agreed to 0.0004; an order of magnitude worse means the two
        # methods no longer see the same separation and the number needs a human look.
        print("  <- LARGE, review before shipping" if delta > 0.01 else "  (consistent)")

    if args.threshold_out:
        write_threshold_config(args.threshold_out, cutoff, span)
        print(f"\nWrote {args.threshold_out}")
        print("  Copy it to models/<model>/decision_threshold.json to activate it.")
    else:
        print("\n(Pass --threshold-out models/<model>/decision_threshold.json to ship this.)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate candidate infestation decision rules against a labeled score manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report-out", default="evaluation_report.csv")
    parser.add_argument("--plots-dir", default=None, help="If set, write comparison PNGs here.")
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-out",
        default=None,
        help="Write the fitted EWMA-peak cutoff here as a decision_threshold.json the "
        "app can load (e.g. models/9_1_2/decision_threshold.json).",
    )
    parser.add_argument(
        "--rms-manifest",
        default=None,
        help="scripts/rms_scan.py output CSV (path,label,...,rms_json), joined by `path` "
        "onto this run's labeled sessions to fit the RMS verdict-confidence curves "
        "(see app/decision/rms_confidence.py). Omit to skip that stage entirely.",
    )
    parser.add_argument(
        "--rms-confidence-out",
        default=None,
        help="Write the fitted RMS verdict-confidence curves here as an rms_confidence.json "
        "the app can load (e.g. models/9_1_2/rms_confidence.json). Requires --rms-manifest.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
