# ============================================================
# evaluation/metrics.py — 성능 지표 계산기
# Precision, Recall, F1 (Phase 1)
# AUROC, AUPRC, 최적 임계값 (Phase 2 — sklearn 있을 때)
# ============================================================

import json
from dataclasses import dataclass, asdict
from typing import List, Optional

from evaluation.matcher import MatchResult

# sklearn 선택적 임포트
try:
    import numpy as np
    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve, auc, roc_curve, f1_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── 데이터 클래스 ──────────────────────────────────────────

@dataclass
class EvaluationMetrics:
    """종별 성능 지표"""
    species: str
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int = 0               # 추정치 (AUROC용)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auroc: Optional[float] = None   # 연속 점수 필요
    auprc: Optional[float] = None   # 연속 점수 필요


# ── Phase 1: 특정 임계값에서의 지표 ────────────────────────

def compute_metrics_at_threshold(
    match_results: List[MatchResult],
    threshold: float = 0.0,
    species: str = None,
) -> EvaluationMetrics:
    """
    특정 임계값에서의 Precision, Recall, F1을 계산한다.

    매칭 결과에서 pred_score >= threshold인 것만 유효한 예측으로 간주.
    threshold=0.0이면 모든 매칭 결과 그대로 사용.
    """
    if species:
        results = [r for r in match_results if r.species == species]
    else:
        results = list(match_results)

    if threshold > 0:
        tp = sum(1 for r in results
                 if r.category == "TP" and r.pred_score is not None
                 and r.pred_score >= threshold)
        fp = sum(1 for r in results
                 if r.category == "FP" and r.pred_score is not None
                 and r.pred_score >= threshold)
        # FN = 원래 FN + 임계값 미달로 인한 추가 FN
        fn_original = sum(1 for r in results if r.category == "FN")
        fn_threshold = sum(1 for r in results
                           if r.category == "TP" and r.pred_score is not None
                           and r.pred_score < threshold)
        fn = fn_original + fn_threshold
    else:
        tp = sum(1 for r in results if r.category == "TP")
        fp = sum(1 for r in results if r.category == "FP")
        fn = sum(1 for r in results if r.category == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return EvaluationMetrics(
        species=species or "전체",
        threshold=threshold,
        tp=tp, fp=fp, fn=fn, tn=0,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        auroc=None, auprc=None,
    )


def compute_all_species_metrics(
    match_results: List[MatchResult],
    threshold: float = 0.0,
) -> List[EvaluationMetrics]:
    """
    전체 종 + 종별 지표를 한 번에 계산한다.

    Returns:
        [종1_metrics, 종2_metrics, ..., 전체_metrics]
    """
    species_set = sorted(set(r.species for r in match_results))
    metrics_list = []

    for sp in species_set:
        m = compute_metrics_at_threshold(match_results, threshold, species=sp)
        metrics_list.append(m)

    # 전체 합산
    overall = compute_metrics_at_threshold(match_results, threshold, species=None)
    metrics_list.append(overall)

    return metrics_list


# ── Phase 2: 임계값 무관 지표 (AUROC, AUPRC) ──────────────

def compute_curve_metrics(
    annotations: List[dict],
    candidates: List[dict],
    species: str,
    config=None,
    audio_duration: float = None,
    neg_window_step: float = 3.0,
) -> Optional[dict]:
    """
    AUROC, AUPRC 및 곡선 데이터를 계산한다.

    candidates_all.csv의 모든 후보에 대해:
      - annotation과 매칭되면 y_true = 1
      - 매칭되지 않으면 y_true = 0
      - composite score = y_score (연속값)

    Two-tier 평가:
      1) 후보 내 평가 (candidates-only): corMatch가 탐지한 후보만으로 AUROC 계산
         → composite score의 실제 변별력 측정
      2) 전체 평가 (with virtual negatives): 미탐지 구간 포함
         → 시스템 전체 성능 (corMatch + composite) 측정

    Returns:
        dict with auroc, auprc, roc_curve, pr_curve, optimal thresholds
        또는 sklearn 없으면 None
    """
    if not HAS_SKLEARN:
        return None

    from evaluation.matcher import MatchingConfig
    tolerance = 1.5
    if config:
        tolerance = config.time_tolerance

    # annotation 필터링
    sp_anns = [a for a in annotations if a["species"] == species]
    sp_cands = [c for c in candidates if c["species"] == species]

    if not sp_anns or not sp_cands:
        return None

    # ── 파일명 정규화 (matcher.py와 동일 로직) ──
    # prediction에 file 정보가 없는 경우(단일 파일 분석),
    # annotation의 file도 무시하여 매칭이 가능하도록 한다.
    pred_files = set(c.get("file", "") for c in sp_cands)
    ann_files = set(a.get("file", "") for a in sp_anns)

    ignore_file = False
    if pred_files == {""} or ann_files == {""} or pred_files <= {""}:
        ignore_file = True

    # 각 candidate에 대해 ground truth 레이블 부여
    y_true = []
    y_scores = []

    for cand in sp_cands:
        matched = False
        cand_file = cand.get("file", "")
        cand_time = cand["time"]

        for ann in sp_anns:
            ann_file = ann.get("file", "")

            # 파일 비교: ignore_file이면 무조건 통과, 아니면 파일명 비교
            if not ignore_file and ann_file != cand_file:
                # 파일명이 다르면 basename 비교도 시도
                import os
                if os.path.basename(ann_file) != os.path.basename(cand_file):
                    continue

            if ann["t_start"] - tolerance <= cand_time <= ann["t_end"] + tolerance:
                matched = True
                break

        y_true.append(1 if matched else 0)
        y_scores.append(cand["composite"])

    n_pos_cand = sum(1 for y in y_true if y == 1)
    n_neg_cand = sum(1 for y in y_true if y == 0)
    print(f"[DEBUG] curve_metrics: 후보 매칭 결과 — 양성={n_pos_cand}, 음성={n_neg_cand} (총 {len(y_true)}건)")

    # ── Tier 1: 후보 내 평가 (candidates-only AUROC) ──
    # composite score의 실제 변별력 = corMatch 후보 내에서 TP vs FP 분류 능력
    y_true_cand = np.array(y_true)
    y_scores_cand = np.array(y_scores)
    auroc_candidates = None
    if len(np.unique(y_true_cand)) >= 2:
        auroc_candidates = float(roc_auc_score(y_true_cand, y_scores_cand))
        print(f"[DEBUG] curve_metrics: 후보 내 AUROC = {auroc_candidates:.4f}")
    else:
        print(f"[DEBUG] curve_metrics: 후보 내 단일 클래스 → 후보 AUROC 생략")

    # ── 가상 음성 생성 (annotation 미포함 구간) ──
    # 가상 음성의 score: corMatch 탐지 실패 = 상관이 cutoff 미만
    # → 실제 FP 후보의 최저 score 근방으로 추정 (score=0은 비현실적)
    fp_scores = [s for s, y in zip(y_scores, y_true) if y == 0]
    if fp_scores:
        # FP 후보 중 하위 25%의 score를 가상 음성 기준으로 사용
        virtual_neg_score = float(np.percentile(fp_scores, 25))
    else:
        # FP가 없으면 전체 후보 최저 score의 절반
        virtual_neg_score = float(min(y_scores) * 0.5) if y_scores else 0.0

    n_virtual = 0
    if audio_duration is not None and audio_duration > 0:
        # 기존 candidate 시간 목록
        cand_times = [c["time"] for c in sp_cands]

        t = neg_window_step / 2  # 첫 윈도우 중심
        while t < audio_duration:
            # 1) annotation 구간 내이면 건너뛰기
            in_annotation = False
            for ann in sp_anns:
                if ann["t_start"] - tolerance <= t <= ann["t_end"] + tolerance:
                    in_annotation = True
                    break
            if in_annotation:
                t += neg_window_step
                continue

            # 2) 기존 candidate 근처이면 건너뛰기 (이미 처리됨)
            near_candidate = False
            for ct in cand_times:
                if abs(t - ct) <= tolerance:
                    near_candidate = True
                    break
            if near_candidate:
                t += neg_window_step
                continue

            # 3) 가상 음성 추가 — FP 분포 기반 합리적 score 추정
            y_true.append(0)
            y_scores.append(virtual_neg_score)
            n_virtual += 1

            t += neg_window_step

    print(f"[DEBUG] curve_metrics: 가상 음성 {n_virtual}건 추가 (score={virtual_neg_score:.4f}, audio_duration={audio_duration}, step={neg_window_step})")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    n_total_pos = int(np.sum(y_true == 1))
    n_total_neg = int(np.sum(y_true == 0))
    print(f"[DEBUG] curve_metrics: 최종 — 양성={n_total_pos}, 음성={n_total_neg}")

    # 양성/음성 모두 있어야 계산 가능
    if len(np.unique(y_true)) < 2:
        print(f"[DEBUG] curve_metrics: 양성/음성 둘 다 필요 → None 반환")
        return None

    # AUROC (전체: 후보 + 가상 음성)
    auroc = roc_auc_score(y_true, y_scores)

    # AUPRC
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(rec_curve, prec_curve)

    # ROC 곡선 데이터
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    # 최적 임계값 (F1 기준) — 0.01부터 탐색
    best_f1 = 0
    best_thresh_f1 = 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        pred_binary = (y_scores >= t).astype(int)
        f1_val = f1_score(y_true, pred_binary, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh_f1 = t

    # 최적 임계값 (Youden's J = TPR - FPR 최대)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_thresh_youden = roc_thresholds[best_j_idx]

    result = {
        "auroc": round(float(auroc), 4),
        "auprc": round(float(auprc), 4),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr_curve": {
            "precision": prec_curve.tolist(),
            "recall": rec_curve.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "optimal_threshold_f1": round(float(best_thresh_f1), 3),
        "optimal_f1": round(float(best_f1), 4),
        "optimal_threshold_youden": round(float(best_thresh_youden), 3),
        # 분포 시각화용 원본 데이터
        "y_true": y_true.tolist(),
        "y_scores": y_scores.tolist(),
    }

    # Tier 1 결과 추가 (후보 내 AUROC — composite score의 순수 변별력)
    if auroc_candidates is not None:
        result["auroc_candidates_only"] = round(auroc_candidates, 4)

    # ── Tier 2: 통과 후보(passed)의 precision ──
    # 실제 사용자가 듣는 결과물의 정밀도 측정
    passed_cands = [c for c in sp_cands if c.get("passed", False)]
    if passed_cands:
        n_passed_tp = 0
        for cand in passed_cands:
            cand_time = cand["time"]
            matched = False
            for ann in sp_anns:
                ann_file = ann.get("file", "")
                cand_file = cand.get("file", "")
                if not ignore_file and ann_file != cand_file:
                    import os
                    if os.path.basename(ann_file) != os.path.basename(cand_file):
                        continue
                if ann["t_start"] - tolerance <= cand_time <= ann["t_end"] + tolerance:
                    matched = True
                    break
            if matched:
                n_passed_tp += 1
        passed_precision = n_passed_tp / len(passed_cands)
        result["passed_precision"] = round(passed_precision, 4)
        result["passed_count"] = len(passed_cands)
        result["passed_tp"] = n_passed_tp
        print(f"[DEBUG] curve_metrics: 통과 후보 정밀도 = {n_passed_tp}/{len(passed_cands)} = {passed_precision:.4f}")

    return result


def find_optimal_thresholds(
    match_results: List[MatchResult],
    species: str = None,
    metric: str = "f1",
    threshold_range: tuple = (0.05, 0.95),
    step: float = 0.01,
) -> dict:
    """
    임계값을 sweep하면서 최적값을 찾는다.

    Returns:
        {
            "optimal_threshold": float,
            "optimal_value": float,
            "threshold_curve": [(threshold, precision, recall, f1), ...],
        }
    """
    results_curve = []
    t = threshold_range[0]
    while t <= threshold_range[1]:
        m = compute_metrics_at_threshold(match_results, threshold=t, species=species)
        results_curve.append((round(t, 3), m.precision, m.recall, m.f1))
        t += step

    if not results_curve:
        return {"optimal_threshold": 0.5, "optimal_value": 0.0, "threshold_curve": []}

    # metric 기준으로 최적값 선택
    if metric == "f1":
        best = max(results_curve, key=lambda r: r[3])
    elif metric == "precision":
        filtered = [r for r in results_curve if r[1] >= 0.9]
        best = max(filtered, key=lambda r: r[2]) if filtered else results_curve[0]
    elif metric == "recall":
        filtered = [r for r in results_curve if r[2] >= 0.9]
        best = max(filtered, key=lambda r: r[1]) if filtered else results_curve[0]
    else:
        best = max(results_curve, key=lambda r: r[3])

    return {
        "optimal_threshold": best[0],
        "optimal_value": best[3] if metric == "f1" else best[1],
        "threshold_curve": results_curve,
    }


# ── 결과 저장 ──────────────────────────────────────────────

def export_evaluation_json(
    metrics_list: List[EvaluationMetrics],
    curve_data: dict = None,
    filepath: str = None,
) -> dict:
    """평가 결과를 JSON 구조로 변환 (+ 선택적 파일 저장)."""
    import datetime

    result = {
        "evaluation_date": datetime.datetime.now().isoformat(),
        "species_metrics": {},
        "overall": None,
    }

    for m in metrics_list:
        entry = {
            "tp": m.tp, "fp": m.fp, "fn": m.fn,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "threshold": m.threshold,
        }
        if m.auroc is not None:
            entry["auroc"] = m.auroc
        if m.auprc is not None:
            entry["auprc"] = m.auprc

        if m.species == "전체":
            result["overall"] = entry
        else:
            result["species_metrics"][m.species] = entry

    if curve_data:
        result["curve_data"] = curve_data

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result
