# ============================================================
# evaluation/matcher.py — 매칭 엔진
# annotation(정답)과 prediction(예측)을 시간+종명 기준으로 대조하여
# TP, FP, FN을 분류한다.
# ============================================================

import csv
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ── 데이터 클래스 ──────────────────────────────────────────

@dataclass
class MatchResult:
    """단일 매칭 결과"""
    category: str          # "TP", "FP", "FN"
    species: str
    file: str
    pred_time: Optional[float] = None
    pred_score: Optional[float] = None
    ann_t_start: Optional[float] = None
    ann_t_end: Optional[float] = None


@dataclass
class MatchingConfig:
    """매칭 파라미터"""
    time_tolerance: float = 1.5    # annotation 경계 밖 허용오차 (초)
    iou_threshold: float = 0.0     # IoU 기반 매칭 시 최소 임계값 (0=비활성)
    one_to_one: bool = True        # 1:1 매칭 (하나의 annotation에 하나의 prediction)
    prefer_highest_score: bool = True  # 복수 후보 시 최고 점수 우선


# ── 매칭 함수 ──────────────────────────────────────────────

def match_predictions_to_annotations(
    annotations: List[dict],
    predictions: List[dict],
    config: MatchingConfig = None,
) -> Tuple[List[MatchResult], dict]:
    """
    annotation과 prediction을 매칭하여 TP/FP/FN을 분류한다.

    Args:
        annotations: [{"file", "t_start", "t_end", "species"}, ...]
        predictions: [{"file", "species", "time", "composite"}, ...]
        config: 매칭 파라미터

    Returns:
        (match_results, summary_dict)
    """
    if config is None:
        config = MatchingConfig()

    results: List[MatchResult] = []

    # --- 0) 파일명 정규화 ---
    # prediction에 file 정보가 없는 경우(단일 파일 분석),
    # annotation의 file도 무시하여 매칭이 가능하도록 한다.
    pred_files = set(p.get("file", "") for p in predictions)
    ann_files = set(a.get("file", "") for a in annotations)

    if pred_files == {""} and ann_files != {""}:
        annotations = [{**a, "file": ""} for a in annotations]
    elif ann_files == {""} and pred_files != {""}:
        predictions = [{**p, "file": ""} for p in predictions]

    # --- 1) 종+파일별 그룹화 ---
    ann_groups = _group_by(annotations, keys=["file", "species"])
    pred_groups = _group_by(predictions, keys=["file", "species"])

    # --- 2) 각 그룹 내에서 매칭 ---
    all_keys = set(ann_groups.keys()) | set(pred_groups.keys())

    for key in all_keys:
        group_anns = ann_groups.get(key, [])
        group_preds = pred_groups.get(key, [])

        # 점수 내림차순 정렬 (높은 점수 우선 매칭)
        if config.prefer_highest_score:
            group_preds = sorted(
                group_preds,
                key=lambda p: p.get("composite", 0),
                reverse=True,
            )

        local_matched_anns = set()

        for pred in group_preds:
            best_ann_idx = None
            best_overlap = -1

            for ai, ann in enumerate(group_anns):
                if config.one_to_one and ai in local_matched_anns:
                    continue

                # 시간 겹침 판정
                overlap = _compute_overlap(
                    pred["time"],
                    ann["t_start"], ann["t_end"],
                    config.time_tolerance,
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ann_idx = ai

            if best_ann_idx is not None and best_overlap >= 0:
                # TP: 매칭 성공
                ann = group_anns[best_ann_idx]
                results.append(MatchResult(
                    category="TP",
                    species=pred["species"],
                    file=pred["file"],
                    pred_time=pred["time"],
                    pred_score=pred.get("composite"),
                    ann_t_start=ann["t_start"],
                    ann_t_end=ann["t_end"],
                ))
                local_matched_anns.add(best_ann_idx)
            else:
                # FP: 매칭 실패 (예측은 있으나 정답 없음)
                results.append(MatchResult(
                    category="FP",
                    species=pred["species"],
                    file=pred["file"],
                    pred_time=pred["time"],
                    pred_score=pred.get("composite"),
                ))

        # FN: 매칭되지 않은 annotation
        for ai, ann in enumerate(group_anns):
            if ai not in local_matched_anns:
                results.append(MatchResult(
                    category="FN",
                    species=ann["species"],
                    file=ann["file"],
                    ann_t_start=ann["t_start"],
                    ann_t_end=ann["t_end"],
                ))

    # --- 3) 요약 통계 ---
    summary = _compute_summary(results)

    return results, summary


# ── 내부 유틸리티 ──────────────────────────────────────────

def _compute_overlap(
    pred_time: float,
    ann_t_start: float,
    ann_t_end: float,
    tolerance: float,
) -> float:
    """
    예측 시점이 annotation 구간에 얼마나 가까운지 계산.

    반환값:
      >= 0: 겹침 있음 (구간 내이면 1.0, tolerance 내이면 0~1)
      < 0: 겹침 없음
    """
    if ann_t_start <= pred_time <= ann_t_end:
        return 1.0  # 구간 내부

    dist = min(abs(pred_time - ann_t_start),
               abs(pred_time - ann_t_end))

    if dist <= tolerance:
        return 1.0 - (dist / tolerance)  # 0~1 (가까울수록 높음)

    return -1.0  # 매칭 실패


def _group_by(items: List[dict], keys: List[str]) -> Dict[tuple, List[dict]]:
    """리스트를 복합 키로 그룹화한다."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for item in items:
        group_key = tuple(item.get(k, "") for k in keys)
        groups[group_key].append(item)
    return dict(groups)


def _compute_summary(results: List[MatchResult]) -> dict:
    """TP/FP/FN 카운트 요약 + 종별 요약."""
    tp = sum(1 for r in results if r.category == "TP")
    fp = sum(1 for r in results if r.category == "FP")
    fn = sum(1 for r in results if r.category == "FN")

    # 종별 요약
    species_set = set(r.species for r in results)
    per_species = {}
    for sp in sorted(species_set):
        sp_results = [r for r in results if r.species == sp]
        sp_tp = sum(1 for r in sp_results if r.category == "TP")
        sp_fp = sum(1 for r in sp_results if r.category == "FP")
        sp_fn = sum(1 for r in sp_results if r.category == "FN")
        per_species[sp] = {"tp": sp_tp, "fp": sp_fp, "fn": sp_fn}

    return {
        "total": {"tp": tp, "fp": fp, "fn": fn},
        "per_species": per_species,
    }


# ── CSV I/O ────────────────────────────────────────────────

def load_annotations_csv(filepath: str) -> List[dict]:
    """
    annotation CSV를 로드한다.
    컬럼: file, t_start, t_end, f_low, f_high, species
    """
    annotations = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append({
                "file": row["file"].strip(),
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "f_low": float(row.get("f_low", 0)),
                "f_high": float(row.get("f_high", 0)),
                "species": row["species"].strip(),
            })
    return annotations


def load_predictions_csv(filepath: str, mode: str = "candidates") -> List[dict]:
    """
    분석 결과 CSV를 로드하여 표준화된 딕셔너리 리스트로 반환.

    Args:
        filepath: candidates_all.csv 또는 results_detailed.csv 경로
        mode: "candidates" (전체 후보) 또는 "results" (통과분만)

    Returns:
        [{"file", "species", "time", "composite", "passed"}, ...]
    """
    predictions = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 컬럼명 정리 (앞뒤 공백 제거)
            row = {k.strip(): v.strip() for k, v in row.items()}

            # R의 'NA' 문자열 처리
            def _safe_float(val, default=0.0):
                if val is None or val == "" or val.upper() == "NA":
                    return default
                return float(val)

            time_val = _safe_float(row.get("time", "0"))
            composite_val = _safe_float(row.get("composite", "0"))

            # time이 0이고 원본이 'NA'이면 유효하지 않은 행 → 건너뛰기
            raw_time = row.get("time", "0").strip().upper()
            if raw_time == "NA":
                continue

            pred = {
                "species": row.get("species", ""),
                "time": time_val,
                "composite": composite_val,
            }

            # file 컬럼: candidates_all.csv에는 없을 수 있음 → 기본값
            pred["file"] = row.get("source_file", row.get("file", ""))

            # passed 필드
            if mode == "candidates":
                passed_str = row.get("passed", "FALSE").upper()
                pred["passed"] = passed_str in ("TRUE", "1", "YES")
            else:
                pred["passed"] = True

            predictions.append(pred)

    return predictions


def save_match_results_csv(results: List[MatchResult], filepath: str):
    """매칭 결과를 CSV로 저장한다."""
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "species", "file",
            "pred_time", "pred_score", "ann_t_start", "ann_t_end",
        ])
        for r in results:
            writer.writerow([
                r.category, r.species, r.file,
                r.pred_time if r.pred_time is not None else "",
                round(r.pred_score, 4) if r.pred_score is not None else "",
                r.ann_t_start if r.ann_t_start is not None else "",
                r.ann_t_end if r.ann_t_end is not None else "",
            ])
