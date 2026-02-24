# ============================================================
# tests/test_metrics.py — 지표 계산기 단위 테스트
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.matcher import MatchResult
from evaluation.metrics import (
    compute_metrics_at_threshold,
    compute_all_species_metrics,
    find_optimal_thresholds,
)


# ── 기본 지표 테스트 ──────────────────────────────────────

def test_known_metrics():
    """알려진 TP/FP/FN에서 정확한 지표가 나오는지"""
    # TP=8, FP=2, FN=3
    # Precision = 8/10 = 0.8
    # Recall = 8/11 ≈ 0.7273
    # F1 = 2 * 0.8 * 0.7273 / (0.8 + 0.7273) ≈ 0.7619
    results = []
    for _ in range(8):
        results.append(MatchResult(category="TP", species="sp1", file="a.wav",
                                   pred_time=10.0, pred_score=0.8))
    for _ in range(2):
        results.append(MatchResult(category="FP", species="sp1", file="a.wav",
                                   pred_time=20.0, pred_score=0.6))
    for _ in range(3):
        results.append(MatchResult(category="FN", species="sp1", file="a.wav",
                                   ann_t_start=30.0, ann_t_end=33.0))

    m = compute_metrics_at_threshold(results, threshold=0.0, species="sp1")
    assert m.tp == 8
    assert m.fp == 2
    assert m.fn == 3
    assert abs(m.precision - 0.8) < 0.001
    assert abs(m.recall - 0.7273) < 0.01
    assert abs(m.f1 - 0.7619) < 0.01
    print("  ✅ test_known_metrics PASSED")


def test_zero_division():
    """TP=0일 때 0으로 나누기 방지"""
    results = [
        MatchResult(category="FP", species="sp1", file="a.wav",
                    pred_time=10.0, pred_score=0.5),
        MatchResult(category="FN", species="sp1", file="a.wav",
                    ann_t_start=20.0, ann_t_end=23.0),
    ]
    m = compute_metrics_at_threshold(results, threshold=0.0, species="sp1")
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    print("  ✅ test_zero_division PASSED")


def test_perfect_scores():
    """TP만 있을 때 Precision=Recall=F1=1.0"""
    results = [
        MatchResult(category="TP", species="sp1", file="a.wav",
                    pred_time=10.0, pred_score=0.9),
    ]
    m = compute_metrics_at_threshold(results, threshold=0.0, species="sp1")
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    print("  ✅ test_perfect_scores PASSED")


def test_threshold_filtering():
    """임계값에 따른 분류 변화"""
    results = [
        MatchResult(category="TP", species="sp1", file="a.wav",
                    pred_time=10.0, pred_score=0.8),
        MatchResult(category="TP", species="sp1", file="a.wav",
                    pred_time=20.0, pred_score=0.3),  # threshold 0.5 미달 → FN으로
        MatchResult(category="FP", species="sp1", file="a.wav",
                    pred_time=30.0, pred_score=0.6),
    ]

    # threshold=0.5: TP(0.8)=1, FP(0.6)=1, FN=0+1(0.3미달)=1
    m = compute_metrics_at_threshold(results, threshold=0.5, species="sp1")
    assert m.tp == 1
    assert m.fp == 1
    assert m.fn == 1
    assert abs(m.precision - 0.5) < 0.001
    assert abs(m.recall - 0.5) < 0.001
    print("  ✅ test_threshold_filtering PASSED")


def test_multi_species():
    """종별 + 전체 지표 계산"""
    results = [
        MatchResult(category="TP", species="sp1", file="a.wav",
                    pred_time=10.0, pred_score=0.8),
        MatchResult(category="FP", species="sp1", file="a.wav",
                    pred_time=20.0, pred_score=0.5),
        MatchResult(category="TP", species="sp2", file="a.wav",
                    pred_time=11.0, pred_score=0.9),
        MatchResult(category="FN", species="sp2", file="a.wav",
                    ann_t_start=30.0, ann_t_end=33.0),
    ]
    metrics_list = compute_all_species_metrics(results)
    assert len(metrics_list) == 3  # sp1, sp2, 전체

    sp1 = [m for m in metrics_list if m.species == "sp1"][0]
    assert sp1.tp == 1 and sp1.fp == 1

    sp2 = [m for m in metrics_list if m.species == "sp2"][0]
    assert sp2.tp == 1 and sp2.fn == 1

    overall = [m for m in metrics_list if m.species == "전체"][0]
    assert overall.tp == 2
    print("  ✅ test_multi_species PASSED")


def test_optimal_threshold():
    """최적 임계값 탐색"""
    results = [
        MatchResult(category="TP", species="sp1", file="a.wav",
                    pred_time=10.0, pred_score=0.9),
        MatchResult(category="TP", species="sp1", file="a.wav",
                    pred_time=11.0, pred_score=0.7),
        MatchResult(category="TP", species="sp1", file="a.wav",
                    pred_time=12.0, pred_score=0.3),
        MatchResult(category="FP", species="sp1", file="a.wav",
                    pred_time=20.0, pred_score=0.4),
        MatchResult(category="FP", species="sp1", file="a.wav",
                    pred_time=21.0, pred_score=0.2),
    ]
    opt = find_optimal_thresholds(results, species="sp1", metric="f1")
    assert "optimal_threshold" in opt
    assert "optimal_value" in opt
    assert "threshold_curve" in opt
    assert len(opt["threshold_curve"]) > 0
    print(f"  ✅ test_optimal_threshold PASSED (opt={opt['optimal_threshold']:.3f}, F1={opt['optimal_value']:.4f})")


# ── 실행 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  지표 계산기 테스트")
    print("=" * 50)

    test_known_metrics()
    test_zero_division()
    test_perfect_scores()
    test_threshold_filtering()
    test_multi_species()
    test_optimal_threshold()

    print()
    print("  🎉 모든 지표 계산기 테스트 통과!")
