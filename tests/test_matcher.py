# ============================================================
# tests/test_matcher.py — 매칭 엔진 단위 테스트
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.matcher import (
    MatchResult,
    MatchingConfig,
    match_predictions_to_annotations,
    _compute_overlap,
)


# ── 기본 매칭 테스트 ──────────────────────────────────────

def test_perfect_match():
    """모든 예측이 정답과 일치하는 경우"""
    anns = [{"file": "a.wav", "t_start": 10, "t_end": 13, "species": "sp1"}]
    preds = [{"file": "a.wav", "time": 11.5, "species": "sp1", "composite": 0.8}]
    results, summary = match_predictions_to_annotations(anns, preds)
    assert sum(1 for r in results if r.category == "TP") == 1
    assert sum(1 for r in results if r.category == "FP") == 0
    assert sum(1 for r in results if r.category == "FN") == 0
    print("  ✅ test_perfect_match PASSED")


def test_false_positive():
    """정답이 없는데 예측한 경우"""
    anns = []
    preds = [{"file": "a.wav", "time": 11.5, "species": "sp1", "composite": 0.8}]
    results, _ = match_predictions_to_annotations(anns, preds)
    assert sum(1 for r in results if r.category == "FP") == 1
    print("  ✅ test_false_positive PASSED")


def test_false_negative():
    """정답이 있는데 예측 못한 경우"""
    anns = [{"file": "a.wav", "t_start": 10, "t_end": 13, "species": "sp1"}]
    preds = []
    results, _ = match_predictions_to_annotations(anns, preds)
    assert sum(1 for r in results if r.category == "FN") == 1
    print("  ✅ test_false_negative PASSED")


def test_time_tolerance():
    """허용오차 내 매칭 (경계 밖이지만 tolerance 이내)"""
    anns = [{"file": "a.wav", "t_start": 10, "t_end": 13, "species": "sp1"}]
    preds = [{"file": "a.wav", "time": 14.0, "species": "sp1", "composite": 0.7}]
    config = MatchingConfig(time_tolerance=1.5)
    results, _ = match_predictions_to_annotations(anns, preds, config)
    assert sum(1 for r in results if r.category == "TP") == 1
    print("  ✅ test_time_tolerance PASSED")


def test_time_tolerance_exceeded():
    """허용오차 초과 → 매칭 실패"""
    anns = [{"file": "a.wav", "t_start": 10, "t_end": 13, "species": "sp1"}]
    preds = [{"file": "a.wav", "time": 16.0, "species": "sp1", "composite": 0.7}]
    config = MatchingConfig(time_tolerance=1.5)
    results, _ = match_predictions_to_annotations(anns, preds, config)
    assert sum(1 for r in results if r.category == "FP") == 1
    assert sum(1 for r in results if r.category == "FN") == 1
    print("  ✅ test_time_tolerance_exceeded PASSED")


def test_species_mismatch():
    """종이 다르면 매칭 안 됨"""
    anns = [{"file": "a.wav", "t_start": 10, "t_end": 13, "species": "sp1"}]
    preds = [{"file": "a.wav", "time": 11.5, "species": "sp2", "composite": 0.8}]
    results, _ = match_predictions_to_annotations(anns, preds)
    assert sum(1 for r in results if r.category == "FP") == 1
    assert sum(1 for r in results if r.category == "FN") == 1
    print("  ✅ test_species_mismatch PASSED")


def test_one_to_one_matching():
    """1:1 매칭: 하나의 annotation에 하나의 prediction만"""
    anns = [{"file": "a.wav", "t_start": 10, "t_end": 13, "species": "sp1"}]
    preds = [
        {"file": "a.wav", "time": 11.0, "species": "sp1", "composite": 0.9},
        {"file": "a.wav", "time": 12.0, "species": "sp1", "composite": 0.6},
    ]
    config = MatchingConfig(one_to_one=True)
    results, _ = match_predictions_to_annotations(anns, preds, config)
    tp_count = sum(1 for r in results if r.category == "TP")
    fp_count = sum(1 for r in results if r.category == "FP")
    assert tp_count == 1  # 최고 점수 하나만 매칭
    assert fp_count == 1  # 나머지는 FP
    print("  ✅ test_one_to_one_matching PASSED")


def test_multi_file():
    """다중 파일에서 매칭"""
    anns = [
        {"file": "a.wav", "t_start": 10, "t_end": 13, "species": "sp1"},
        {"file": "b.wav", "t_start": 5, "t_end": 8, "species": "sp1"},
    ]
    preds = [
        {"file": "a.wav", "time": 11.0, "species": "sp1", "composite": 0.8},
        {"file": "b.wav", "time": 6.5, "species": "sp1", "composite": 0.7},
    ]
    results, summary = match_predictions_to_annotations(anns, preds)
    tp_count = sum(1 for r in results if r.category == "TP")
    assert tp_count == 2
    assert summary["total"]["tp"] == 2
    print("  ✅ test_multi_file PASSED")


# ── overlap 함수 테스트 ────────────────────────────────────

def test_overlap_inside():
    """예측이 구간 내부"""
    assert _compute_overlap(11.0, 10.0, 13.0, 1.5) == 1.0
    print("  ✅ test_overlap_inside PASSED")


def test_overlap_edge():
    """예측이 구간 경계"""
    assert _compute_overlap(13.0, 10.0, 13.0, 1.5) == 1.0
    print("  ✅ test_overlap_edge PASSED")


def test_overlap_tolerance():
    """예측이 tolerance 내"""
    v = _compute_overlap(14.0, 10.0, 13.0, 1.5)
    assert 0 < v < 1
    print("  ✅ test_overlap_tolerance PASSED")


def test_overlap_outside():
    """예측이 tolerance 밖"""
    assert _compute_overlap(20.0, 10.0, 13.0, 1.5) < 0
    print("  ✅ test_overlap_outside PASSED")


def test_empty_file_in_predictions():
    """prediction에 file이 없을 때도 annotation과 매칭되어야 한다 (단일 파일 분석)"""
    anns = [{"file": "test.wav", "t_start": 10, "t_end": 13, "species": "sp1"}]
    preds = [{"file": "", "time": 11.5, "species": "sp1", "composite": 0.8}]
    results, summary = match_predictions_to_annotations(anns, preds)
    assert sum(1 for r in results if r.category == "TP") == 1
    assert sum(1 for r in results if r.category == "FP") == 0
    assert sum(1 for r in results if r.category == "FN") == 0
    print("  ✅ test_empty_file_in_predictions PASSED")


def test_empty_file_in_annotations():
    """annotation에 file이 없을 때도 prediction과 매칭되어야 한다"""
    anns = [{"file": "", "t_start": 10, "t_end": 13, "species": "sp1"}]
    preds = [{"file": "test.wav", "time": 11.5, "species": "sp1", "composite": 0.8}]
    results, summary = match_predictions_to_annotations(anns, preds)
    assert sum(1 for r in results if r.category == "TP") == 1
    assert sum(1 for r in results if r.category == "FP") == 0
    assert sum(1 for r in results if r.category == "FN") == 0
    print("  ✅ test_empty_file_in_annotations PASSED")


# ── 실행 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  매칭 엔진 테스트")
    print("=" * 50)

    test_perfect_match()
    test_false_positive()
    test_false_negative()
    test_time_tolerance()
    test_time_tolerance_exceeded()
    test_species_mismatch()
    test_one_to_one_matching()
    test_multi_file()
    test_overlap_inside()
    test_overlap_edge()
    test_overlap_tolerance()
    test_overlap_outside()
    test_empty_file_in_predictions()
    test_empty_file_in_annotations()

    print()
    print("  🎉 모든 매칭 엔진 테스트 통과!")

