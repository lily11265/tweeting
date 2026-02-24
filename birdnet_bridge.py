# ============================================================
# birdnet_bridge.py — BirdNET 딥러닝 모델 브릿지
# BirdNET 예측 결과를 annotation 형식으로 변환
# ============================================================

import os
import sys
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Callable

# BirdNET 선택적 임포트 확인
try:
    import birdnet  # noqa: F401
    HAS_BIRDNET = True
except ImportError:
    HAS_BIRDNET = False


def check_birdnet_available() -> bool:
    """BirdNET 사용 가능 여부 확인"""
    return HAS_BIRDNET


# ── 서브프로세스 워커 스크립트 ──
# Windows에서 BirdNET은 multiprocessing을 사용하므로
# __main__ 가드가 있는 별도 프로세스에서 실행해야 한다.
_WORKER_SCRIPT = '''
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def main():
    import birdnet
    args = json.loads(sys.argv[1])
    wav_path = args["wav_path"]
    confidence = args["confidence"]
    lang = args["lang"]

    print(json.dumps({"progress": "BirdNET 모델 로딩 중..."}), flush=True)
    model = birdnet.load("acoustic", "2.4", "tf", lang=lang)

    print(json.dumps({"progress": f"'{os.path.basename(wav_path)}' 분석 중..."}), flush=True)
    predictions = model.predict(
        wav_path,
        default_confidence_threshold=confidence,
    )

    print(json.dumps({"progress": "결과 변환 중..."}), flush=True)
    arr = predictions.to_structured_array()
    filename = os.path.basename(wav_path)

    results = []
    for row in arr:
        species_raw = str(row["species_name"])
        conf = float(row["confidence"])

        # 종명 파싱: "Otus scops_소쩍새" → "소쩍새"
        species = species_raw.split("_", 1)[1] if "_" in species_raw else species_raw

        # 시간 파싱
        t_start = _parse_time(str(row["start_time"]))
        t_end = _parse_time(str(row["end_time"]))

        results.append({
            "file": filename,
            "t_start": round(t_start, 4),
            "t_end": round(t_end, 4),
            "f_low": 0,
            "f_high": 24000,
            "species": species,
            "confidence": round(conf, 4),
        })

    print(json.dumps({"result": results}), flush=True)


def _parse_time(time_str):
    try:
        return float(time_str)
    except ValueError:
        pass
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0


if __name__ == "__main__":
    main()
'''


def run_birdnet_prediction(
    wav_path: str,
    confidence_threshold: float = 0.5,
    lang: str = "ko",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """
    BirdNET으로 WAV 파일을 분석하여 annotation 형식으로 반환.
    Windows 호환을 위해 서브프로세스에서 실행.

    Args:
        wav_path: WAV 파일 경로
        confidence_threshold: 최소 신뢰도 임계값 (기본 0.5)
        lang: 종명 언어 (기본 "ko" 한국어)
        progress_callback: 진행 상태 콜백 (메시지 문자열)

    Returns:
        [{
            "file": "파일명.wav",
            "t_start": float,
            "t_end": float,
            "f_low": 0,
            "f_high": 24000,
            "species": "종명",
            "confidence": float,
        }, ...]
    """
    if not HAS_BIRDNET:
        raise RuntimeError(
            "BirdNET이 설치되지 않았습니다.\n"
            "설치: pip install birdnet"
        )

    if progress_callback:
        progress_callback("BirdNET 서브프로세스 시작 중...")

    # 인자를 JSON으로 전달
    args_json = json.dumps({
        "wav_path": os.path.abspath(wav_path),
        "confidence": confidence_threshold,
        "lang": lang,
    }, ensure_ascii=False)

    # 워커 스크립트를 임시 파일로 저장
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(_WORKER_SCRIPT)
        worker_path = f.name

    try:
        proc = subprocess.Popen(
            [sys.executable, worker_path, args_json],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        results = None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if "progress" in msg and progress_callback:
                    progress_callback(msg["progress"])
                elif "result" in msg:
                    results = msg["result"]
            except json.JSONDecodeError:
                pass

        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(f"BirdNET 프로세스 오류 (코드 {proc.returncode}):\n{stderr}")

        if results is None:
            raise RuntimeError("BirdNET에서 결과를 받지 못했습니다.")

        if progress_callback:
            progress_callback(f"완료: {len(results)}건 검출")

        return results

    finally:
        try:
            os.unlink(worker_path)
        except OSError:
            pass


def run_birdnet_batch(
    wav_paths: List[str],
    confidence_threshold: float = 0.5,
    lang: str = "ko",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """
    여러 WAV 파일에 대해 BirdNET 예측을 실행.
    """
    all_annotations = []
    for i, wav_path in enumerate(wav_paths, 1):
        if progress_callback:
            progress_callback(
                f"[{i}/{len(wav_paths)}] {os.path.basename(wav_path)} 분석 중..."
            )
        anns = run_birdnet_prediction(
            wav_path,
            confidence_threshold=confidence_threshold,
            lang=lang,
            progress_callback=None,
        )
        all_annotations.extend(anns)

    if progress_callback:
        progress_callback(f"전체 완료: {len(all_annotations)}건")

    return all_annotations


# ── 유틸리티 (테스트용 직접 접근) ──

def _parse_species_name(raw: str) -> str:
    """BirdNET 종명 파싱: "Otus scops_소쩍새" → "소쩍새" """
    if "_" in raw:
        return raw.split("_", 1)[1]
    return raw


def _parse_time(time_str: str) -> float:
    """시간 문자열 → 초: "00:01:30.50" → 90.5"""
    try:
        return float(time_str)
    except ValueError:
        pass
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0
