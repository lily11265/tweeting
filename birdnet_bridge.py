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
        n_workers=1,
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
    - 개발 모드: 서브프로세스에서 실행 (sys.executable = python)
    - PyInstaller 번들 모드: 인프로세스 실행 (sys.executable = .exe)
    """
    if not HAS_BIRDNET:
        raise RuntimeError(
            "BirdNET이 설치되지 않았습니다.\n"
            "설치: pip install birdnet"
        )

    is_frozen = getattr(sys, 'frozen', False)

    if is_frozen:
        # ── PyInstaller 번들 모드: 인프로세스 실행 ──
        return _run_birdnet_inprocess(
            wav_path, confidence_threshold, lang, progress_callback
        )
    else:
        # ── 개발 모드: 서브프로세스 실행 ──
        return _run_birdnet_subprocess(
            wav_path, confidence_threshold, lang, progress_callback
        )


def _run_birdnet_inprocess(
    wav_path: str,
    confidence_threshold: float,
    lang: str,
    progress_callback: Optional[Callable[[str], None]],
) -> List[dict]:
    """PyInstaller 번들일 때 인프로세스로 BirdNET 실행."""
    import birdnet

    if progress_callback:
        progress_callback("BirdNET 모델 로딩 중...")

    model = birdnet.load("acoustic", "2.4", "tf", lang=lang)

    if progress_callback:
        progress_callback(f"'{os.path.basename(wav_path)}' 분석 중...")

    predictions = model.predict(
        os.path.abspath(wav_path),
        default_confidence_threshold=confidence_threshold,
        n_workers=1,
    )

    if progress_callback:
        progress_callback("결과 변환 중...")

    arr = predictions.to_structured_array()
    filename = os.path.basename(wav_path)

    results = []
    for row in arr:
        species_raw = str(row["species_name"])
        conf = float(row["confidence"])
        species = species_raw.split("_", 1)[1] if "_" in species_raw else species_raw

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

    if progress_callback:
        progress_callback(f"완료: {len(results)}건 검출")

    return results


def _run_birdnet_subprocess(
    wav_path: str,
    confidence_threshold: float,
    lang: str,
    progress_callback: Optional[Callable[[str], None]],
) -> List[dict]:
    """개발 모드: 서브프로세스에서 BirdNET 실행."""

    print(f"[BirdNET] === 시작 ===")
    print(f"[BirdNET] wav_path: {wav_path}")
    print(f"[BirdNET] confidence: {confidence_threshold}, lang: {lang}")
    print(f"[BirdNET] sys.executable: {sys.executable}")

    if progress_callback:
        progress_callback("BirdNET 서브프로세스 시작 중...")

    # 인자를 JSON으로 전달
    args_json = json.dumps({
        "wav_path": os.path.abspath(wav_path),
        "confidence": confidence_threshold,
        "lang": lang,
    }, ensure_ascii=False)

    print(f"[BirdNET] args_json: {args_json}")

    # 워커 스크립트를 임시 파일로 저장
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(_WORKER_SCRIPT)
        worker_path = f.name

    print(f"[BirdNET] worker_path: {worker_path}")

    cmd = [sys.executable, worker_path, args_json]
    print(f"[BirdNET] cmd: {cmd}")

    try:
        # stderr→stdout 통합: TensorFlow 경고가 stderr 파이프 버퍼를
        # 가득 채워 데드락을 유발하는 문제를 방지
        print(f"[BirdNET] Popen 시작...")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        print(f"[BirdNET] Popen 완료, PID={proc.pid}")

        results = None
        diag_lines = []  # non-JSON lines for error diagnosis
        line_count = 0
        print(f"[BirdNET] stdout 읽기 시작...")

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                msg = json.loads(line)
                if "progress" in msg:
                    print(f"[BirdNET] 진행: {msg['progress']}")
                    if progress_callback:
                        progress_callback(msg["progress"])
                elif "result" in msg:
                    results = msg["result"]
                    print(f"[BirdNET] 결과 수신: {len(results)}건")
                else:
                    print(f"[BirdNET] JSON(기타): {line[:200]}")
            except json.JSONDecodeError:
                diag_lines.append(line)
                print(f"[BirdNET] 비-JSON: {line[:200]}")

        print(f"[BirdNET] stdout 읽기 완료 (총 {line_count}줄)")
        print(f"[BirdNET] proc.wait() 호출 중...")

        proc.wait(timeout=600)  # 10-minute timeout

        print(f"[BirdNET] proc.wait() 완료, returncode={proc.returncode}")

        if proc.returncode != 0:
            diag = "\n".join(diag_lines[-30:])  # last 30 lines
            raise RuntimeError(f"BirdNET 프로세스 오류 (코드 {proc.returncode}):\n{diag}")

        if results is None:
            diag = "\n".join(diag_lines[-30:])
            raise RuntimeError(f"BirdNET에서 결과를 받지 못했습니다.\n진단 로그:\n{diag}")

        if progress_callback:
            progress_callback(f"완료: {len(results)}건 검출")

        print(f"[BirdNET] === 완료: {len(results)}건 ===")
        return results

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"[BirdNET] === 타임아웃 ===")
        raise RuntimeError("BirdNET 프로세스가 시간 초과(10분)되었습니다.")

    except Exception as e:
        print(f"[BirdNET] === 예외: {e} ===")
        raise

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
