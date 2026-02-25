# ============================================================
# parallel_runner.py — R 분석 병렬 실행 워커
# ============================================================
"""
ProcessPoolExecutor 워커는 pickle 가능해야 하므로 tkinter와 완전 분리된
순수 함수로 작성. 배치 분석 시 파일 단위 병렬화에 사용.
"""

import json
import subprocess
import traceback
from pathlib import Path
from typing import Optional

from audio.sanitizer import ensure_wav


def run_single_analysis(
    rscript_path: str,
    r_script: str,
    audio_file: str,
    species_data: list,
    global_weights: dict,
    output_dir: str,
    extra_config: Optional[dict] = None,
    timeout: int = 600,
) -> dict:
    """
    단일 파일에 대한 R 분석 실행 (워커 프로세스에서 호출).

    Args:
        rscript_path: Rscript 실행 파일 경로
        r_script: R 분석 스크립트 경로
        audio_file: 분석 대상 음원 파일 경로
        species_data: 종별 설정 리스트 (순수 dict, tk 변수 없음)
        global_weights: 전역 가중치 dict
        output_dir: 결과 출력 디렉토리
        extra_config: 추가 config 옵션 (staged_eval 등)
        timeout: R 실행 타임아웃 (초)

    Returns:
        {
            "audio_file": str,
            "basename": str,
            "output_dir": str,
            "returncode": int,
            "stdout": str,
            "stderr": str,
            "results_csv": str | None,
            "preprocess_log": list[str],
            "error": str | None,
        }
    """
    result = {
        "audio_file": audio_file,
        "basename": Path(audio_file).name,
        "output_dir": output_dir,
        "returncode": -1,
        "stdout": "",
        "stderr": "",
        "results_csv": None,
        "preprocess_log": [],
        "error": None,
    }

    try:
        sub_dir = Path(output_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)

        # 1. 전처리
        main_wav, main_log = ensure_wav(audio_file, sub_dir)
        result["preprocess_log"].append(main_log)

        # 2. 템플릿 전처리
        sp_data_copy = []
        for sp in species_data:
            sp_copy = {k: v for k, v in sp.items() if k != "templates"}
            sp_copy["templates"] = []
            for tmpl in sp["templates"]:
                tmpl_copy = dict(tmpl)
                try:
                    sp_wav, sp_log = ensure_wav(tmpl["wav_path"], sub_dir)
                    tmpl_copy["wav_path"] = sp_wav
                    result["preprocess_log"].append(sp_log)
                except Exception as e:
                    result["preprocess_log"].append(
                        f"⚠ {sp.get('name','?')}/{tmpl.get('label','?')} 전처리 실패: {e}"
                    )
                sp_copy["templates"].append(tmpl_copy)
            sp_data_copy.append(sp_copy)

        # 3. config JSON
        config = {
            "main_wav": main_wav,
            "output_dir": str(sub_dir),
            "weights": global_weights,
            "species": sp_data_copy,
            **(extra_config or {}),
        }
        config_path = sub_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 4. 디버깅 로그 저장
        result["preprocess_log"].append(
            f"config: {config_path}, main_wav: {main_wav}"
        )

        # 5. Rscript 실행
        proc = subprocess.run(
            [rscript_path, "--encoding=UTF-8", r_script, str(config_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr

        # 디버그 로그 저장
        log_path = sub_dir / "debug_log.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== R 실행 결과 (코드: {proc.returncode}) ===\n\n")
                f.write(f"--- stdout ---\n{proc.stdout or '(없음)'}\n\n")
                f.write(f"--- stderr ---\n{proc.stderr or '(없음)'}\n")
        except Exception:
            pass

        if proc.returncode == 0:
            csv_path = sub_dir / "results.csv"
            if csv_path.exists():
                result["results_csv"] = str(csv_path)

    except subprocess.TimeoutExpired:
        result["error"] = f"분석 시간이 {timeout}초를 초과했습니다."
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return result
