# ============================================================
# r_bridge.py — R 연동 (Rscript 탐색 및 번들 FFmpeg 설정)
# ============================================================

import os
import sys
import shutil
from pathlib import Path


def _setup_bundled_ffmpeg():
    """설치 폴더에 번들된 ffmpeg를 PATH에 추가한다."""
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent
    else:
        app_dir = Path(__file__).parent
    ffmpeg_exe = app_dir / "ffmpeg" / "bin" / "ffmpeg.exe"
    if ffmpeg_exe.exists():
        ffmpeg_dir = str(ffmpeg_exe.parent)
        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
        return str(ffmpeg_exe)
    return None


# 모듈 로딩 시 자동 실행
bundled_ffmpeg = _setup_bundled_ffmpeg()


def find_rscript():
    """시스템에서 Rscript.exe를 자동으로 찾는다."""
    # 0) 번들된 Portable R (인스톨러 배포 시 최우선)
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent
    else:
        app_dir = Path(__file__).parent
    portable = app_dir / "R-Portable" / "bin" / "Rscript.exe"
    if portable.exists():
        # Portable R의 라이브러리 경로도 설정
        r_libs = app_dir / "R-Portable" / "library"
        if r_libs.exists():
            r_libs_str = str(r_libs)
            os.environ["R_LIBS_USER"] = r_libs_str
            os.environ["R_LIBS"] = r_libs_str
            os.environ["R_LIBS_SITE"] = r_libs_str
        return str(portable)

    # 1) PATH에 있으면 바로 사용
    rscript_in_path = shutil.which("Rscript")
    if rscript_in_path:
        return rscript_in_path

    if sys.platform == "win32":
        # 2) 윈도우 레지스트리에서 R 설치 경로 확인
        try:
            import winreg
            for hkey in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    key = winreg.OpenKey(hkey, r"SOFTWARE\R-core\R")
                    install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                    winreg.CloseKey(key)
                    candidate = Path(install_path) / "bin" / "Rscript.exe"
                    if candidate.exists():
                        return str(candidate)
                except (FileNotFoundError, OSError):
                    pass
        except ImportError:
            pass

        # 3) 일반적인 설치 경로 탐색
        for base in [Path(r"C:\Program Files\R"), Path(r"C:\Program Files (x86)\R")]:
            if base.exists():
                # 가장 최신 버전 폴더를 우선 사용
                versions = sorted(base.iterdir(), reverse=True)
                for ver_dir in versions:
                    candidate = ver_dir / "bin" / "Rscript.exe"
                    if candidate.exists():
                        return str(candidate)

    return None  # 찾지 못함


def export_r_spectrogram(
    rscript_path,
    r_script_path,
    wav_path,
    output_path,
    t_start=None,
    t_end=None,
    f_low=None,
    f_high=None,
    width=1600,
    height=800,
    wl=512,
    ovlp=75,
    collevels=30,
    palette="spectro.colors",
    detections=None,
    timeout=300,
    dB_min=-60,
    dB_max=0,
    res=150,
    show_title=True,
    show_scale=True,
    show_osc=False,
    show_detections=True,
    det_cex=0.7,
):
    """R seewave::spectro()로 연구용 스펙트로그램 PNG를 생성한다.

    Args:
        rscript_path: Rscript.exe 경로
        r_script_path: new_analysis.R 경로
        wav_path: 대상 WAV 파일 경로
        output_path: 저장할 PNG 파일 경로
        t_start/t_end: 시간 범위 (초, None이면 전체)
        f_low/f_high: 주파수 범위 (Hz, None이면 전체)
        width/height: 이미지 크기 (px)
        wl: FFT 윈도우 길이
        ovlp: 오버랩 %
        collevels: dB 레벨 수
        palette: 팔레트 이름
        detections: 검출 결과 [{species, time, score}, ...]
        timeout: R 실행 제한 시간 (초)
        dB_min/dB_max: dB 범위 (밝기/대비)
        res: DPI
        show_title: 제목 표시
        show_scale: 스케일바 표시
        show_osc: 오실로그램 표시
        show_detections: 검출 오버레이 표시
        det_cex: 검출 라벨 크기

    Returns:
        str: 생성된 PNG 파일 경로

    Raises:
        RuntimeError: R 실행 실패 시
    """
    import json
    import subprocess
    import tempfile

    config = {
        "mode": "spectrogram",
        "wav_path": str(wav_path),
        "output_path": str(output_path),
        "output_dir": str(Path(output_path).parent),
        "width": width,
        "height": height,
        "wl": wl,
        "ovlp": ovlp,
        "collevels": collevels,
        "palette": palette,
        "dB_min": dB_min,
        "dB_max": dB_max,
        "res": res,
        "show_title": show_title,
        "show_scale": show_scale,
        "show_osc": show_osc,
        "show_detections": show_detections,
        "det_cex": det_cex,
    }

    if t_start is not None:
        config["t_start"] = t_start
    if t_end is not None:
        config["t_end"] = t_end
    if f_low is not None:
        config["f_low"] = f_low
    if f_high is not None:
        config["f_high"] = f_high
    if detections and show_detections:
        config["detections"] = detections

    # 임시 config JSON 생성
    config_dir = Path(output_path).parent
    config_path = config_dir / "_spectro_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    try:
        result = subprocess.run(
            [rscript_path, "--encoding=UTF-8", str(r_script_path), str(config_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"R 스펙트로그램 생성 실패:\n{error_msg}")

        if not Path(output_path).exists():
            raise RuntimeError(
                f"R 실행은 완료되었으나 출력 파일이 없습니다: {output_path}\n"
                f"R stdout: {result.stdout}"
            )

        return str(output_path)

    finally:
        # 임시 config 삭제
        try:
            config_path.unlink(missing_ok=True)
        except Exception:
            pass

