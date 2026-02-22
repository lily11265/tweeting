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
            os.environ["R_LIBS_USER"] = str(r_libs)
            os.environ["R_LIBS"] = str(r_libs)
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
