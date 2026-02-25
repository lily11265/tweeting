# ============================================================
# tweeting - Python tkinter GUI
# 내부 분석: R (monitoR, seewave, tuneR)
# 외부 GUI: Python tkinter
# ============================================================
# 필요: Python 3.x, R 설치 (Rscript 실행 가능해야 함)
# R 패키지: seewave, tuneR, monitoR, jsonlite
# Python 패키지: Pillow, pydub, numpy, scipy, matplotlib
#   pip install Pillow pydub numpy scipy matplotlib
# 시스템: ffmpeg 설치 필요 (MP3 변환용)
# ============================================================

import tkinter as tk
from tkinter import ttk
import os
import sys
import atexit
import shutil
import tempfile
from pathlib import Path

# 선택적 라이브러리 플래그
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    import numpy as np
    from scipy.io import wavfile as scipy_wavfile
    from scipy.signal import spectrogram as scipy_spectrogram
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── 리팩토링된 모듈 임포트 ──
from r_bridge import find_rscript
from ui.analysis_tab import AnalysisTabMixin
from ui.batch_tab import BatchTabMixin
from ui.evaluation_tab import EvaluationTabMixin
from ui.converter_tab import ConverterTabMixin


class BirdSongDetectorApp(AnalysisTabMixin, BatchTabMixin, EvaluationTabMixin, ConverterTabMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("tweeting")
        self.root.geometry("1050x800")
        self.root.minsize(950, 700)

        # 라이브러리 가용성 플래그 (탭 mixin에서 참조)
        self._HAS_PIL = HAS_PIL
        self._HAS_PYDUB = HAS_PYDUB
        self._HAS_SCIPY = HAS_SCIPY

        # R 스크립트 경로 (PyInstaller 번들 시 _MEIPASS 사용)
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            self.script_dir = Path(sys._MEIPASS)
        else:
            self.script_dir = Path(__file__).parent
        self.r_script = self.script_dir / "new_analysis.R"

        # Rscript 실행 파일 경로 자동 탐색
        self.rscript_path = find_rscript()

        # 결과 임시 폴더
        self._created_temp_dirs = []  # 종료 시 정리할 임시 디렉터리 목록
        self.output_dir = Path(tempfile.mkdtemp(prefix="birdsong_"))
        self._created_temp_dirs.append(str(self.output_dir))

        # 종 정보 저장 리스트
        self.species_frames = []

        # 배치 분석용
        self._batch_files = []
        self.batch_species_frames = []
        self._batch_running = False
        self._batch_cancel = False
        self._batch_results = []
        self._batch_wav_map = {}

        self._build_ui()

        # 종료 시 임시 폴더 정리 등록
        atexit.register(self._cleanup_temp_dirs)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ========================================
    # UI 구성
    # ========================================
    def _build_ui(self):
        # 노트북 (탭)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # --- 탭 1: 음성 분석 ---
        tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(tab_analysis, text="  🔍 음성 분석  ")
        self._build_analysis_tab(tab_analysis)

        # --- 탭 2: 배치 분석 ---
        tab_batch = ttk.Frame(self.notebook)
        self.notebook.add(tab_batch, text="  📂 배치 분석  ")
        self._build_batch_tab(tab_batch)

        # --- 탭 3: 자동 튜닝 ---
        tab_autotune = ttk.Frame(self.notebook)
        self.notebook.add(tab_autotune, text="  🎛 자동 튜닝  ")
        self._build_autotune_tab(tab_autotune)

        # --- 탭 4: 성능 평가 ---
        tab_eval = ttk.Frame(self.notebook)
        self.notebook.add(tab_eval, text="  📊 성능 평가  ")
        self._build_evaluation_tab(tab_eval)

        # --- 탭 5: MP3/MP4 → WAV 변환기 ---
        tab_converter = ttk.Frame(self.notebook)
        self.notebook.add(tab_converter, text="  🔄 MP3/MP4 → WAV 변환  ")
        self._build_converter_tab(tab_converter)

    # ========================================
    # 임시 디렉터리 정리
    # ========================================
    def _cleanup_temp_dirs(self):
        """추적된 모든 임시 디렉터리를 삭제한다."""
        for d in self._created_temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        self._created_temp_dirs.clear()

    def _on_close(self):
        """프로그램 종료 시 정리 후 종료."""
        self._cleanup_temp_dirs()
        self.root.destroy()


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # PyInstaller --windowed 모드에서는 sys.stdout/stderr 가 None이므로
    # print() 호출 시 'NoneType' has no attribute 'write' 방지
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    root = tk.Tk()
    app = BirdSongDetectorApp(root)
    root.mainloop()
