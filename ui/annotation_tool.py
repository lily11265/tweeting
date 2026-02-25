# ============================================================
# ui/annotation_tool.py — 스펙트로그램 기반 Annotation 도구
# TemplateSelector의 렌더링 엔진을 재활용하여
# 사용자가 조류 음성 구간을 시각적으로 태깅
# ============================================================

import os
import csv
import time
import wave
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path

# numpy / scipy
try:
    import numpy as np
    from scipy.signal import spectrogram as scipy_spectrogram
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Pillow
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# 오디오 재생
from audio.playback import (
    play_wav_async, play_numpy_async, stop_playback as _stop_audio,
    prepare_playback_wav, HAS_PLAYBACK,
)

# 오디오 필터 (밴드패스, 폴리곤 마스킹)
from audio.audio_filter import (
    prepare_filtered_wav, prepare_polygon_wav,
    prepare_filtered_pcm, prepare_polygon_pcm,
)

from colormaps import COLORMAPS


# ── 색상 상수 ─────────────────────────────────────────────

ANN_COLORS = [
    ("#4CAF50", "#A5D6A7"),   # 초록
    ("#2196F3", "#90CAF9"),   # 파랑
    ("#FF9800", "#FFCC80"),   # 주황
    ("#E91E63", "#F48FB1"),   # 분홍
    ("#9C27B0", "#CE93D8"),   # 보라
    ("#00BCD4", "#80DEEA"),   # 청록
    ("#FF5722", "#FFAB91"),   # 빨강
    ("#795548", "#BCAAA4"),   # 갈색
]


class Annotation:
    """단일 annotation 데이터"""
    __slots__ = ("file", "t_start", "t_end", "f_low", "f_high", "species")

    def __init__(self, file, t_start, t_end, f_low, f_high, species):
        self.file = file
        self.t_start = t_start
        self.t_end = t_end
        self.f_low = f_low
        self.f_high = f_high
        self.species = species

    def to_dict(self):
        return {
            "file": self.file,
            "t_start": round(self.t_start, 4),
            "t_end": round(self.t_end, 4),
            "f_low": round(self.f_low, 1),
            "f_high": round(self.f_high, 1),
            "species": self.species,
        }


class AnnotationTool:
    """
    스펙트로그램 위에서 드래그로 구간을 지정하고
    종명을 입력하여 annotation CSV를 생성하는 도구.
    """

    RENDER_W = 1200
    RENDER_H = 600

    def __init__(self, parent, wav_path, species_list=None, callback=None):
        self.parent = parent
        self.species_list = species_list or []
        self.callback = callback

        # ── 다중 파일 지원 ──
        # wav_path가 리스트이면 다중 파일, 문자열이면 단일 파일
        if isinstance(wav_path, (list, tuple)):
            self._wav_files = [(str(p), Path(p).name) for p in wav_path]
        else:
            self._wav_files = [(str(wav_path), Path(wav_path).name)]
        self._multi_file = len(self._wav_files) > 1
        self._current_file_idx = 0
        self._file_cache = {}  # {idx: (sr, data, duration, max_freq)}
        self._file_tab_btns = []  # 파일 탭 버튼들

        # 현재 파일 설정
        self.wav_path = self._wav_files[0][0]
        self.wav_name = self._wav_files[0][1]

        self.annotations: list[Annotation] = []

        # WAV 데이터
        self.sr = None
        self.data = None
        self.duration = 0.0
        self.max_freq = 22050.0

        # 뷰 범위
        self.t_start = 0.0
        self.t_end = 1.0
        self.f_low = 0.0
        self.f_high = 22050.0

        # 렌더링 상태
        self._loaded = False
        self._rendering = False
        self._render_gen = 0
        self._render_after_id = None
        self._rendered_view = None  # (t_start, t_end, f_low, f_high) — 이미지 렌더 시점
        self._tk_img = None
        self._img_id = None  # 캔버스 이미지 아이템 ID

        # 선택 드래그 상태
        self._sel_start = None        # 좌클릭 드래그 캔버스 좌표 (px)
        self._sel_start_data = None   # 좌클릭 드래그 데이터 좌표 (t, f)
        self._sel_rect_id = None

        # 팬 상태
        self._pan_start = None
        self._pan_view = None
        self._pan_img_origin = None  # 드래그 시작 시 이미지 위치

        # 캔버스 annotation 아이템
        self._ann_canvas_ids = []

        # 종별 색상
        self._species_color_map = {}

        # ── 재생 상태 ──
        self._playing = False
        self._play_thread = None
        self._playhead_id = None
        self._playhead_after_id = None
        self._play_speed = 1.0
        self._stop_event = threading.Event()
        self._play_start_wall = 0.0
        self._play_start_time = 0.0
        self._play_end_time = 0.0
        self._play_temp_wav = None
        self._play_generation = 0

        # ── 박스/폴리곤 필터 재생 상태 ──
        self._filter_box_start = None    # Shift+드래그 박스
        self._filter_box_rect = None
        self._poly_points = []           # Ctrl+클릭 폴리곤
        self._poly_canvas_ids = []
        self._poly_snap_dist = 15

        # 창 빌드
        self._build_window()
        self._load_wav()

    def _build_window(self):
        self.win = tk.Toplevel(self.parent)
        title = f"🖊 Annotation 도구 — {self.wav_name}"
        if self._multi_file:
            title = f"🖊 Annotation 도구 — {len(self._wav_files)}개 파일"
        self.win.title(title)
        self.win.geometry("1300x800")
        self.win.minsize(900, 550)
        self.win.transient(self.parent)

        # ── 파일 탭 바 (다중 파일 시) ──
        if self._multi_file:
            file_bar = ttk.Frame(self.win)
            file_bar.pack(fill="x", padx=5, pady=(5, 0))
            ttk.Label(file_bar, text="📁 파일:", font=("Arial", 9, "bold")).pack(side="left")
            self._file_tab_btns = []
            for i, (_, fname) in enumerate(self._wav_files):
                # 파일명이 너무 길면 축약
                display = fname if len(fname) <= 25 else fname[:22] + "..."
                btn = tk.Button(
                    file_bar, text=display, relief="sunken" if i == 0 else "raised",
                    padx=6, pady=2, font=("Arial", 8),
                    command=lambda idx=i: self._switch_file(idx),
                )
                btn.pack(side="left", padx=1)
                self._file_tab_btns.append(btn)

        # ── 상단: 종명 선택 + 도구 ──
        top_bar = ttk.Frame(self.win)
        top_bar.pack(fill="x", padx=5, pady=5)

        ttk.Label(top_bar, text="종명:", font=("Arial", 10, "bold")).pack(side="left")
        self._species_var = tk.StringVar()
        self._species_combo = ttk.Combobox(
            top_bar, textvariable=self._species_var,
            values=self.species_list, width=20,
        )
        self._species_combo.pack(side="left", padx=5)
        if self.species_list:
            self._species_combo.set(self.species_list[0])

        ttk.Button(top_bar, text="+ 종 추가",
                   command=self._add_species).pack(side="left", padx=3)

        ttk.Separator(top_bar, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(top_bar, text="좌클릭: 구간지정 | 우클릭: 이동 | Shift+드래그: 📦박스재생 | Ctrl+클릭: ✏폴리곤재생",
                  foreground="gray").pack(side="left")

        ttk.Button(top_bar, text="↺ 전체 보기",
                   command=self._reset_view).pack(side="right", padx=3)

        # ── 재생 컨트롤 바 ──
        play_bar = ttk.Frame(self.win)
        play_bar.pack(fill="x", padx=5, pady=(0, 3))

        self._btn_play = ttk.Button(play_bar, text="▶ 현재 뷰 재생", width=14,
                                     command=self._play_view)
        self._btn_play.pack(side="left", padx=(0, 2))

        self._btn_stop = ttk.Button(play_bar, text="⏹ 정지", width=8,
                                     command=self._stop_playback, state="disabled")
        self._btn_stop.pack(side="left", padx=(0, 8))

        ttk.Separator(play_bar, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        ttk.Label(play_bar, text="속도:").pack(side="left", padx=(4, 2))
        self._speed_var = tk.DoubleVar(value=1.0)
        for spd in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            ttk.Radiobutton(play_bar, text=f"{spd}x", value=spd,
                            variable=self._speed_var,
                            command=lambda s=spd: self._set_speed(s),
                            ).pack(side="left", padx=1)

        ttk.Separator(play_bar, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)

        ttk.Label(play_bar, text="🔊").pack(side="left", padx=(2, 0))
        self._volume_var = tk.IntVar(value=80)
        vol_scale = ttk.Scale(play_bar, from_=0, to=100,
                              variable=self._volume_var,
                              orient="horizontal", length=80)
        vol_scale.pack(side="left", padx=(0, 2))
        self._vol_label = ttk.Label(play_bar, text="80%", width=4)
        self._vol_label.pack(side="left", padx=(0, 4))
        vol_scale.config(command=self._on_volume_change)

        # 재생 상태 라벨
        self._play_status_var = tk.StringVar(value="")
        ttk.Label(play_bar, textvariable=self._play_status_var,
                  foreground="#FF6B6B", font=("Consolas", 8)).pack(side="right", padx=5)

        # ── 메인 영역: 캔버스 + 리스트 ──
        paned = ttk.PanedWindow(self.win, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=5, pady=2)

        # 좌: 스펙트로그램 캔버스
        canvas_frame = ttk.Frame(paned)
        self.canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        paned.add(canvas_frame, weight=3)

        # 우: Annotation 리스트
        list_frame = ttk.LabelFrame(paned, text=" Annotation 목록 ", padding=5)
        paned.add(list_frame, weight=1)

        if self._multi_file:
            cols = ("file", "species", "t_start", "t_end", "freq")
        else:
            cols = ("species", "t_start", "t_end", "freq")
        self._ann_tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                       height=15)
        tree_cols = {
            "file":     ("파일", 80),
            "species":  ("종명", 80),
            "t_start":  ("시작(s)", 70),
            "t_end":    ("끝(s)", 70),
            "freq":     ("주파수(Hz)", 100),
        }
        for col_id in cols:
            heading, width = tree_cols[col_id]
            self._ann_tree.heading(col_id, text=heading)
            self._ann_tree.column(col_id, width=width, anchor="center")
        self._ann_tree.pack(fill="both", expand=True)
        self._ann_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._ann_tree.bind("<Double-1>", self._on_tree_double_click)
        self._ann_tree.bind("<Button-3>", self._show_tree_context_menu)

        tree_btn_frame = ttk.Frame(list_frame)
        tree_btn_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(tree_btn_frame, text="🗑 선택 삭제",
                   command=self._delete_selected).pack(side="left")
        ttk.Button(tree_btn_frame, text="✏ 종명 변경",
                   command=self._edit_selected_species).pack(side="left", padx=3)

        self._ann_count_var = tk.StringVar(value="0건")
        ttk.Label(list_frame, textvariable=self._ann_count_var,
                  font=("Arial", 10, "bold")).pack(anchor="e")

        # ── 하단: 저장/취소 ──
        bottom = ttk.Frame(self.win)
        bottom.pack(fill="x", padx=5, pady=5)

        self._info_var = tk.StringVar(value="WAV 로딩 중...")
        ttk.Label(bottom, textvariable=self._info_var, foreground="gray",
                  font=("Consolas", 9)).pack(side="left")

        ttk.Button(bottom, text="💾 CSV로 저장",
                   command=self._save_csv).pack(side="right", padx=3)
        ttk.Button(bottom, text="📂 기존 CSV 불러오기",
                   command=self._load_csv).pack(side="right", padx=3)
        ttk.Button(bottom, text="✅ 완료 (평가 탭에 전달)",
                   command=self._confirm).pack(side="right", padx=3)

        # 캔버스 이벤트
        self.canvas.bind("<ButtonPress-1>", self._on_sel_start)
        self.canvas.bind("<B1-Motion>", self._on_sel_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_sel_end)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<Configure>", lambda e: self._schedule_render(100))
        self.canvas.bind("<Escape>", lambda e: self._cancel_polygon())

        # 창 닫힐 때 정리
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── WAV 로딩 ──────────────────────────────────────────

    def _load_wav(self):
        threading.Thread(target=self._load_wav_worker, daemon=True).start()

    def _load_wav_worker(self):
        try:
            idx = self._current_file_idx
            wav_path = self._wav_files[idx][0]

            from scipy.io import wavfile
            sr, data = wavfile.read(wav_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype != np.float64:
                if np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.float64) / np.iinfo(data.dtype).max
                else:
                    data = data.astype(np.float64)

            duration = len(data) / sr
            max_freq = sr / 2.0

            # 캐시 저장
            self._file_cache[idx] = (sr, data, duration, max_freq)

            self.sr = sr
            self.data = data
            self.duration = duration
            self.max_freq = max_freq

            self.t_start = 0.0
            self.t_end = self.duration
            self.f_low = 0.0
            self.f_high = self.max_freq
            self._loaded = True

            self.win.after(0, self._on_loaded)
        except Exception as e:
            self.win.after(0, lambda: self._info_var.set(f"로드 오류: {e}"))

    def _on_loaded(self):
        file_info = f"[{self._current_file_idx + 1}/{len(self._wav_files)}] " if self._multi_file else ""
        self._info_var.set(f"{file_info}로드 완료: {self.wav_name} — {self.duration:.1f}초, {self.sr}Hz")
        self._refresh_tree()
        self._schedule_render(0)

    def _switch_file(self, idx):
        """파일 탭 전환"""
        if idx == self._current_file_idx:
            return
        if self._playing:
            self._stop_playback()

        self._current_file_idx = idx
        self.wav_path = self._wav_files[idx][0]
        self.wav_name = self._wav_files[idx][1]

        # 탭 버튼 스타일 갱신
        for i, btn in enumerate(self._file_tab_btns):
            btn.config(relief="sunken" if i == idx else "raised")

        # 캐시에 있으면 바로 전환, 없으면 로드
        if idx in self._file_cache:
            sr, data, duration, max_freq = self._file_cache[idx]
            self.sr = sr
            self.data = data
            self.duration = duration
            self.max_freq = max_freq
            self.t_start = 0.0
            self.t_end = self.duration
            self.f_low = 0.0
            self.f_high = self.max_freq
            self._loaded = True
            self._on_loaded()
        else:
            self._loaded = False
            self._info_var.set(f"파일 로딩 중... {self.wav_name}")
            self._load_wav()

    # ── 렌더링 ────────────────────────────────────────────

    def _schedule_render(self, delay_ms=200):
        if self._render_after_id:
            self.canvas.after_cancel(self._render_after_id)
        self._render_after_id = self.canvas.after(delay_ms, self._render)

    def _render(self):
        self._render_after_id = None
        if not self._loaded or self._rendering:
            return
        self._rendering = True
        self._render_gen += 1
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        params = {
            "t_start": self.t_start, "t_end": self.t_end,
            "f_low": self.f_low, "f_high": self.f_high,
            "cw": min(cw, self.RENDER_W), "ch": min(ch, self.RENDER_H),
            "gen": self._render_gen,
        }
        threading.Thread(target=self._render_worker, args=(params,), daemon=True).start()

    def _render_worker(self, params):
        try:
            t_start, t_end = params["t_start"], params["t_end"]
            f_low, f_high = params["f_low"], params["f_high"]
            cw, ch, gen = params["cw"], params["ch"], params["gen"]

            i_start = max(0, int(t_start * self.sr))
            i_end = min(len(self.data), int(t_end * self.sr))
            segment = self.data[i_start:i_end]
            if len(segment) < 64:
                self.win.after(0, self._on_render_done, None, None, gen)
                return

            # 다운샘플링
            max_samples = cw * 512
            if len(segment) > max_samples:
                from scipy.signal import decimate as _decimate
                step = len(segment) // max_samples
                if step >= 2:
                    segment = _decimate(segment, step)
                    effective_sr = self.sr / step
                else:
                    effective_sr = self.sr
            else:
                effective_sr = self.sr

            # FFT 파라미터
            view_duration = t_end - t_start
            total_ratio = self.duration / max(view_duration, 0.001)
            if total_ratio > 20:
                nperseg = min(2048, len(segment))
            elif total_ratio > 5:
                nperseg = min(1024, len(segment))
            else:
                nperseg = min(512, len(segment))
            noverlap = int(nperseg * 0.75)

            freqs, times, Sxx = scipy_spectrogram(
                segment, fs=effective_sr, nperseg=nperseg,
                noverlap=noverlap, window="hann"
            )
            f_mask = (freqs >= f_low) & (freqs <= f_high)
            Sxx = Sxx[f_mask, :]
            if Sxx.size == 0:
                self.win.after(0, self._on_render_done, None, None, gen)
                return

            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            vmin = np.percentile(Sxx_db, 2)
            vmax = np.percentile(Sxx_db, 99.5)
            if vmax <= vmin:
                vmax = vmin + 1
            normalized = np.clip((Sxx_db - vmin) / (vmax - vmin), 0, 1)

            lut = COLORMAPS.get("Magma", list(COLORMAPS.values())[0])
            indices = (normalized * 255).astype(np.uint8)
            rgb = lut[indices][::-1, :, :]

            pil_img = Image.fromarray(rgb, mode="RGB")
            resample = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
            pil_img = pil_img.resize((cw, ch), resample)

            info = f"시간: {t_start:.2f}~{t_end:.2f}s  |  주파수: {f_low:.0f}~{f_high:.0f}Hz"
            self.win.after(0, self._on_render_done, pil_img, info, gen)
        except Exception as e:
            self.win.after(0, self._on_render_done, None, f"렌더링 오류: {e}", gen)

    def _on_render_done(self, pil_img, info, gen):
        self._rendering = False
        if gen != self._render_gen:
            return
        if pil_img:
            self._tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("specimg")
            self._img_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self._tk_img, tags="specimg"
            )
            self._rendered_view = (self.t_start, self.t_end, self.f_low, self.f_high)
            self._redraw_annotations()
        if isinstance(info, str):
            self._info_var.set(info)

    # ── 좌클릭: 구간 선택 ─────────────────────────────────
    # 좌표 변환에 _rendered_view (이미지가 실제 렌더된 범위) 사용

    def _on_sel_start(self, event):
        if not self._loaded:
            return

        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004)

        # Ctrl+클릭 → 폴리곤 점 추가
        if ctrl:
            self._on_polygon_click(event)
            return

        # Shift+드래그 → 박스 필터 재생
        if shift:
            self._filter_box_start = (event.x, event.y)
            self._clear_filter_box()
            return

        # 일반 좌클릭: annotation 생성
        self._sel_start = (event.x, event.y)
        self._sel_start_data = self._px_to_data(event.x, event.y)
        if self._sel_rect_id:
            self.canvas.delete(self._sel_rect_id)
            self._sel_rect_id = None

    def _on_sel_move(self, event):
        # 박스 필터 모드
        if self._filter_box_start:
            self._update_filter_box(event.x, event.y)
            return

        if not self._sel_start or not self._loaded:
            return
        x0, y0 = self._sel_start
        x1, y1 = event.x, event.y
        if self._sel_rect_id:
            self.canvas.coords(self._sel_rect_id, x0, y0, x1, y1)
        else:
            self._sel_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="#FFD600", width=2, dash=(6, 3),
            )

    def _on_sel_end(self, event):
        # 박스 필터 재생 완료
        if self._filter_box_start:
            sx, sy = self._filter_box_start
            self._filter_box_start = None
            t0, f0 = self._px_to_data(min(sx, event.x), min(sy, event.y))
            t1, f1 = self._px_to_data(max(sx, event.x), max(sy, event.y))
            f_low, f_high = min(f0, f1), max(f0, f1)
            t0 = max(0, t0)
            t1 = min(self.duration, t1)
            if t1 - t0 > 0.01 and f_high - f_low > 10:
                self._play_filtered_box(t0, t1, f_low, f_high)
            return

        if not self._sel_start or not self._loaded:
            return
        x0, y0 = self._sel_start
        x1, y1 = event.x, event.y
        self._sel_start = None

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            if self._sel_rect_id:
                self.canvas.delete(self._sel_rect_id)
                self._sel_rect_id = None
            return

        t0_data, f0_data = self._sel_start_data
        t1_data, f1_data = self._px_to_data(x1, y1)

        ta = min(t0_data, t1_data)
        tb = max(t0_data, t1_data)
        fa = min(f0_data, f1_data)
        fb = max(f0_data, f1_data)

        ta = max(0, min(ta, self.duration))
        tb = max(0, min(tb, self.duration))
        fa = max(0, fa)
        fb = min(self.max_freq, fb)

        species = self._species_var.get().strip()
        if not species:
            messagebox.showwarning("종명 필요",
                                   "구간을 추가하려면 먼저 종명을 선택해 주세요.",
                                   parent=self.win)
            if self._sel_rect_id:
                self.canvas.delete(self._sel_rect_id)
                self._sel_rect_id = None
            return

        ann = Annotation(self.wav_name, ta, tb, fa, fb, species)  # 현재 파일명 사용
        self.annotations.append(ann)
        self._refresh_tree()
        self._redraw_annotations()

        if self._sel_rect_id:
            self.canvas.delete(self._sel_rect_id)
            self._sel_rect_id = None

    # ── 박스 필터 오버레이 ──

    def _update_filter_box(self, cx, cy):
        sx, sy = self._filter_box_start
        self._clear_filter_box()
        self._filter_box_rect = self.canvas.create_rectangle(
            sx, sy, cx, cy,
            outline="#00BFFF", width=2, dash=(4, 2),
            fill="#00BFFF", stipple="gray25"
        )

    def _clear_filter_box(self):
        if self._filter_box_rect:
            self.canvas.delete(self._filter_box_rect)
            self._filter_box_rect = None

    # ── 폴리곤 선택 ──

    def _on_polygon_click(self, event):
        """Ctrl+클릭: 폴리곤 꼭짓점 추가"""
        t, f = self._px_to_data(event.x, event.y)

        if len(self._poly_points) >= 3:
            sx, sy = self._data_to_px(*self._poly_points[0])
            dist = ((event.x - sx)**2 + (event.y - sy)**2)**0.5
            if dist < self._poly_snap_dist:
                self._close_polygon()
                return

        self._poly_points.append((t, f))
        self._redraw_polygon()

    def _redraw_polygon(self):
        for cid in self._poly_canvas_ids:
            self.canvas.delete(cid)
        self._poly_canvas_ids.clear()

        if not self._poly_points:
            return

        for i, (t, f) in enumerate(self._poly_points):
            px, py = self._data_to_px(t, f)
            r = 5 if i == 0 else 3
            color = "#FF4444" if i == 0 else "#00FF88"
            cid = self.canvas.create_oval(
                px - r, py - r, px + r, py + r,
                fill=color, outline="white", width=1
            )
            self._poly_canvas_ids.append(cid)

        if len(self._poly_points) >= 2:
            coords = []
            for t, f in self._poly_points:
                px, py = self._data_to_px(t, f)
                coords.extend([px, py])
            cid = self.canvas.create_line(
                *coords, fill="#00FF88", width=2, dash=(4, 2)
            )
            self._poly_canvas_ids.append(cid)

        if len(self._poly_points) >= 3:
            last_px, last_py = self._data_to_px(*self._poly_points[-1])
            first_px, first_py = self._data_to_px(*self._poly_points[0])
            cid = self.canvas.create_line(
                last_px, last_py, first_px, first_py,
                fill="#FF4444", width=1, dash=(2, 4)
            )
            self._poly_canvas_ids.append(cid)

    def _close_polygon(self):
        points = list(self._poly_points)
        self._cancel_polygon()
        if len(points) >= 3:
            self._play_filtered_polygon(points)

    def _cancel_polygon(self):
        self._poly_points.clear()
        for cid in self._poly_canvas_ids:
            self.canvas.delete(cid)
        self._poly_canvas_ids.clear()

    # ── 필터 재생 ──

    def _play_filtered_box(self, t0, t1, f_low, f_high):
        if not self._loaded or self.data is None or not HAS_PLAYBACK:
            return
        self._stop_playback()
        speed = self._play_speed
        volume = self._volume_var.get() / 100.0
        pcm, sr, duration = prepare_filtered_pcm(
            self.data, self.sr, t0, t1, f_low, f_high,
            speed=speed, volume=volume
        )
        if pcm is None:
            return
        self._play_status_var.set(
            f"📦 박스 재생: {t0:.1f}-{t1:.1f}s, {f_low:.0f}-{f_high:.0f}Hz"
        )
        self._start_filtered_playback_pcm(pcm, sr, duration, t0, t1)

    def _play_filtered_polygon(self, points):
        if not self._loaded or self.data is None or not HAS_PLAYBACK:
            return
        self._stop_playback()
        speed = self._play_speed
        volume = self._volume_var.get() / 100.0
        self._play_status_var.set("✏ 폴리곤 필터 처리 중...")
        self.win.update_idletasks()
        pcm, sr, duration = prepare_polygon_pcm(
            self.data, self.sr, points,
            speed=speed, volume=volume
        )
        if pcm is None:
            self._play_status_var.set("")
            return
        times = [p[0] for p in points]
        t0, t1 = min(times), max(times)
        self._play_status_var.set(
            f"✏ 폴리곤 재생: {t0:.1f}-{t1:.1f}s ({len(points)}점)"
        )
        self._start_filtered_playback_pcm(pcm, sr, duration, t0, t1)

    def _start_filtered_playback_pcm(self, pcm_data, sr, duration, t0, t1):
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._playing = True
        self._play_generation += 1
        gen = self._play_generation
        self._play_start_time = t0
        self._play_end_time = t1
        self._play_start_wall = time.time()

        self._btn_play.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._update_playhead()

        def _on_done(error):
            if self._play_generation == gen:
                if error:
                    self.win.after(0, lambda m=error: self._play_status_var.set(f"오류: {m}"))
                self.win.after(0, lambda g=gen: self._on_playback_done(g))

        self._play_thread = play_numpy_async(
            pcm_data, sr, stop_event, duration, on_done=_on_done
        )

    # ── 우클릭: 팬 (즉시 시각 이동) ──────────────────────

    def _on_pan_start(self, event):
        if not self._loaded:
            return
        self._pan_start = (event.x, event.y)
        self._pan_view = (self.t_start, self.t_end, self.f_low, self.f_high)
        # 드래그 시작 시 캔버스 이미지 좌표 저장
        self._pan_img_origin = (0, 0)
        if self._img_id:
            coords = self.canvas.coords(self._img_id)
            if coords:
                self._pan_img_origin = (coords[0], coords[1])

    def _on_pan_move(self, event):
        if not self._pan_start or not self._loaded:
            return

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)

        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]

        orig = self._pan_view
        dt = orig[1] - orig[0]
        df = orig[3] - orig[2]

        # 왼쪽 드래그 → 시간 전진, 위쪽 드래그 → 주파수 상승
        t_shift = -dx / cw * dt
        f_shift = dy / ch * df

        self.t_start = orig[0] + t_shift
        self.t_end = orig[1] + t_shift
        self.f_low = orig[2] + f_shift
        self.f_high = orig[3] + f_shift
        self._clamp_view()

        # 즉시 캔버스 이미지를 픽셀 단위로 이동 (FFT 재계산 없이)
        if self._img_id:
            actual_t_shift = self.t_start - orig[0]
            actual_f_shift = self.f_low - orig[2]
            px_dx = -actual_t_shift / max(dt, 0.001) * cw
            px_dy = actual_f_shift / max(df, 1) * ch
            ox, oy = self._pan_img_origin
            self.canvas.coords(self._img_id, ox + px_dx, oy + px_dy)

        # annotation 즉시 갱신
        self._redraw_annotations()

        # 정밀 렌더링 디바운스
        self._schedule_render(250)

    def _on_pan_end(self, event):
        self._pan_start = None
        self._pan_view = None
        # 드래그 종료 시 즉시 정밀 렌더링
        self._schedule_render(0)

    # ── 휠: 줌 ────────────────────────────────────────────

    def _on_wheel(self, event):
        if not self._loaded:
            return
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 1.3
        else:
            factor = 0.75
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        rx = event.x / cw
        ry = 1.0 - event.y / ch

        t_cursor = self.t_start + rx * (self.t_end - self.t_start)
        f_cursor = self.f_low + ry * (self.f_high - self.f_low)

        new_dt = (self.t_end - self.t_start) * factor
        new_df = (self.f_high - self.f_low) * factor

        self.t_start = t_cursor - rx * new_dt
        self.t_end = t_cursor + (1 - rx) * new_dt
        self.f_low = f_cursor - ry * new_df
        self.f_high = f_cursor + (1 - ry) * new_df
        self._clamp_view()

        # annotation 즉시 갱신 (현재 뷰 기준)
        self._redraw_annotations()
        self._schedule_render(100)

    def _clamp_view(self):
        # 최소 뷰 크기 한도
        min_dt = 0.05
        min_df = 50
        if (self.t_end - self.t_start) < min_dt:
            mid = (self.t_start + self.t_end) / 2
            self.t_start = mid - min_dt / 2
            self.t_end = mid + min_dt / 2
        if (self.f_high - self.f_low) < min_df:
            mid = (self.f_low + self.f_high) / 2
            self.f_low = mid - min_df / 2
            self.f_high = mid + min_df / 2

        if self.t_start < 0:
            self.t_end -= self.t_start
            self.t_start = 0
        if self.t_end > self.duration:
            self.t_start -= (self.t_end - self.duration)
            self.t_end = self.duration
        if self.t_start < 0:
            self.t_start = 0
        if self.f_low < 0:
            self.f_high -= self.f_low
            self.f_low = 0
        if self.f_high > self.max_freq:
            self.f_low -= (self.f_high - self.max_freq)
            self.f_high = self.max_freq
        if self.f_low < 0:
            self.f_low = 0

    def _reset_view(self):
        self.t_start = 0.0
        self.t_end = self.duration
        self.f_low = 0.0
        self.f_high = self.max_freq
        self._schedule_render(0)

    # ── 좌표 변환 ─────────────────────────────────────────
    # 항상 현재 뷰 범위 기준으로 변환 (pan/zoom 중에도 정확)

    def _px_to_data(self, px_x, px_y):
        """캔버스 픽셀 → (시간, 주파수) 데이터 좌표"""
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        view_dt = self.t_end - self.t_start
        view_df = self.f_high - self.f_low
        t = self.t_start + (px_x / cw) * view_dt
        f = self.f_high - (px_y / ch) * view_df
        return t, f

    def _data_to_px(self, t, f):
        """(시간, 주파수) 데이터 좌표 → 캔버스 픽셀"""
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        view_dt = self.t_end - self.t_start
        view_df = self.f_high - self.f_low
        if view_dt <= 0 or view_df <= 0:
            return 0, 0
        px_x = (t - self.t_start) / view_dt * cw
        px_y = (1.0 - (f - self.f_low) / view_df) * ch
        return px_x, px_y

    def _range_to_px(self, t0, t1, f0, f1):
        """annotation 범위 → 캔버스 좌표 (left, right, top, bottom)"""
        px_left, px_bottom = self._data_to_px(t0, f0)
        px_right, px_top = self._data_to_px(t1, f1)
        return px_left, px_right, px_top, px_bottom

    # ── Annotation 시각화 ─────────────────────────────────

    def _get_species_color(self, species):
        if species not in self._species_color_map:
            idx = len(self._species_color_map)
            self._species_color_map[species] = ANN_COLORS[idx % len(ANN_COLORS)]
        return self._species_color_map[species]

    def _redraw_annotations(self):
        for cid in self._ann_canvas_ids:
            self.canvas.delete(cid)
        self._ann_canvas_ids.clear()

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        for i, ann in enumerate(self.annotations):
            # 다중 파일: 현재 파일의 annotation만 표시
            if self._multi_file and ann.file != self.wav_name:
                continue

            px_l, px_r, px_t, px_b = self._range_to_px(
                ann.t_start, ann.t_end, ann.f_low, ann.f_high
            )
            # 뷰 안에 있는지 대략 확인
            if px_r < 0 or px_l > cw or px_b < 0 or px_t > ch:
                continue

            outline, fill_light = self._get_species_color(ann.species)

            # 반투명 배경 (fill 사용)
            rect = self.canvas.create_rectangle(
                px_l, px_t, px_r, px_b,
                outline=outline, width=2, fill=fill_light,
                stipple="gray25",
            )
            # 라벨
            label_text = f"{ann.species} [{ann.t_start:.1f}~{ann.t_end:.1f}s]"
            label = self.canvas.create_text(
                px_l + 3, px_t - 3, anchor="sw",
                text=label_text, fill=outline,
                font=("Arial", 8, "bold"),
            )
            self._ann_canvas_ids.extend([rect, label])

    # ── 재생 ──────────────────────────────────────────────

    def _set_speed(self, speed):
        self._play_speed = speed
        if self._playing:
            # 재생 중 속도를 바꾸면: 현재 위치에서 다시 재생
            elapsed = time.time() - self._play_start_wall
            current_time = self._play_start_time + elapsed * self._play_speed
            current_time = min(current_time, self._play_end_time)
            self._stop_event.set()
            self._playing = False
            self._clear_playhead()
            self.win.after(100, lambda: self._start_playback(current_time, self._play_end_time))

    def _on_volume_change(self, val):
        v = int(float(val))
        self._vol_label.config(text=f"{v}%")

    def _play_view(self):
        """현재 보이는 범위를 재생"""
        if not self._loaded or not HAS_PLAYBACK:
            if not HAS_PLAYBACK:
                messagebox.showwarning("경고",
                                       "오디오 재생 라이브러리를 찾을 수 없습니다.\n"
                                       "pip install sounddevice 또는 soundfile",
                                       parent=self.win)
            return
        if self._playing:
            self._stop_playback()
            return
        self._start_playback(self.t_start, self.t_end)

    def _start_playback(self, t0, t1):
        """지정 구간 재생"""
        if not self._loaded or self.sr is None:
            return

        speed = self._play_speed
        sr = self.sr

        # 샘플 추출
        i0 = max(0, int(t0 * sr))
        i1 = min(len(self.data), int(t1 * sr))
        segment = self.data[i0:i1].copy()

        if len(segment) < 64:
            return

        # 속도 변경: 리샘플링
        if abs(speed - 1.0) > 0.01 and HAS_SCIPY:
            from scipy.signal import resample
            new_len = int(len(segment) / speed)
            if new_len < 64:
                new_len = 64
            segment = resample(segment, new_len)

        # float64 → int16 PCM (볼륨 적용)
        max_val = np.max(np.abs(segment))
        if max_val > 0:
            segment = segment / max_val
        volume = self._volume_var.get() / 100.0
        pcm = (segment * 32767 * volume).astype(np.int16)

        self._play_temp_wav = None

        # 세션별 stop event
        stop_event = threading.Event()
        self._stop_event = stop_event

        # 재생 상태 설정
        self._playing = True
        self._play_generation += 1
        gen = self._play_generation
        self._play_start_time = t0
        self._play_end_time = t1
        actual_duration = (t1 - t0) / speed
        self._play_start_wall = time.time()

        # UI 상태 갱신
        self._btn_play.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._play_status_var.set(f"▶ 재생 중 ({speed}x)  {t0:.1f}s ~ {t1:.1f}s")

        # 플레이헤드 업데이트 시작
        self._update_playhead()

        # 백그라운드 재생 (인메모리 numpy 재생)
        def _on_play_done(error):
            is_current = (self._play_generation == gen)
            if error and is_current:
                self.win.after(0, lambda m=error: self._play_status_var.set(f"재생 오류: {m}"))
            if is_current:
                self.win.after(0, lambda g=gen: self._on_playback_done(g))

        self._play_thread = play_numpy_async(
            pcm, sr, stop_event, actual_duration, on_done=_on_play_done
        )

    def _stop_playback(self):
        """재생 중지"""
        self._stop_event.set()
        self._playing = False
        self._clear_playhead()
        self._btn_play.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._play_status_var.set("")

    def _on_playback_done(self, gen=None):
        """재생 완료 시 호출 (메인 스레드)"""
        if gen is not None and gen != self._play_generation:
            return
        if not self._stop_event.is_set():
            self._playing = False
            self._clear_playhead()
            self._btn_play.config(state="normal")
            self._btn_stop.config(state="disabled")
            self._play_status_var.set("✅ 재생 완료")
            self.win.after(3000, lambda: self._play_status_var.set(""))

    def _update_playhead(self):
        """재생 중 플레이헤드 위치를 업데이트"""
        if not self._playing:
            return

        elapsed = time.time() - self._play_start_wall
        current_time = self._play_start_time + elapsed * self._play_speed

        if current_time > self._play_end_time:
            return

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        view_dt = self.t_end - self.t_start

        if view_dt > 0:
            px_x = (current_time - self.t_start) / view_dt * cw
        else:
            px_x = 0

        # 기존 플레이헤드 삭제 + 새로 그리기
        self._clear_playhead()
        if 0 <= px_x <= cw:
            self._playhead_id = self.canvas.create_line(
                px_x, 0, px_x, ch,
                fill="#FF4444", width=2, dash=(4, 2),
            )

        self._playhead_after_id = self.win.after(30, self._update_playhead)

    def _clear_playhead(self):
        if self._playhead_id:
            self.canvas.delete(self._playhead_id)
            self._playhead_id = None
        if self._playhead_after_id:
            self.win.after_cancel(self._playhead_after_id)
            self._playhead_after_id = None

    # ── Annotation 리스트 ─────────────────────────────────

    def _refresh_tree(self):
        for item in self._ann_tree.get_children():
            self._ann_tree.delete(item)
        cur_file_count = 0
        for ann in self.annotations:
            if self._multi_file:
                # 현재 파일과 같으면 볼드 태그 표시
                is_current = (ann.file == self.wav_name)
                # 파일명 축약
                fn = ann.file if len(ann.file) <= 15 else ann.file[:12] + "..."
                vals = (
                    fn,
                    ann.species,
                    f"{ann.t_start:.2f}",
                    f"{ann.t_end:.2f}",
                    f"{ann.f_low:.0f}~{ann.f_high:.0f}",
                )
                tag = ("current",) if is_current else ()
                self._ann_tree.insert("", "end", values=vals, tags=tag)
                if is_current:
                    cur_file_count += 1
            else:
                self._ann_tree.insert("", "end", values=(
                    ann.species,
                    f"{ann.t_start:.2f}",
                    f"{ann.t_end:.2f}",
                    f"{ann.f_low:.0f}~{ann.f_high:.0f}",
                ))
        if self._multi_file:
            self._ann_tree.tag_configure("current", background="#E3F2FD")
            self._ann_count_var.set(f"전체 {len(self.annotations)}건 (현재 파일 {cur_file_count}건)")
        else:
            self._ann_count_var.set(f"{len(self.annotations)}건")

    def _on_tree_select(self, event):
        """선택한 annotation으로 뷰 이동"""
        sel = self._ann_tree.selection()
        if not sel:
            return
        idx = self._ann_tree.index(sel[0])
        if 0 <= idx < len(self.annotations):
            ann = self.annotations[idx]

            # 다중 파일: 다른 파일의 annotation이면 해당 파일로 전환
            if self._multi_file and ann.file != self.wav_name:
                for fi, (_, fname) in enumerate(self._wav_files):
                    if fname == ann.file:
                        self._switch_file(fi)
                        break

            margin_t = max(0.5, (ann.t_end - ann.t_start) * 0.5)
            margin_f = max(500, (ann.f_high - ann.f_low) * 0.5)
            self.t_start = max(0, ann.t_start - margin_t)
            self.t_end = min(self.duration, ann.t_end + margin_t)
            self.f_low = max(0, ann.f_low - margin_f)
            self.f_high = min(self.max_freq, ann.f_high + margin_f)
            self._schedule_render(0)

    def _delete_selected(self):
        sel = self._ann_tree.selection()
        if not sel:
            return
        indices = sorted(
            [self._ann_tree.index(s) for s in sel],
            reverse=True,
        )
        for idx in indices:
            if 0 <= idx < len(self.annotations):
                self.annotations.pop(idx)
        self._refresh_tree()
        self._redraw_annotations()

    # ── 종명 편집 ─────────────────────────────────────────

    def _on_tree_double_click(self, event):
        """트리뷰 더블클릭: 해당 annotation의 종명 변경"""
        item = self._ann_tree.identify_row(event.y)
        if not item:
            return
        idx = self._ann_tree.index(item)
        if 0 <= idx < len(self.annotations):
            self._edit_species(idx)

    def _edit_selected_species(self):
        """버튼: 선택된 annotation의 종명 변경"""
        sel = self._ann_tree.selection()
        if not sel:
            messagebox.showinfo("알림", "종명을 변경할 항목을 선택하세요.", parent=self.win)
            return
        idx = self._ann_tree.index(sel[0])
        if 0 <= idx < len(self.annotations):
            self._edit_species(idx)

    def _edit_species(self, idx):
        """특정 annotation의 종명을 변경"""
        ann = self.annotations[idx]
        old_name = ann.species
        new_name = simpledialog.askstring(
            "종명 변경",
            f"'{old_name}' → 새 종명을 입력하세요:",
            initialvalue=old_name,
            parent=self.win,
        )
        if new_name and new_name.strip() and new_name.strip() != old_name:
            new_name = new_name.strip()
            ann.species = new_name
            # 종 목록에 새 이름 추가
            if new_name not in self.species_list:
                self.species_list.append(new_name)
                self._species_combo["values"] = self.species_list
            self._refresh_tree()
            self._redraw_annotations()

    def _rename_all_species(self, old_name):
        """동일 종명의 모든 annotation을 일괄 변경"""
        count = sum(1 for a in self.annotations if a.species == old_name)
        new_name = simpledialog.askstring(
            "종명 일괄 변경",
            f"'{old_name}' ({count}건) → 새 종명을 입력하세요:",
            initialvalue=old_name,
            parent=self.win,
        )
        if new_name and new_name.strip() and new_name.strip() != old_name:
            new_name = new_name.strip()
            changed = 0
            for a in self.annotations:
                if a.species == old_name:
                    a.species = new_name
                    changed += 1
            # 종 목록 갱신
            if old_name in self.species_list:
                idx = self.species_list.index(old_name)
                self.species_list[idx] = new_name
            elif new_name not in self.species_list:
                self.species_list.append(new_name)
            self._species_combo["values"] = self.species_list
            self._refresh_tree()
            self._redraw_annotations()
            self._info_var.set(f"✅ '{old_name}' → '{new_name}' ({changed}건 변경)")

    def _show_tree_context_menu(self, event):
        """트리뷰 우클릭 컨텍스트 메뉴"""
        item = self._ann_tree.identify_row(event.y)
        if not item:
            return
        # 해당 행 선택
        self._ann_tree.selection_set(item)
        idx = self._ann_tree.index(item)
        if idx < 0 or idx >= len(self.annotations):
            return

        ann = self.annotations[idx]
        species = ann.species
        same_count = sum(1 for a in self.annotations if a.species == species)

        menu = tk.Menu(self.win, tearoff=0)
        menu.add_command(
            label=f"✏ '{species}' 종명 변경",
            command=lambda: self._edit_species(idx),
        )
        if same_count > 1:
            menu.add_command(
                label=f"✏ '{species}' 전체 변경 ({same_count}건)",
                command=lambda s=species: self._rename_all_species(s),
            )
        menu.add_separator()
        menu.add_command(
            label="▶ 이 구간 재생",
            command=lambda: self._start_playback(ann.t_start, ann.t_end),
        )
        menu.add_separator()
        menu.add_command(
            label="🗑 삭제",
            command=self._delete_selected,
        )
        menu.tk_popup(event.x_root, event.y_root)

    # ── 종명 관리 ─────────────────────────────────────────

    def _add_species(self):
        name = simpledialog.askstring("종 추가", "추가할 종명을 입력하세요:", parent=self.win)
        if name and name.strip():
            name = name.strip()
            if name not in self.species_list:
                self.species_list.append(name)
            self._species_combo["values"] = self.species_list
            self._species_var.set(name)

    # ── CSV 저장/불러오기 ─────────────────────────────────

    def _save_csv(self):
        if not self.annotations:
            messagebox.showwarning("경고", "저장할 annotation이 없습니다.",
                                   parent=self.win)
            return
        path = filedialog.asksaveasfilename(
            title="Annotation CSV 저장",
            defaultextension=".csv",
            filetypes=[("CSV 파일", "*.csv")],
            parent=self.win,
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "t_start", "t_end", "f_low", "f_high", "species"]
            )
            writer.writeheader()
            for ann in self.annotations:
                writer.writerow(ann.to_dict())

        messagebox.showinfo("저장 완료",
                            f"Annotation {len(self.annotations)}건 저장:\n{path}",
                            parent=self.win)

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Annotation CSV 불러오기",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")],
            parent=self.win,
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ann = Annotation(
                        file=row.get("file", self.wav_name),
                        t_start=float(row["t_start"]),
                        t_end=float(row["t_end"]),
                        f_low=float(row.get("f_low", 0)),
                        f_high=float(row.get("f_high", self.max_freq)),
                        species=row["species"],
                    )
                    self.annotations.append(ann)

                    # 종명 리스트 갱신
                    if ann.species not in self.species_list:
                        self.species_list.append(ann.species)

            self._species_combo["values"] = self.species_list
            self._refresh_tree()
            self._redraw_annotations()
            self._info_var.set(f"CSV에서 {len(self.annotations)}건 로드됨")
        except Exception as e:
            messagebox.showerror("오류", f"CSV 로드 실패:\n{e}", parent=self.win)

    # ── 완료/콜백/정리 ────────────────────────────────────

    def _confirm(self):
        if not self.annotations:
            messagebox.showwarning("경고", "Annotation이 없습니다.", parent=self.win)
            return

        result = [ann.to_dict() for ann in self.annotations]
        if self.callback:
            self.callback(result)
        self._on_close()

    def _on_close(self):
        """창 닫을 때 정리"""
        if self._playing:
            self._stop_event.set()
            self._playing = False
        self._clear_playhead()
        self.win.destroy()

    def get_annotations(self):
        return [ann.to_dict() for ann in self.annotations]
