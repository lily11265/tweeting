# ============================================================
# ui/template_selector.py — 스펙트로그램 기반 템플릿 구간 선택기
# ============================================================

import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import wave
import tempfile
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

# 오디오 필터
from audio.audio_filter import (
    prepare_filtered_wav, prepare_polygon_wav,
    prepare_filtered_pcm, prepare_polygon_pcm,
)

# 모듈 내 참조
from colormaps import COLORMAPS


class _TabState:
    """탭 하나의 렌더링/선택 상태를 보관하는 내부 클래스."""
    def __init__(self, wav_path, display_name):
        self.wav_path = wav_path
        self.display_name = display_name

        # WAV 데이터
        self.sr = None
        self.data = None
        self.duration = 0
        self.max_freq = 22050
        self.loaded = False

        # 뷰 범위
        self.t_start = 0.0
        self.t_end = 1.0
        self.f_low = 0.0
        self.f_high = 22050

        # 렌더 상태
        self.rendered_view = None
        self.render_after_id = None
        self.rendering = False
        self.render_gen = 0
        self.tk_img = None

        # 캔버스 (UI 빌드 시 설정)
        self.canvas = None

        # 선택 (좌클릭 드래그) 상태 — 단일 선택용
        self.sel_start = None
        self.sel_rect_id = None
        self.sel_info_id = None
        self.selection = None  # (t0, t1, f0, f1)

        # 다중 선택 상태
        self.selections = []         # [(t0, t1, f0, f1), ...]
        self.sel_canvas_ids = []     # [canvas_item_id, ...]

        # 팬 상태
        self.pan_start = None
        self.pan_view = None

        # ── 박스/폴리곤 필터 재생 상태 ──
        self.filter_box_start = None
        self.filter_box_rect = None
        self.poly_points = []
        self.poly_canvas_ids = []
        self.poly_snap_dist = 15


class TemplateSelector:
    """종 음원의 스펙트로그램을 표시하고 드래그로 시간/주파수 구간을 선택."""

    RENDER_W = 1200
    RENDER_H = 600

    # 다중 선택 시 색상 팔레트
    MULTI_COLORS = ["#FF4444", "#44DD44", "#4488FF", "#FFAA00", "#FF44FF", "#44DDDD"]

    def __init__(self, parent, wav_path, callback, multi_select=False):
        """
        parent: 부모 위젯
        wav_path: WAV 파일 경로 (str) 또는
                  [(wav_path, display_name), ...] 리스트 (탭 모드)
        callback: 선택 완료 시 호출
            multi_select=False: callback(t_start, t_end, f_low, f_high)
            multi_select=True (단일파일):  callback([(t_start, t_end, f_low, f_high), ...])
            multi_select=True (탭모드):    callback([(t_start, t_end, f_low, f_high, file_path), ...])
        multi_select: True이면 여러 구간을 선택 가능
        """
        self.callback = callback
        self.multi_select = multi_select

        # wav_path를 정규화: 항상 [(path, name), ...] 리스트로
        if isinstance(wav_path, (str, Path)):
            self._wav_files = [(str(wav_path), Path(wav_path).name)]
        elif isinstance(wav_path, (list, tuple)):
            self._wav_files = [(str(p), n) for p, n in wav_path]
        else:
            self._wav_files = [(str(wav_path), str(wav_path))]

        self._tabbed = len(self._wav_files) > 1

        # 탭별 상태 생성
        self._tabs: list[_TabState] = []
        for wp, dn in self._wav_files:
            self._tabs.append(_TabState(wp, dn))

        self._active_tab_idx = 0

        # 창 설정
        self.win = tk.Toplevel(parent)
        if self._tabbed:
            title = "📊 다중 구간 선택 (탭 전환 가능)" if multi_select else "📊 구간 선택"
        else:
            title = "📊 다중 구간 선택" if multi_select else "📊 구간 선택"
            title += f" — {self._wav_files[0][1]}"
        self.win.title(title)
        self.win.geometry("1100x700")
        self.win.transient(parent)
        self.win.grab_set()

        self._build_ui()

        # 모든 탭의 WAV 로드 시작
        for tab in self._tabs:
            threading.Thread(target=self._load_wav, args=(tab,), daemon=True).start()

    # ============================================================
    # UI 빌드
    # ============================================================
    def _build_ui(self):
        # 상단 정보바
        top = ttk.Frame(self.win)
        top.pack(fill="x", padx=5, pady=3)
        hint = "좌클릭: 구간선택 | 우클릭: 이동 | 휠: 줌 | Shift+드래그: 📦박스재생 | Ctrl+클릭: ✏폴리곤재생"
        if self.multi_select:
            hint += "  |  여러 구간 드래그 가능"
        if self._tabbed:
            hint += "  |  탭 위에서 음원 전환"
        ttk.Label(top, text=hint, foreground="gray").pack(side="left")
        ttk.Button(top, text="↺ 전체 보기", command=self._reset_view).pack(side="right", padx=5)

        # ── 재생 컨트롤 바 ──
        play_bar = ttk.Frame(self.win)
        play_bar.pack(fill="x", padx=5, pady=(0, 2))

        self._btn_play = ttk.Button(play_bar, text="▶ 현재 뷰 재생", width=14,
                                     command=self._play_view)
        self._btn_play.pack(side="left", padx=(0, 2))

        self._btn_stop = ttk.Button(play_bar, text="⏹ 정지", width=8,
                                     command=self._stop_playback, state="disabled")
        self._btn_stop.pack(side="left", padx=(0, 8))

        ttk.Separator(play_bar, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        ttk.Label(play_bar, text="속도:").pack(side="left", padx=(4, 2))
        self._speed_var = tk.DoubleVar(value=1.0)
        self._play_speed = 1.0
        for spd in [0.5, 0.75, 1.0, 1.5, 2.0]:
            ttk.Radiobutton(play_bar, text=f"{spd}x", value=spd,
                            variable=self._speed_var,
                            command=lambda s=spd: self._set_speed(s)).pack(side="left", padx=1)

        ttk.Separator(play_bar, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)
        ttk.Label(play_bar, text="🔊").pack(side="left", padx=(2, 0))
        self._volume_var = tk.IntVar(value=80)
        vol_scale = ttk.Scale(play_bar, from_=0, to=100,
                              variable=self._volume_var,
                              orient="horizontal", length=80)
        vol_scale.pack(side="left", padx=(0, 2))
        self._vol_label = ttk.Label(play_bar, text="80%", width=4)
        self._vol_label.pack(side="left", padx=(0, 4))
        vol_scale.config(command=lambda v: self._vol_label.config(text=f"{int(float(v))}%"))

        self._play_status_var = tk.StringVar(value="")
        ttk.Label(play_bar, textvariable=self._play_status_var,
                  foreground="#FF6B6B", font=("Consolas", 8)).pack(side="right", padx=5)

        # ── 재생 상태 ──
        self._playing = False
        self._play_thread = None
        self._playhead_id = None
        self._playhead_after_id = None
        self._stop_event = threading.Event()
        self._play_start_wall = 0.0
        self._play_start_time = 0.0
        self._play_end_time = 0.0
        self._play_generation = 0

        # 탭 모드: Notebook  /  단일 모드: Canvas 직접
        if self._tabbed:
            self.notebook = ttk.Notebook(self.win)
            self.notebook.pack(fill="both", expand=True, padx=5, pady=2)
            for i, tab in enumerate(self._tabs):
                frame = ttk.Frame(self.notebook)
                self.notebook.add(frame, text=f" {tab.display_name} ")
                canvas = tk.Canvas(frame, bg="black", highlightthickness=0)
                canvas.pack(fill="both", expand=True)
                tab.canvas = canvas
                self._bind_canvas_events(canvas, i)
            self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        else:
            tab = self._tabs[0]
            tab.canvas = tk.Canvas(self.win, bg="black", highlightthickness=0)
            tab.canvas.pack(fill="both", expand=True, padx=5, pady=2)
            self._bind_canvas_events(tab.canvas, 0)

        # 선택 정보 + 버튼 바
        bottom = ttk.Frame(self.win)
        bottom.pack(fill="x", padx=5, pady=5)

        self.sel_label = ttk.Label(bottom, text="선택 영역: (없음)", font=("Consolas", 10))
        self.sel_label.pack(side="left", padx=10)

        ttk.Button(bottom, text="✅ 확인 (선택 적용)", command=self._confirm).pack(side="right", padx=5)
        ttk.Button(bottom, text="❌ 취소", command=self.win.destroy).pack(side="right")

        if self.multi_select:
            ttk.Button(bottom, text="↩ 마지막 취소", command=self._undo_last_selection).pack(side="right", padx=5)

        # 하단 상태바
        self.info_var = tk.StringVar(value="WAV 파일 로딩 중...")
        ttk.Label(self.win, textvariable=self.info_var, foreground="gray",
                  font=("Consolas", 9)).pack(fill="x", padx=5, pady=(0, 3))

    def _bind_canvas_events(self, canvas, tab_idx):
        """캔버스에 마우스 이벤트를 바인딩 (탭 인덱스 캡처)."""
        canvas.bind("<ButtonPress-1>", lambda e, ti=tab_idx: self._on_sel_start(e, ti))
        canvas.bind("<B1-Motion>", lambda e, ti=tab_idx: self._on_sel_move(e, ti))
        canvas.bind("<ButtonRelease-1>", lambda e, ti=tab_idx: self._on_sel_end(e, ti))
        canvas.bind("<ButtonPress-3>", lambda e, ti=tab_idx: self._on_pan_start(e, ti))
        canvas.bind("<B3-Motion>", lambda e, ti=tab_idx: self._on_pan_move(e, ti))
        canvas.bind("<MouseWheel>", lambda e, ti=tab_idx: self._on_wheel(e, ti))
        canvas.bind("<Button-4>", lambda e, ti=tab_idx: self._on_wheel(e, ti))
        canvas.bind("<Button-5>", lambda e, ti=tab_idx: self._on_wheel(e, ti))
        canvas.bind("<Configure>", lambda e, ti=tab_idx: self._schedule_render(ti, 100))
        canvas.bind("<Escape>", lambda e, ti=tab_idx: self._cancel_polygon(ti))

    # ============================================================
    # 탭 전환
    # ============================================================
    def _on_tab_changed(self, event=None):
        if not self._tabbed:
            return
        idx = self.notebook.index(self.notebook.select())
        self._active_tab_idx = idx
        tab = self._tabs[idx]
        if tab.loaded:
            self.info_var.set(
                f"[{tab.display_name}] {tab.duration:.1f}초, {tab.sr}Hz"
            )
            self._schedule_render(idx, 0)
        else:
            self.info_var.set(f"[{tab.display_name}] 로딩 중...")
        self._update_sel_label()

    # ============================================================
    # WAV 로딩
    # ============================================================
    def _load_wav(self, tab: _TabState):
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(tab.wav_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype != np.float64:
                if np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.float64) / np.iinfo(data.dtype).max
                else:
                    data = data.astype(np.float64)
            tab.sr = sr
            tab.data = data
            tab.duration = len(data) / sr
            tab.max_freq = sr // 2
            tab.t_start = 0.0
            tab.t_end = tab.duration
            tab.f_low = 0.0
            tab.f_high = tab.max_freq
            tab.loaded = True
            tab_idx = self._tabs.index(tab)
            self.win.after(0, self._on_loaded, tab_idx)
        except Exception as e:
            self.win.after(0, lambda: self.info_var.set(f"로드 오류: {e}"))

    def _on_loaded(self, tab_idx):
        tab = self._tabs[tab_idx]
        if tab_idx == self._active_tab_idx:
            self.info_var.set(f"[{tab.display_name}] 로드 완료: {tab.duration:.1f}초, {tab.sr}Hz")
        self._schedule_render(tab_idx, 0)

    # ============================================================
    # 렌더링
    # ============================================================
    def _schedule_render(self, tab_idx, delay_ms=200):
        tab = self._tabs[tab_idx]
        if tab.render_after_id:
            tab.canvas.after_cancel(tab.render_after_id)
        tab.render_after_id = tab.canvas.after(
            delay_ms, lambda: self._render(tab_idx)
        )

    def _render(self, tab_idx):
        tab = self._tabs[tab_idx]
        if not tab.loaded or tab.rendering:
            return
        tab.rendering = True
        tab.render_gen += 1
        cw = max(tab.canvas.winfo_width(), 100)
        ch = max(tab.canvas.winfo_height(), 100)
        params = {
            "t_start": tab.t_start, "t_end": tab.t_end,
            "f_low": tab.f_low, "f_high": tab.f_high,
            "cw": min(cw, self.RENDER_W), "ch": min(ch, self.RENDER_H),
            "gen": tab.render_gen,
        }
        threading.Thread(
            target=self._render_worker, args=(tab_idx, params), daemon=True
        ).start()

    def _render_worker(self, tab_idx, params):
        tab = self._tabs[tab_idx]
        try:
            t_start, t_end = params["t_start"], params["t_end"]
            f_low, f_high = params["f_low"], params["f_high"]
            cw, ch, gen = params["cw"], params["ch"], params["gen"]

            i_start = max(0, int(t_start * tab.sr))
            i_end = min(len(tab.data), int(t_end * tab.sr))
            segment = tab.data[i_start:i_end]
            if len(segment) < 64:
                self.win.after(0, self._on_render_done, tab_idx, None, None, gen)
                return

            max_samples = cw * 512
            if len(segment) > max_samples:
                from scipy.signal import decimate as _decimate
                step = len(segment) // max_samples
                if step >= 2:
                    segment = _decimate(segment, step)
                    effective_sr = tab.sr / step
                else:
                    effective_sr = tab.sr
            else:
                effective_sr = tab.sr

            view_duration = t_end - t_start
            total_ratio = tab.duration / max(view_duration, 0.001)
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
                self.win.after(0, self._on_render_done, tab_idx, None, None, gen)
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
            self.win.after(0, self._on_render_done, tab_idx, pil_img, info, gen)
        except Exception as e:
            self.win.after(0, self._on_render_done, tab_idx, None, f"렌더링 오류: {e}", gen)

    def _on_render_done(self, tab_idx, pil_img, info, gen):
        tab = self._tabs[tab_idx]
        tab.rendering = False
        if gen != tab.render_gen:
            return
        if pil_img:
            tab.tk_img = ImageTk.PhotoImage(pil_img)
            tab.canvas.delete("specimg")
            tab.canvas.create_image(0, 0, anchor="nw", image=tab.tk_img, tags="specimg")
            tab.rendered_view = (tab.t_start, tab.t_end, tab.f_low, tab.f_high)
            self._redraw_selection(tab_idx)
        if isinstance(info, str) and tab_idx == self._active_tab_idx:
            prefix = f"[{tab.display_name}] " if self._tabbed else ""
            self.info_var.set(prefix + info)

    # ============================================================
    # 좌클릭 드래그: 구간 선택
    # ============================================================
    def _on_sel_start(self, event, tab_idx):
        tab = self._tabs[tab_idx]
        if not tab.loaded:
            return

        shift = bool(event.state & 0x0001)
        ctrl = bool(event.state & 0x0004)

        # Ctrl+클릭 → 폴리곤
        if ctrl:
            self._on_polygon_click(event, tab_idx)
            return

        # Shift+드래그 → 박스 필터 재생
        if shift:
            tab.filter_box_start = (event.x, event.y)
            self._clear_filter_box(tab_idx)
            return

        # 일반 드래그: 구간 선택
        tab.sel_start = (event.x, event.y)
        if tab.sel_rect_id:
            tab.canvas.delete(tab.sel_rect_id)
            tab.sel_rect_id = None
        if tab.sel_info_id:
            tab.canvas.delete(tab.sel_info_id)
            tab.sel_info_id = None

    def _on_sel_move(self, event, tab_idx):
        tab = self._tabs[tab_idx]

        # 박스 필터 모드
        if tab.filter_box_start:
            self._update_filter_box(event, tab_idx)
            return

        if not tab.sel_start or not tab.loaded:
            return
        x0, y0 = tab.sel_start
        x1, y1 = event.x, event.y
        color = self._current_sel_color(tab_idx)
        if tab.sel_rect_id:
            tab.canvas.coords(tab.sel_rect_id, x0, y0, x1, y1)
        else:
            tab.sel_rect_id = tab.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline=color, width=2, dash=(6, 3)
            )
        t0, t1, f0, f1 = self._px_to_range(tab_idx, x0, y0, x1, y1)
        self.sel_label.config(
            text=f"선택 중: {t0:.2f}~{t1:.2f}초, {f0:.0f}~{f1:.0f} Hz"
        )

    def _on_sel_end(self, event, tab_idx):
        tab = self._tabs[tab_idx]

        # 박스 필터 재생
        if tab.filter_box_start:
            sx, sy = tab.filter_box_start
            tab.filter_box_start = None
            t0, t1, f0, f1 = self._px_to_range(tab_idx, sx, sy, event.x, event.y)
            if t1 - t0 > 0.01 and f1 - f0 > 10:
                self._play_filtered_box(tab_idx, t0, t1, f0, f1)
            return

        if not tab.sel_start or not tab.loaded:
            return
        x0, y0 = tab.sel_start
        x1, y1 = event.x, event.y
        tab.sel_start = None

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return

        t0, t1, f0, f1 = self._px_to_range(tab_idx, x0, y0, x1, y1)

        if self.multi_select:
            tab.selections.append((t0, t1, f0, f1))
            if tab.sel_rect_id:
                tab.canvas.delete(tab.sel_rect_id)
                tab.sel_rect_id = None
            self._redraw_all_selections(tab_idx)
            self._update_sel_label()
        else:
            tab.selection = (t0, t1, f0, f1)
            self.sel_label.config(
                text=f"✅ 선택됨: {t0:.2f}~{t1:.2f}초, {f0:.0f}~{f1:.0f} Hz"
            )

    def _current_sel_color(self, tab_idx):
        """현재 선택에 사용할 색상"""
        if not self.multi_select:
            return "#FF4444"
        # 전체 선택 수 기준으로 색상 순환
        total = self._total_selection_count()
        return self.MULTI_COLORS[total % len(self.MULTI_COLORS)]

    def _total_selection_count(self):
        """모든 탭의 총 선택 수"""
        return sum(len(t.selections) for t in self._tabs)

    def _update_sel_label(self):
        """선택 수 라벨 갱신"""
        n = self._total_selection_count()
        if n == 0:
            self.sel_label.config(text="선택 영역: (없음)")
        elif n < 2:
            self.sel_label.config(text=f"✅ {n}개 구간 선택됨 (최소 2개 필요)")
        else:
            tab_info = ""
            if self._tabbed:
                parts = []
                for tab in self._tabs:
                    if tab.selections:
                        parts.append(f"{tab.display_name}: {len(tab.selections)}개")
                tab_info = f"  ({', '.join(parts)})"
            self.sel_label.config(text=f"✅ {n}개 구간 선택됨{tab_info}")

    def _undo_last_selection(self):
        """다중 선택 모드: 현재 활성 탭에서 마지막 선택 취소"""
        tab = self._tabs[self._active_tab_idx]
        if not tab.selections:
            # 현재 탭에 선택이 없으면 다른 탭에서 찾기
            for t in reversed(self._tabs):
                if t.selections:
                    tab = t
                    break
            else:
                return
        tab.selections.pop()
        tab_idx = self._tabs.index(tab)
        self._redraw_all_selections(tab_idx)
        self._update_sel_label()

    # ============================================================
    # 좌표 변환
    # ============================================================
    def _px_to_range(self, tab_idx, x0, y0, x1, y1):
        """픽셀 좌표를 시간/주파수 범위로 변환"""
        tab = self._tabs[tab_idx]
        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        px_left, px_right = min(x0, x1), max(x0, x1)
        px_top, px_bottom = min(y0, y1), max(y0, y1)
        view_dt = tab.t_end - tab.t_start
        t0 = tab.t_start + (px_left / cw) * view_dt
        t1 = tab.t_start + (px_right / cw) * view_dt
        view_df = tab.f_high - tab.f_low
        f1 = tab.f_high - (px_top / ch) * view_df
        f0 = tab.f_high - (px_bottom / ch) * view_df
        t0 = max(0, min(t0, tab.duration))
        t1 = max(0, min(t1, tab.duration))
        f0 = max(0, f0)
        f1 = min(tab.max_freq, f1)
        return t0, t1, f0, f1

    # ============================================================
    # 선택 영역 다시 그리기
    # ============================================================
    def _redraw_selection(self, tab_idx):
        """렌더링 후 선택 영역을 다시 그리기"""
        if self.multi_select:
            self._redraw_all_selections(tab_idx)
            return
        tab = self._tabs[tab_idx]
        if tab.sel_rect_id:
            tab.canvas.delete(tab.sel_rect_id)
            tab.sel_rect_id = None
        if tab.sel_info_id:
            tab.canvas.delete(tab.sel_info_id)
            tab.sel_info_id = None
        if not tab.selection:
            return
        t0, t1, f0, f1 = tab.selection
        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        view_dt = tab.t_end - tab.t_start
        view_df = tab.f_high - tab.f_low
        if view_dt <= 0 or view_df <= 0:
            return
        px_left = (t0 - tab.t_start) / view_dt * cw
        px_right = (t1 - tab.t_start) / view_dt * cw
        px_top = (1.0 - (f1 - tab.f_low) / view_df) * ch
        px_bottom = (1.0 - (f0 - tab.f_low) / view_df) * ch
        tab.sel_rect_id = tab.canvas.create_rectangle(
            px_left, px_top, px_right, px_bottom,
            outline="#FF4444", width=2, dash=(6, 3)
        )

    def _redraw_all_selections(self, tab_idx):
        """다중 선택 모드: 해당 탭의 모든 확정된 선택 영역을 다시 그리기"""
        tab = self._tabs[tab_idx]
        for cid in tab.sel_canvas_ids:
            tab.canvas.delete(cid)
        tab.sel_canvas_ids.clear()

        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        view_dt = tab.t_end - tab.t_start
        view_df = tab.f_high - tab.f_low
        if view_dt <= 0 or view_df <= 0:
            return

        # 이 탭 이전 탭들의 선택 수를 계산 (색상 오프셋용)
        color_offset = sum(len(self._tabs[j].selections) for j in range(tab_idx))

        for i, (t0, t1, f0, f1) in enumerate(tab.selections):
            color = self.MULTI_COLORS[(color_offset + i) % len(self.MULTI_COLORS)]
            px_left = (t0 - tab.t_start) / view_dt * cw
            px_right = (t1 - tab.t_start) / view_dt * cw
            px_top = (1.0 - (f1 - tab.f_low) / view_df) * ch
            px_bottom = (1.0 - (f0 - tab.f_low) / view_df) * ch
            rect_id = tab.canvas.create_rectangle(
                px_left, px_top, px_right, px_bottom,
                outline=color, width=2
            )
            global_idx = color_offset + i + 1
            label_id = tab.canvas.create_text(
                px_left + 3, px_top + 2, anchor="nw",
                text=f"#{global_idx}", fill=color, font=("Arial", 10, "bold")
            )
            tab.sel_canvas_ids.extend([rect_id, label_id])

    # ============================================================
    # 우클릭 드래그: 팬
    # ============================================================
    def _on_pan_start(self, event, tab_idx):
        tab = self._tabs[tab_idx]
        tab.pan_start = (event.x, event.y)
        tab.pan_view = (tab.t_start, tab.t_end, tab.f_low, tab.f_high)

    def _on_pan_move(self, event, tab_idx):
        tab = self._tabs[tab_idx]
        if not tab.pan_start or not tab.loaded:
            return
        dx = event.x - tab.pan_start[0]
        dy = event.y - tab.pan_start[1]
        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        t0, t1, fl, fh = tab.pan_view
        dt = (t1 - t0) * dx / cw
        df = (fh - fl) * dy / ch
        tab.t_start = t0 - dt
        tab.t_end = t1 - dt
        tab.f_low = fl + df
        tab.f_high = fh + df
        self._clamp_view(tab_idx)
        self._apply_visual_transform(tab_idx)
        self._schedule_render(tab_idx, 150)

    # ============================================================
    # 휠: 줌
    # ============================================================
    def _on_wheel(self, event, tab_idx):
        tab = self._tabs[tab_idx]
        if not tab.loaded:
            return
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 1.3
        else:
            factor = 0.75
        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        rx = event.x / cw
        ry = 1.0 - event.y / ch

        t_cursor = tab.t_start + rx * (tab.t_end - tab.t_start)
        f_cursor = tab.f_low + ry * (tab.f_high - tab.f_low)

        new_dt = (tab.t_end - tab.t_start) * factor
        new_df = (tab.f_high - tab.f_low) * factor

        tab.t_start = t_cursor - rx * new_dt
        tab.t_end = t_cursor + (1 - rx) * new_dt
        tab.f_low = f_cursor - ry * new_df
        tab.f_high = f_cursor + (1 - ry) * new_df
        self._clamp_view(tab_idx)
        self._apply_visual_transform(tab_idx)
        self._schedule_render(tab_idx, 100)

    def _apply_visual_transform(self, tab_idx):
        """뷰 변경 시 기존 이미지를 즉시 이동/스케일링 (렌더링 완료 전 시각 피드백)"""
        tab = self._tabs[tab_idx]
        if not tab.rendered_view or not tab.tk_img:
            return
        old_t0, old_t1, old_fl, old_fh = tab.rendered_view
        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        new_dt = tab.t_end - tab.t_start
        new_df = tab.f_high - tab.f_low
        old_dt = old_t1 - old_t0
        old_df = old_fh - old_fl
        if new_dt <= 0 or new_df <= 0 or old_dt <= 0 or old_df <= 0:
            return
        ox = (old_t0 - tab.t_start) / new_dt * cw
        oy = (tab.f_high - old_fh) / new_df * ch
        items = tab.canvas.find_withtag("specimg")
        if items:
            tab.canvas.coords(items[0], ox, oy)

    def _clamp_view(self, tab_idx):
        tab = self._tabs[tab_idx]
        if tab.t_start < 0:
            tab.t_end -= tab.t_start
            tab.t_start = 0
        if tab.t_end > tab.duration:
            tab.t_start -= (tab.t_end - tab.duration)
            tab.t_end = tab.duration
        if tab.t_start < 0:
            tab.t_start = 0
        if tab.f_low < 0:
            tab.f_high -= tab.f_low
            tab.f_low = 0
        if tab.f_high > tab.max_freq:
            tab.f_low -= (tab.f_high - tab.max_freq)
            tab.f_high = tab.max_freq
        if tab.f_low < 0:
            tab.f_low = 0

    def _reset_view(self):
        tab = self._tabs[self._active_tab_idx]
        tab.t_start = 0.0
        tab.t_end = tab.duration
        tab.f_low = 0.0
        tab.f_high = tab.max_freq
        self._schedule_render(self._active_tab_idx, 0)

    # ============================================================
    # 확인/취소
    # ============================================================
    def _confirm(self):
        if self.multi_select:
            total = self._total_selection_count()
            if total < 2:
                self.sel_label.config(text="⚠ 최소 2개 구간을 선택하세요!")
                return
            if self._tabbed:
                # 탭 모드: 파일 경로 포함 5-tuple
                all_selections = []
                for tab in self._tabs:
                    for (t0, t1, f0, f1) in tab.selections:
                        all_selections.append((t0, t1, f0, f1, tab.wav_path))
                self.callback(all_selections)
            else:
                # 단일 파일 모드: 기존 4-tuple
                self.callback(self._tabs[0].selections)
            self.win.destroy()
        else:
            tab = self._tabs[self._active_tab_idx]
            if not tab.selection:
                self.sel_label.config(text="⚠ 먼저 영역을 드래그로 선택하세요!")
                return
            t0, t1, f0, f1 = tab.selection
            self.callback(t0, t1, f0, f1)
            self.win.destroy()

    # ============================================================
    # 하위 호환 속성 (단일 탭 모드에서 외부 접근 시)
    # ============================================================
    @property
    def canvas(self):
        return self._tabs[self._active_tab_idx].canvas if self._tabs else None

    # ============================================================
    # 좌표 변환 (박스/폴리곤 재생용)
    # ============================================================
    def _px_to_data(self, tab_idx, px_x, px_y):
        tab = self._tabs[tab_idx]
        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        t = tab.t_start + (px_x / cw) * (tab.t_end - tab.t_start)
        f = tab.f_high - (px_y / ch) * (tab.f_high - tab.f_low)
        return t, f

    def _data_to_px(self, tab_idx, t, f):
        tab = self._tabs[tab_idx]
        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        px_x = (t - tab.t_start) / max(tab.t_end - tab.t_start, 0.001) * cw
        px_y = (tab.f_high - f) / max(tab.f_high - tab.f_low, 1) * ch
        return px_x, px_y

    # ============================================================
    # 박스 필터 오버레이
    # ============================================================
    def _update_filter_box(self, event, tab_idx):
        tab = self._tabs[tab_idx]
        sx, sy = tab.filter_box_start
        self._clear_filter_box(tab_idx)
        tab.filter_box_rect = tab.canvas.create_rectangle(
            sx, sy, event.x, event.y,
            outline="#00BFFF", width=2, dash=(4, 2),
            fill="#00BFFF", stipple="gray25"
        )

    def _clear_filter_box(self, tab_idx):
        tab = self._tabs[tab_idx]
        if tab.filter_box_rect:
            tab.canvas.delete(tab.filter_box_rect)
            tab.filter_box_rect = None

    # ============================================================
    # 폴리곤 선택
    # ============================================================
    def _on_polygon_click(self, event, tab_idx):
        tab = self._tabs[tab_idx]
        t, f = self._px_to_data(tab_idx, event.x, event.y)

        if len(tab.poly_points) >= 3:
            sx, sy = self._data_to_px(tab_idx, *tab.poly_points[0])
            dist = ((event.x - sx)**2 + (event.y - sy)**2)**0.5
            if dist < tab.poly_snap_dist:
                self._close_polygon(tab_idx)
                return

        tab.poly_points.append((t, f))
        self._redraw_polygon_overlay(tab_idx)

    def _redraw_polygon_overlay(self, tab_idx):
        tab = self._tabs[tab_idx]
        for cid in tab.poly_canvas_ids:
            tab.canvas.delete(cid)
        tab.poly_canvas_ids.clear()

        if not tab.poly_points:
            return

        for i, (t, f) in enumerate(tab.poly_points):
            px, py = self._data_to_px(tab_idx, t, f)
            r = 5 if i == 0 else 3
            color = "#FF4444" if i == 0 else "#00FF88"
            cid = tab.canvas.create_oval(
                px - r, py - r, px + r, py + r,
                fill=color, outline="white", width=1
            )
            tab.poly_canvas_ids.append(cid)

        if len(tab.poly_points) >= 2:
            coords = []
            for t, f in tab.poly_points:
                px, py = self._data_to_px(tab_idx, t, f)
                coords.extend([px, py])
            cid = tab.canvas.create_line(
                *coords, fill="#00FF88", width=2, dash=(4, 2)
            )
            tab.poly_canvas_ids.append(cid)

        if len(tab.poly_points) >= 3:
            last_px, last_py = self._data_to_px(tab_idx, *tab.poly_points[-1])
            first_px, first_py = self._data_to_px(tab_idx, *tab.poly_points[0])
            cid = tab.canvas.create_line(
                last_px, last_py, first_px, first_py,
                fill="#FF4444", width=1, dash=(2, 4)
            )
            tab.poly_canvas_ids.append(cid)

    def _close_polygon(self, tab_idx):
        tab = self._tabs[tab_idx]
        points = list(tab.poly_points)
        self._cancel_polygon(tab_idx)
        if len(points) >= 3:
            self._play_filtered_polygon(tab_idx, points)

    def _cancel_polygon(self, tab_idx):
        tab = self._tabs[tab_idx]
        tab.poly_points.clear()
        for cid in tab.poly_canvas_ids:
            tab.canvas.delete(cid)
        tab.poly_canvas_ids.clear()

    # ============================================================
    # 재생
    # ============================================================
    def _set_speed(self, speed):
        self._play_speed = speed

    def _play_view(self):
        tab = self._tabs[self._active_tab_idx]
        if not tab.loaded or not HAS_PLAYBACK:
            return
        if self._playing:
            self._stop_playback()
            return
        self._start_playback(self._active_tab_idx, tab.t_start, tab.t_end)

    def _start_playback(self, tab_idx, t0, t1):
        tab = self._tabs[tab_idx]
        if not tab.loaded or tab.sr is None:
            return

        speed = self._play_speed
        volume = self._volume_var.get() / 100.0

        i0 = max(0, int(t0 * tab.sr))
        i1 = min(len(tab.data), int(t1 * tab.sr))
        segment = tab.data[i0:i1].copy()
        if len(segment) < 64:
            return

        if abs(speed - 1.0) > 0.01 and HAS_SCIPY:
            from scipy.signal import resample
            new_len = max(64, int(len(segment) / speed))
            segment = resample(segment, new_len)

        max_val = np.max(np.abs(segment))
        if max_val > 0:
            segment = segment / max_val
        pcm = (segment * 32767 * volume).astype(np.int16)

        actual_duration = (t1 - t0) / speed
        self._do_play_pcm(pcm, tab.sr, actual_duration, t0, t1, tab_idx)

    def _play_filtered_box(self, tab_idx, t0, t1, f_low, f_high):
        tab = self._tabs[tab_idx]
        if not tab.loaded or not HAS_PLAYBACK:
            return
        self._stop_playback()
        speed = self._play_speed
        volume = self._volume_var.get() / 100.0
        pcm, sr, duration = prepare_filtered_pcm(
            tab.data, tab.sr, t0, t1, f_low, f_high,
            speed=speed, volume=volume
        )
        if pcm is None:
            return
        self._play_status_var.set(
            f"📦 박스: {t0:.1f}-{t1:.1f}s, {f_low:.0f}-{f_high:.0f}Hz"
        )
        self._do_play_pcm(pcm, sr, duration, t0, t1, tab_idx)

    def _play_filtered_polygon(self, tab_idx, points):
        tab = self._tabs[tab_idx]
        if not tab.loaded or not HAS_PLAYBACK:
            return
        self._stop_playback()
        speed = self._play_speed
        volume = self._volume_var.get() / 100.0
        self._play_status_var.set("✏ 폴리곤 처리 중...")
        self.win.update_idletasks()
        pcm, sr, duration = prepare_polygon_pcm(
            tab.data, tab.sr, points,
            speed=speed, volume=volume
        )
        if pcm is None:
            self._play_status_var.set("")
            return
        times = [p[0] for p in points]
        t0, t1 = min(times), max(times)
        self._play_status_var.set(
            f"✏ 폴리곤: {t0:.1f}-{t1:.1f}s ({len(points)}점)"
        )
        self._do_play_pcm(pcm, sr, duration, t0, t1, tab_idx)

    def _do_play_pcm(self, pcm_data, sr, duration, t0, t1, tab_idx):
        """인메모리 PCM 재생 공통"""
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._playing = True
        self._play_generation += 1
        gen = self._play_generation
        self._play_start_time = t0
        self._play_end_time = t1
        self._play_start_wall = time.time()
        self._play_tab_idx = tab_idx

        self._btn_play.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._play_status_var.set(f"▶ {t0:.1f}-{t1:.1f}s ({self._play_speed}x)")
        self._update_playhead()

        def _on_done(error):
            if self._play_generation == gen:
                if error:
                    self.win.after(0, lambda m=error: self._play_status_var.set(f"오류: {m}"))
                self.win.after(0, lambda: self._on_playback_done(gen))

        self._play_thread = play_numpy_async(
            pcm_data, sr, stop_event, duration, on_done=_on_done
        )

    def _stop_playback(self):
        self._stop_event.set()
        self._playing = False
        self._clear_playhead()
        self._btn_play.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._play_status_var.set("")

    def _on_playback_done(self, gen):
        if gen != self._play_generation:
            return
        if not self._stop_event.is_set():
            self._playing = False
            self._clear_playhead()
            self._btn_play.config(state="normal")
            self._btn_stop.config(state="disabled")
            self._play_status_var.set("✅ 재생 완료")
            self.win.after(3000, lambda: self._play_status_var.set(""))

    def _update_playhead(self):
        if not self._playing:
            return
        elapsed = time.time() - self._play_start_wall
        current = self._play_start_time + elapsed * self._play_speed
        if current > self._play_end_time:
            return

        tab_idx = getattr(self, '_play_tab_idx', self._active_tab_idx)
        tab = self._tabs[tab_idx]
        cw = max(tab.canvas.winfo_width(), 1)
        ch = max(tab.canvas.winfo_height(), 1)
        view_dt = tab.t_end - tab.t_start
        if view_dt > 0:
            px = (current - tab.t_start) / view_dt * cw
            if 0 <= px <= cw:
                if self._playhead_id:
                    tab.canvas.coords(self._playhead_id, px, 0, px, ch)
                else:
                    self._playhead_id = tab.canvas.create_line(
                        px, 0, px, ch,
                        fill="#FF3333", width=2, tags="playhead"
                    )
            else:
                self._clear_playhead()

        self._playhead_after_id = self.win.after(33, self._update_playhead)

    def _clear_playhead(self):
        if self._playhead_after_id:
            self.win.after_cancel(self._playhead_after_id)
            self._playhead_after_id = None
        if self._playhead_id:
            tab_idx = getattr(self, '_play_tab_idx', self._active_tab_idx)
            self._tabs[tab_idx].canvas.delete(self._playhead_id)
            self._playhead_id = None
