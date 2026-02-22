# ============================================================
# ui/spectrogram_tab.py — 실시간 스펙트로그램 탭 위젯
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
    from scipy.io import wavfile as scipy_wavfile
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

# 오디오 재생 (sounddevice 우선, winsound 폴백)
from audio.playback import (
    play_wav_async, stop_playback as _stop_audio,
    prepare_playback_wav, HAS_PLAYBACK,
)

# 모듈 내 참조
from colormaps import COLORMAPS, MAGMA_LUT, DETECTION_COLORS


class SpectrogramTab:
    """WAV 파일을 로드하고, 현재 보이는 시간/주파수 범위에 맞춰
    scipy + matplotlib로 스펙트로그램을 실시간 재생성하는 뷰어."""

    # 렌더링 해상도 (픽셀)
    RENDER_W = 1400
    RENDER_H = 700

    def __init__(self, parent_notebook, wav_path, title, toplevel_win, detections=None):
        self.frame = ttk.Frame(parent_notebook)
        self.title = title
        self.win = toplevel_win
        self.wav_path = wav_path

        # 데이터 (백그라운드에서 로드)
        self.sr = None
        self.data = None
        self.duration = 0
        self.max_freq = 22050

        # 뷰 범위 (전체) — 로드 후 갱신됨
        self.t_start = 0.0
        self.t_end = 1.0
        self.f_low = 0.0
        self.f_high = 22050

        # 마지막 렌더링된 뷰 상태 (즉시 변환용)
        self._rendered_view = None  # (t_start, t_end, f_low, f_high)

        # 드래그 상태
        self._drag_start = None
        self._drag_view = None

        # 디바운스 타이머
        self._render_after_id = None
        self._rendering = False      # 렌더링 중 플래그
        self._render_gen = 0         # 렌더 세대 (스레드 결과 유효성 확인)
        self._loaded = False         # WAV 로드 완료 여부

        # 줌 애니메이션 상태
        self._zoom_anim_id = None

        # 검출 결과 (오버레이용)
        # [{"species": str, "time": float, "score": float}, ...]
        self._detections = detections or []
        self._show_detections = True  # 오버레이 표시 여부
        self._detection_items = []    # 캔버스 아이템 ID 목록

        # 종별 색상 매핑
        species_names = list(set(d.get("species", "") for d in self._detections))
        self._species_colors = {}
        for idx, name in enumerate(sorted(species_names)):
            self._species_colors[name] = DETECTION_COLORS[idx % len(DETECTION_COLORS)]

        # ── 재생 상태 ──
        self._playing = False
        self._play_thread = None
        self._playhead_id = None
        self._playhead_after_id = None
        self._play_speed = 1.0
        self._stop_event = threading.Event()
        self._play_start_wall = 0.0   # 재생 시작 wall-clock
        self._play_start_time = 0.0   # 재생 시작 시점 (초)
        self._play_end_time = 0.0     # 재생 종료 시점 (초)
        self._play_temp_wav = None    # 임시 WAV 파일 경로
        self._play_generation = 0     # 재생 세대 카운터 (스레드 경합 방지)

        self._build_ui()

        # WAV 로드를 백그라운드 스레드에서 실행
        self.info_var.set("WAV 파일 로딩 중...")
        load_thread = threading.Thread(target=self._load_wav, daemon=True)
        load_thread.start()

    def _load_wav(self):
        """백그라운드에서 WAV 파일을 로드."""
        try:
            sr, data = scipy_wavfile.read(self.wav_path)
            if data.ndim > 1:
                data = data[:, 0]  # 스테레오 → 모노
            data = data.astype(np.float64)

            self.sr = sr
            self.data = data
            self.duration = len(data) / sr
            self.max_freq = sr / 2.0

            self.t_start = 0.0
            self.t_end = self.duration
            self.f_low = 0.0
            self.f_high = self.max_freq
            self._loaded = True

            # 메인 스레드에서 초기 렌더링
            self.frame.after(100, self._render)
        except Exception as e:
            self.frame.after(0, lambda: self.info_var.set(f"로드 오류: {e}"))

    def _build_ui(self):
        # ---- 툴바 1행: 줌 조작 ----
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill="x", padx=5, pady=(5, 0))

        self.info_var = tk.StringVar(value="로딩 중...")
        ttk.Label(toolbar, textvariable=self.info_var, font=("Consolas", 9)).pack(side="left")

        ttk.Button(toolbar, text="🔍 확대 (+)", width=10,
                   command=lambda: self._zoom_center(1.4)).pack(side="right", padx=2)
        ttk.Button(toolbar, text="🔍 축소 (−)", width=10,
                   command=lambda: self._zoom_center(0.7)).pack(side="right", padx=2)
        ttk.Button(toolbar, text="📐 전체 보기", width=10,
                   command=self._reset_view).pack(side="right", padx=5)

        ttk.Label(toolbar, text="  휠: 확대/축소 | Shift+휠: 좌우 | Ctrl+휠: 상하 | 드래그: 이동  ",
                  foreground="gray").pack(side="right")

        # ---- 툴바 2행: 색상 조절 ----
        toolbar2 = ttk.Frame(self.frame)
        toolbar2.pack(fill="x", padx=5, pady=(2, 0))

        # 컬러맵 선택
        ttk.Label(toolbar2, text="🎨 컬러맵:").pack(side="left", padx=(0, 2))
        self._colormap_var = tk.StringVar(value="Magma")
        cmap_combo = ttk.Combobox(toolbar2, textvariable=self._colormap_var,
                                  values=list(COLORMAPS.keys()),
                                  state="readonly", width=10)
        cmap_combo.pack(side="left", padx=(0, 10))
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self._on_color_change())

        # 밝기 슬라이더
        ttk.Label(toolbar2, text="☀ 밝기:").pack(side="left", padx=(0, 2))
        self._brightness_var = tk.IntVar(value=0)
        bright_scale = ttk.Scale(toolbar2, from_=-50, to=50,
                                 variable=self._brightness_var,
                                 orient="horizontal", length=120,
                                 command=lambda v: self._on_color_change())
        bright_scale.pack(side="left", padx=(0, 5))
        self._bright_label = ttk.Label(toolbar2, text="0", width=4)
        self._bright_label.pack(side="left", padx=(0, 10))

        # 대비 슬라이더
        ttk.Label(toolbar2, text="🔆 대비:").pack(side="left", padx=(0, 2))
        self._contrast_var = tk.IntVar(value=0)
        contrast_scale = ttk.Scale(toolbar2, from_=-50, to=50,
                                   variable=self._contrast_var,
                                   orient="horizontal", length=120,
                                   command=lambda v: self._on_color_change())
        contrast_scale.pack(side="left", padx=(0, 5))
        self._contrast_label = ttk.Label(toolbar2, text="0", width=4)
        self._contrast_label.pack(side="left", padx=(0, 10))

        # 초기화 버튼
        ttk.Button(toolbar2, text="↺ 초기화", width=8,
                   command=self._reset_color).pack(side="left", padx=(0, 10))

        # 검출 오버레이 토글 (검출 결과가 있을 때만)
        if self._detections:
            self._show_det_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(toolbar2, text="📍 식별구간 표시",
                            variable=self._show_det_var,
                            command=self._toggle_detections).pack(side="right", padx=5)

        # ---- 툴바 3행: 재생 컨트롤 ----
        toolbar3 = ttk.Frame(self.frame)
        toolbar3.pack(fill="x", padx=5, pady=(2, 0))

        self._btn_play = ttk.Button(toolbar3, text="▶ 현재 뷰 재생", width=14,
                                     command=self._play_view)
        self._btn_play.pack(side="left", padx=(0, 2))

        self._btn_stop = ttk.Button(toolbar3, text="⏹ 정지", width=8,
                                     command=self._stop_playback, state="disabled")
        self._btn_stop.pack(side="left", padx=(0, 8))

        ttk.Separator(toolbar3, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        ttk.Label(toolbar3, text="속도:").pack(side="left", padx=(4, 2))
        self._speed_var = tk.DoubleVar(value=1.0)
        self._speed_buttons = {}
        for spd in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            label = f"{spd}x"
            btn = ttk.Radiobutton(toolbar3, text=label, value=spd,
                                   variable=self._speed_var,
                                   command=lambda s=spd: self._set_speed(s))
            btn.pack(side="left", padx=1)
            self._speed_buttons[spd] = btn

        # 검출 구간 재생 드롭다운 (검출 결과가 있을 때만)
        if self._detections:
            ttk.Separator(toolbar3, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)
            ttk.Label(toolbar3, text="📋 검출 구간:").pack(side="left", padx=(2, 2))

            # 검출 목록 문자열 생성
            self._det_options = []
            self._det_map = {}  # display_str → detection dict
            for i, det in enumerate(self._detections):
                sp = det.get("species", "?")
                t = det.get("time", 0)
                sc = det.get("score", 0)
                display = f"{sp} @ {t:.1f}s ({sc:.0%})"
                self._det_options.append(display)
                self._det_map[display] = det

            self._det_combo_var = tk.StringVar(value="")
            det_combo = ttk.Combobox(toolbar3, textvariable=self._det_combo_var,
                                     values=self._det_options,
                                     state="readonly", width=28)
            det_combo.pack(side="left", padx=(0, 2))
            det_combo.bind("<<ComboboxSelected>>", self._on_detection_select)

            ttk.Button(toolbar3, text="▶ 구간 재생", width=10,
                       command=self._play_selected_detection).pack(side="left", padx=2)

        # 볼륨 슬라이더
        ttk.Separator(toolbar3, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)
        ttk.Label(toolbar3, text="🔊").pack(side="left", padx=(2, 0))
        self._volume_var = tk.IntVar(value=80)
        vol_scale = ttk.Scale(toolbar3, from_=0, to=100,
                              variable=self._volume_var,
                              orient="horizontal", length=80)
        vol_scale.pack(side="left", padx=(0, 2))
        self._vol_label = ttk.Label(toolbar3, text="80%", width=4)
        self._vol_label.pack(side="left", padx=(0, 4))
        vol_scale.config(command=self._on_volume_change)

        # 재생 상태 라벨
        self._play_status_var = tk.StringVar(value="")
        ttk.Label(toolbar3, textvariable=self._play_status_var,
                  foreground="#FF6B6B", font=("Consolas", 8)).pack(side="right", padx=5)

        # ---- 캔버스 ----
        self.canvas = tk.Canvas(self.frame, bg="#1a1a2e", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)

        self._photo = None
        self._img_id = None

        # 이벤트 바인딩
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<Configure>", self._on_resize)

    # ---- 렌더링 (백그라운드 스레드) ----
    def _render(self):
        """렌더링을 백그라운드 스레드에서 실행."""
        self._render_after_id = None

        if not self._loaded or self._rendering:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        self._rendering = True
        self._render_gen += 1
        gen = self._render_gen

        # 현재 뷰 상태를 스냅샷
        params = {
            "t_start": self.t_start, "t_end": self.t_end,
            "f_low": self.f_low, "f_high": self.f_high,
            "cw": cw, "ch": ch, "gen": gen,
        }

        self.info_var.set("렌더링 중...")
        thread = threading.Thread(target=self._render_worker, args=(params,), daemon=True)
        thread.start()

    def _render_worker(self, params):
        """백그라운드에서 FFT → numpy 컬러 매핑 → PIL 이미지 생성 (matplotlib 미사용)."""
        try:
            t_start = params["t_start"]
            t_end = params["t_end"]
            f_low = params["f_low"]
            f_high = params["f_high"]
            cw = params["cw"]
            ch = params["ch"]
            gen = params["gen"]

            # 시간 범위에 해당하는 샘플 추출
            i_start = max(0, int(t_start * self.sr))
            i_end = min(len(self.data), int(t_end * self.sr))
            segment = self.data[i_start:i_end]

            if len(segment) < 64:
                self.frame.after(0, self._on_render_done, None, None, gen)
                return

            # 성능 최적화: 너무 큰 세그먼트는 다운샘플링 (안티앨리어싱 필터 적용)
            max_samples = cw * 512  # 화면 너비 * 512배 정도면 충분
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

            # 줌 레벨에 따른 적응형 FFT 파라미터
            view_duration = t_end - t_start
            total_ratio = self.duration / max(view_duration, 0.001)

            if total_ratio > 20:
                nperseg = min(2048, len(segment))
            elif total_ratio > 5:
                nperseg = min(1024, len(segment))
            else:
                nperseg = min(512, len(segment))

            noverlap = int(nperseg * 0.75)

            # scipy 스펙트로그램 계산
            freqs, times, Sxx = scipy_spectrogram(
                segment, fs=effective_sr,
                nperseg=nperseg, noverlap=noverlap,
                window="hann"
            )

            # 주파수 범위 필터링
            f_mask = (freqs >= f_low) & (freqs <= f_high)
            freqs = freqs[f_mask]
            Sxx = Sxx[f_mask, :]

            if Sxx.size == 0:
                self.frame.after(0, self._on_render_done, None, None, gen)
                return

            # dB 변환
            Sxx_db = 10 * np.log10(Sxx + 1e-12)

            # 밝기/대비 값 가져오기
            brightness = self._brightness_var.get()  # -50 ~ +50
            contrast = self._contrast_var.get()      # -50 ~ +50

            # 정규화 (0~255) — 밝기/대비 적용
            pct_low = max(0.1, 2 - contrast * 0.04)   # 대비↑ → 범위 좁아짐
            pct_high = min(99.9, 99.5 + contrast * 0.01)
            vmin = np.percentile(Sxx_db, pct_low)
            vmax = np.percentile(Sxx_db, pct_high)
            # 밝기 오프셋 (dB 단위)
            bright_offset = brightness * 0.5  # -25dB ~ +25dB
            vmin -= bright_offset
            vmax -= bright_offset
            if vmax <= vmin:
                vmax = vmin + 1
            normalized = np.clip((Sxx_db - vmin) / (vmax - vmin), 0, 1)

            # 선택된 컬러맵 LUT 적용
            cmap_name = self._colormap_var.get()
            lut = COLORMAPS.get(cmap_name, MAGMA_LUT)
            indices = (normalized * 255).astype(np.uint8)
            rgb = lut[indices]  # (freq, time, 3)

            # 이미지 생성: 주파수축 뒤집기 (높은 주파수가 위)
            rgb = rgb[::-1, :, :]

            # PIL 이미지로 변환 후 화면 크기에 맞춤
            pil_img = Image.fromarray(rgb, mode="RGB")
            resample = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
            pil_img = pil_img.resize((cw, ch), resample)

            # 정보 문자열
            t_range = f"{t_start:.2f}s ~ {t_end:.2f}s"
            f_range = f"{f_low:.0f}Hz ~ {f_high:.0f}Hz"
            zoom_t = self.duration / max(t_end - t_start, 0.001)
            info = (f"시간: {t_range}  |  주파수: {f_range}  |  "
                    f"FFT: {nperseg}  |  줌: {zoom_t:.1f}x")

            self.frame.after(0, self._on_render_done, pil_img, info, gen)

        except Exception as e:
            self.frame.after(0, self._on_render_done, None, f"렌더링 오류: {e}", gen)

    def _on_render_done(self, pil_img, info, gen):
        """메인 스레드에서 캔버스에 이미지를 표시."""
        self._rendering = False

        # 오래된 세대의 결과는 무시
        if gen != self._render_gen:
            self._schedule_render(50)
            return

        if pil_img is not None:
            self._last_pil_img = pil_img  # 즉시 변환용 원본 보관
            self._photo = ImageTk.PhotoImage(pil_img)
            self.win._refs.append(self._photo)

            if self._img_id:
                self.canvas.coords(self._img_id, 0, 0)
                self.canvas.itemconfigure(self._img_id, image=self._photo)
            else:
                self._img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

            # 렌더링 완료 시 현재 뷰 상태 저장
            self._rendered_view = (self.t_start, self.t_end, self.f_low, self.f_high)

        if info:
            self.info_var.set(info)

        # 오버레이 그리기
        self._draw_detections()

    def _on_color_change(self):
        """밝기/대비/컬러맵 변경 시 즉시 재렌더링"""
        self._bright_label.config(text=str(self._brightness_var.get()))
        self._contrast_label.config(text=str(self._contrast_var.get()))
        self._schedule_render(100)

    def _reset_color(self):
        """색상 설정을 기본값으로 초기화"""
        self._brightness_var.set(0)
        self._contrast_var.set(0)
        self._colormap_var.set("Magma")
        self._bright_label.config(text="0")
        self._contrast_label.config(text="0")
        self._schedule_render(50)

    def _toggle_detections(self):
        """검출 오버레이 표시/숨기기"""
        self._show_detections = self._show_det_var.get()
        if self._show_detections:
            self._draw_detections()
        else:
            self._clear_detection_overlay()

    def _clear_detection_overlay(self):
        """캔버스에서 검출 오버레이 아이템 삭제"""
        for item_id in self._detection_items:
            self.canvas.delete(item_id)
        self._detection_items.clear()

    def _draw_detections(self):
        """현재 뷰에 보이는 검출 결과를 캔버스에 오버레이로 표시"""
        self._clear_detection_overlay()

        if not self._detections or not self._show_detections:
            return
        if not self._loaded:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        view_dt = self.t_end - self.t_start
        if view_dt <= 0:
            return

        # 검출 시간 ± 마진 (줌 레벨에 비례)
        margin = max(0.3, view_dt * 0.01)

        for det in self._detections:
            det_time = det.get("time", 0)
            det_species = det.get("species", "")
            det_score = det.get("score", 0)

            # 검출 범위
            t0 = det_time - margin
            t1 = det_time + margin

            # 현재 뷰에 보이는지 확인
            if t1 < self.t_start or t0 > self.t_end:
                continue

            # 시간 → 픽셀 X 변환
            px_left = max(0, (t0 - self.t_start) / view_dt * cw)
            px_right = min(cw, (t1 - self.t_start) / view_dt * cw)

            if px_right - px_left < 2:
                px_right = px_left + 2

            # 종별 색상
            colors = self._species_colors.get(det_species, DETECTION_COLORS[0])
            outline_color = colors[0]

            # 반투명 사각형 (전체 높이)
            rect_id = self.canvas.create_rectangle(
                px_left, 2, px_right, ch - 2,
                outline=outline_color, width=2,
                fill="", dash=(4, 2)
            )
            self._detection_items.append(rect_id)

            # 상단 라벨 (종명 + 점수)
            label_x = (px_left + px_right) / 2
            score_pct = f"{det_score:.0%}" if isinstance(det_score, float) else str(det_score)
            label_text = f"{det_species}\n{score_pct}"
            text_id = self.canvas.create_text(
                label_x, 12, text=label_text,
                fill=outline_color, font=("Arial", 8, "bold"),
                anchor="n"
            )
            self._detection_items.append(text_id)

            # 중심선 (정확한 검출 시점)
            px_center = (det_time - self.t_start) / view_dt * cw
            if 0 <= px_center <= cw:
                line_id = self.canvas.create_line(
                    px_center, 0, px_center, ch,
                    fill=outline_color, width=1, dash=(2, 4)
                )
                self._detection_items.append(line_id)

    def _schedule_render(self, delay_ms=200):
        """디바운스된 렌더링 예약"""
        if self._render_after_id:
            self.canvas.after_cancel(self._render_after_id)
        self._render_after_id = self.canvas.after(delay_ms, self._render)

    # ---- 뷰 조작 ----
    def _clamp_view(self):
        """뷰 범위를 유효 영역으로 제한"""
        dt = self.t_end - self.t_start
        df = self.f_high - self.f_low

        # 최소 범위
        if dt < 0.01:
            mid = (self.t_start + self.t_end) / 2
            self.t_start = mid - 0.005
            self.t_end = mid + 0.005
        if df < 50:
            mid = (self.f_low + self.f_high) / 2
            self.f_low = mid - 25
            self.f_high = mid + 25

        # 경계 제한
        if self.t_start < 0:
            self.t_end -= self.t_start
            self.t_start = 0
        if self.t_end > self.duration:
            self.t_start -= (self.t_end - self.duration)
            self.t_end = self.duration
        if self.f_low < 0:
            self.f_high -= self.f_low
            self.f_low = 0
        if self.f_high > self.max_freq:
            self.f_low -= (self.f_high - self.max_freq)
            self.f_high = self.max_freq

        self.t_start = max(0, self.t_start)
        self.f_low = max(0, self.f_low)

    def _on_wheel(self, event):
        """마우스 휠로 확대/축소 (Shift: 시간축만, Ctrl: 주파수축만)"""
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 1.3   # 축소
        else:
            factor = 0.75  # 확대

        # 수정키 확인
        shift = bool(event.state & 0x0001)  # Shift
        ctrl = bool(event.state & 0x0004)   # Ctrl

        # 커서 위치를 뷰 비율로 변환
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        rx = event.x / max(cw, 1)  # 0~1 (왼쪽~오른쪽 = 시간)
        ry = 1.0 - event.y / max(ch, 1)  # 0~1 (아래~위 = 주파수)

        # 현재 커서가 가리키는 시간/주파수
        t_cursor = self.t_start + rx * (self.t_end - self.t_start)
        f_cursor = self.f_low + ry * (self.f_high - self.f_low)

        # Shift: 시간축만 / Ctrl: 주파수축만 / 기본: 양쪽 모두
        if shift and not ctrl:
            dt = (self.t_end - self.t_start) * factor
            self.t_start = t_cursor - rx * dt
            self.t_end = t_cursor + (1 - rx) * dt
        elif ctrl and not shift:
            df = (self.f_high - self.f_low) * factor
            self.f_low = f_cursor - ry * df
            self.f_high = f_cursor + (1 - ry) * df
        else:
            dt = (self.t_end - self.t_start) * factor
            df = (self.f_high - self.f_low) * factor
            self.t_start = t_cursor - rx * dt
            self.t_end = t_cursor + (1 - rx) * dt
            self.f_low = f_cursor - ry * df
            self.f_high = f_cursor + (1 - ry) * df

        self._clamp_view()

        # 즉시 캔버스 이미지를 변환하여 시각적 피드백 제공
        self._apply_canvas_transform()
        self._draw_detections()

        # 정밀 렌더링은 디바운스로 예약
        self._schedule_render(200)

    def _on_drag_start(self, event):
        # 재생 중 클릭 → 해당 시점으로 탐색
        if self._playing:
            cw = max(self.canvas.winfo_width(), 1)
            rx = event.x / cw
            click_time = self.t_start + rx * (self.t_end - self.t_start)
            click_time = max(self.t_start, min(click_time, self.t_end))
            end_time = getattr(self, '_play_range_t1', self._play_end_time)
            if click_time < end_time - 0.05:
                self._stop_event.set()
                self._clear_playhead()
                self._playing = False
                self._start_playback(click_time, end_time)
            return
        self._drag_start = (event.x, event.y)
        self._drag_view = (self.t_start, self.t_end, self.f_low, self.f_high)
        # 드래그 시작 시 캔버스 이미지 원점 저장
        self._drag_img_origin = (0, 0)
        if self._img_id:
            coords = self.canvas.coords(self._img_id)
            if coords:
                self._drag_img_origin = (coords[0], coords[1])

    def _on_drag_move(self, event):
        if not self._drag_start or not self._drag_view:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]

        orig = self._drag_view
        dt = orig[1] - orig[0]
        df = orig[3] - orig[2]

        # 드래그 방향: 왼쪽 드래그 = 시간 전진, 위로 드래그 = 주파수 상승
        t_shift = -dx / max(cw, 1) * dt
        f_shift = dy / max(ch, 1) * df

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
            ox, oy = self._drag_img_origin
            self.canvas.coords(self._img_id, ox + px_dx, oy + px_dy)

        # 드래그 중에는 250ms 디바운스로 정밀 렌더링 예약
        self._draw_detections()
        self._schedule_render(250)

    def _on_drag_end(self, event):
        self._drag_start = None
        self._drag_view = None
        # 드래그 종료 시 즉시 정밀 렌더링
        self._schedule_render(0)

    def _on_resize(self, event):
        self._schedule_render(300)

    def _zoom_center(self, factor):
        """화면 중앙 기준 확대/축소 — 부드러운 애니메이션"""
        if self._zoom_anim_id:
            self.canvas.after_cancel(self._zoom_anim_id)
            self._zoom_anim_id = None

        # 애니메이션 단계 수
        steps = 5
        step_factor = factor ** (1.0 / steps)
        self._zoom_animate(step_factor, steps)

    def _zoom_animate(self, step_factor, remaining):
        """단계적으로 줌을 적용하는 애니메이션"""
        if remaining <= 0:
            self._schedule_render(0)
            return

        t_mid = (self.t_start + self.t_end) / 2
        f_mid = (self.f_low + self.f_high) / 2
        dt = (self.t_end - self.t_start) * step_factor / 2
        df = (self.f_high - self.f_low) * step_factor / 2

        self.t_start = t_mid - dt
        self.t_end = t_mid + dt
        self.f_low = f_mid - df
        self.f_high = f_mid + df

        self._clamp_view()
        self._apply_canvas_transform()
        self._draw_detections()

        # 다음 프레임 예약 (~16ms ≈ 60fps)
        self._zoom_anim_id = self.canvas.after(
            16, self._zoom_animate, step_factor, remaining - 1
        )

    def _apply_canvas_transform(self):
        """마지막 렌더링 뷰와 현재 뷰의 차이로 캔버스 이미지를 즉시 변환"""
        if not self._img_id or not self._rendered_view:
            return

        rv = self._rendered_view
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        rv_dt = rv[1] - rv[0]
        rv_df = rv[3] - rv[2]
        if rv_dt <= 0 or rv_df <= 0:
            return

        px_x = -(self.t_start - rv[0]) / rv_dt * cw
        px_y = (self.f_high - rv[3]) / rv_df * ch

        scale_x = rv_dt / (self.t_end - self.t_start) if (self.t_end - self.t_start) > 0 else 1.0
        scale_y = rv_df / (self.f_high - self.f_low) if (self.f_high - self.f_low) > 0 else 1.0

        if self._photo and hasattr(self, '_last_pil_img'):
            try:
                new_w = max(1, int(cw * scale_x))
                new_h = max(1, int(ch * scale_y))
                resized = self._last_pil_img.resize((new_w, new_h), Image.NEAREST)
                self._photo_preview = ImageTk.PhotoImage(resized)
                self.win._refs.append(self._photo_preview)
                self.canvas.itemconfigure(self._img_id, image=self._photo_preview)
                self.canvas.coords(self._img_id, px_x, px_y)
            except Exception:
                pass
        else:
            self.canvas.coords(self._img_id, px_x, px_y)

    def _reset_view(self):
        """전체 보기로 복원"""
        if self._zoom_anim_id:
            self.canvas.after_cancel(self._zoom_anim_id)
            self._zoom_anim_id = None
        self.t_start = 0.0
        self.t_end = self.duration
        self.f_low = 0.0
        self.f_high = self.max_freq
        self._schedule_render(0)

    # ---- 오디오 재생 ----

    def _set_speed(self, speed):
        """재생 속도 변경 (재생 중이면 현재 위치에서 재시작)"""
        self._play_speed = speed
        self._speed_var.set(speed)
        if self._playing:
            self._restart_from_current()

    def _on_volume_change(self, value):
        """볼륨 슬라이더 변경 콜백 (디바운스 200ms)"""
        vol = int(float(value))
        self._vol_label.config(text=f"{vol}%")
        if self._playing:
            if hasattr(self, '_vol_debounce_id') and self._vol_debounce_id:
                self.frame.after_cancel(self._vol_debounce_id)
            self._vol_debounce_id = self.frame.after(
                200, self._restart_from_current)

    def _get_current_play_time(self):
        """현재 재생 위치를 원본 타임라인 기준으로 반환"""
        if not self._playing:
            return self._play_start_time
        elapsed = time.time() - self._play_start_wall
        current = self._play_start_time + elapsed * self._play_speed
        return min(current, self._play_end_time)

    def _restart_from_current(self):
        """현재 위치에서 재생을 재시작 (속도/볼륨 변경 시)"""
        if not self._playing:
            return
        current_time = self._get_current_play_time()
        end_time = getattr(self, '_play_range_t1', self._play_end_time)
        if current_time >= end_time - 0.05:
            return
        self._stop_event.set()
        self._clear_playhead()
        self._playing = False
        self._start_playback(current_time, end_time)

    def _play_view(self):
        """현재 뷰 범위의 음원을 재생"""
        if not self._loaded or self.data is None:
            return
        if not HAS_PLAYBACK:
            self._play_status_var.set("⚠ 오디오 재생 불가 (pip install sounddevice soundfile)")
            return
        self._stop_playback()
        t0 = max(0, self.t_start)
        t1 = min(self.duration, self.t_end)
        self._start_playback(t0, t1)

    def _play_detection(self, det):
        """특정 검출 구간 재생 (전후 1초 마진)"""
        if not self._loaded or self.data is None:
            return
        if not HAS_PLAYBACK:
            self._play_status_var.set("⚠ 오디오 재생 불가 (pip install sounddevice soundfile)")
            return
        self._stop_playback()
        det_time = det.get("time", 0)
        margin = 1.5
        t0 = max(0, det_time - margin)
        t1 = min(self.duration, det_time + margin)

        view_margin = margin + 1.0
        self.t_start = max(0, det_time - view_margin)
        self.t_end = min(self.duration, det_time + view_margin)
        self._clamp_view()
        self._schedule_render(0)

        self.frame.after(150, lambda: self._start_playback(t0, t1))

    def _play_selected_detection(self):
        """드롭다운에서 선택된 검출 구간 재생"""
        if not hasattr(self, '_det_combo_var'):
            return
        selected = self._det_combo_var.get()
        if selected and selected in self._det_map:
            self._play_detection(self._det_map[selected])

    def _on_detection_select(self, event):
        """검출 구간 드롭다운 선택 시 해당 위치로 뷰 이동"""
        selected = self._det_combo_var.get()
        if not selected or selected not in self._det_map:
            return
        det = self._det_map[selected]
        det_time = det.get("time", 0)
        margin = 2.5
        self.t_start = max(0, det_time - margin)
        self.t_end = min(self.duration, det_time + margin)
        self._clamp_view()
        self._schedule_render(0)

    def _start_playback(self, t0, t1):
        """지정 구간 재생 시작 (백그라운드 스레드)"""
        if self._playing:
            self._stop_playback()

        # 원래 구간 저장 (restart 시 필요)
        self._play_range_t0 = t0
        self._play_range_t1 = t1

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

        # 임시 WAV 파일 생성
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
        finally:
            os.close(tmp_fd)

        self._play_temp_wav = tmp_path

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

        # 백그라운드 재생 (audio.playback 모듈 사용)
        def _on_play_done(error):
            is_current = (self._play_generation == gen)
            if error and is_current:
                self.frame.after(0, lambda m=error: self._play_status_var.set(f"재생 오류: {m}"))
            if is_current:
                self.frame.after(0, lambda g=gen: self._on_playback_done(g))

        self._play_thread = play_wav_async(
            tmp_path, stop_event, actual_duration, on_done=_on_play_done
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
            self.frame.after(3000, lambda: self._play_status_var.set(""))

    def _update_playhead(self):
        """재생 중 플레이헤드 위치를 업데이트"""
        if not self._playing:
            return

        elapsed = time.time() - self._play_start_wall
        current_time = self._play_start_time + elapsed * self._play_speed

        if current_time > self._play_end_time:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        view_dt = self.t_end - self.t_start
        if view_dt > 0 and cw > 0:
            px = (current_time - self.t_start) / view_dt * cw
            if 0 <= px <= cw:
                if self._playhead_id:
                    self.canvas.coords(self._playhead_id, px, 0, px, ch)
                else:
                    self._playhead_id = self.canvas.create_line(
                        px, 0, px, ch,
                        fill="#FF3333", width=2, tags="playhead"
                    )
            else:
                if self._playhead_id:
                    self.canvas.delete(self._playhead_id)
                    self._playhead_id = None

        self._play_status_var.set(
            f"▶ {current_time:.1f}s / {self._play_end_time:.1f}s  ({self._play_speed}x)"
        )

        self._playhead_after_id = self.frame.after(33, self._update_playhead)

    def _clear_playhead(self):
        """플레이헤드 제거"""
        if self._playhead_after_id:
            self.frame.after_cancel(self._playhead_after_id)
            self._playhead_after_id = None
        if self._playhead_id:
            self.canvas.delete(self._playhead_id)
            self._playhead_id = None
