# ============================================================
# ui/template_selector.py — 스펙트로그램 기반 템플릿 구간 선택기
# ============================================================

import tkinter as tk
from tkinter import ttk
import threading
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

# 모듈 내 참조
from colormaps import COLORMAPS


class TemplateSelector:
    """종 음원의 스펙트로그램을 표시하고 드래그로 시간/주파수 구간을 선택."""

    RENDER_W = 1200
    RENDER_H = 600

    def __init__(self, parent, wav_path, callback):
        """
        parent: 부모 위젯
        wav_path: WAV 파일 경로
        callback: 선택 완료 시 호출 — callback(t_start, t_end, f_low, f_high)
        """
        self.callback = callback
        self.wav_path = wav_path

        # 창 설정
        self.win = tk.Toplevel(parent)
        self.win.title(f"📊 구간 선택 — {Path(wav_path).name}")
        self.win.geometry("1100x650")
        self.win.transient(parent)
        self.win.grab_set()

        # 데이터
        self.sr = None
        self.data = None
        self.duration = 0
        self.max_freq = 22050

        # 뷰 범위
        self.t_start = 0.0
        self.t_end = 1.0
        self.f_low = 0.0
        self.f_high = 22050

        # 렌더 상태
        self._rendered_view = None
        self._render_after_id = None
        self._rendering = False
        self._render_gen = 0
        self._loaded = False

        # 팬 (우클릭) 상태
        self._pan_start = None
        self._pan_view = None

        # 선택 (좌클릭 드래그) 상태
        self._sel_start = None   # (x, y) 픽셀
        self._sel_rect_id = None
        self._sel_info_id = None
        self._selection = None   # (t0, t1, f0, f1) 최종 선택값

        self._build_ui()

        # 로딩
        self.info_var.set("WAV 파일 로딩 중...")
        threading.Thread(target=self._load_wav, daemon=True).start()

    def _build_ui(self):
        # 상단 정보바
        top = ttk.Frame(self.win)
        top.pack(fill="x", padx=5, pady=3)
        ttk.Label(top, text="좌클릭 드래그: 구간 선택  |  우클릭 드래그: 이동  |  휠: 확대/축소",
                  foreground="gray").pack(side="left")
        ttk.Button(top, text="↺ 전체 보기", command=self._reset_view).pack(side="right", padx=5)

        # 캔버스
        self.canvas = tk.Canvas(self.win, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=2)

        # 선택 정보 + 버튼 바
        bottom = ttk.Frame(self.win)
        bottom.pack(fill="x", padx=5, pady=5)

        self.sel_label = ttk.Label(bottom, text="선택 영역: (없음)", font=("Consolas", 10))
        self.sel_label.pack(side="left", padx=10)

        ttk.Button(bottom, text="✅ 확인 (선택 적용)", command=self._confirm).pack(side="right", padx=5)
        ttk.Button(bottom, text="❌ 취소", command=self.win.destroy).pack(side="right")

        # 하단 상태바
        self.info_var = tk.StringVar()
        ttk.Label(self.win, textvariable=self.info_var, foreground="gray",
                  font=("Consolas", 9)).pack(fill="x", padx=5, pady=(0, 3))

        # 이벤트 바인딩
        self.canvas.bind("<ButtonPress-1>", self._on_sel_start)
        self.canvas.bind("<B1-Motion>", self._on_sel_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_sel_end)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<Configure>", lambda e: self._schedule_render(100))

    def _load_wav(self):
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(self.wav_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype != np.float64:
                if np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.float64) / np.iinfo(data.dtype).max
                else:
                    data = data.astype(np.float64)
            self.sr = sr
            self.data = data
            self.duration = len(data) / sr
            self.max_freq = sr // 2
            self.t_start = 0.0
            self.t_end = self.duration
            self.f_low = 0.0
            self.f_high = self.max_freq
            self._loaded = True
            self.win.after(0, self._on_loaded)
        except Exception as e:
            self.win.after(0, lambda: self.info_var.set(f"로드 오류: {e}"))

    def _on_loaded(self):
        self.info_var.set(f"로드 완료: {self.duration:.1f}초, {self.sr}Hz")
        self._schedule_render(0)

    # ---- 렌더링 ----
    def _schedule_render(self, delay_ms=200):
        if self._render_after_id:
            self.canvas.after_cancel(self._render_after_id)
        self._render_after_id = self.canvas.after(delay_ms, self._render)

    def _render(self):
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
            self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img, tags="specimg")
            self._rendered_view = (self.t_start, self.t_end, self.f_low, self.f_high)
            self._redraw_selection()
        if isinstance(info, str):
            self.info_var.set(info)

    # ---- 좌클릭 드래그: 구간 선택 ----
    def _on_sel_start(self, event):
        self._sel_start = (event.x, event.y)
        if self._sel_rect_id:
            self.canvas.delete(self._sel_rect_id)
            self._sel_rect_id = None
        if self._sel_info_id:
            self.canvas.delete(self._sel_info_id)
            self._sel_info_id = None

    def _on_sel_move(self, event):
        if not self._sel_start or not self._loaded:
            return
        x0, y0 = self._sel_start
        x1, y1 = event.x, event.y
        if self._sel_rect_id:
            self.canvas.coords(self._sel_rect_id, x0, y0, x1, y1)
        else:
            self._sel_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="#FF4444", width=2, dash=(6, 3)
            )
        t0, t1, f0, f1 = self._px_to_range(x0, y0, x1, y1)
        self.sel_label.config(
            text=f"선택: {t0:.2f}~{t1:.2f}초, {f0:.0f}~{f1:.0f} Hz"
        )

    def _on_sel_end(self, event):
        if not self._sel_start or not self._loaded:
            return
        x0, y0 = self._sel_start
        x1, y1 = event.x, event.y
        self._sel_start = None

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return

        t0, t1, f0, f1 = self._px_to_range(x0, y0, x1, y1)
        self._selection = (t0, t1, f0, f1)
        self.sel_label.config(
            text=f"✅ 선택됨: {t0:.2f}~{t1:.2f}초, {f0:.0f}~{f1:.0f} Hz"
        )

    def _px_to_range(self, x0, y0, x1, y1):
        """픽셀 좌표를 시간/주파수 범위로 변환"""
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        px_left, px_right = min(x0, x1), max(x0, x1)
        px_top, px_bottom = min(y0, y1), max(y0, y1)
        view_dt = self.t_end - self.t_start
        t0 = self.t_start + (px_left / cw) * view_dt
        t1 = self.t_start + (px_right / cw) * view_dt
        view_df = self.f_high - self.f_low
        f1 = self.f_high - (px_top / ch) * view_df
        f0 = self.f_high - (px_bottom / ch) * view_df
        t0 = max(0, min(t0, self.duration))
        t1 = max(0, min(t1, self.duration))
        f0 = max(0, f0)
        f1 = min(self.max_freq, f1)
        return t0, t1, f0, f1

    def _redraw_selection(self):
        """렌더링 후 선택 영역을 다시 그리기"""
        if self._sel_rect_id:
            self.canvas.delete(self._sel_rect_id)
            self._sel_rect_id = None
        if self._sel_info_id:
            self.canvas.delete(self._sel_info_id)
            self._sel_info_id = None
        if not self._selection:
            return
        t0, t1, f0, f1 = self._selection
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        view_dt = self.t_end - self.t_start
        view_df = self.f_high - self.f_low
        if view_dt <= 0 or view_df <= 0:
            return
        px_left = (t0 - self.t_start) / view_dt * cw
        px_right = (t1 - self.t_start) / view_dt * cw
        px_top = (1.0 - (f1 - self.f_low) / view_df) * ch
        px_bottom = (1.0 - (f0 - self.f_low) / view_df) * ch
        self._sel_rect_id = self.canvas.create_rectangle(
            px_left, px_top, px_right, px_bottom,
            outline="#FF4444", width=2, dash=(6, 3)
        )

    # ---- 우클릭 드래그: 팬 ----
    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)
        self._pan_view = (self.t_start, self.t_end, self.f_low, self.f_high)

    def _on_pan_move(self, event):
        if not self._pan_start or not self._loaded:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        t0, t1, fl, fh = self._pan_view
        dt = (t1 - t0) * dx / cw
        df = (fh - fl) * dy / ch
        self.t_start = t0 - dt
        self.t_end = t1 - dt
        self.f_low = fl + df
        self.f_high = fh + df
        self._clamp_view()
        self._apply_visual_transform()
        self._schedule_render(150)

    # ---- 휠: 줌 ----
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
        self._apply_visual_transform()
        self._schedule_render(100)

    def _apply_visual_transform(self):
        """뷰 변경 시 기존 이미지를 즉시 이동/스케일링 (렌더링 완료 전 시각 피드백)"""
        if not self._rendered_view or not hasattr(self, '_tk_img') or not self._tk_img:
            return
        old_t0, old_t1, old_fl, old_fh = self._rendered_view
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        new_dt = self.t_end - self.t_start
        new_df = self.f_high - self.f_low
        old_dt = old_t1 - old_t0
        old_df = old_fh - old_fl
        if new_dt <= 0 or new_df <= 0 or old_dt <= 0 or old_df <= 0:
            return
        ox = (old_t0 - self.t_start) / new_dt * cw
        oy = (self.f_high - old_fh) / new_df * ch
        items = self.canvas.find_withtag("specimg")
        if items:
            self.canvas.coords(items[0], ox, oy)

    def _clamp_view(self):
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

    def _confirm(self):
        if not self._selection:
            self.sel_label.config(text="⚠ 먼저 영역을 드래그로 선택하세요!")
            return
        t0, t1, f0, f1 = self._selection
        self.callback(t0, t1, f0, f1)
        self.win.destroy()
