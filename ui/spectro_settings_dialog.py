# ============================================================
#  R 스펙토그램 내보내기 설정 다이얼로그 (실시간 미리보기 포함)
# ============================================================

import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from PIL import Image, ImageTk
from colormaps import COLORMAPS, MAGMA_LUT


class SpectroSettingsDialog:
    """R 스펙토그램 내보내기 파라미터를 설정하는 모달 다이얼로그.
    오른쪽에 Python 기반 실시간 미리보기를 표시한다."""

    PALETTES = [
        ("spectro.colors", "spectro.colors"),
        ("reverse.gray", "reverse.gray"),
        ("heat", "heat"),
        ("terrain", "terrain"),
    ]
    WL_OPTIONS = [128, 256, 512, 1024, 2048, 4096]

    # Python 미리보기 ↔ R 팔레트 매핑
    _PY_CMAP = {
        "spectro.colors": "Turbo",
        "reverse.gray": "Grayscale",
        "heat": "Inferno",
        "terrain": "Viridis",
    }

    PREVIEW_W = 700
    PREVIEW_H = 400

    def __init__(self, parent, has_detections=False, defaults=None,
                 wav_data=None, sr=None, wav_path=None,
                 t_start=None, t_end=None):
        """
        Args:
            parent:         부모 tkinter 위젯
            has_detections: 검출 결과 여부
            defaults:       기본값 dict
            wav_data:       numpy 오디오 array (미리보기용, 이미 로드됨)
            sr:             샘플레이트
            wav_path:       WAV 경로 (wav_data 없으면 내부에서 로드)
            t_start/t_end:  시간 범위
        """
        self.result = None
        self._sr = sr
        self._data = wav_data
        self._t_start_audio = t_start or 0.0
        self._t_end_audio = t_end
        self._render_after_id = None
        self._render_gen = 0
        self._rendering = False
        self._drag_start = None
        self._drag_view = None
        self._rendered_view = None
        self._photo = None  # ImageTk 참조 유지

        # WAV 로드 (데이터가 없으면)
        if self._data is None and wav_path:
            self._load_wav(wav_path)

        if self._data is not None and self._t_end_audio is None:
            self._t_end_audio = len(self._data) / self._sr

        self.win = tk.Toplevel(parent)
        self.win.title("R 스펙토그램 내보내기 설정")
        self.win.resizable(True, True)
        self.win.transient(parent)
        self.win.grab_set()

        d = defaults or {}

        # ---- 최상단 2-pane ----
        paned = ttk.PanedWindow(self.win, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        # ==== 왼쪽: 설정 패널 ====
        left = ttk.Frame(paned, width=340)
        paned.add(left, weight=0)

        self._build_settings(left, d, has_detections)

        # ==== 오른쪽: 미리보기 ====
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._build_preview(right)

        # ==== 하단 버튼 ====
        frm_btn = ttk.Frame(self.win)
        frm_btn.pack(pady=(0, 8))
        ttk.Button(frm_btn, text="📄 내보내기", command=self._on_ok).pack(side="left", padx=(0, 8))
        ttk.Button(frm_btn, text="취소", command=self._on_cancel).pack(side="left")

        # 모달
        self.win.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.win.update_idletasks()
        # 화면 중앙
        w = max(self.win.winfo_width(), 1060)
        h = max(self.win.winfo_height(), 600)
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.win.geometry(f"{w}x{h}+{max(0,x)}+{max(0,y)}")

        # 초기 미리보기 렌더링
        self.win.after(100, self._schedule_render)

        self.win.wait_window()

    # ──────────────────────────────────────────
    #  WAV 로드
    # ──────────────────────────────────────────
    def _load_wav(self, wav_path):
        try:
            from scipy.io import wavfile as scipy_wavfile
            sr, data = scipy_wavfile.read(wav_path)
            if data.ndim > 1:
                data = data[:, 0]
            self._sr = sr
            self._data = data.astype(np.float64)
        except Exception:
            self._data = None

    # ──────────────────────────────────────────
    #  왼쪽 설정 패널
    # ──────────────────────────────────────────
    def _build_settings(self, parent, d, has_detections):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=8)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 주파수
        lf = ttk.LabelFrame(inner, text=" 🎵 주파수 범위 (Hz) ", padding=6)
        lf.pack(fill="x", pady=(0, 4))

        frm = ttk.Frame(lf)
        frm.pack(fill="x")
        ttk.Label(frm, text="하한:").pack(side="left")
        self.var_f_low = tk.IntVar(value=d.get("f_low", 500))
        ttk.Spinbox(frm, from_=0, to=24000, increment=100,
                    textvariable=self.var_f_low, width=7).pack(side="left", padx=4)
        ttk.Label(frm, text="상한:").pack(side="left", padx=(8, 0))
        self.var_f_high = tk.IntVar(value=d.get("f_high", 12000))
        ttk.Spinbox(frm, from_=100, to=48000, increment=100,
                    textvariable=self.var_f_high, width=7).pack(side="left", padx=4)

        self.var_f_low.trace_add("write", lambda *_: self._on_setting_change())
        self.var_f_high.trace_add("write", lambda *_: self._on_setting_change())

        # dB
        lf = ttk.LabelFrame(inner, text=" 🔆 밝기 / 대비 (dB) ", padding=6)
        lf.pack(fill="x", pady=(0, 4))

        frm = ttk.Frame(lf)
        frm.pack(fill="x")
        ttk.Label(frm, text="하한:").pack(side="left")
        self.var_db_min = tk.IntVar(value=d.get("dB_min", -60))
        sc = ttk.Scale(frm, from_=-120, to=-10, variable=self.var_db_min,
                       orient="horizontal", length=100, command=lambda _: self._on_setting_change())
        sc.pack(side="left", padx=4)
        self.lbl_db_min = ttk.Label(frm, text=str(self.var_db_min.get()), width=4)
        self.lbl_db_min.pack(side="left")

        frm = ttk.Frame(lf)
        frm.pack(fill="x", pady=(2, 0))
        ttk.Label(frm, text="상한:").pack(side="left")
        self.var_db_max = tk.IntVar(value=d.get("dB_max", 0))
        sc = ttk.Scale(frm, from_=-60, to=0, variable=self.var_db_max,
                       orient="horizontal", length=100, command=lambda _: self._on_setting_change())
        sc.pack(side="left", padx=4)
        self.lbl_db_max = ttk.Label(frm, text=str(self.var_db_max.get()), width=4)
        self.lbl_db_max.pack(side="left")

        frm = ttk.Frame(lf)
        frm.pack(fill="x", pady=(2, 0))
        ttk.Label(frm, text="레벨:").pack(side="left")
        self.var_collevels = tk.IntVar(value=d.get("collevels", 30))
        ttk.Spinbox(frm, from_=10, to=100, increment=5,
                    textvariable=self.var_collevels, width=5).pack(side="left", padx=4)

        self.var_db_min.trace_add("write", lambda *_: self.lbl_db_min.config(text=str(self.var_db_min.get())))
        self.var_db_max.trace_add("write", lambda *_: self.lbl_db_max.config(text=str(self.var_db_max.get())))

        # FFT
        lf = ttk.LabelFrame(inner, text=" 📐 FFT ", padding=6)
        lf.pack(fill="x", pady=(0, 4))

        frm = ttk.Frame(lf)
        frm.pack(fill="x")
        ttk.Label(frm, text="wl:").pack(side="left")
        self.var_wl = tk.IntVar(value=d.get("wl", 512))
        cb = ttk.Combobox(frm, textvariable=self.var_wl, width=6,
                          values=self.WL_OPTIONS, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _: self._on_setting_change())

        ttk.Label(frm, text="ovlp:").pack(side="left", padx=(8, 0))
        self.var_ovlp = tk.IntVar(value=d.get("ovlp", 75))
        sc = ttk.Scale(frm, from_=0, to=99, variable=self.var_ovlp,
                       orient="horizontal", length=80, command=lambda _: self._on_setting_change())
        sc.pack(side="left", padx=4)
        self.lbl_ovlp = ttk.Label(frm, text=f"{self.var_ovlp.get()}%", width=4)
        self.lbl_ovlp.pack(side="left")
        self.var_ovlp.trace_add("write", lambda *_: self.lbl_ovlp.config(text=f"{self.var_ovlp.get()}%"))

        # 이미지 크기
        lf = ttk.LabelFrame(inner, text=" 🖼️ 이미지 ", padding=6)
        lf.pack(fill="x", pady=(0, 4))

        frm = ttk.Frame(lf)
        frm.pack(fill="x")
        ttk.Label(frm, text="W:").pack(side="left")
        self.var_width = tk.IntVar(value=d.get("width", 1600))
        ttk.Spinbox(frm, from_=400, to=6000, increment=100,
                    textvariable=self.var_width, width=6).pack(side="left", padx=4)
        ttk.Label(frm, text="H:").pack(side="left", padx=(4, 0))
        self.var_height = tk.IntVar(value=d.get("height", 800))
        ttk.Spinbox(frm, from_=200, to=4000, increment=100,
                    textvariable=self.var_height, width=6).pack(side="left", padx=4)
        ttk.Label(frm, text="DPI:").pack(side="left", padx=(4, 0))
        self.var_res = tk.IntVar(value=d.get("res", 150))
        ttk.Spinbox(frm, from_=72, to=600, increment=10,
                    textvariable=self.var_res, width=4).pack(side="left", padx=4)

        # 팔레트
        lf = ttk.LabelFrame(inner, text=" 🎨 팔레트 ", padding=6)
        lf.pack(fill="x", pady=(0, 4))

        self.var_palette = tk.StringVar(value=d.get("palette", "spectro.colors"))
        for label, val in self.PALETTES:
            ttk.Radiobutton(lf, text=label, variable=self.var_palette,
                            value=val, command=self._on_setting_change).pack(anchor="w")

        # 표시 요소
        lf = ttk.LabelFrame(inner, text=" 👁️ 표시 요소 ", padding=6)
        lf.pack(fill="x", pady=(0, 4))

        self.var_show_title = tk.BooleanVar(value=d.get("show_title", True))
        ttk.Checkbutton(lf, text="제목", variable=self.var_show_title).pack(anchor="w")

        self.var_show_scale = tk.BooleanVar(value=d.get("show_scale", True))
        ttk.Checkbutton(lf, text="스케일바", variable=self.var_show_scale).pack(anchor="w")

        self.var_show_osc = tk.BooleanVar(value=d.get("show_osc", False))
        ttk.Checkbutton(lf, text="오실로그램", variable=self.var_show_osc).pack(anchor="w")

        self.var_show_det = tk.BooleanVar(
            value=d.get("show_detections", True) if has_detections else False)
        self.cb_det = ttk.Checkbutton(lf, text="검출 오버레이",
                                      variable=self.var_show_det,
                                      state="normal" if has_detections else "disabled")
        self.cb_det.pack(anchor="w")

        frm = ttk.Frame(lf)
        frm.pack(fill="x", pady=(2, 0))
        ttk.Label(frm, text="라벨 크기:").pack(side="left")
        self.var_det_cex = tk.DoubleVar(value=d.get("det_cex", 0.7))
        sc = ttk.Scale(frm, from_=0.3, to=1.5, variable=self.var_det_cex,
                       orient="horizontal", length=80)
        sc.pack(side="left", padx=4)
        self.lbl_cex = ttk.Label(frm, text=f"{self.var_det_cex.get():.1f}", width=3)
        self.lbl_cex.pack(side="left")
        self.var_det_cex.trace_add("write", lambda *_: self.lbl_cex.config(
            text=f"{self.var_det_cex.get():.1f}"))

    # ──────────────────────────────────────────
    #  오른쪽 미리보기 패널
    # ──────────────────────────────────────────
    def _build_preview(self, parent):
        ttk.Label(parent, text="미리보기  (마우스 휠: 주파수 줌 / 드래그: 패닝 / Shift+휠: 시간 줌)",
                  font=("", 9)).pack(anchor="w", padx=4, pady=(0, 2))

        self.canvas = tk.Canvas(parent, bg="#1a1a2e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # 마우스 이벤트
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<Configure>", lambda e: self._schedule_render(delay_ms=150))

        # 정보 라벨
        self.info_var = tk.StringVar(value="로딩 중...")
        ttk.Label(parent, textvariable=self.info_var, font=("", 8)).pack(anchor="w", padx=4)

    # ──────────────────────────────────────────
    #  설정 변경 콜백
    # ──────────────────────────────────────────
    def _on_setting_change(self):
        self._schedule_render(delay_ms=300)

    # ──────────────────────────────────────────
    #  렌더링
    # ──────────────────────────────────────────
    def _schedule_render(self, delay_ms=200):
        if self._render_after_id:
            self.win.after_cancel(self._render_after_id)
        self._render_after_id = self.win.after(delay_ms, self._render)

    def _render(self):
        if self._data is None or self._rendering:
            return
        self._rendering = True
        self._render_gen += 1
        gen = self._render_gen

        try:
            cw = max(self.canvas.winfo_width(), 100)
            ch = max(self.canvas.winfo_height(), 100)
        except Exception:
            self._rendering = False
            return

        params = {
            "t_start": self._t_start_audio,
            "t_end": self._t_end_audio,
            "f_low": max(0, self.var_f_low.get()),
            "f_high": max(1, self.var_f_high.get()),
            "dB_min": self.var_db_min.get(),
            "dB_max": self.var_db_max.get(),
            "wl": self.var_wl.get(),
            "ovlp": self.var_ovlp.get(),
            "cw": cw,
            "ch": ch,
            "gen": gen,
            "palette": self.var_palette.get(),
        }
        threading.Thread(target=self._render_worker, args=(params,), daemon=True).start()

    def _render_worker(self, p):
        try:
            t_start = p["t_start"]
            t_end = p["t_end"]
            f_low = p["f_low"]
            f_high = p["f_high"]
            cw = p["cw"]
            ch = p["ch"]
            gen = p["gen"]

            i_start = max(0, int(t_start * self._sr))
            i_end = min(len(self._data), int(t_end * self._sr))
            segment = self._data[i_start:i_end]

            if len(segment) < 64:
                self._on_render_done_safe(None, "세그먼트 너무 짧음", gen)
                return

            # 다운샘플링 (성능 최적화)
            effective_sr = self._sr
            max_samples = cw * 256
            if len(segment) > max_samples:
                from scipy.signal import decimate as _decimate
                step = len(segment) // max_samples
                if step >= 2:
                    segment = _decimate(segment, step)
                    effective_sr = self._sr / step

            # FFT
            nperseg = min(p["wl"], len(segment))
            if nperseg < 32:
                nperseg = 32
            noverlap = int(nperseg * p["ovlp"] / 100.0)
            if noverlap >= nperseg:
                noverlap = nperseg - 1

            freqs, times, Sxx = scipy_spectrogram(
                segment, fs=effective_sr,
                nperseg=nperseg, noverlap=noverlap,
                window="hann"
            )

            f_mask = (freqs >= f_low) & (freqs <= f_high)
            Sxx = Sxx[f_mask, :]

            if Sxx.size == 0:
                self._on_render_done_safe(None, "데이터 없음", gen)
                return

            Sxx_db = 10 * np.log10(Sxx + 1e-12)

            # dB 범위 클리핑 (사용자 설정)
            dB_min = p["dB_min"]
            dB_max = p["dB_max"]
            if dB_max <= dB_min:
                dB_max = dB_min + 1
            normalized = np.clip((Sxx_db - dB_min) / (dB_max - dB_min), 0, 1)

            # 컬러맵 적용
            py_cmap = self._PY_CMAP.get(p["palette"], "Magma")
            lut = COLORMAPS.get(py_cmap, MAGMA_LUT)
            indices = (normalized * 255).astype(np.uint8)
            rgb = lut[indices]
            rgb = rgb[::-1, :, :]  # 높은 주파수가 위

            pil_img = Image.fromarray(rgb, mode="RGB")
            resample = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
            pil_img = pil_img.resize((cw, ch), resample)

            info = (f"시간 {t_start:.1f}~{t_end:.1f}s | "
                    f"주파수 {f_low:.0f}~{f_high:.0f} Hz | "
                    f"FFT {nperseg}")

            self._on_render_done_safe(pil_img, info, gen)

        except Exception as e:
            self._on_render_done_safe(None, f"오류: {e}", gen)

    def _on_render_done_safe(self, img, info, gen):
        try:
            self.win.after(0, self._on_render_done, img, info, gen)
        except Exception:
            pass

    def _on_render_done(self, pil_img, info, gen):
        self._rendering = False
        if gen != self._render_gen:
            return  # 구세대 결과 무시
        if pil_img:
            self._photo = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
            self._rendered_view = (
                self._t_start_audio, self._t_end_audio,
                self.var_f_low.get(), self.var_f_high.get()
            )
        if info:
            self.info_var.set(info)

    # ──────────────────────────────────────────
    #  마우스 인터랙션
    # ──────────────────────────────────────────
    def _on_wheel(self, event):
        if self._data is None:
            return
        # 줌 방향
        if hasattr(event, "delta"):
            zoom_in = event.delta > 0
        else:
            zoom_in = event.num == 4

        factor = 0.85 if zoom_in else 1.18

        shift = bool(event.state & 0x1)

        if shift:
            # Shift + 휠: 시간축 줌
            t_mid = (self._t_start_audio + self._t_end_audio) / 2
            t_half = (self._t_end_audio - self._t_start_audio) / 2 * factor
            total_dur = len(self._data) / self._sr
            self._t_start_audio = max(0, t_mid - t_half)
            self._t_end_audio = min(total_dur, t_mid + t_half)
        else:
            # 일반 휠: 주파수축 줌 (마우스 위치 기준)
            ch = self.canvas.winfo_height()
            if ch < 1:
                return
            f_low = self.var_f_low.get()
            f_high = self.var_f_high.get()

            # 마우스 Y → 주파수 (위=높은 주파수)
            ratio = 1.0 - (event.y / ch)
            f_at_mouse = f_low + ratio * (f_high - f_low)

            new_half = (f_high - f_low) / 2 * factor
            new_low = f_at_mouse - new_half * (1 - ratio) * 2  # 마우스 위치 유지
            new_high = f_at_mouse + new_half * ratio * 2

            # 클램프
            nyquist = self._sr / 2 if self._sr else 24000
            new_low = max(0, new_low)
            new_high = min(nyquist, new_high)
            if new_high - new_low < 100:
                return

            self.var_f_low.set(int(new_low))
            self.var_f_high.set(int(new_high))
            return  # _on_setting_change가 trace에서 호출됨

        self._schedule_render(delay_ms=100)

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)
        self._drag_view = (
            self._t_start_audio, self._t_end_audio,
            self.var_f_low.get(), self.var_f_high.get()
        )

    def _on_drag_move(self, event):
        if self._drag_start is None or self._drag_view is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)

        t0, t1, fl, fh = self._drag_view
        total_dur = len(self._data) / self._sr if self._data is not None else 1
        nyquist = self._sr / 2 if self._sr else 24000

        # 시간 패닝
        t_range = t1 - t0
        dt = -dx / cw * t_range
        new_t0 = max(0, min(total_dur - t_range, t0 + dt))
        new_t1 = new_t0 + t_range

        # 주파수 패닝 (Y축: 위=고주파)
        f_range = fh - fl
        df = dy / ch * f_range
        new_fl = max(0, min(nyquist - f_range, fl + df))
        new_fh = new_fl + f_range

        self._t_start_audio = new_t0
        self._t_end_audio = new_t1
        self.var_f_low.set(int(new_fl))
        self.var_f_high.set(int(new_fh))

        # 캔버스 즉시 변환 (부드러움)
        self._apply_canvas_transform()

    def _on_drag_end(self, event):
        if self._drag_start:
            self._drag_start = None
            self._schedule_render(delay_ms=100)

    def _apply_canvas_transform(self):
        """드래그 중 마지막 렌더링과의 차이로 캔버스를 즉시 변환."""
        if not self._rendered_view or not self._photo:
            return
        rt0, rt1, rfl, rfh = self._rendered_view
        ct0 = self._t_start_audio
        ct1 = self._t_end_audio
        cfl = self.var_f_low.get()
        cfh = self.var_f_high.get()

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)

        # 시간축 오프셋
        t_range_r = rt1 - rt0
        if t_range_r > 0:
            dx = -(ct0 - rt0) / t_range_r * cw
        else:
            dx = 0

        # 주파수축 오프셋 (Y 반전)
        f_range_r = rfh - rfl
        if f_range_r > 0:
            dy = (cfl - rfl) / f_range_r * ch
        else:
            dy = 0

        self.canvas.delete("all")
        self.canvas.create_image(int(dx), int(dy), anchor="nw", image=self._photo)

    # ──────────────────────────────────────────
    #  확인 / 취소
    # ──────────────────────────────────────────
    def _on_ok(self):
        self.result = {
            "t_start": self._t_start_audio,
            "t_end": self._t_end_audio,
            "f_low": self.var_f_low.get(),
            "f_high": self.var_f_high.get(),
            "dB_min": self.var_db_min.get(),
            "dB_max": self.var_db_max.get(),
            "collevels": self.var_collevels.get(),
            "wl": self.var_wl.get(),
            "ovlp": self.var_ovlp.get(),
            "width": self.var_width.get(),
            "height": self.var_height.get(),
            "res": self.var_res.get(),
            "palette": self.var_palette.get(),
            "show_title": self.var_show_title.get(),
            "show_scale": self.var_show_scale.get(),
            "show_osc": self.var_show_osc.get(),
            "show_detections": self.var_show_det.get(),
            "det_cex": round(self.var_det_cex.get(), 2),
        }
        self.win.destroy()

    def _on_cancel(self):
        self.result = None
        self.win.destroy()
