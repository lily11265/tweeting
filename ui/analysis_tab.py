# ============================================================
# AnalysisTabMixin — 음성 분석 탭 + 자동 튜닝 탭
# ============================================================

import os
import csv
import json
import threading
import subprocess
import re
import tempfile
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from collections import defaultdict

from audio.sanitizer import ensure_wav, AUDIO_FILETYPES
from ui.spectrogram_tab import SpectrogramTab as _SpectrogramTab
from ui.template_selector import TemplateSelector as _TemplateSelector
from ui.species_form import create_species_form


class AnalysisTabMixin:
    """음성 분석 탭 + 자동 튜닝 탭 메서드 모음 (Mixin)"""

    # ----------------------------------------
    # 탭 1: 음성 분석
    # ----------------------------------------
    def _build_analysis_tab(self, parent):
        # --- 상단: 전체 음원 ---
        frm_main = ttk.LabelFrame(parent, text=" 1. 전체 음원 (분석 대상 - WAV/MP3) ", padding=10)
        frm_main.pack(fill="x", padx=10, pady=(10, 5))

        self.var_main_wav = tk.StringVar()
        ttk.Entry(frm_main, textvariable=self.var_main_wav, width=80).pack(side="left", fill="x", expand=True)
        ttk.Button(frm_main, text="파일 선택", command=self._select_main_wav).pack(side="right", padx=(10, 0))

        # MP3 자동 변환 안내
        ttk.Label(frm_main, text="※ MP3 선택 시 자동으로 WAV 변환 후 분석합니다.",
                  foreground="gray").pack(side="bottom", anchor="w")

        # --- 중단: 종 목록 (스크롤 가능) ---
        frm_species_outer = ttk.LabelFrame(parent, text=" 2. 찾을 종 설정 (WAV/MP3) ", padding=5)
        frm_species_outer.pack(fill="both", expand=True, padx=10, pady=5)

        # 종 추가/삭제 버튼
        btn_bar = ttk.Frame(frm_species_outer)
        btn_bar.pack(fill="x", pady=(0, 5))
        ttk.Button(btn_bar, text="+ 종 추가", command=self._add_species).pack(side="left")
        ttk.Button(btn_bar, text="- 마지막 종 삭제", command=self._remove_species).pack(side="left", padx=5)

        # 스크롤 캔버스
        canvas = tk.Canvas(frm_species_outer)
        scrollbar = ttk.Scrollbar(frm_species_outer, orient="vertical", command=canvas.yview)
        self.species_container = ttk.Frame(canvas)

        self.species_container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.species_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 기본 1종 추가
        self._add_species()

        # --- 가중치 설정 ---
        frm_weights = ttk.LabelFrame(parent, text=" 3. 종합 판별 가중치 ", padding=5)
        frm_weights.pack(fill="x", padx=10, pady=5)

        self.weight_vars = {}
        weight_defs = [
            ("cor_score",      "스펙트로그램 상관", 0.20),
            ("mfcc_score",     "MFCC 유사도",      0.20),
            ("dtw_freq",       "주파수궤적 DTW",   0.15),
            ("dtw_env",        "진폭포락선 DTW",   0.10),
            ("band_energy",    "대역 에너지",      0.15),
            ("harmonic_ratio", "조화 비율",        0.20),
        ]
        wrow = ttk.Frame(frm_weights)
        wrow.pack(fill="x")
        for key, label, default in weight_defs:
            ttk.Label(wrow, text=f"{label}:").pack(side="left", padx=(8, 0))
            var = tk.DoubleVar(value=default)
            var.trace_add("write", lambda *_: self._update_weight_sum())
            ttk.Spinbox(wrow, textvariable=var, from_=0.0, to=1.0,
                        width=5, increment=0.05).pack(side="left", padx=(2, 5))
            self.weight_vars[key] = var

        self.weight_sum_label = ttk.Label(wrow, text="합계: 1.00", foreground="green",
                                          font=("Consolas", 9, "bold"))
        self.weight_sum_label.pack(side="right", padx=10)

        # --- 하단: 실행 버튼 & 결과 ---
        frm_bottom = ttk.Frame(parent)
        frm_bottom.pack(fill="x", padx=10, pady=5)

        self.btn_run = ttk.Button(frm_bottom, text="🔍 분석 실행", command=self._run_analysis)
        self.btn_run.pack(side="left")

        self.btn_spectro = ttk.Button(frm_bottom, text="📊 스펙트로그램 보기",
                                       command=self._show_spectrograms, state="disabled")
        self.btn_spectro.pack(side="left", padx=5)

        self.btn_export = ttk.Button(frm_bottom, text="📥 결과 CSV 저장",
                                      command=self._export_csv, state="disabled")
        self.btn_export.pack(side="left", padx=5)

        self.btn_r_spectro = ttk.Button(frm_bottom, text="📄 R 스펙토그램 저장",
                                         command=self._export_r_spectrogram_analysis,
                                         state="disabled")
        self.btn_r_spectro.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(frm_bottom, mode="determinate", length=200)
        self.progress.pack(side="right")

        # 결과 텍스트
        frm_result = ttk.LabelFrame(parent, text=" 4. 검출 결과 ", padding=5)
        frm_result.pack(fill="both", expand=False, padx=10, pady=(5, 10))

        self.txt_result = scrolledtext.ScrolledText(frm_result, height=12, font=("Consolas", 10))
        self.txt_result.pack(fill="both", expand=True)

    # ----------------------------------------
    # 탭: 자동 튜닝
    # ----------------------------------------
    def _build_autotune_tab(self, parent):
        # 안내
        frm_info = ttk.Frame(parent, padding=10)
        frm_info.pack(fill="x")
        ttk.Label(frm_info, text="종 음원을 자가진단하여 최적 가중치를 자동 결정합니다.",
                  font=("Arial", 12, "bold")).pack(anchor="w")
        ttk.Label(frm_info,
                  text="스펙트로그램에서 새소리 구간을 2곳 이상 선택하면,\n"
                       "선택한 구간과 유사한 음을 양성, 나머지를 음성으로 분류하여\n"
                       "변별력이 높은 지표에 더 큰 가중치를 부여합니다.",
                  foreground="gray").pack(anchor="w", pady=(2, 0))

        # --- 1. 종 음원 설정 (스크롤 가능) ---
        frm_sp = ttk.LabelFrame(parent, text=" 1. 종 음원 설정 ", padding=5)
        frm_sp.pack(fill="both", expand=True, padx=10, pady=5)

        btn_bar = ttk.Frame(frm_sp)
        btn_bar.pack(fill="x", pady=(0, 5))
        ttk.Button(btn_bar, text="+ 종 추가", command=self._at_add_species).pack(side="left")
        ttk.Button(btn_bar, text="- 마지막 종 삭제", command=self._at_remove_species).pack(side="left", padx=5)

        at_canvas = tk.Canvas(frm_sp)
        at_scroll = ttk.Scrollbar(frm_sp, orient="vertical", command=at_canvas.yview)
        self.at_species_container = ttk.Frame(at_canvas)
        self.at_species_container.bind(
            "<Configure>", lambda e: at_canvas.configure(scrollregion=at_canvas.bbox("all"))
        )
        at_canvas.create_window((0, 0), window=self.at_species_container, anchor="nw")
        at_canvas.configure(yscrollcommand=at_scroll.set)
        at_canvas.pack(side="left", fill="both", expand=True)
        at_scroll.pack(side="right", fill="y")

        self.at_species_frames = []
        self._at_add_species()

        # --- 2. 실행 버튼 ---
        frm_run = ttk.Frame(parent)
        frm_run.pack(fill="x", padx=10, pady=5)

        self.at_btn_run = ttk.Button(frm_run, text="🎛 자동 튜닝 실행",
                                      command=self._at_run_tuning)
        self.at_btn_run.pack(side="left")

        self.at_btn_apply = ttk.Button(frm_run, text="✅ 분석/배치 탭에 적용",
                                        command=self._at_apply_weights, state="disabled")
        self.at_btn_apply.pack(side="left", padx=5)

        self.at_progress = ttk.Progressbar(frm_run, mode="indeterminate", length=200)
        self.at_progress.pack(side="right")

        # --- 3. 결과 ---
        frm_result = ttk.LabelFrame(parent, text=" 2. 튜닝 결과 ", padding=5)
        frm_result.pack(fill="both", expand=False, padx=10, pady=(5, 10))

        self.at_txt_result = scrolledtext.ScrolledText(frm_result, height=14, font=("Consolas", 10))
        self.at_txt_result.pack(fill="both", expand=True)

        # 튜닝 결과 저장용
        self._at_tune_results = {}

    def _at_add_species(self):
        idx = len(self.at_species_frames) + 1
        sp_info = create_species_form(
            self.at_species_container, idx,
            on_template_select=self._open_template_selector,
            include_cutoff=False,
            include_weights=False,
        )
        # 다중 구간 선택 버튼 추가
        multi_btn = ttk.Button(
            sp_info["frame"],
            text="📊 새소리 구간 다중 선택 (자동 튜닝용)",
            command=lambda sp=sp_info: self._open_multi_template_selector(sp),
        )
        multi_btn.pack(fill="x", pady=(3, 0))
        self.at_species_frames.append(sp_info)

    def _at_remove_species(self):
        if len(self.at_species_frames) > 1:
            sp = self.at_species_frames.pop()
            sp["frame"].destroy()

    def _at_run_tuning(self):
        """자동 튜닝 실행"""
        species_data = []
        for sp in self.at_species_frames:
            if not sp["templates"]:
                continue
            # 모든 템플릿 수집
            tmpls = []
            wav_path = None
            for tmpl in sp["templates"]:
                tp = tmpl["path"].get().strip()
                if tp and os.path.isfile(tp):
                    if wav_path is None:
                        wav_path = tp
                    tmpls.append({
                        "wav_path": tp,
                        "t_start": tmpl["t_start"].get(),
                        "t_end":   tmpl["t_end"].get(),
                        "f_low":   tmpl["f_low"].get(),
                        "f_high":  tmpl["f_high"].get(),
                    })
            if tmpls:
                species_data.append({
                    "name":      sp["name"].get().strip(),
                    "wav_path":  wav_path,  # 하위 호환용
                    "templates": tmpls,
                })

        if not species_data:
            messagebox.showwarning("경고", "최소 1종의 음원을 선택해 주세요.")
            return

        self.at_btn_run.config(state="disabled")
        self.at_btn_apply.config(state="disabled")
        self.at_progress.start(10)
        self.at_txt_result.delete("1.0", "end")
        self.at_txt_result.insert("end", "자동 튜닝 시작...\n\n")

        thread = threading.Thread(
            target=self._at_run_thread,
            args=(species_data,),
            daemon=True
        )
        thread.start()

    def _at_run_thread(self, species_data):
        """자동 튜닝 워커 스레드"""
        try:
            tune_dir = Path(tempfile.mkdtemp(prefix="birdsong_tune_"))

            # MP3→WAV 변환 + sanitize
            for sp in species_data:
                original = sp["wav_path"]
                converted, conv_log = ensure_wav(original, tune_dir)
                if converted != original:
                    self.root.after(0, self._at_log,
                                   f"  ✅ {sp['name']} 음원 전처리 완료\n")
                sp["wav_path"] = converted
                # 템플릿별 wav_path도 갱신
                for tmpl in sp.get("templates", []):
                    if tmpl["wav_path"] == original:
                        tmpl["wav_path"] = converted

            # config.json 생성 (auto_tune 모드)
            config = {
                "mode": "auto_tune",
                "main_wav": species_data[0]["wav_path"],  # dummy (auto_tune에선 안 씀)
                "output_dir": str(tune_dir),
                "species": species_data,
            }
            config_path = tune_dir / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            if not self.rscript_path:
                self.root.after(0, self._at_on_error, "Rscript를 찾을 수 없습니다.")
                return

            self.root.after(0, self._at_log, "R 자동 튜닝 실행 중...\n")

            result = subprocess.run(
                [self.rscript_path, "--encoding=UTF-8",
                 str(self.r_script), str(config_path)],
                capture_output=True, text=True, encoding="utf-8", timeout=300,
            )

            # 결과 파싱
            result_json_path = tune_dir / "auto_tune_results.json"
            tune_results = {}
            if result_json_path.exists():
                with open(result_json_path, "r", encoding="utf-8") as f:
                    tune_results = json.load(f)

            self.root.after(0, self._at_on_done,
                           result.stdout, result.stderr, result.returncode,
                           tune_results)

        except subprocess.TimeoutExpired:
            self.root.after(0, self._at_on_error, "튜닝 시간이 5분을 초과했습니다.")
        except Exception as e:
            self.root.after(0, self._at_on_error, str(e))

    def _at_log(self, msg):
        self.at_txt_result.insert("end", msg)
        self.at_txt_result.see("end")

    def _at_on_error(self, msg):
        self.at_progress.stop()
        self.at_btn_run.config(state="normal")
        messagebox.showerror("오류", msg)

    def _at_on_done(self, stdout, stderr, returncode, tune_results):
        self.at_progress.stop()
        self.at_btn_run.config(state="normal")

        if stdout:
            self.at_txt_result.insert("end", stdout + "\n")

        if returncode != 0:
            self.at_txt_result.insert("end", f"\n⚠ R 실행 오류 (코드: {returncode})\n")
            if stderr:
                self.at_txt_result.insert("end", f"오류: {stderr}\n")
            return

        if not tune_results:
            self.at_txt_result.insert("end", "\n결과를 파싱할 수 없습니다.\n")
            return

        self._at_tune_results = tune_results

        # 결과 표시
        self.at_txt_result.insert("end", "\n" + "=" * 65 + "\n")
        self.at_txt_result.insert("end", "  ★ 자동 튜닝 결과 요약\n")
        self.at_txt_result.insert("end", "=" * 65 + "\n\n")

        labels = {"cor_score": "스펙트로그램상관",
                  "mfcc_score": "MFCC유사도",
                  "dtw_freq": "주파수궤적DTW",
                  "dtw_env": "진폭포락선DTW",
                  "band_energy": "대역에너지",
                  "harmonic_ratio": "조화비율"}
        defaults = {"cor_score": 0.20, "mfcc_score": 0.20,
                    "dtw_freq": 0.15, "dtw_env": 0.10,
                    "band_energy": 0.15, "harmonic_ratio": 0.20}

        for sp_name, res in tune_results.items():
            self.at_txt_result.insert("end", f"▶ {sp_name}\n")

            if "error" in res:
                self.at_txt_result.insert("end", f"  오류: {res['error']}\n\n")
                continue

            diag = res.get("diagnostics", {})
            if diag:
                n_tmpl = diag.get("n_templates", 1)
                self.at_txt_result.insert("end",
                    f"  분석: 템플릿 {n_tmpl}개, "
                    f"양성 {diag.get('n_positive', 0)}건, "
                    f"음성 {diag.get('n_negative', 0)}건\n\n")

            w = res.get("weights", {})
            dp = diag.get("discriminative_power", {})
            pm = diag.get("positive_means", {})
            nm_ = diag.get("negative_means", {})

            # 테이블 헤더
            self.at_txt_result.insert("end",
                f"  {'지표':>16}  {'기본값':>6}  {'최적값':>6}  {'변별력':>6}  "
                f"{'양성평균':>7}  {'음성평균':>7}  {'방향':>4}\n")
            self.at_txt_result.insert("end",
                f"  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*6}  "
                f"{'-'*7}  {'-'*7}  {'-'*4}\n")

            for key in ["cor_score", "mfcc_score", "dtw_freq", "dtw_env", "band_energy", "harmonic_ratio"]:
                tuned = w.get(key, 0)
                default = defaults[key]
                disc = dp.get(key, 0)
                p_mean = pm.get(key, 0)
                n_mean = nm_.get(key, 0)
                arrow = "↑↑" if tuned > default + 0.05 else \
                        "↑" if tuned > default + 0.02 else \
                        "↓↓" if tuned < default - 0.05 else \
                        "↓" if tuned < default - 0.02 else "="
                self.at_txt_result.insert("end",
                    f"  {labels[key]:>16}  {default:>6.3f}  {tuned:>6.3f}  "
                    f"{disc:>6.2f}  {p_mean:>7.3f}  {n_mean:>7.3f}  {arrow:>4}\n")

            self.at_txt_result.insert("end", "\n")

        self.at_txt_result.insert("end",
            "※ '분석 탭에 적용' 버튼으로 이 가중치를 음성 분석 탭에 반영할 수 있습니다.\n")
        self.at_txt_result.see("end")

        self.at_btn_apply.config(state="normal")

    def _at_apply_weights(self):
        """자동 튜닝 결과를 분석 탭 + 배치 분석 탭의 전역 가중치 + 종별 가중치에 적용"""
        if not self._at_tune_results:
            messagebox.showinfo("안내", "먼저 자동 튜닝을 실행하세요.")
            return

        applied_count = 0
        batch_applied_count = 0

        # 첫 번째 종의 가중치를 전역 가중치에 적용
        first_sp = list(self._at_tune_results.keys())[0]
        first_w = self._at_tune_results[first_sp].get("weights", {})
        if first_w:
            # 분석 탭 전역 가중치 적용
            for key, var in self.weight_vars.items():
                if key in first_w:
                    var.set(round(first_w[key], 3))
            applied_count += 1

            # 배치 탭 전역 가중치 적용 (존재하는 키만)
            if hasattr(self, 'batch_weight_vars'):
                for key, var in self.batch_weight_vars.items():
                    if key in first_w:
                        var.set(round(first_w[key], 3))
                batch_applied_count += 1

        # 동일 이름의 종이 분석 탭에 있으면 종별 가중치도 적용
        for sp_name, res in self._at_tune_results.items():
            w = res.get("weights", {})
            if not w:
                continue
            # 분석 탭 종별 가중치
            for sp_frame in self.species_frames:
                if sp_frame["name"].get().strip() == sp_name:
                    sp_frame["use_custom_weights"].set(True)
                    for key, wvar in sp_frame["sp_weights"].items():
                        if key in w:
                            wvar.set(round(w[key], 3))
                    applied_count += 1
            # 배치 탭 종별 가중치
            if hasattr(self, 'batch_species_frames'):
                for sp_frame in self.batch_species_frames:
                    if sp_frame["name"].get().strip() == sp_name:
                        sp_frame["use_custom_weights"].set(True)
                        for key, wvar in sp_frame["sp_weights"].items():
                            if key in w:
                                wvar.set(round(w[key], 3))
                        batch_applied_count += 1

        self.notebook.select(0)  # 분석 탭으로 전환
        messagebox.showinfo("완료",
            f"자동 튜닝 가중치가 적용되었습니다.\n\n"
            f"[분석 탭]\n"
            f"  전역 가중치: {first_sp}의 결과 적용\n"
            f"  종별 가중치: 동일 이름 {max(0, applied_count-1)}종에 적용\n\n"
            f"[배치 분석 탭]\n"
            f"  전역 가중치: 적용 완료\n"
            f"  종별 가중치: 동일 이름 {max(0, batch_applied_count-1)}종에 적용")

    # ========================================
    # 종 프레임 추가/삭제
    # ========================================
    def _add_species(self):
        idx = len(self.species_frames) + 1
        sp_info = create_species_form(
            self.species_container, idx,
            on_template_select=self._open_template_selector,
            include_cutoff=True,
            include_weights=True,
        )
        self.species_frames.append(sp_info)

    def _update_weight_sum(self):
        """전역 가중치 합계 라벨 갱신"""
        try:
            total = sum(v.get() for v in self.weight_vars.values())
            color = "green" if abs(total - 1.0) < 0.01 else "red"
            self.weight_sum_label.config(text=f"합계: {total:.2f}", foreground=color)
        except Exception:
            pass

    def _open_template_selector(self, var_path, var_t_start, var_t_end, var_f_low, var_f_high):
        """스펙트로그램 기반 구간 선택기 열기"""
        file_path = var_path.get()
        if not file_path or not os.path.isfile(file_path):
            messagebox.showwarning("파일 없음", "먼저 종 음원 파일을 선택하세요.")
            return

        # MP3 → WAV 변환 + sanitize
        wav_path = file_path
        if file_path.lower().endswith(".mp3"):
            try:
                tmp_dir = Path(tempfile.mkdtemp(prefix="birdsong_sel_"))
                wav_path, _log = ensure_wav(file_path, tmp_dir)
            except Exception as e:
                messagebox.showerror("변환 오류", f"MP3→WAV 변환 실패:\n{e}")
                return
        else:
            # WAV도 sanitize (template selector에서는 임시 폴더 사용)
            try:
                tmp_dir = Path(tempfile.mkdtemp(prefix="birdsong_sel_"))
                wav_path, _log = ensure_wav(file_path, tmp_dir)
            except Exception:
                wav_path = file_path  # sanitize 실패 시 원본 사용

        def on_selected(t0, t1, f0, f1):
            var_t_start.set(round(t0, 2))
            var_t_end.set(round(t1, 2))
            var_f_low.set(round(f0, 0))
            var_f_high.set(round(f1, 0))

        _TemplateSelector(self.root, wav_path, on_selected)

    def _open_multi_template_selector(self, sp_info):
        """다중 구간 선택: 스펙트로그램에서 여러 새소리 구간을 선택하여 템플릿으로 등록"""
        if not sp_info["templates"]:
            messagebox.showwarning("경고", "먼저 음원 파일을 선택하세요.")
            return

        # 모든 템플릿에서 고유 파일 경로 수집
        seen = set()
        file_list = []           # [(original_path, display_name), ...]
        wav_map = {}             # original_path → converted_wav_path
        tmp_dir = Path(tempfile.mkdtemp(prefix="birdsong_msel_"))

        for tmpl in sp_info["templates"]:
            fp = tmpl["path"].get().strip()
            if not fp or not os.path.isfile(fp) or fp in seen:
                continue
            seen.add(fp)
            display_name = os.path.basename(fp)
            # WAV 변환/sanitize
            try:
                wav_path, _log = ensure_wav(fp, tmp_dir)
            except Exception:
                wav_path = fp
            file_list.append((wav_path, display_name))
            wav_map[wav_path] = fp   # 변환된 경로 → 원본 파일 경로

        if not file_list:
            messagebox.showwarning("파일 없음", "먼저 종 음원 파일을 선택하세요.")
            return

        def on_multi_selected(selections):
            """다중 선택 콜백: 선택된 구간들로 템플릿 행 생성"""
            templates = sp_info["templates"]

            # 기존 추가 템플릿 제거 (2번째부터)
            while len(templates) > 1:
                old = templates.pop()
                old["frame"].destroy()

            for i, sel in enumerate(selections):
                # 탭 모드(5-tuple) / 단일 모드(4-tuple) 분기
                if len(sel) == 5:
                    t0, t1, f0, f1, wav_p = sel
                    orig_path = wav_map.get(wav_p, wav_p)
                else:
                    t0, t1, f0, f1 = sel
                    orig_path = wav_map.get(file_list[0][0], file_list[0][0])

                if i == 0:
                    # 첫 번째 선택 → 첫 번째 템플릿에 적용
                    templates[0]["path"].set(orig_path)
                    templates[0]["t_start"].set(round(t0, 2))
                    templates[0]["t_end"].set(round(t1, 2))
                    templates[0]["f_low"].set(round(f0, 0))
                    templates[0]["f_high"].set(round(f1, 0))
                    templates[0]["label"].set("call1")
                else:
                    from ui.species_form import _create_template_row
                    tmpl_container = templates[0]["frame"].master
                    tmpl = _create_template_row(
                        tmpl_container, i + 1,
                        on_template_select=self._open_template_selector,
                    )
                    tmpl["path"].set(orig_path)
                    tmpl["t_start"].set(round(t0, 2))
                    tmpl["t_end"].set(round(t1, 2))
                    tmpl["f_low"].set(round(f0, 0))
                    tmpl["f_high"].set(round(f1, 0))
                    tmpl["label"].set(f"call{i + 1}")
                    templates.append(tmpl)

        # 단일 파일이면 기존 방식, 복수 파일이면 탭 모드
        wav_input = file_list[0][0] if len(file_list) == 1 else file_list
        _TemplateSelector(self.root, wav_input, on_multi_selected, multi_select=True)

    def _remove_species(self):
        if len(self.species_frames) > 1:
            sp = self.species_frames.pop()
            sp["frame"].destroy()

    # ========================================
    # 파일 선택 (WAV + MP3 지원)
    # ========================================
    def _select_main_wav(self):
        path = filedialog.askopenfilename(filetypes=AUDIO_FILETYPES)
        if path:
            self.var_main_wav.set(path)

    # ========================================
    # 분석 실행
    # ========================================
    def _run_analysis(self):
        # 입력 검증
        main_file = self.var_main_wav.get().strip()
        if not main_file or not os.path.isfile(main_file):
            messagebox.showwarning("경고", "전체 음원 파일을 선택해 주세요.")
            return

        # 유효한 종만 수집
        species_data = []
        for sp in self.species_frames:
            sp_entry = {
                "name":   sp["name"].get().strip(),
                "cutoff": sp["cutoff"].get(),
                "templates": [],
            }
            # C5: 멀티 템플릿 수집
            for tmpl in sp["templates"]:
                tmpl_path = tmpl["path"].get().strip()
                if tmpl_path and os.path.isfile(tmpl_path):
                    sp_entry["templates"].append({
                        "wav_path": tmpl_path,
                        "t_start":  tmpl["t_start"].get(),
                        "t_end":    tmpl["t_end"].get(),
                        "f_low":    tmpl["f_low"].get(),
                        "f_high":   tmpl["f_high"].get(),
                        "label":    tmpl["label"].get().strip(),
                    })
            # 종별 가중치 (사용자 지정 시)
            if sp["use_custom_weights"].get():
                sp_entry["weights"] = {
                    k: v.get() for k, v in sp["sp_weights"].items()
                }
            if sp_entry["templates"]:
                species_data.append(sp_entry)

        if not species_data:
            messagebox.showwarning("경고", "최소 1종의 음원을 선택해 주세요.")
            return

        # UI 상태 변경
        self.btn_run.config(state="disabled")
        self.btn_spectro.config(state="disabled")
        self.btn_export.config(state="disabled")
        self.progress.start(10)
        self.txt_result.delete("1.0", "end")
        self.txt_result.insert("end", "MP3 파일 확인 및 변환 중...\n")

        # 별도 스레드에서 변환 + R 실행
        thread = threading.Thread(
            target=self._convert_and_run,
            args=(main_file, species_data),
            daemon=True
        )
        thread.start()

    def _convert_and_run(self, main_file, species_data):
        """MP3→WAV 자동 변환 + WAV 전처리 후 R 스크립트 호출"""
        try:
            # 전체 음원 변환 + 전처리
            main_wav, main_log = ensure_wav(main_file, self.output_dir)
            self.root.after(0, self._log, f"  [전체 음원 전처리]\n")
            for line in main_log.splitlines():
                self.root.after(0, self._log, f"    {line}\n")

            # C5: 템플릿별 음원 변환 + 전처리
            for sp in species_data:
                for tmpl in sp["templates"]:
                    original = tmpl["wav_path"]
                    converted, sp_log = ensure_wav(original, self.output_dir)
                    tmpl["wav_path"] = converted
                    self.root.after(0, self._log,
                        f"  [{sp['name']}/{tmpl.get('label','')} 전처리]\n")
                    for line in sp_log.splitlines():
                        self.root.after(0, self._log, f"    {line}\n")

            self.root.after(0, self._log, "\nR 분석 스크립트 호출 중...\n\n")

            # Rscript 경로 확인
            if not self.rscript_path:
                self.root.after(0, self._on_r_error,
                    "Rscript를 찾을 수 없습니다.\n\n"
                    "R이 설치되어 있는지 확인하세요.\n"
                    "설치 경로 예: C:\\Program Files\\R\\R-x.x.x\\bin")
                return

            # 설정 JSON 생성 (종합 판별 가중치 포함)
            global_weights = {
                k: v.get() for k, v in self.weight_vars.items()
            }
            config = {
                "main_wav":   main_wav,
                "output_dir": str(self.output_dir),
                "weights":    global_weights,
                "species":    species_data,
            }
            config_path = self.output_dir / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            # Rscript 실행 (C3: Popen 스트리밍으로 실시간 진행)
            self.root.after(0, self._log, f"Rscript 경로: {self.rscript_path}\n")
            proc = subprocess.Popen(
                [self.rscript_path, "--encoding=UTF-8", str(self.r_script), str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # 라인 버퍼링
            )

            stdout_lines = []
            for line in proc.stdout:
                stdout_lines.append(line)
                self.root.after(0, self._log, line)

                # C3: 진행 패턴 파싱 [N/M]
                m = re.search(r'\[(\d+)/(\d+)\]', line)
                if m:
                    current, total = int(m.group(1)), int(m.group(2))
                    self.root.after(0, self._update_progress, current, total)

            proc.wait(timeout=600)
            stderr = proc.stderr.read()
            stdout = ''.join(stdout_lines)

            self.root.after(0, self._on_r_output, stdout, stderr, proc.returncode)

        except FileNotFoundError as e:
            err_str = str(e).lower()
            if "ffmpeg" in err_str or "pydub" in err_str or "mp3" in err_str:
                self.root.after(0, self._on_r_error,
                    "MP3 변환에 실패했습니다.\n\n"
                    "pydub + ffmpeg를 설치하세요:\n"
                    "  pip install pydub\n"
                    "  + ffmpeg 설치 (https://ffmpeg.org)")
            else:
                self.root.after(0, self._on_r_error,
                    f"실행 파일을 찾을 수 없습니다: {e}\n\n"
                    "R이 올바르게 설치되어 있는지 확인하세요.")
        except subprocess.TimeoutExpired:
            self.root.after(0, self._on_r_error, "분석 시간이 10분을 초과했습니다.")
        except Exception as e:
            self.root.after(0, self._on_r_error, str(e))

    def _log(self, msg):
        self.txt_result.insert("end", msg)
        self.txt_result.see("end")

    def _update_progress(self, current, total):
        """C3: R 진행 상황에 맞춰 프로그레스 바 업데이트"""
        self.progress["maximum"] = total
        self.progress["value"] = current

    def _on_r_output(self, stdout, stderr, returncode):
        self.progress.stop()
        self.progress["value"] = 0
        self.btn_run.config(state="normal")

        # 디버그 로그 파일 항상 저장
        log_path = self.output_dir / "debug_log.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== R 실행 결과 (코드: {returncode}) ===\n\n")
                f.write(f"--- stdout ---\n{stdout or '(없음)'}\n\n")
                f.write(f"--- stderr ---\n{stderr or '(없음)'}\n")
        except Exception:
            pass

        if returncode != 0:
            self.txt_result.insert("end", f"⚠ R 실행 오류 (코드: {returncode})\n\n")
            if stdout:
                self.txt_result.insert("end", f"--- 출력 ---\n{stdout}\n")
            if stderr:
                self.txt_result.insert("end", f"--- 오류 메시지 ---\n{stderr}\n")
            self.txt_result.insert("end", f"\n📄 상세 로그: {log_path}\n")
            self.txt_result.see("end")
            return

        if stdout:
            self.txt_result.insert("end", stdout + "\n")

        csv_path = self.output_dir / "results.csv"
        if csv_path.exists():
            self._display_results(csv_path)
            self.btn_spectro.config(state="normal")
            self.btn_export.config(state="normal")
            self.btn_r_spectro.config(state="normal")
        else:
            self.txt_result.insert("end", "결과 파일이 생성되지 않았습니다.\n")

    def _on_r_error(self, msg):
        self.progress.stop()
        self.btn_run.config(state="normal")
        messagebox.showerror("오류", msg)

    def _export_r_spectrogram_analysis(self):
        """분석 결과 + 스펙토그램을 R seewave::spectro()로 PNG로 내보낸다."""
        config_path = self.output_dir / "config.json"
        if not config_path.exists():
            messagebox.showwarning("경고", "먼저 분석을 실행해 주세요.")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        main_wav = config.get("main_wav", "")
        if not main_wav or not os.path.isfile(main_wav):
            messagebox.showwarning("경고", "분석에 사용된 WAV 파일을 찾을 수 없습니다.")
            return

        if not self.rscript_path:
            messagebox.showerror("오류", "Rscript를 찾을 수 없습니다.")
            return

        # 검출 결과 로드
        detections = None
        csv_path = self.output_dir / "results.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    detections = []
                    for row in reader:
                        detections.append({
                            "species": row.get("species", ""),
                            "time": float(row.get("time", 0)),
                            "score": float(row.get("score", 0)),
                        })
            except Exception:
                detections = None

        # 설정 다이얼로그
        from ui.spectro_settings_dialog import SpectroSettingsDialog
        has_det = bool(detections)
        dlg = SpectroSettingsDialog(self.root, has_detections=has_det,
                                    wav_path=main_wav)
        if dlg.result is None:
            return
        settings = dlg.result

        # 저장 경로 선택
        wav_stem = Path(main_wav).stem
        default_name = f"{wav_stem}_R_spectrogram.png"
        save_path = filedialog.asksaveasfilename(
            title="R 스펙토그램 저장",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG 이미지", "*.png"), ("모든 파일", "*.*")],
        )
        if not save_path:
            return

        self.btn_r_spectro.config(state="disabled")
        self.txt_result.insert("end", "\nR 스펙토그램 생성 중...\n")
        self.txt_result.see("end")

        def _worker():
            try:
                from r_bridge import export_r_spectrogram
                result_path = export_r_spectrogram(
                    rscript_path=self.rscript_path,
                    r_script_path=str(self.r_script),
                    wav_path=main_wav,
                    output_path=save_path,
                    t_start=settings.get("t_start"),
                    t_end=settings.get("t_end"),
                    detections=detections,
                    f_low=settings["f_low"],
                    f_high=settings["f_high"],
                    width=settings["width"],
                    height=settings["height"],
                    wl=settings["wl"],
                    ovlp=settings["ovlp"],
                    collevels=settings["collevels"],
                    palette=settings["palette"],
                    dB_min=settings["dB_min"],
                    dB_max=settings["dB_max"],
                    res=settings["res"],
                    show_title=settings["show_title"],
                    show_scale=settings["show_scale"],
                    show_osc=settings["show_osc"],
                    show_detections=settings["show_detections"],
                    det_cex=settings["det_cex"],
                )
                self.root.after(0, self._on_r_spectro_done, result_path)
            except Exception as e:
                self.root.after(0, self._on_r_spectro_error, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_r_spectro_done(self, path):
        self.btn_r_spectro.config(state="normal")
        self.txt_result.insert("end", f"✅ R 스펙토그램 저장 완료: {path}\n")
        self.txt_result.see("end")
        result = messagebox.askyesno("완료",
            f"R 스펙토그램이 저장되었습니다:\n{path}\n\n파일 위치를 열겠습니까?")
        if result:
            import subprocess as sp
            sp.Popen(["explorer", "/select,", str(path)])

    def _on_r_spectro_error(self, msg):
        self.btn_r_spectro.config(state="normal")
        self.txt_result.insert("end", f"❌ R 스펙토그램 생성 실패: {msg}\n")
        self.txt_result.see("end")
        messagebox.showerror("R 스펙토그램 오류", msg)

    # ========================================
    # 결과 표시
    # ========================================
    def _display_results(self, csv_path):
        self.txt_result.insert("end", "\n" + "=" * 70 + "\n")
        self.txt_result.insert("end", "  검출 결과 (종합 판별)\n")
        self.txt_result.insert("end", "=" * 70 + "\n\n")

        # 상세 결과 파일 우선 시도
        detailed_path = self.output_dir / "results_detailed.csv"
        use_detailed = detailed_path.exists()
        read_path = str(detailed_path) if use_detailed else csv_path

        try:
            with open(read_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                self.txt_result.insert("end", "검출된 종이 없습니다.\n")
                return

            by_species = defaultdict(list)
            for row in rows:
                by_species[row["species"]].append(row)

            for sp_name, detections in by_species.items():
                self.txt_result.insert("end", f"▶ {sp_name} ({len(detections)}건 검출)\n")
                if use_detailed and "composite" in detections[0]:
                    self.txt_result.insert("end",
                        f"  {'시간':>10}  {'종합':>6}  {'corM':>6}  {'MFCC':>6}  "
                        f"{'freq':>6}  {'env':>6}  {'band':>6}\n")
                    self.txt_result.insert("end", f"  {'-'*10}  {'-'*6}  "
                        f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}\n")
                    for det in detections:
                        self.txt_result.insert("end",
                            f"  {det['time_display']:>10}  "
                            f"{float(det.get('composite', 0)):>6.3f}  "
                            f"{float(det.get('cor_score', 0)):>6.3f}  "
                            f"{float(det.get('mfcc_score', 0)):>6.3f}  "
                            f"{float(det.get('dtw_freq', 0)):>6.3f}  "
                            f"{float(det.get('dtw_env', 0)):>6.3f}  "
                            f"{float(det.get('band_energy', 0)):>6.3f}\n")
                else:
                    # 기존 형식 폴백
                    self.txt_result.insert("end", f"  {'시간':>10}  {'점수':>8}\n")
                    self.txt_result.insert("end", f"  {'-'*10}  {'-'*8}\n")
                    for det in detections:
                        score = det.get('composite', det.get('score', '0'))
                        self.txt_result.insert("end",
                            f"  {det['time_display']:>10}  {float(score):>8.4f}\n")
                self.txt_result.insert("end", "\n")

            self.result_rows = rows

        except Exception as e:
            self.txt_result.insert("end", f"결과 읽기 오류: {e}\n")

    # ========================================
    # 스펙트로그램 보기 (실시간 재생성)
    # ========================================
    def _show_spectrograms(self):
        if not self._HAS_SCIPY:
            messagebox.showerror("오류",
                "실시간 스펙트로그램에는 numpy, scipy, matplotlib가 필요합니다.\n\n"
                "pip install numpy scipy matplotlib")
            return
        if not self._HAS_PIL:
            messagebox.showinfo("안내", "Pillow도 필요합니다.\npip install Pillow")
            return

        # 분석에 사용된 WAV 파일 목록 수집
        wav_files = []
        config_path = self.output_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 전체 음원
            main_wav = config.get("main_wav", "")
            if main_wav and os.path.isfile(main_wav):
                wav_files.append(("전체 음원", main_wav))
            # 종별 음원
            for sp in config.get("species", []):
                sp_name = sp.get("name", "종")
                sp_wav = sp.get("wav_path", "")
                if sp_wav and os.path.isfile(sp_wav):
                    wav_files.append((sp_name, sp_wav))

        if not wav_files:
            messagebox.showinfo("안내", "분석에 사용된 WAV 파일을 찾을 수 없습니다.")
            return

        win = tk.Toplevel(self.root)
        win.title("📊 실시간 스펙트로그램 뷰어")
        win.geometry("1200x750")
        win._refs = []  # GC 방지

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # 검출 결과 로드 (있으면)
        detections = []
        csv_path = self.output_dir / "results.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        detections.append({
                            "species": row.get("species", ""),
                            "time": float(row.get("time", 0)),
                            "score": float(row.get("score", 0)),
                        })
            except Exception:
                pass

        viewers = []
        for tab_name, wav_path in wav_files:
            # 전체 음원 탭에만 검출 결과 오버레이 표시
            tab_detections = detections if tab_name == "전체 음원" else []
            viewer = _SpectrogramTab(notebook, wav_path, tab_name, win,
                                     detections=tab_detections)
            notebook.add(viewer.frame, text=f"  {tab_name}  ")
            viewers.append(viewer)

        # 창 닫힘 시 재생 중지
        def on_close():
            for v in viewers:
                try:
                    v._stop_playback()
                except Exception:
                    pass
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)

    # ========================================
    # CSV 내보내기
    # ========================================
    def _export_csv(self):
        src = self.output_dir / "results.csv"
        if not src.exists():
            messagebox.showwarning("경고", "결과 파일이 없습니다.")
            return

        dst = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV 파일", "*.csv")],
            initialfile="검출결과.csv"
        )
        if dst:
            shutil.copy2(src, dst)
            messagebox.showinfo("완료", f"저장 완료: {dst}")
