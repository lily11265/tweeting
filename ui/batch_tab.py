# ============================================================
# BatchTabMixin — 배치 분석 탭
# ============================================================

import os
import csv
import json
import threading
import subprocess
import tempfile
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from collections import defaultdict

from audio.sanitizer import ensure_wav, AUDIO_FILETYPES
from ui.spectrogram_tab import SpectrogramTab as _SpectrogramTab
from ui.species_form import create_species_form


class BatchTabMixin:
    """배치 분석 탭 메서드 모음 (Mixin)"""

    # ============================================================
    # 배치 분석 탭 UI 구성
    # ============================================================
    def _build_batch_tab(self, parent):
        """배치 분석 탭 UI 구성"""
        # --- 1. 분석 대상 음원 목록 ---
        frm_files = ttk.LabelFrame(parent, text=" 1. 분석 대상 음원 목록 (WAV/MP3) ", padding=10)
        frm_files.pack(fill="x", padx=10, pady=(10, 5))

        btn_row = ttk.Frame(frm_files)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="📂 파일 추가", command=self._batch_add_files).pack(side="left")
        ttk.Button(btn_row, text="📁 폴더 전체 추가", command=self._batch_add_folder).pack(side="left", padx=5)
        ttk.Button(btn_row, text="🗑 선택 삭제", command=self._batch_remove_files).pack(side="left", padx=5)
        ttk.Button(btn_row, text="🗑 전체 비우기", command=self._batch_clear_files).pack(side="left", padx=5)

        list_frame = ttk.Frame(frm_files)
        list_frame.pack(fill="x", pady=(5, 0))
        self.batch_file_listbox = tk.Listbox(list_frame, height=6, font=("Consolas", 9),
                                              selectmode="extended")
        batch_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.batch_file_listbox.yview)
        self.batch_file_listbox.configure(yscrollcommand=batch_scroll.set)
        self.batch_file_listbox.pack(side="left", fill="both", expand=True)
        batch_scroll.pack(side="right", fill="y")

        self.batch_file_count_var = tk.StringVar(value="0 개 파일")
        ttk.Label(frm_files, textvariable=self.batch_file_count_var,
                  foreground="gray").pack(anchor="w", pady=(2, 0))

        # --- 2. 찾을 종 설정 ---
        frm_species_outer = ttk.LabelFrame(parent, text=" 2. 찾을 종 설정 (WAV/MP3) ", padding=5)
        frm_species_outer.pack(fill="both", expand=True, padx=10, pady=5)

        btn_bar = ttk.Frame(frm_species_outer)
        btn_bar.pack(fill="x", pady=(0, 5))
        ttk.Button(btn_bar, text="+ 종 추가", command=self._batch_add_species).pack(side="left")
        ttk.Button(btn_bar, text="- 마지막 종 삭제", command=self._batch_remove_species).pack(side="left", padx=5)

        canvas = tk.Canvas(frm_species_outer)
        scrollbar = ttk.Scrollbar(frm_species_outer, orient="vertical", command=canvas.yview)
        self.batch_species_container = ttk.Frame(canvas)
        self.batch_species_container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.batch_species_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 기본 1종 추가
        self._batch_add_species()

        # --- 3. 가중치 ---
        frm_weights = ttk.LabelFrame(parent, text=" 3. 종합 판별 가중치 ", padding=5)
        frm_weights.pack(fill="x", padx=10, pady=5)

        self.batch_weight_vars = {}
        weight_defs = [
            ("cor_score",   "스펙트로그램 상관", 0.25),
            ("mfcc_score",  "MFCC 유사도",      0.25),
            ("dtw_freq",    "주파수궤적 DTW",   0.20),
            ("dtw_env",     "진폭포락선 DTW",   0.15),
            ("band_energy", "대역 에너지",      0.15),
        ]
        wrow = ttk.Frame(frm_weights)
        wrow.pack(fill="x")
        for key, label, default in weight_defs:
            ttk.Label(wrow, text=f"{label}:").pack(side="left", padx=(8, 0))
            var = tk.DoubleVar(value=default)
            ttk.Spinbox(wrow, textvariable=var, from_=0.0, to=1.0,
                        width=5, increment=0.05).pack(side="left", padx=(2, 5))
            self.batch_weight_vars[key] = var

        # --- 4. 실행 / 결과 ---
        frm_run = ttk.Frame(parent, padding=5)
        frm_run.pack(fill="x", padx=10, pady=5)

        self.btn_batch_run = ttk.Button(frm_run, text="🔍 배치 분석 실행",
                                         command=self._run_batch_analysis)
        self.btn_batch_run.pack(side="left")

        self.btn_batch_cancel = ttk.Button(frm_run, text="⛔ 중단",
                                            command=self._batch_cancel_analysis,
                                            state="disabled")
        self.btn_batch_cancel.pack(side="left", padx=5)

        self.btn_batch_export = ttk.Button(frm_run, text="📥 통합 CSV 저장",
                                            command=self._batch_export_csv,
                                            state="disabled")
        self.btn_batch_export.pack(side="left", padx=5)

        self.btn_batch_spectro = ttk.Button(frm_run, text="📊 스펙트로그램",
                                             command=self._batch_show_spectrograms,
                                             state="disabled")
        self.btn_batch_spectro.pack(side="left", padx=5)

        self.batch_progress = ttk.Progressbar(frm_run, mode="determinate", length=250)
        self.batch_progress.pack(side="left", padx=10, fill="x", expand=True)

        self.batch_status_var = tk.StringVar(value="대기 중")
        ttk.Label(frm_run, textvariable=self.batch_status_var).pack(side="right")

        # 결과 텍스트
        frm_result = ttk.LabelFrame(parent, text=" 5. 배치 결과 ", padding=5)
        frm_result.pack(fill="both", expand=False, padx=10, pady=(5, 10))

        self.batch_result_text = scrolledtext.ScrolledText(frm_result, height=10,
                                                           font=("Consolas", 10))
        self.batch_result_text.pack(fill="both", expand=True)

    # --- 배치: 파일/폴더 관리 ---

    def _batch_add_files(self):
        """배치 분석 대상 파일 추가 (다중 선택)"""
        files = filedialog.askopenfilenames(filetypes=AUDIO_FILETYPES)
        for f in files:
            if f and f not in self._batch_files:
                self._batch_files.append(f)
                self.batch_file_listbox.insert("end", f)
        self.batch_file_count_var.set(f"{len(self._batch_files)} 개 파일")

    def _batch_add_folder(self):
        """폴더 내 모든 WAV/MP3 파일 추가"""
        folder = filedialog.askdirectory(title="음원 폴더 선택")
        if not folder:
            return
        added = 0
        for root_dir, _, files in os.walk(folder):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".wav", ".mp3"):
                    full = os.path.join(root_dir, fname)
                    if full not in self._batch_files:
                        self._batch_files.append(full)
                        self.batch_file_listbox.insert("end", full)
                        added += 1
        self.batch_file_count_var.set(f"{len(self._batch_files)} 개 파일")
        if added == 0:
            messagebox.showinfo("안내", "해당 폴더에 WAV/MP3 파일이 없습니다.")

    def _batch_remove_files(self):
        """선택된 파일 제거"""
        sel = list(self.batch_file_listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            self.batch_file_listbox.delete(idx)
            del self._batch_files[idx]
        self.batch_file_count_var.set(f"{len(self._batch_files)} 개 파일")

    def _batch_clear_files(self):
        """전체 파일 비우기"""
        self._batch_files.clear()
        self.batch_file_listbox.delete(0, "end")
        self.batch_file_count_var.set("0 개 파일")

    # --- 배치: 종 관리 ---

    def _batch_add_species(self):
        """배치 분석용 종 추가 (공통 팩토리 사용)"""
        idx = len(self.batch_species_frames) + 1
        sp_info = create_species_form(
            self.batch_species_container, idx,
            on_template_select=self._open_template_selector,
            include_cutoff=True,
            include_weights=True,
        )
        self.batch_species_frames.append(sp_info)

    def _batch_remove_species(self):
        """배치 분석 마지막 종 삭제"""
        if self.batch_species_frames:
            sp = self.batch_species_frames.pop()
            sp["frame"].destroy()

    # --- 배치: 실행 ---

    def _run_batch_analysis(self):
        """배치 분석 실행"""
        if not self._batch_files:
            messagebox.showwarning("경고", "분석 대상 음원 파일을 추가해 주세요.")
            return

        # 유효한 종만 수집
        species_data = []
        for sp in self.batch_species_frames:
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
        self._batch_running = True
        self._batch_cancel = False
        self._batch_results = []
        self._batch_wav_map = {}
        self.btn_batch_run.config(state="disabled")
        self.btn_batch_cancel.config(state="normal")
        self.btn_batch_export.config(state="disabled")
        self.btn_batch_spectro.config(state="disabled")
        self.batch_progress["value"] = 0
        self.batch_progress["maximum"] = len(self._batch_files)
        self.batch_result_text.delete("1.0", "end")

        thread = threading.Thread(
            target=self._batch_worker,
            args=(list(self._batch_files), species_data),
            daemon=True
        )
        thread.start()

    def _batch_cancel_analysis(self):
        """배치 분석 중단"""
        self._batch_cancel = True
        self.batch_status_var.set("중단 요청됨...")

    def _batch_log(self, msg):
        """배치 결과 텍스트에 로그 추가 (메인 스레드 전용)"""
        self.batch_result_text.insert("end", msg)
        self.batch_result_text.see("end")

    def _batch_worker(self, audio_files, species_data):
        """배경 스레드에서 각 음원별로 R 분석 실행"""
        total = len(audio_files)
        batch_output_base = Path(tempfile.mkdtemp(prefix="birdsong_batch_"))

        # Rscript 확인
        if not self.rscript_path:
            self.root.after(0, lambda: messagebox.showerror("오류",
                "Rscript를 찾을 수 없습니다.\n\n"
                "R이 설치되어 있는지 확인하세요."))
            self.root.after(0, self._batch_finish)
            return

        global_weights = {
            k: v.get() for k, v in self.batch_weight_vars.items()
        }

        for i, audio_file in enumerate(audio_files):
            if self._batch_cancel:
                self.root.after(0, self._batch_log,
                    f"\n⛔ 사용자에 의해 중단됨 ({i}/{total} 완료)\n")
                break

            basename = os.path.basename(audio_file)
            self.root.after(0, lambda b=basename, idx=i: (
                self.batch_status_var.set(f"{idx+1}/{total}: {b}"),
                self.batch_progress.configure(value=idx)
            ))
            self.root.after(0, self._batch_log,
                f"\n{'='*60}\n  [{i+1}/{total}] {basename}\n{'='*60}\n")

            try:
                # 1) 음원별 출력 폴더 생성
                sub_dir = batch_output_base / f"batch_{i+1:04d}"
                sub_dir.mkdir(parents=True, exist_ok=True)

                # 2) 전체 음원 WAV 전처리 (sanitize)
                self.root.after(0, self._batch_log, "  [Python 전처리]\n")
                try:
                    main_wav, main_log = ensure_wav(audio_file, sub_dir)
                    for line in main_log.splitlines():
                        self.root.after(0, self._batch_log, f"    {line}\n")
                except Exception as e:
                    self.root.after(0, self._batch_log,
                        f"    ⚠ 전체 음원 전처리 실패: {e}\n")
                    self.root.after(0, self._batch_log,
                        f"    {traceback.format_exc()}\n")
                    continue

                # WAV 경로 저장 (스펙트로그램 뷰어용)
                self._batch_wav_map[basename] = main_wav

                # 3) C5: 템플릿별 음원 전처리
                sp_data_copy = []
                for sp in species_data:
                    sp_copy = {k: v for k, v in sp.items() if k != "templates"}
                    sp_copy["templates"] = []
                    for tmpl in sp["templates"]:
                        tmpl_copy = dict(tmpl)
                        try:
                            sp_wav, sp_log = ensure_wav(tmpl["wav_path"], sub_dir)
                            tmpl_copy["wav_path"] = sp_wav
                            for line in sp_log.splitlines():
                                self.root.after(0, self._batch_log, f"    {line}\n")
                        except Exception as e:
                            self.root.after(0, self._batch_log,
                                f"    ⚠ {sp.get('name','?')}/{tmpl.get('label','?')} 전처리 실패: {e}\n")
                        sp_copy["templates"].append(tmpl_copy)
                    sp_data_copy.append(sp_copy)

                # 4) config.json 생성
                config = {
                    "main_wav":   main_wav,
                    "output_dir": str(sub_dir),
                    "weights":    global_weights,
                    "species":    sp_data_copy,
                }
                config_path = sub_dir / "config.json"
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)

                self.root.after(0, self._batch_log,
                    f"  [R 분석 실행]\n"
                    f"    config: {config_path}\n"
                    f"    main_wav: {main_wav}\n")

                # 5) Rscript 실행
                result = subprocess.run(
                    [self.rscript_path, "--encoding=UTF-8",
                     str(self.r_script), str(config_path)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=600,
                )

                # 디버그 로그 항상 저장
                log_path = sub_dir / "debug_log.txt"
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(f"=== R 실행 결과 (코드: {result.returncode}) ===\n\n")
                        f.write(f"--- stdout ---\n{result.stdout or '(없음)'}\n\n")
                        f.write(f"--- stderr ---\n{result.stderr or '(없음)'}\n")
                except Exception:
                    pass

                if result.returncode != 0:
                    self.root.after(0, self._batch_log,
                        f"  ⚠ R 오류 (코드: {result.returncode})\n")

                    # === stdout 전체 출력 ===
                    if result.stdout:
                        stdout_lines = result.stdout.strip().splitlines()
                        self.root.after(0, self._batch_log,
                            f"  --- R stdout ({len(stdout_lines)}줄) ---\n")
                        for line in stdout_lines:
                            self.root.after(0, self._batch_log,
                                f"  {line}\n")

                    # === stderr 전체 출력 ===
                    if result.stderr:
                        stderr_lines = result.stderr.strip().splitlines()
                        self.root.after(0, self._batch_log,
                            f"  --- R stderr ({len(stderr_lines)}줄) ---\n")
                        for line in stderr_lines:
                            self.root.after(0, self._batch_log,
                                f"  {line}\n")

                    if not result.stdout and not result.stderr:
                        self.root.after(0, self._batch_log,
                            f"  (R 출력 없음)\n")

                    self.root.after(0, self._batch_log,
                        f"  📄 상세 로그: {log_path}\n")
                    continue

                # ✅ 성공 시에도 주요 R 로그 표시
                if result.stdout:
                    info_lines = [
                        l for l in result.stdout.splitlines()
                        if any(k in l for k in ["[INFO]", "[ERROR]", "★", "검출", "완료", "cutoff"])
                    ]
                    if info_lines:
                        for line in info_lines[-10:]:  # 마지막 10줄
                            self.root.after(0, self._batch_log,
                                f"  {line.strip()}\n")

                # 6) 결과 CSV 수집
                csv_path = sub_dir / "results_detailed.csv"
                if not csv_path.exists():
                    csv_path = sub_dir / "results.csv"

                if csv_path.exists():
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)

                    detect_count = len(rows)
                    for row in rows:
                        row["source_file"] = basename
                        self._batch_results.append(row)

                    # 결과 요약 표시
                    if detect_count > 0:
                        by_sp = defaultdict(list)
                        for r in rows:
                            by_sp[r.get("species", "?")].append(r)
                        for sp_name, dets in by_sp.items():
                            times = ", ".join(
                                d.get("time_display", f"{float(d.get('time',0)):.1f}s")
                                for d in dets[:5]
                            )
                            suffix = "..." if len(dets) > 5 else ""
                            self.root.after(0, self._batch_log,
                                f"  ✅ {sp_name}: {len(dets)}건 ({times}{suffix})\n")
                    else:
                        self.root.after(0, self._batch_log,
                            f"  검출 없음\n")
                else:
                    self.root.after(0, self._batch_log,
                        f"  결과 파일 없음\n")

            except subprocess.TimeoutExpired:
                self.root.after(0, self._batch_log,
                    f"  ⚠ 분석 시간 초과 (10분)\n")
            except Exception as e:
                self.root.after(0, self._batch_log,
                    f"  ⚠ Python 오류: {e}\n"
                    f"  {traceback.format_exc()}\n")

        # 완료
        self.root.after(0, lambda: self.batch_progress.configure(value=total))
        self.root.after(0, self._batch_finish)

    def _batch_finish(self):
        """배치 분석 완료 처리 (메인 스레드)"""
        self._batch_running = False
        self.btn_batch_run.config(state="normal")
        self.btn_batch_cancel.config(state="disabled")

        total_detections = len(self._batch_results)
        cancelled = " (중단됨)" if self._batch_cancel else ""

        self.batch_result_text.insert("end",
            f"\n{'='*60}\n"
            f"  배치 분석 완료{cancelled}\n"
            f"  총 검출: {total_detections}건\n"
            f"{'='*60}\n")
        self.batch_result_text.see("end")

        if total_detections > 0:
            self.btn_batch_export.config(state="normal")
            self.btn_batch_spectro.config(state="normal")
            self.batch_status_var.set(f"완료 — {total_detections}건 검출")

            # 종별 요약
            summary = defaultdict(lambda: defaultdict(int))
            for r in self._batch_results:
                summary[r.get("species", "?")][r.get("source_file", "?")] += 1

            self.batch_result_text.insert("end", "\n  [종별 요약]\n")
            for sp, files in summary.items():
                total_sp = sum(files.values())
                file_count = len(files)
                self.batch_result_text.insert("end",
                    f"  {sp}: 총 {total_sp}건 ({file_count}개 파일에서 검출)\n")
        else:
            self.batch_status_var.set(f"완료 — 검출 없음{cancelled}")

        self.batch_result_text.see("end")

    # --- 배치: 스펙트로그램 보기 ---

    def _batch_show_spectrograms(self):
        """배치 결과를 스펙트로그램 오버레이로 표시 (파일별 탭)"""
        if not self._HAS_SCIPY:
            messagebox.showerror("오류",
                "실시간 스펙트로그램에는 numpy, scipy, matplotlib가 필요합니다.\n\n"
                "pip install numpy scipy matplotlib")
            return
        if not self._HAS_PIL:
            messagebox.showinfo("안내", "Pillow도 필요합니다.\npip install Pillow")
            return

        if not self._batch_results:
            messagebox.showinfo("안내", "배치 분석 결과가 없습니다.")
            return

        # 파일별 검출 결과 그룹화
        by_file = defaultdict(list)
        for r in self._batch_results:
            src = r.get("source_file", "?")
            by_file[src].append(r)

        # WAV 경로가 있는 파일만 표시
        wav_tabs = []
        for src_name, rows in by_file.items():
            wav_path = self._batch_wav_map.get(src_name)
            if wav_path and os.path.isfile(wav_path):
                # 검출 결과를 _SpectrogramTab이 기대하는 형식으로 변환
                detections = []
                for row in rows:
                    try:
                        detections.append({
                            "species": row.get("species", ""),
                            "time": float(row.get("time", 0)),
                            "score": float(row.get("composite",
                                          row.get("score", 0))),
                        })
                    except (ValueError, TypeError):
                        pass
                wav_tabs.append((src_name, wav_path, detections))

        if not wav_tabs:
            messagebox.showinfo("안내", "표시할 수 있는 WAV 파일이 없습니다.\n"
                                "(MP3→WAV 변환된 임시 파일이 삭제되었을 수 있습니다)")
            return

        win = tk.Toplevel(self.root)
        win.title("📊 배치 분석 — 스펙트로그램 뷰어")
        win.geometry("1200x750")
        win._refs = []  # GC 방지

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)

        viewers = []
        for tab_name, wav_path, detections in wav_tabs:
            # 파일명이 길면 축약
            short_name = tab_name if len(tab_name) <= 25 else tab_name[:22] + "..."
            viewer = _SpectrogramTab(notebook, wav_path, tab_name, win,
                                     detections=detections)
            notebook.add(viewer.frame, text=f"  {short_name}  ")
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

    # --- 배치: CSV 내보내기 ---

    def _batch_export_csv(self):
        """배치 결과를 통합 CSV로 저장"""
        if not self._batch_results:
            messagebox.showwarning("경고", "저장할 배치 결과가 없습니다.")
            return

        dst = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV 파일", "*.csv")],
            initialfile="배치_검출결과.csv"
        )
        if not dst:
            return

        # 컬럼 순서 결정
        columns = ["source_file", "species", "time_display", "time",
                   "composite", "cor_score", "mfcc_score",
                   "dtw_freq", "dtw_env", "band_energy"]
        # 실제 데이터에 있는 키 확인
        all_keys = set()
        for r in self._batch_results:
            all_keys.update(r.keys())
        # columns에 없는 키 추가
        extra = sorted(all_keys - set(columns))
        final_columns = [c for c in columns if c in all_keys] + extra

        try:
            with open(dst, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=final_columns,
                                        extrasaction="ignore")
                writer.writeheader()
                for row in self._batch_results:
                    writer.writerow(row)
            messagebox.showinfo("완료", f"저장 완료: {dst}\n총 {len(self._batch_results)}건")
        except Exception as e:
            messagebox.showerror("오류", f"CSV 저장 실패: {e}")
