# ============================================================
# ui/evaluation_tab.py — 성능 평가 탭 (Mixin)
# ============================================================

import os
import csv
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

from birdnet_bridge import check_birdnet_available, run_birdnet_prediction, run_birdnet_batch, HAS_BIRDNET

from evaluation.matcher import (
    MatchingConfig,
    match_predictions_to_annotations,
    load_annotations_csv,
    load_predictions_csv,
    save_match_results_csv,
)
from evaluation.metrics import (
    compute_metrics_at_threshold,
    compute_all_species_metrics,
    compute_curve_metrics,
    find_optimal_thresholds,
    export_evaluation_json,
    HAS_SKLEARN,
)


class EvaluationTabMixin:
    """성능 평가 탭 메서드 모음 (Mixin)"""

    # ── 탭 UI 구성 ────────────────────────────────────────

    def _build_evaluation_tab(self, parent):
        # 내부 상태 초기화
        self._eval_annotations = []
        self._eval_predictions = []
        self._eval_match_results = []
        self._eval_metrics = []
        self._eval_curve_data = {}

        # 스크롤 가능한 컨테이너
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 마우스 휠 스크롤
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        # ── §1. Ground Truth ──
        frm_gt = ttk.LabelFrame(scroll_frame, text=" 1. Ground Truth (정답지) ", padding=10)
        frm_gt.pack(fill="x", padx=10, pady=(10, 5))

        row_gt = ttk.Frame(frm_gt)
        row_gt.pack(fill="x")
        ttk.Button(row_gt, text="📂 CSV 불러오기",
                   command=self._eval_load_annotations).pack(side="left")
        ttk.Button(row_gt, text="🖊 새 Annotation 만들기",
                   command=self._eval_create_annotations).pack(side="left", padx=5)

        # BirdNET 자동 생성 버튼
        if HAS_BIRDNET:
            ttk.Button(row_gt, text="🤖 BirdNET 자동 생성",
                       command=self._eval_birdnet_generate).pack(side="left", padx=5)

        self._eval_ann_label = tk.StringVar(value="파일 없음")
        ttk.Label(frm_gt, textvariable=self._eval_ann_label,
                  foreground="gray").pack(anchor="w", pady=(5, 0))

        # BirdNET 설정 (confidence 임계값)
        if HAS_BIRDNET:
            row_bn = ttk.Frame(frm_gt)
            row_bn.pack(fill="x", pady=(5, 0))
            ttk.Label(row_bn, text="BirdNET 신뢰도 임계값:").pack(side="left")
            self._eval_bn_confidence = tk.DoubleVar(value=0.5)
            ttk.Spinbox(row_bn, from_=0.05, to=0.95, increment=0.05,
                        textvariable=self._eval_bn_confidence, width=6).pack(side="left", padx=5)
            ttk.Label(row_bn, text="(높을수록 정확, 낮을수록 많이 검출)",
                      foreground="gray").pack(side="left")

        # ── §2. 예측 결과 ──
        frm_pred = ttk.LabelFrame(scroll_frame, text=" 2. 예측 결과 ", padding=10)
        frm_pred.pack(fill="x", padx=10, pady=5)

        self._eval_pred_source = tk.StringVar(value="csv")
        modes = [
            ("분석 탭 결과 사용 (results_detailed.csv)", "analysis"),
            ("배치 분석 결과 사용", "batch"),
            ("CSV 직접 불러오기", "csv"),
        ]
        for text, val in modes:
            ttk.Radiobutton(
                frm_pred, text=text, variable=self._eval_pred_source, value=val,
            ).pack(anchor="w")

        row_pred = ttk.Frame(frm_pred)
        row_pred.pack(fill="x", pady=(5, 0))
        ttk.Button(row_pred, text="📂 예측 CSV 불러오기",
                   command=self._eval_load_predictions).pack(side="left")

        self._eval_pred_label = tk.StringVar(value="파일 없음")
        ttk.Label(frm_pred, textvariable=self._eval_pred_label,
                  foreground="gray").pack(anchor="w", pady=(5, 0))

        # ── §3. 매칭 설정 ──
        frm_config = ttk.LabelFrame(scroll_frame, text=" 3. 매칭 설정 ", padding=10)
        frm_config.pack(fill="x", padx=10, pady=5)

        row_cfg1 = ttk.Frame(frm_config)
        row_cfg1.pack(fill="x")

        ttk.Label(row_cfg1, text="시간 허용오차 (초):").pack(side="left")
        self._eval_tolerance = tk.DoubleVar(value=1.5)
        ttk.Spinbox(row_cfg1, from_=0.0, to=5.0, increment=0.5,
                    textvariable=self._eval_tolerance, width=6).pack(side="left", padx=5)

        self._eval_one_to_one = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_cfg1, text="1:1 매칭",
                        variable=self._eval_one_to_one).pack(side="left", padx=15)

        row_cfg2 = ttk.Frame(frm_config)
        row_cfg2.pack(fill="x", pady=(5, 0))

        ttk.Label(row_cfg2, text="임계값 (cutoff):").pack(side="left")
        self._eval_threshold = tk.DoubleVar(value=0.0)
        ttk.Spinbox(row_cfg2, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self._eval_threshold, width=6).pack(side="left", padx=5)
        ttk.Label(row_cfg2, text="(0 = 매칭 결과 그대로)",
                  foreground="gray").pack(side="left")

        # ── 실행 버튼 ──
        frm_run = ttk.Frame(scroll_frame, padding=5)
        frm_run.pack(fill="x", padx=10, pady=5)

        ttk.Button(frm_run, text="▶ 평가 실행",
                   command=self._eval_run).pack(side="left")
        ttk.Button(frm_run, text="📈 곡선 보기",
                   command=self._eval_show_curves).pack(side="left", padx=5)
        ttk.Button(frm_run, text="✅ 최적 cutoff 찾기",
                   command=self._eval_find_optimal).pack(side="left", padx=5)
        ttk.Button(frm_run, text="🔍 오류 시각화",
                   command=self._eval_show_error_overlay).pack(side="left", padx=5)
        ttk.Button(frm_run, text="📥 결과 저장",
                   command=self._eval_export).pack(side="right")

        # ── §4. 결과 ──
        frm_result = ttk.LabelFrame(scroll_frame, text=" 4. 결과 ", padding=10)
        frm_result.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # Treeview로 종별 성능 테이블
        cols = ("species", "tp", "fp", "fn", "precision", "recall", "f1")
        self._eval_tree = ttk.Treeview(frm_result, columns=cols, show="headings",
                                        height=8)
        col_cfg = {
            "species":   ("종명", 100),
            "tp":        ("TP", 50),
            "fp":        ("FP", 50),
            "fn":        ("FN", 50),
            "precision": ("Precision", 80),
            "recall":    ("Recall", 80),
            "f1":        ("F1", 80),
        }
        for col_id, (heading, width) in col_cfg.items():
            self._eval_tree.heading(col_id, text=heading)
            self._eval_tree.column(col_id, width=width, anchor="center")

        self._eval_tree.pack(fill="both", expand=True)

        # 텍스트 요약
        self._eval_summary_text = scrolledtext.ScrolledText(
            frm_result, height=6, font=("Consolas", 9), state="disabled",
        )
        self._eval_summary_text.pack(fill="x", pady=(5, 0))

    # ── 이벤트 핸들러 ─────────────────────────────────────

    def _eval_load_annotations(self):
        """annotation CSV 파일 불러오기"""
        path = filedialog.askopenfilename(
            title="Annotation CSV 불러오기",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")],
        )
        if not path:
            return

        try:
            self._eval_annotations = load_annotations_csv(path)
            species_counts = {}
            for ann in self._eval_annotations:
                sp = ann["species"]
                species_counts[sp] = species_counts.get(sp, 0) + 1

            summary = ", ".join(f"{sp} {cnt}건" for sp, cnt in species_counts.items())
            self._eval_ann_label.set(
                f"✅ {os.path.basename(path)} ({len(self._eval_annotations)}건: {summary})"
            )
        except Exception as e:
            messagebox.showerror("오류", f"Annotation CSV 로드 실패:\n{e}")

    def _eval_create_annotations(self):
        """내장 Annotation 도구로 annotation 생성"""
        # WAV 파일 선택
        wav_path = filedialog.askopenfilename(
            title="Annotation할 WAV 파일 선택",
            filetypes=[("WAV 파일", "*.wav"), ("모든 파일", "*.*")],
        )
        if not wav_path:
            return

        # 분석 탭에서 종 목록 가져오기
        species_list = []
        if hasattr(self, "species_entries"):
            for entry in self.species_entries:
                sp = entry.get().strip()
                if sp:
                    species_list.append(sp)

        try:
            from ui.annotation_tool import AnnotationTool
            AnnotationTool(
                self.root, wav_path,
                species_list=species_list,
                callback=self._eval_on_annotations_created,
            )
        except Exception as e:
            messagebox.showerror("오류", f"Annotation 도구 실행 실패:\n{e}")

    # ── BirdNET 자동 정답지 ────────────────────────────────

    def _eval_birdnet_generate(self):
        """BirdNET으로 자동 annotation 생성 → Annotation 도구에서 검토 (다중 파일 지원)"""
        wav_paths = filedialog.askopenfilenames(
            title="BirdNET으로 분석할 음원 파일 선택 (다중 선택 가능)",
            filetypes=[("WAV 파일", "*.wav"), ("모든 오디오", "*.wav *.mp3 *.flac *.ogg"), ("모든 파일", "*.*")],
        )
        if not wav_paths:
            return

        wav_paths = list(wav_paths)  # tuple → list
        confidence = self._eval_bn_confidence.get()
        n_files = len(wav_paths)
        self._eval_ann_label.set(f"🤖 BirdNET 분석 중... ({n_files}개 파일)")

        def worker():
            try:
                def on_progress(msg):
                    self.root.after(0, lambda m=msg: self._eval_ann_label.set(f"🤖 {m}"))

                if n_files == 1:
                    annotations = run_birdnet_prediction(
                        wav_paths[0],
                        confidence_threshold=confidence,
                        lang="ko",
                        progress_callback=on_progress,
                    )
                else:
                    annotations = run_birdnet_batch(
                        wav_paths,
                        confidence_threshold=confidence,
                        lang="ko",
                        progress_callback=on_progress,
                    )

                self.root.after(0, self._eval_birdnet_on_done, wav_paths, annotations)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "BirdNET 오류", f"BirdNET 분석 실패:\n{e}"
                ))
                self.root.after(0, lambda: self._eval_ann_label.set("파일 없음"))

        threading.Thread(target=worker, daemon=True).start()

    def _eval_birdnet_on_done(self, wav_paths, annotations):
        """BirdNET 결과를 Annotation 도구로 전달하여 검토 (다중 파일 지원)"""
        if not annotations:
            messagebox.showinfo(
                "BirdNET 결과",
                "검출된 조류 음성이 없습니다.\n"
                "신뢰도 임계값을 낮추거나 다른 파일을 시도해 보세요.",
            )
            self._eval_ann_label.set("파일 없음")
            return

        # 종별 요약 메시지
        species_counts = {}
        for ann in annotations:
            sp = ann["species"]
            species_counts[sp] = species_counts.get(sp, 0) + 1
        summary = ", ".join(f"{sp} {cnt}건" for sp, cnt in species_counts.items())

        n_files = len(wav_paths)
        file_info = f"{n_files}개 파일, " if n_files > 1 else ""
        self._eval_ann_label.set(
            f"🤖 BirdNET 검출: {file_info}{len(annotations)}건 ({summary}) → Annotation 도구로 검토 중..."
        )

        # Annotation 도구를 열어서 BirdNET 결과를 사전 로드
        # 다중 파일이면 리스트 전달 (AnnotationTool이 자동 처리)
        try:
            from ui.annotation_tool import AnnotationTool, Annotation

            species_list = list(species_counts.keys())

            tool = AnnotationTool(
                self.root, wav_paths,
                species_list=species_list,
                callback=self._eval_on_annotations_created,
            )

            # BirdNET 결과를 Annotation 객체로 변환하여 도구에 사전 로드
            for ann in annotations:
                a = Annotation(
                    file=ann["file"],
                    t_start=ann["t_start"],
                    t_end=ann["t_end"],
                    f_low=ann.get("f_low", 0),
                    f_high=ann.get("f_high", 0),
                    species=ann["species"],
                )
                tool.annotations.append(a)

            # 트리뷰 갱신
            tool._refresh_tree()
            tool._redraw_annotations()

        except Exception as e:
            messagebox.showerror("오류", f"Annotation 도구 실행 실패:\n{e}")
            # 폴백: BirdNET 결과를 직접 정답지로 사용
            self._eval_annotations = annotations
            self._eval_ann_label.set(
                f"🤖 BirdNET 자동 생성 ({file_info}{len(annotations)}건: {summary})"
            )

    def _eval_on_annotations_created(self, annotations):
        """Annotation 도구에서 완료된 annotation을 수신"""
        self._eval_annotations = annotations
        species_counts = {}
        for ann in annotations:
            sp = ann["species"]
            species_counts[sp] = species_counts.get(sp, 0) + 1

        summary = ", ".join(f"{sp} {cnt}건" for sp, cnt in species_counts.items())
        self._eval_ann_label.set(
            f"✅ Annotation 도구에서 생성 ({len(annotations)}건: {summary})"
        )

    def _eval_load_predictions(self):
        """예측 결과 불러오기"""
        source = self._eval_pred_source.get()

        if source == "analysis":
            # 분석 탭의 마지막 결과 파일 탐색
            self._eval_load_predictions_from_analysis()
        elif source == "batch":
            self._eval_load_predictions_from_batch()
        else:
            self._eval_load_predictions_from_csv()

    def _eval_load_predictions_from_csv(self):
        """CSV 파일에서 예측 결과 불러오기"""
        path = filedialog.askopenfilename(
            title="예측 결과 CSV 불러오기",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")],
        )
        if not path:
            return

        try:
            # candidates_all.csv인지 results_detailed.csv인지 자동 판별
            with open(path, "r", encoding="utf-8-sig") as f:
                header = f.readline().lower()

            mode = "candidates" if "passed" in header else "results"
            preds = load_predictions_csv(path, mode=mode)

            # 곡선 평가를 위해 전체 후보 사용 (passed/failed 모두 필요)

            self._eval_predictions = preds
            self._eval_update_pred_label()
        except Exception as e:
            messagebox.showerror("오류", f"예측 CSV 로드 실패:\n{e}")

    def _eval_load_predictions_from_analysis(self):
        """분석 탭의 결과 디렉토리에서 전체 후보 로드"""
        if not hasattr(self, "output_dir") or not self.output_dir:
            messagebox.showwarning("경고", "먼저 분석을 실행해 주세요.")
            return

        candidates_path = self.output_dir / "candidates_all.csv"
        results_path = self.output_dir / "results_detailed.csv"

        print(f"[DEBUG] 예측로드: output_dir={self.output_dir}")
        print(f"[DEBUG] 예측로드: candidates_all.csv 존재={candidates_path.exists()}")
        print(f"[DEBUG] 예측로드: results_detailed.csv 존재={results_path.exists()}")

        try:
            if candidates_path.exists():
                # 곡선 평가를 위해 전체 후보 사용 (passed/failed 모두 필요)
                self._eval_predictions = load_predictions_csv(
                    str(candidates_path), mode="candidates"
                )
                print(f"[DEBUG] 예측로드: candidates_all.csv에서 {len(self._eval_predictions)}건 로드")
            elif results_path.exists():
                # fallback: 최종 결과만 있는 경우
                self._eval_predictions = load_predictions_csv(
                    str(results_path), mode="results"
                )
                print(f"[DEBUG] 예측로드: results_detailed.csv에서 {len(self._eval_predictions)}건 로드")
            else:
                messagebox.showwarning(
                    "경고",
                    f"분석 결과 파일을 찾을 수 없습니다.\n"
                    f"경로: {self.output_dir}",
                )
                return
        except Exception as e:
            print(f"[DEBUG] 예측로드 오류: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("오류", f"예측 파일 로드 실패:\n{e}")
            return

        self._eval_update_pred_label()

    def _eval_load_predictions_from_batch(self):
        """배치 분석 결과를 메모리에서 로드"""
        if not hasattr(self, "_batch_results") or not self._batch_results:
            messagebox.showwarning("경고", "먼저 배치 분석을 실행해 주세요.")
            return

        # _batch_results를 표준 prediction 형식으로 변환
        preds = []
        for result in self._batch_results:
            # 배치 결과에서 detection 정보 추출
            source_file = result.get("source_file", "")
            detections = result.get("detections", [])
            for det in detections:
                preds.append({
                    "file": os.path.basename(source_file),
                    "species": det.get("species", ""),
                    "time": det.get("time", 0),
                    "composite": det.get("composite", 0),
                    "passed": True,
                })

        if not preds:
            # 대안: 배치 CSV 내보내기 후 불러오기
            messagebox.showinfo(
                "안내",
                "배치 결과를 직접 변환할 수 없습니다.\n"
                "배치 분석 탭에서 CSV를 내보낸 후 '직접 불러오기'를 사용해 주세요.",
            )
            return

        self._eval_predictions = preds
        self._eval_update_pred_label()

    def _eval_update_pred_label(self):
        """예측 결과 요약 라벨 갱신"""
        species_counts = {}
        for p in self._eval_predictions:
            sp = p["species"]
            species_counts[sp] = species_counts.get(sp, 0) + 1

        summary = ", ".join(f"{sp} {cnt}건" for sp, cnt in species_counts.items())
        self._eval_pred_label.set(
            f"✅ {len(self._eval_predictions)}건 로드 완료 ({summary})"
        )

    # ── 평가 실행 ──────────────────────────────────────────

    def _eval_run(self):
        """매칭 + 지표 계산 실행"""
        if not self._eval_annotations:
            messagebox.showwarning("경고", "Ground Truth(annotation)를 먼저 불러와 주세요.")
            return
        if not self._eval_predictions:
            messagebox.showwarning("경고", "예측 결과를 먼저 불러와 주세요.")
            return

        # 매칭 설정
        config = MatchingConfig(
            time_tolerance=self._eval_tolerance.get(),
            one_to_one=self._eval_one_to_one.get(),
        )

        # 매칭 실행
        self._eval_match_results, summary = match_predictions_to_annotations(
            self._eval_annotations,
            self._eval_predictions,
            config,
        )

        # 지표 계산
        threshold = self._eval_threshold.get()
        self._eval_metrics = compute_all_species_metrics(
            self._eval_match_results,
            threshold=threshold,
        )

        # 결과 표시
        self._eval_display_results(summary)

    def _eval_display_results(self, summary: dict):
        """결과 테이블 + 요약 텍스트 갱신"""
        # Treeview 초기화
        for item in self._eval_tree.get_children():
            self._eval_tree.delete(item)

        # 종별 행 추가
        for m in self._eval_metrics:
            tag = "overall" if m.species == "전체" else ""
            self._eval_tree.insert("", "end", values=(
                m.species, m.tp, m.fp, m.fn,
                f"{m.precision:.3f}", f"{m.recall:.3f}", f"{m.f1:.3f}",
            ), tags=(tag,))

        # '전체' 행 볼드 처리
        self._eval_tree.tag_configure("overall", font=("Arial", 10, "bold"))

        # 요약 텍스트
        total = summary.get("total", {})
        tp = total.get("tp", 0)
        fp = total.get("fp", 0)
        fn = total.get("fn", 0)

        text_lines = [
            "═" * 50,
            f"  매칭 결과 요약",
            f"  TP(정탐): {tp}건  |  FP(오탐): {fp}건  |  FN(미탐): {fn}건",
            "═" * 50,
            "",
        ]

        for sp, sp_data in summary.get("per_species", {}).items():
            text_lines.append(
                f"  {sp}: TP={sp_data['tp']}, FP={sp_data['fp']}, FN={sp_data['fn']}"
            )

        if HAS_SKLEARN:
            text_lines.append("")
            text_lines.append("  ℹ scikit-learn 사용 가능 — [📈 곡선 보기]로 AUROC/AUPRC 확인 가능")
        else:
            text_lines.append("")
            text_lines.append("  ⚠ scikit-learn 미설치 — AUROC/AUPRC 기능 비활성")

        self._eval_summary_text.config(state="normal")
        self._eval_summary_text.delete("1.0", "end")
        self._eval_summary_text.insert("1.0", "\n".join(text_lines))
        self._eval_summary_text.config(state="disabled")

    # ── 곡선 & 최적화 (Phase 2) ────────────────────────────

    def _eval_show_curves(self):
        """ROC/PR 곡선 표시 (Phase 2)"""
        if not HAS_SKLEARN:
            messagebox.showinfo(
                "안내",
                "AUROC/AUPRC 곡선에는 scikit-learn이 필요합니다.\n\n"
                "설치: pip install scikit-learn",
            )
            return

        if not self._eval_annotations or not self._eval_predictions:
            print(f"[DEBUG] 곡선보기: annotations={len(self._eval_annotations) if self._eval_annotations else 0}, "
                  f"predictions={len(self._eval_predictions) if self._eval_predictions else 0}")
            messagebox.showwarning("경고", "먼저 annotation과 예측 결과를 불러온 후 평가를 실행해 주세요.")
            return

        # 종별 곡선 계산
        species_set = sorted(set(a["species"] for a in self._eval_annotations))
        pred_species = sorted(set(p["species"] for p in self._eval_predictions))
        config = MatchingConfig(time_tolerance=self._eval_tolerance.get())
        curve_results = {}

        print(f"[DEBUG] 곡선보기: annotation 종={species_set}, prediction 종={pred_species}")
        print(f"[DEBUG] 곡선보기: annotation {len(self._eval_annotations)}건, prediction {len(self._eval_predictions)}건")

        # 오디오 길이 추정 (가상 음성 생성용)
        # 1순위: annotation의 최대 t_end
        # 2순위: prediction의 최대 time
        audio_duration = None
        ann_max = max((a["t_end"] for a in self._eval_annotations), default=0)
        pred_max = max((p["time"] for p in self._eval_predictions), default=0)
        if ann_max > 0 or pred_max > 0:
            audio_duration = max(ann_max, pred_max)

        print(f"[DEBUG] 곡선보기: audio_duration={audio_duration}")

        for sp in species_set:
            print(f"[DEBUG] 곡선보기: '{sp}' 계산 중...")
            sp_anns = sum(1 for a in self._eval_annotations if a["species"] == sp)
            sp_preds = sum(1 for p in self._eval_predictions if p["species"] == sp)
            print(f"[DEBUG]   annotation={sp_anns}건, prediction={sp_preds}건")

            try:
                result = compute_curve_metrics(
                    self._eval_annotations,
                    self._eval_predictions,
                    species=sp,
                    config=config,
                    audio_duration=audio_duration,
                )
                if result:
                    curve_results[sp] = result
                    print(f"[DEBUG]   → AUROC={result['auroc']}, AUPRC={result['auprc']}")
                else:
                    print(f"[DEBUG]   → 결과 없음 (None 반환)")
            except Exception as e:
                print(f"[DEBUG]   → 오류: {e}")
                import traceback
                traceback.print_exc()

        if not curve_results:
            # 진단 정보 생성
            diag = []
            for sp in species_set:
                n_ann = sum(1 for a in self._eval_annotations if a["species"] == sp)
                n_pred = sum(1 for p in self._eval_predictions if p["species"] == sp)
                diag.append(f"  • {sp}: annotation {n_ann}건, 예측 {n_pred}건")

            diag_text = "\n".join(diag)
            messagebox.showinfo(
                "안내",
                f"곡선 계산을 위한 충분한 데이터가 없습니다.\n\n"
                f"종별 현황:\n{diag_text}\n\n"
                f"가능한 원인:\n"
                f"  1) 예측 결과에 매칭/비매칭 후보가 모두 필요합니다\n"
                f"     → candidates_all.csv (전체 후보) 사용을 권장합니다\n"
                f"  2) annotation과 예측의 종명이 일치해야 합니다\n"
                f"  3) 양성(TP)과 음성(FP) 후보가 모두 있어야 합니다",
            )
            return

        self._eval_curve_data = curve_results

        # 결과를 텍스트로 표시
        text_lines = ["═" * 50, "  AUROC / AUPRC 결과", "═" * 50, ""]
        for sp, data in curve_results.items():
            text_lines.append(f"  {sp}:")
            text_lines.append(f"    AUROC = {data['auroc']}")
            text_lines.append(f"    AUPRC = {data['auprc']}")
            text_lines.append(f"    최적 F1 임계값 = {data['optimal_threshold_f1']} (F1={data['optimal_f1']})")
            text_lines.append(f"    최적 Youden 임계값 = {data['optimal_threshold_youden']}")
            text_lines.append("")

        self._eval_summary_text.config(state="normal")
        self._eval_summary_text.delete("1.0", "end")
        self._eval_summary_text.insert("1.0", "\n".join(text_lines))
        self._eval_summary_text.config(state="disabled")

        # matplotlib 있으면 곡선 그래프 표시
        try:
            from evaluation.plots import show_curves_window
            show_curves_window(self.root, curve_results)
        except ImportError:
            pass  # plots.py 아직 구현 전이면 텍스트만

    def _eval_find_optimal(self):
        """최적 cutoff 탐색"""
        if not self._eval_match_results:
            messagebox.showwarning("경고", "먼저 [▶ 평가 실행]을 해주세요.")
            return

        species_set = sorted(set(
            r.species for r in self._eval_match_results
        ))

        text_lines = ["═" * 50, "  최적 임계값 탐색 결과 (F1 기준)", "═" * 50, ""]

        for sp in species_set:
            result = find_optimal_thresholds(
                self._eval_match_results, species=sp, metric="f1",
            )
            text_lines.append(
                f"  {sp}: 최적 cutoff = {result['optimal_threshold']:.3f} "
                f"(F1 = {result['optimal_value']:.4f})"
            )

        # 전체
        overall = find_optimal_thresholds(
            self._eval_match_results, species=None, metric="f1",
        )
        text_lines.append("")
        text_lines.append(
            f"  전체: 최적 cutoff = {overall['optimal_threshold']:.3f} "
            f"(F1 = {overall['optimal_value']:.4f})"
        )

        self._eval_summary_text.config(state="normal")
        self._eval_summary_text.delete("1.0", "end")
        self._eval_summary_text.insert("1.0", "\n".join(text_lines))
        self._eval_summary_text.config(state="disabled")

    # ── 저장 ───────────────────────────────────────────────

    def _eval_export(self):
        """평가 결과 저장 (JSON + 매칭 CSV + 곡선 PNG)"""
        if not self._eval_metrics:
            messagebox.showwarning("경고", "먼저 [▶ 평가 실행]을 해주세요.")
            return

        save_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")
        if not save_dir:
            return

        try:
            saved_files = []

            # JSON 저장
            json_path = os.path.join(save_dir, "evaluation_result.json")
            export_evaluation_json(
                self._eval_metrics,
                curve_data=self._eval_curve_data,
                filepath=json_path,
            )
            saved_files.append(json_path)

            # 매칭 결과 CSV 저장
            csv_path = os.path.join(save_dir, "matched_results.csv")
            save_match_results_csv(self._eval_match_results, csv_path)
            saved_files.append(csv_path)

            # 곡선 PNG 저장 (curve_data 있을 때)
            if self._eval_curve_data:
                try:
                    from evaluation.plots import save_roc_plot, save_pr_plot
                    roc_path = os.path.join(save_dir, "roc_curve.png")
                    pr_path = os.path.join(save_dir, "pr_curve.png")
                    save_roc_plot(self._eval_curve_data, roc_path)
                    save_pr_plot(self._eval_curve_data, pr_path)
                    saved_files.append(roc_path)
                    saved_files.append(pr_path)
                except Exception:
                    pass  # matplotlib 없으면 PNG 생략

            file_list = "\n".join(f"• {f}" for f in saved_files)
            messagebox.showinfo(
                "저장 완료",
                f"평가 결과가 저장되었습니다:\n\n{file_list}",
            )
        except Exception as e:
            messagebox.showerror("저장 실패", str(e))

    # ── 오류 시각화 (Phase 3) ──────────────────────────────

    def _eval_show_error_overlay(self):
        """매칭 결과를 스펙트로그램 위에 TP/FP/FN 색상으로 시각화"""
        if not self._eval_match_results:
            messagebox.showwarning("경고", "먼저 [▶ 평가 실행]을 해주세요.")
            return

        # 파일별로 매칭 결과 그룹화
        files_in_results = sorted(set(
            r.file for r in self._eval_match_results if r.file
        ))

        if not files_in_results:
            messagebox.showinfo("안내", "파일 정보가 없어 시각화할 수 없습니다.")
            return

        # WAV 파일 선택 (또는 이미 연 파일 역참조)
        if len(files_in_results) == 1:
            target_file = files_in_results[0]
        else:
            target_file = self._eval_select_file_for_overlay(files_in_results)
            if not target_file:
                return

        # WAV 경로 확인
        wav_path = self._eval_find_wav_path(target_file)
        if not wav_path:
            wav_path = filedialog.askopenfilename(
                title=f"'{target_file}'의 WAV 파일 위치를 지정해 주세요",
                filetypes=[("WAV 파일", "*.wav")],
            )
            if not wav_path:
                return

        # 해당 파일의 매칭 결과를 detection 형식으로 변환
        detections = []
        for r in self._eval_match_results:
            if r.file != target_file:
                continue

            if r.category == "TP":
                det_time = r.pred_time if r.pred_time is not None else r.ann_t_start
                detections.append({
                    "species": f"✅ TP: {r.species}",
                    "time": det_time or 0,
                    "score": r.pred_score or 0,
                    "_eval_category": "TP",
                })
            elif r.category == "FP":
                detections.append({
                    "species": f"❌ FP: {r.species}",
                    "time": r.pred_time or 0,
                    "score": r.pred_score or 0,
                    "_eval_category": "FP",
                })
            elif r.category == "FN":
                mid_time = ((r.ann_t_start or 0) + (r.ann_t_end or 0)) / 2
                detections.append({
                    "species": f"⚠ FN: {r.species}",
                    "time": mid_time,
                    "score": 0,
                    "_eval_category": "FN",
                })

        if not detections:
            messagebox.showinfo("안내", "표시할 매칭 결과가 없습니다.")
            return

        # SpectrogramTab 열기
        try:
            from ui.spectrogram_tab import SpectrogramTab
            win = tk.Toplevel(self.root)
            win.title(f"🔍 오류 시각화 — {target_file}")
            win.geometry("1200x700")
            win._refs = []

            notebook = ttk.Notebook(win)
            notebook.pack(fill="both", expand=True)

            tab = SpectrogramTab(
                notebook, wav_path, target_file, win,
                detections=detections,
            )
            notebook.add(tab.frame, text=f" {target_file} ")

            # 범례
            legend = ttk.Frame(win)
            legend.pack(fill="x", padx=10, pady=5)
            ttk.Label(legend, text="범례:", font=("Arial", 10, "bold")).pack(side="left")
            ttk.Label(legend, text="  ✅ TP (정탐)", foreground="#4CAF50").pack(side="left", padx=5)
            ttk.Label(legend, text="  ❌ FP (오탐)", foreground="#F44336").pack(side="left", padx=5)
            ttk.Label(legend, text="  ⚠ FN (미탐)", foreground="#FF9800").pack(side="left", padx=5)

            tp_count = sum(1 for d in detections if d["_eval_category"] == "TP")
            fp_count = sum(1 for d in detections if d["_eval_category"] == "FP")
            fn_count = sum(1 for d in detections if d["_eval_category"] == "FN")
            ttk.Label(legend, text=f"  |  TP={tp_count} FP={fp_count} FN={fn_count}",
                      font=("Consolas", 9)).pack(side="left", padx=10)

            ttk.Button(win, text="닫기", command=win.destroy).pack(pady=3)
        except Exception as e:
            messagebox.showerror("오류", f"스펙트로그램 열기 실패:\n{e}")

    def _eval_select_file_for_overlay(self, files):
        """여러 파일 중 시각화할 파일 선택 대화상자"""
        win = tk.Toplevel(self.root)
        win.title("파일 선택")
        win.geometry("400x300")
        win.transient(self.root)
        win.grab_set()

        selected = [None]

        ttk.Label(win, text="시각화할 파일을 선택하세요:",
                  font=("Arial", 11)).pack(padx=10, pady=10)

        listbox = tk.Listbox(win, font=("Consolas", 10))
        for f in files:
            # 파일별 TP/FP/FN 카운트
            tp = sum(1 for r in self._eval_match_results
                     if r.file == f and r.category == "TP")
            fp = sum(1 for r in self._eval_match_results
                     if r.file == f and r.category == "FP")
            fn = sum(1 for r in self._eval_match_results
                     if r.file == f and r.category == "FN")
            listbox.insert("end", f"{f}  (TP={tp} FP={fp} FN={fn})")
        listbox.pack(fill="both", expand=True, padx=10)

        def on_ok():
            sel = listbox.curselection()
            if sel:
                selected[0] = files[sel[0]]
            win.destroy()

        ttk.Button(win, text="확인", command=on_ok).pack(pady=5)
        win.wait_window()
        return selected[0]

    def _eval_find_wav_path(self, filename):
        """파일명으로 WAV 경로 탐색"""
        # 분석 결과 디렉토리에서 검색
        search_dirs = []
        if hasattr(self, "output_dir") and self.output_dir:
            search_dirs.append(Path(self.output_dir).parent)
        if hasattr(self, "wav_path_var"):
            wav_dir = self.wav_path_var.get()
            if wav_dir:
                search_dirs.append(Path(wav_dir).parent)

        for d in search_dirs:
            candidate = d / filename
            if candidate.exists():
                return str(candidate)
            # 하위 폴더도 검색
            for sub in d.rglob(filename):
                return str(sub)
        return None
