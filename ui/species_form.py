# ============================================================
# ui/species_form.py — 종 추가 폼 공통 팩토리 (C5: 멀티 템플릿)
# ============================================================
"""
_add_species()와 _batch_add_species()의 중복 코드를 제거하기 위한
공통 위젯 팩토리 모듈.

create_species_form()  — 분석·배치 탭용 (cutoff + 종별 가중치 포함)

C5: 동일 종에 여러 울음 유형(경계음, 구애음 등)을 별도 템플릿으로 등록 가능.
     종 단위: name, cutoff, weights
     템플릿 단위: wav_path, t_start, t_end, f_low, f_high, label
"""

import tkinter as tk
from tkinter import ttk, filedialog

from audio.sanitizer import AUDIO_FILETYPES


def _create_template_row(
    parent_frame,
    index,
    on_template_select=None,
) -> dict:
    """단일 템플릿 행 위젯 생성.

    Returns:
        딕셔너리 — frame, path, t_start, t_end, f_low, f_high, label
    """
    tfrm = ttk.Frame(parent_frame, padding=(0, 1))
    tfrm.pack(fill="x")

    # 라벨(유형명)
    var_label = tk.StringVar(value=f"type{index}")
    ttk.Entry(tfrm, textvariable=var_label, width=8).pack(side="left", padx=(0, 4))

    # 음원 파일
    var_path = tk.StringVar()
    ttk.Entry(tfrm, textvariable=var_path, width=30).pack(side="left", padx=(0, 2), fill="x", expand=True)
    ttk.Button(
        tfrm, text="선택",
        command=lambda v=var_path: v.set(
            filedialog.askopenfilename(filetypes=AUDIO_FILETYPES)
        ),
    ).pack(side="left", padx=(0, 6))

    # 시간/주파수
    var_t_start = tk.DoubleVar(value=0.0)
    var_t_end = tk.DoubleVar(value=3.0)
    var_f_low = tk.DoubleVar(value=0)
    var_f_high = tk.DoubleVar(value=4000)

    for lbl, var, w in [
        ("시작:", var_t_start, 5), ("종료:", var_t_end, 5),
        ("Lo:", var_f_low, 6), ("Hi:", var_f_high, 6),
    ]:
        ttk.Label(tfrm, text=lbl).pack(side="left")
        ttk.Spinbox(tfrm, textvariable=var, from_=0,
                     to=22050 if ":" in lbl and lbl.startswith(("L", "H")) else 9999,
                     width=w, increment=0.1 if lbl.startswith(("시", "종")) else 100,
                     ).pack(side="left", padx=(0, 3))

    # 📊 구간 선택 버튼
    if on_template_select is not None:
        ttk.Button(
            tfrm, text="📊",
            command=lambda: on_template_select(
                var_path, var_t_start, var_t_end, var_f_low, var_f_high
            ),
        ).pack(side="right", padx=(2, 0))

    return {
        "frame":   tfrm,
        "path":    var_path,
        "t_start": var_t_start,
        "t_end":   var_t_end,
        "f_low":   var_f_low,
        "f_high":  var_f_high,
        "label":   var_label,
    }


def create_species_form(
    parent_container,
    index,
    on_template_select=None,
    include_cutoff=True,
    include_weights=True,
) -> dict:
    """
    종 추가 폼을 생성하고 변수 딕셔너리를 반환합니다.

    Args:
        parent_container: 폼이 추가될 부모 ttk.Frame
        index: 종 번호 (1-based)
        on_template_select: 구간 선택 콜백 fn(var_path, var_t_start, var_t_end, var_f_low, var_f_high)
        include_cutoff: cutoff 스핀박스 포함 여부
        include_weights: 종별 가중치 UI 포함 여부

    Returns:
        딕셔너리 — frame, name, templates[], [cutoff], [use_custom_weights, sp_weights]
    """
    frm = ttk.LabelFrame(parent_container, text=f" 종 {index} ", padding=5)
    frm.pack(fill="x", pady=2, padx=5)

    # ── 종 이름 + cutoff (종 단위) ──
    row0 = ttk.Frame(frm)
    row0.pack(fill="x")

    ttk.Label(row0, text="종 이름:").pack(side="left")
    var_name = tk.StringVar(value=f"종{index}")
    ttk.Entry(row0, textvariable=var_name, width=15).pack(side="left", padx=5)

    var_cutoff = None
    if include_cutoff:
        ttk.Label(row0, text="  cutoff:").pack(side="left")
        var_cutoff = tk.DoubleVar(value=0.4)
        ttk.Spinbox(row0, textvariable=var_cutoff, from_=0.0, to=1.0,
                    width=5, increment=0.05).pack(side="left")

    # ── 앙상블 전략 (멀티 템플릿 시) ──
    var_ensemble = tk.StringVar(value="max")
    ttk.Label(row0, text="  앙상블:").pack(side="left")
    ttk.Combobox(row0, textvariable=var_ensemble,
                 values=["max", "mean", "weighted_max"],
                 width=12, state="readonly").pack(side="left", padx=2)

    # ── C5: 멀티 템플릿 영역 ──
    tmpl_container = ttk.LabelFrame(frm, text="템플릿 목록", padding=2)
    tmpl_container.pack(fill="x", pady=(3, 0))

    # 헤더
    hdr = ttk.Frame(tmpl_container)
    hdr.pack(fill="x")
    for txt, w in [("유형", 8), ("음원 파일", 30), ("", 4),
                    ("시작", 5), ("종료", 5), ("Lo Hz", 6), ("Hi Hz", 6)]:
        if txt:
            ttk.Label(hdr, text=txt, font=("", 7)).pack(side="left", padx=(0, 2))

    templates = []  # 템플릿 딕셔너리 리스트

    def add_template():
        idx = len(templates) + 1
        tmpl = _create_template_row(tmpl_container, idx, on_template_select)
        templates.append(tmpl)

    def remove_template():
        if len(templates) > 1:
            tmpl = templates.pop()
            tmpl["frame"].destroy()

    # 버튼 행
    btn_row = ttk.Frame(frm)
    btn_row.pack(fill="x", pady=(2, 0))
    ttk.Button(btn_row, text="➕ 템플릿 추가", command=add_template).pack(side="left")
    ttk.Button(btn_row, text="➖ 마지막 제거", command=remove_template).pack(side="left", padx=4)

    # 기본 1개 템플릿 생성
    add_template()

    # ── 종별 가중치 (접이식 — 분석·배치 탭 전용) ──
    var_use_custom_w = None
    sp_weight_vars = {}

    if include_weights:
        var_use_custom_w = tk.BooleanVar(value=False)
        w_frame = ttk.Frame(frm)

        def toggle_weights(wf=w_frame, uw=var_use_custom_w):
            if uw.get():
                wf.pack(fill="x", pady=(3, 0))
            else:
                wf.pack_forget()

        row_wtoggle = ttk.Frame(frm)
        row_wtoggle.pack(fill="x", pady=(3, 0))
        ttk.Checkbutton(
            row_wtoggle, text="⚙ 종별 가중치 설정",
            variable=var_use_custom_w, command=toggle_weights,
        ).pack(side="left")

        w_defs = [
            ("cor_score", "corM", 0.18),
            ("mfcc_score", "MFCC", 0.18),
            ("dtw_freq", "freq", 0.13),
            ("dtw_env", "env", 0.08),
            ("band_energy", "band", 0.13),
            ("harmonic_ratio", "HR", 0.18),
            ("snr", "SNR", 0.12),
        ]
        for key, label, default in w_defs:
            ttk.Label(w_frame, text=f"{label}:").pack(side="left", padx=(5, 0))
            wvar = tk.DoubleVar(value=default)
            ttk.Spinbox(w_frame, textvariable=wvar, from_=0.0, to=1.0,
                        width=5, increment=0.05).pack(side="left", padx=(1, 3))
            sp_weight_vars[key] = wvar

    # ── 결과 딕셔너리 조립 ──
    result = {
        "frame":     frm,
        "name":      var_name,
        "templates": templates,   # C5: 템플릿 리스트 (동적)
        "ensemble":  var_ensemble,  # 앙상블 전략
    }
    if include_cutoff:
        result["cutoff"] = var_cutoff
    if include_weights:
        result["use_custom_weights"] = var_use_custom_w
        result["sp_weights"] = sp_weight_vars

    return result
