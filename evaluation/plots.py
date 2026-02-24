# ============================================================
# evaluation/plots.py — ROC/PR 곡선 시각화
# matplotlib + FigureCanvasTkAgg로 tkinter 창에 표시
# matplotlib 미설치 시: graceful fallback (텍스트만)
# ============================================================

import tkinter as tk
from tkinter import ttk

# matplotlib 선택적 임포트
try:
    import matplotlib
    matplotlib.use("Agg")  # 비대화형 백엔드 (PIL 변환용)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np
    HAS_MATPLOTLIB = True

    # ── 한글 폰트 설정 ──
    import platform
    if platform.system() == "Windows":
        matplotlib.rc("font", family="Malgun Gothic")
    elif platform.system() == "Darwin":  # macOS
        matplotlib.rc("font", family="AppleGothic")
    else:  # Linux
        matplotlib.rc("font", family="NanumGothic")
    matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
except ImportError:
    HAS_MATPLOTLIB = False


# ── 색상 팔레트 ────────────────────────────────────────────

SPECIES_COLORS = [
    "#2196F3",  # 파랑
    "#4CAF50",  # 초록
    "#FF9800",  # 주황
    "#E91E63",  # 분홍
    "#9C27B0",  # 보라
    "#00BCD4",  # 청록
    "#FF5722",  # 빨강
    "#795548",  # 갈색
]


def _get_color(idx: int) -> str:
    return SPECIES_COLORS[idx % len(SPECIES_COLORS)]


# ── 메인 창 ───────────────────────────────────────────────

def show_curves_window(parent, curve_results: dict):
    """
    ROC/PR/Threshold-F1 곡선을 별도 Toplevel 창에 표시.

    Args:
        parent: 부모 tkinter 위젯
        curve_results: {species_name: curve_data_dict, ...}
    """
    if not HAS_MATPLOTLIB:
        _show_text_fallback(parent, curve_results)
        return

    win = tk.Toplevel(parent)
    win.title("📈 성능 곡선 (ROC / PR / Threshold-F1 / 분포)")
    win.geometry("1200x800")
    win.minsize(900, 600)

    # 탭 노트북
    notebook = ttk.Notebook(win)
    notebook.pack(fill="both", expand=True, padx=5, pady=5)

    # ── 탭 1: 점수 분포 (진단용) ──
    tab_dist = ttk.Frame(notebook)
    notebook.add(tab_dist, text="  📊 점수 분포  ")
    _build_distribution_tab(tab_dist, curve_results)

    # ── 탭 2: ROC 곡선 ──
    tab_roc = ttk.Frame(notebook)
    notebook.add(tab_roc, text="  ROC 곡선  ")
    _build_roc_tab(tab_roc, curve_results)

    # ── 탭 3: PR 곡선 ──
    tab_pr = ttk.Frame(notebook)
    notebook.add(tab_pr, text="  PR 곡선  ")
    _build_pr_tab(tab_pr, curve_results)

    # ── 탭 4: 임계값-F1 곡선 ──
    tab_thresh = ttk.Frame(notebook)
    notebook.add(tab_thresh, text="  임계값-성능  ")
    _build_threshold_tab(tab_thresh, curve_results)

    # ── 탭 5: 요약 테이블 ──
    tab_summary = ttk.Frame(notebook)
    notebook.add(tab_summary, text="  요약  ")
    _build_summary_tab(tab_summary, curve_results)

    # 닫기 버튼
    ttk.Button(win, text="닫기", command=win.destroy).pack(pady=5)

# ── 점수 분포 히스토그램 ───────────────────────────────────

def _build_distribution_tab(parent, curve_results: dict):
    """양성(TP)과 음성(FP) 후보의 composite score 분포 시각화."""
    species_list = list(curve_results.keys())
    n_sp = len(species_list)
    if n_sp == 0:
        return

    cols = min(n_sp, 2)
    rows = (n_sp + cols - 1) // cols

    fig = Figure(figsize=(6 * cols, 5 * rows), dpi=100)
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    for idx, sp in enumerate(species_list, 1):
        data = curve_results[sp]
        y_true = data.get("y_true", [])
        y_scores = data.get("y_scores", [])

        if not y_true or not y_scores:
            continue

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        pos_scores = y_scores[y_true == 1]
        neg_scores = y_scores[y_true == 0]

        ax = fig.add_subplot(rows, cols, idx)

        # 히스토그램 범위 통일
        all_min = min(y_scores.min(), 0)
        all_max = max(y_scores.max(), 1)
        bins = np.linspace(all_min, all_max, 30)

        ax.hist(pos_scores, bins=bins, alpha=0.6, color="#4CAF50",
                label=f"양성 (n={len(pos_scores)})", edgecolor="white", linewidth=0.5)
        ax.hist(neg_scores, bins=bins, alpha=0.6, color="#F44336",
                label=f"음성 (n={len(neg_scores)})", edgecolor="white", linewidth=0.5)

        # 최적 임계값 표시
        opt_f1_thresh = data.get("optimal_threshold_f1")
        opt_youden_thresh = data.get("optimal_threshold_youden")
        if opt_f1_thresh is not None:
            ax.axvline(opt_f1_thresh, color="#2196F3", linestyle="--",
                       linewidth=1.5, label=f"F1 임계값={opt_f1_thresh:.3f}")
        if opt_youden_thresh is not None:
            ax.axvline(opt_youden_thresh, color="#FF9800", linestyle=":",
                       linewidth=1.5, label=f"Youden 임계값={opt_youden_thresh:.3f}")

        # AUROC 표시
        auroc = data.get("auroc", 0)
        title_color = "#D32F2F" if auroc < 0.5 else "#333333"
        ax.set_title(f"{sp}  (AUROC={auroc:.4f})", fontsize=11,
                     fontweight="bold", color=title_color)
        ax.set_xlabel("Composite Score")
        ax.set_ylabel("빈도")
        ax.legend(fontsize=8, loc="upper right")

        # 진단 텍스트
        pos_mean = pos_scores.mean() if len(pos_scores) > 0 else 0
        neg_mean = neg_scores.mean() if len(neg_scores) > 0 else 0
        pos_med = np.median(pos_scores) if len(pos_scores) > 0 else 0
        neg_med = np.median(neg_scores) if len(neg_scores) > 0 else 0
        separation = pos_mean - neg_mean

        diag_lines = [
            f"양성 평균={pos_mean:.3f} 중앙={pos_med:.3f}",
            f"음성 평균={neg_mean:.3f} 중앙={neg_med:.3f}",
            f"분리도={separation:+.3f}",
        ]
        if auroc < 0.5:
            diag_lines.append("⚠ 점수 반전! 음성 > 양성")

        diag_text = "\n".join(diag_lines)
        ax.text(0.02, 0.98, diag_text, transform=ax.transAxes,
                fontsize=7.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="#999", alpha=0.9))

    canvas = FigureCanvasTkAgg(fig, parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ── ROC 곡선 ───────────────────────────────────────────────

def _build_roc_tab(parent, curve_results: dict):
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)

    for idx, (sp, data) in enumerate(curve_results.items()):
        roc = data.get("roc_curve", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auroc = data.get("auroc", "N/A")
        color = _get_color(idx)

        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{sp} (AUROC={auroc})")

        # 최적 Youden 지점 표시
        opt_thresh = data.get("optimal_threshold_youden", None)
        if opt_thresh is not None:
            thresholds = roc.get("thresholds", [])
            if thresholds:
                # 가장 가까운 threshold 인덱스 찾기 (inf 값 제외)
                arr = np.array(thresholds, dtype=float)
                finite_mask = np.isfinite(arr)
                if finite_mask.any():
                    search_arr = np.where(finite_mask, arr, np.nan)
                    best_idx = np.nanargmin(np.abs(search_arr - opt_thresh))
                else:
                    best_idx = 0
                ax.plot(fpr[best_idx], tpr[best_idx], "o",
                        color=color, markersize=10, markeredgecolor="white",
                        markeredgewidth=2, zorder=5)

    # 대각선 기준선
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="기준선 (AUC=0.5)")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("ROC 곡선", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _embed_figure(parent, fig)


# ── PR 곡선 ────────────────────────────────────────────────

def _build_pr_tab(parent, curve_results: dict):
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)

    for idx, (sp, data) in enumerate(curve_results.items()):
        pr = data.get("pr_curve", {})
        precision = pr.get("precision", [])
        recall = pr.get("recall", [])
        auprc = data.get("auprc", "N/A")
        color = _get_color(idx)

        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{sp} (AUPRC={auprc})")

        # 최적 F1 지점 표시
        opt_f1 = data.get("optimal_f1", None)
        opt_thresh = data.get("optimal_threshold_f1", None)
        if opt_f1 is not None and opt_thresh is not None:
            thresholds = pr.get("thresholds", [])
            if thresholds:
                arr = np.array(thresholds, dtype=float)
                finite_mask = np.isfinite(arr)
                if finite_mask.any():
                    search_arr = np.where(finite_mask, arr, np.nan)
                    best_idx = np.nanargmin(np.abs(search_arr - opt_thresh))
                else:
                    best_idx = 0
                # precision_recall_curve의 길이: precision/recall은 thresholds보다 1 큼
                best_idx = min(best_idx, len(precision) - 1)
                ax.plot(recall[best_idx], precision[best_idx], "o",
                        color=color, markersize=10, markeredgecolor="white",
                        markeredgewidth=2, zorder=5)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall 곡선", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _embed_figure(parent, fig)


# ── 임계값-성능 곡선 ───────────────────────────────────────

def _build_threshold_tab(parent, curve_results: dict):
    n_species = len(curve_results)
    cols = min(n_species, 3)
    rows = (n_species + cols - 1) // cols

    fig = Figure(figsize=(5 * cols, 4 * rows), dpi=100)

    for idx, (sp, data) in enumerate(curve_results.items()):
        ax = fig.add_subplot(rows, cols, idx + 1)

        pr = data.get("pr_curve", {})
        precision_arr = pr.get("precision", [])
        recall_arr = pr.get("recall", [])
        thresholds = pr.get("thresholds", [])

        if not thresholds:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
            ax.set_title(sp)
            continue

        t_arr = np.array(thresholds)
        p_arr = np.array(precision_arr[:len(thresholds)])
        r_arr = np.array(recall_arr[:len(thresholds)])

        # F1 계산 (divide-by-zero 방지)
        denom = p_arr + r_arr
        f1_arr = np.zeros_like(denom)
        np.divide(2 * p_arr * r_arr, denom, out=f1_arr, where=denom > 0)

        ax.plot(t_arr, p_arr, color="#2196F3", linewidth=1.5, label="Precision")
        ax.plot(t_arr, r_arr, color="#FF9800", linewidth=1.5, label="Recall")
        ax.plot(t_arr, f1_arr, color="#4CAF50", linewidth=2, label="F1")

        # 최적 F1 지점
        opt_thresh = data.get("optimal_threshold_f1", None)
        if opt_thresh is not None:
            ax.axvline(x=opt_thresh, color="#E91E63", linestyle="--",
                       linewidth=1.5, alpha=0.7,
                       label=f"최적={opt_thresh:.3f}")

        ax.set_xlabel("임계값")
        ax.set_ylabel("점수")
        ax.set_title(f"{sp}", fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _embed_figure(parent, fig)


# ── 요약 테이블 탭 ─────────────────────────────────────────

def _build_summary_tab(parent, curve_results: dict):
    cols = ("species", "auroc", "auprc", "opt_f1_thresh", "opt_f1",
            "opt_youden_thresh")
    tree = ttk.Treeview(parent, columns=cols, show="headings", height=10)

    headings = {
        "species": ("종명", 120),
        "auroc": ("AUROC", 80),
        "auprc": ("AUPRC", 80),
        "opt_f1_thresh": ("최적 F1 임계값", 120),
        "opt_f1": ("최적 F1", 80),
        "opt_youden_thresh": ("Youden 임계값", 120),
    }
    for col_id, (heading, width) in headings.items():
        tree.heading(col_id, text=heading)
        tree.column(col_id, width=width, anchor="center")

    for sp, data in curve_results.items():
        tree.insert("", "end", values=(
            sp,
            data.get("auroc", "N/A"),
            data.get("auprc", "N/A"),
            data.get("optimal_threshold_f1", "N/A"),
            data.get("optimal_f1", "N/A"),
            data.get("optimal_threshold_youden", "N/A"),
        ))

    tree.pack(fill="both", expand=True, padx=10, pady=10)


# ── 유틸리티 ──────────────────────────────────────────────

def _embed_figure(parent, fig):
    """matplotlib Figure를 tkinter Frame에 임베드."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def _show_text_fallback(parent, curve_results: dict):
    """matplotlib 미설치 시 텍스트 요약만 표시."""
    from tkinter import scrolledtext

    win = tk.Toplevel(parent)
    win.title("📈 성능 곡선 (텍스트)")
    win.geometry("600x400")

    text = scrolledtext.ScrolledText(win, font=("Consolas", 10))
    text.pack(fill="both", expand=True, padx=10, pady=10)

    lines = [
        "⚠ matplotlib이 설치되지 않아 그래프를 표시할 수 없습니다.",
        "  pip install matplotlib 로 설치 후 다시 시도해 주세요.",
        "",
        "=" * 50,
        "  AUROC / AUPRC 요약 (텍스트)",
        "=" * 50,
        "",
    ]

    for sp, data in curve_results.items():
        lines.append(f"  {sp}:")
        lines.append(f"    AUROC = {data.get('auroc', 'N/A')}")
        lines.append(f"    AUPRC = {data.get('auprc', 'N/A')}")
        lines.append(f"    최적 F1 임계값 = {data.get('optimal_threshold_f1', 'N/A')} "
                     f"(F1={data.get('optimal_f1', 'N/A')})")
        lines.append(f"    Youden 임계값 = {data.get('optimal_threshold_youden', 'N/A')}")
        lines.append("")

    text.insert("1.0", "\n".join(lines))
    text.config(state="disabled")

    ttk.Button(win, text="닫기", command=win.destroy).pack(pady=5)


# ── 단독 PNG 저장 (유틸리티) ───────────────────────────────

def save_roc_plot(curve_results: dict, filepath: str):
    """ROC 곡선을 PNG로 저장."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (sp, data) in enumerate(curve_results.items()):
        roc = data.get("roc_curve", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auroc = data.get("auroc", "N/A")
        ax.plot(fpr, tpr, color=_get_color(idx), linewidth=2,
                label=f"{sp} (AUROC={auroc})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pr_plot(curve_results: dict, filepath: str):
    """PR 곡선을 PNG로 저장."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (sp, data) in enumerate(curve_results.items()):
        pr = data.get("pr_curve", {})
        precision = pr.get("precision", [])
        recall = pr.get("recall", [])
        auprc = data.get("auprc", "N/A")
        ax.plot(recall, precision, color=_get_color(idx), linewidth=2,
                label=f"{sp} (AUPRC={auprc})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
