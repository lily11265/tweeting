# ============================================================
# ConverterTabMixin — MP3 → WAV 변환기 탭
# ============================================================

import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

from audio.sanitizer import convert_mp3_to_wav


class ConverterTabMixin:
    """MP3 → WAV 변환기 탭 메서드 모음 (Mixin)"""

    # ----------------------------------------
    # 탭: MP3 → WAV 변환기
    # ----------------------------------------
    def _build_converter_tab(self, parent):
        # 안내
        frm_info = ttk.Frame(parent, padding=10)
        frm_info.pack(fill="x")
        ttk.Label(frm_info, text="MP3 파일을 WAV로 변환합니다.",
                  font=("Arial", 12, "bold")).pack(anchor="w")
        ttk.Label(frm_info, text="여러 파일을 한번에 선택하여 일괄 변환할 수 있습니다.",
                  foreground="gray").pack(anchor="w", pady=(2, 0))

        # 입력 파일
        frm_input = ttk.LabelFrame(parent, text=" MP3 파일 선택 ", padding=10)
        frm_input.pack(fill="x", padx=10, pady=5)

        btn_row = ttk.Frame(frm_input)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="📂 파일 추가", command=self._conv_add_files).pack(side="left")
        ttk.Button(btn_row, text="📁 폴더 전체 추가", command=self._conv_add_folder).pack(side="left", padx=5)
        ttk.Button(btn_row, text="🗑 목록 비우기", command=self._conv_clear_files).pack(side="left", padx=5)

        self.conv_file_listbox = tk.Listbox(frm_input, height=8, font=("Consolas", 9))
        conv_scroll = ttk.Scrollbar(frm_input, orient="vertical", command=self.conv_file_listbox.yview)
        self.conv_file_listbox.configure(yscrollcommand=conv_scroll.set)
        self.conv_file_listbox.pack(side="left", fill="both", expand=True, pady=(5, 0))
        conv_scroll.pack(side="right", fill="y", pady=(5, 0))

        # 출력 폴더
        frm_output = ttk.LabelFrame(parent, text=" 저장 위치 ", padding=10)
        frm_output.pack(fill="x", padx=10, pady=5)

        self.var_conv_output_dir = tk.StringVar()
        row_out = ttk.Frame(frm_output)
        row_out.pack(fill="x")
        ttk.Entry(row_out, textvariable=self.var_conv_output_dir, width=70).pack(side="left", fill="x", expand=True)
        ttk.Button(row_out, text="폴더 선택", command=self._conv_select_output).pack(side="right", padx=(10, 0))

        # 옵션
        frm_opt = ttk.Frame(frm_output)
        frm_opt.pack(fill="x", pady=(5, 0))
        self.var_conv_same_dir = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_opt, text="원본과 같은 폴더에 저장 (체크 해제 시 위 폴더에 저장)",
                        variable=self.var_conv_same_dir).pack(anchor="w")

        # 변환 실행
        frm_run = ttk.Frame(parent, padding=10)
        frm_run.pack(fill="x", padx=10)

        self.btn_convert = ttk.Button(frm_run, text="🔄 변환 실행", command=self._conv_run)
        self.btn_convert.pack(side="left")

        self.conv_progress = ttk.Progressbar(frm_run, mode="determinate", length=300)
        self.conv_progress.pack(side="left", padx=10, fill="x", expand=True)

        self.conv_status = tk.StringVar(value="대기 중")
        ttk.Label(frm_run, textvariable=self.conv_status).pack(side="right")

        # 변환 로그
        frm_log = ttk.LabelFrame(parent, text=" 변환 로그 ", padding=5)
        frm_log.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.conv_log = scrolledtext.ScrolledText(frm_log, height=10, font=("Consolas", 9))
        self.conv_log.pack(fill="both", expand=True)

        # 내부 파일 목록
        self._conv_files = []

    def _conv_add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("MP3 파일", "*.mp3")])
        for f in files:
            if f not in self._conv_files:
                self._conv_files.append(f)
                self.conv_file_listbox.insert("end", f)

    def _conv_add_folder(self):
        """폴더 내 모든 MP3 파일 추가"""
        folder = filedialog.askdirectory()
        if folder:
            count = 0
            for f in sorted(Path(folder).glob("*.mp3")):
                fp = str(f)
                if fp not in self._conv_files:
                    self._conv_files.append(fp)
                    self.conv_file_listbox.insert("end", fp)
                    count += 1
            if count == 0:
                messagebox.showinfo("안내", "해당 폴더에 MP3 파일이 없습니다.")
            else:
                self.conv_status.set(f"{count}개 파일 추가됨")

    def _conv_clear_files(self):
        self._conv_files.clear()
        self.conv_file_listbox.delete(0, "end")

    def _conv_select_output(self):
        d = filedialog.askdirectory()
        if d:
            self.var_conv_output_dir.set(d)

    def _conv_run(self):
        if not self._conv_files:
            messagebox.showwarning("경고", "변환할 MP3 파일을 추가해 주세요.")
            return

        if not self._HAS_PYDUB:
            # ffmpeg 직접 사용 가능한지 체크
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            except FileNotFoundError:
                messagebox.showerror("오류",
                    "MP3 변환에 pydub 또는 ffmpeg가 필요합니다.\n\n"
                    "설치 방법:\n"
                    "  pip install pydub\n"
                    "  + ffmpeg 설치 (https://ffmpeg.org)")
                return

        self.btn_convert.config(state="disabled")
        self.conv_log.delete("1.0", "end")
        self.conv_progress["value"] = 0
        self.conv_progress["maximum"] = len(self._conv_files)

        thread = threading.Thread(target=self._conv_process, daemon=True)
        thread.start()

    def _conv_process(self):
        total = len(self._conv_files)
        success = 0
        fail = 0

        for i, mp3_path in enumerate(self._conv_files):
            mp3 = Path(mp3_path)
            self.root.after(0, self._conv_log_msg, f"[{i+1}/{total}] 변환 중: {mp3.name}")
            self.root.after(0, self._conv_update_status, f"변환 중... ({i+1}/{total})")

            try:
                # 저장 경로 결정
                if self.var_conv_same_dir.get():
                    wav_path = mp3.with_suffix(".wav")
                else:
                    out_dir = self.var_conv_output_dir.get().strip()
                    if not out_dir:
                        out_dir = str(mp3.parent)
                    wav_path = Path(out_dir) / (mp3.stem + ".wav")

                convert_mp3_to_wav(mp3_path, wav_path)
                self.root.after(0, self._conv_log_msg, f"  ✅ 완료 → {wav_path}\n")
                success += 1
            except Exception as e:
                self.root.after(0, self._conv_log_msg, f"  ❌ 실패: {e}\n")
                fail += 1

            self.root.after(0, self._conv_update_progress, i + 1)

        summary = f"변환 완료! (성공: {success}건, 실패: {fail}건)"
        self.root.after(0, self._conv_log_msg, f"\n{'='*40}\n{summary}")
        self.root.after(0, self._conv_update_status, summary)
        self.root.after(0, lambda: self.btn_convert.config(state="normal"))

    def _conv_log_msg(self, msg):
        self.conv_log.insert("end", msg + "\n")
        self.conv_log.see("end")

    def _conv_update_progress(self, value):
        self.conv_progress["value"] = value

    def _conv_update_status(self, text):
        self.conv_status.set(text)
