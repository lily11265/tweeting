"""Validate R script syntax by calling parse()."""
import subprocess, sys

rscript = r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
r_file  = r"g:\birdtweet\new_analysis.R"

result = subprocess.run(
    [rscript, "-e", f'tryCatch({{ parse("{r_file}"); cat("PARSE OK\\n") }}, error=function(e) {{ cat("PARSE ERROR:", conditionMessage(e), "\\n"); quit(status=1) }})'],
    capture_output=True, text=True, encoding="utf-8", errors="replace"
)
print(result.stdout)
if result.stderr.strip():
    print("STDERR:", result.stderr[-300:])
sys.exit(result.returncode)
